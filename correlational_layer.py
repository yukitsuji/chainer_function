import numpy

from chainer import cuda
from chainer import function_node
from chainer.utils import type_check
import string

if cuda.available:
    import cupy as cp
    @cp.util.memoize(for_each_device=True)
    def _load_kernel(kernel_name, code, options=()):
        assert isinstance(options, tuple)
        kernel_code = cp.cuda.compile_with_cache(code, options=options)
        return kernel_code.get_function(kernel_name)

_corr_foward_code = '''
extern "C" __global__
void corr_foward(
  const float *bottom0, const float *bottom1, float *top)
{
  __shared__ float patch_data[${bottomchannels}];

  const int x1 = blockIdx.x;
  const int y1 = blockIdx.y;
  const int item = blockIdx.z;
  const int ch_off = threadIdx.x;

  // Load 3D patch into shared shared memory
  for(int ch = ch_off; ch < ${bottomchannels}; ch += ${threads_per_block}) { // CHANNELS
      int idx1 = ((item * ${topheight} + y1) * ${topwidth} + x1) * ${bottomchannels} + ch;
      patch_data[ch] = bottom0[idx1];
  }

  __syncthreads();

  __shared__ float sum[${threads_per_block}];

  // Compute inner product at each threads
  for(int top_channel = 0; top_channel < ${topchannels}; top_channel++) {
    sum[ch_off] = 0;
    int s2o = top_channel % ${topchannels};
    int x2 = x1 + s2o; //top_channel;
    for(int ch = ch_off; ch < ${bottomchannels}; ch += ${threads_per_block}) { // CHANNELS
      int idx2 = ((item * ${bottomheight} + y1) * ${bottomwidth} + x2) * ${bottomchannels} + ch;
      sum[ch_off] += patch_data[ch] * bottom1[idx2];
    }

    __syncthreads();

    // sum over all threads
    if(ch_off == 0) {
        float total_sum = 0;
        for(int idx = 0; idx < ${threads_per_block}; idx++) {
            total_sum += sum[idx];
        }
        const int index = ((top_channel*${topheight} + y1)*${topwidth})+x1;
        top[index + item*${topcount}] = total_sum / (float)${bottomchannels};
    }
  }
}
'''

class Correlation(function_node.FunctionNode):

    def __init__(self, max_displacement=40):
        self.max_displacement = max_displacement

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        left_x, right_x = inputs
        self.retain_inputs((0, 1))

        xp = cuda.get_array_module(left_x)
        B, C, H, W = left_x.shape

        top_ch = 2 * self.max_displacement + 1
        top_count = top_ch * H * W

        left_x = xp.transpose(left_x, (0, 2, 3, 1))
        right_x = xp.pad(right_x, ((0, 0), (0, 0), (0, 0), (self.max_displacement, self.max_displacement)), mode='constant')
        right_x = xp.transpose(right_x, (0, 2, 3, 1))
        bottomwidth = W + 2 * self.max_displacement

        threads_per_block = 32
        blocks = (W, H, B)
        threads = (threads_per_block, 1, 1)
        shared_mem = C
        #left_x = cp.ascontiguousarray(left_x, dtype=numpy.float32)
        right_x = cp.ascontiguousarray(right_x, dtype=numpy.float32)
        top_data = cuda.cupy.empty((B, top_ch, H, W), dtype=numpy.float32)
        kern = _load_kernel('corr_foward',
                            string.Template(_corr_foward_code).substitute(
                                topwidth=W,
                                topheight=H,
                                topchannels=top_ch,
                                topcount=top_count,
                                max_displacement=self.max_displacement,
                                bottomwidth=bottomwidth,
                                bottomheight=H,
                                bottomchannels=C,
                                threads_per_block=threads_per_block))
        kern(blocks, threads, shared_mem=shared_mem,
             args=(left_x, right_x, top_data))
        return top_data,

    def backward(self, indexes, grad_outputs):
        left_x, right_x = self.get_retained_inputs()
        return CorrelationGrad(self.max_displacement).apply((left_x, right_x, grad_outputs[0]))


class CorrelationGrad(function_node.FunctionNode):

    def __init__(self, max_displacement):
        self.max_displacement = max_displacement

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        left_x, right_x, gy = inputs
        right_x = cuda.cupy.pad(right_x, ((0, 0), (0, 0), (0, 0), (self.max_displacement, self.max_displacement)), mode='constant')
        right_x = cp.ascontiguousarray(right_x, dtype=numpy.float32)
        xp = cuda.get_array_module(left_x)
        B, C, H, W = left_x.shape

        top_ch = 2 * self.max_displacement + 1
        top_count = top_ch * H * W
        gx_left = cuda.cupy.empty((B, C, H, W), dtype=numpy.float32)
        gx_right = cuda.cupy.empty((B, C, H, W), dtype=numpy.float32)
        cuda.cupy.ElementwiseKernel(
            'raw float32 gy, raw float32 right_x',
            'float32 top_data',
            string.Template('''
            const int w = i % ${width};
            const int h = (i / ${width}) % ${height};
            const int c = (i / ${width} / ${height}) % ${bottomchannels};
            const int item = i / ${width} / ${height} / ${bottomchannels};
            float sum = 0;
            for (int ch=0; ch < ${topchannels}; ch++) {
                int idx1 = ((item * ${topchannels} + ch) * ${height} + h) * ${width} + w;
                int idx2 = ((item * ${bottomchannels} + c) * ${height} + h) * (${xwidth}) + w+ch;
                sum += gy[idx1] * right_x[idx2];  
            }
            top_data = sum / ${bottomchannels};
            ''').substitute(
                    width=W,
                    height=H,
                    bottomchannels=C,
                    topchannels=top_ch,
                    xwidth=int(right_x.shape[3]),
                    max_displacement=self.max_displacement,
                ), 'corr_backward'
        )(gy, right_x, gx_left)

        cuda.cupy.ElementwiseKernel(
            'raw float32 gy, raw float32 left_x',
            'float32 top_data',
            string.Template('''
            const int w = i % ${width};
            const int h = (i / ${width}) % ${height};
            const int c = (i / ${width} / ${height}) % ${bottomchannels};
            const int item = i / ${width} / ${height} / ${bottomchannels};
            float sum = 0;
            int idx2 = ((item * ${bottomchannels} + c) * ${height} + h) * ${width} + w;
            float tmp_left_x = left_x[idx2];
            for (int ch=0; ch < ${topchannels}; ch++) {
                int idx1 = ((item * ${topchannels} + ch) * ${height} + h) * ${width} + w;
                sum += gy[idx1] * tmp_left_x;
            }
            top_data = sum / ${bottomchannels};
            ''').substitute(
                    width=W,
                    height=H,
                    bottomchannels=C,
                    topchannels=top_ch,
                    xwidth=int(right_x.shape[3]),
                    max_displacement=self.max_displacement,
                ), 'corr_backward'
        )(gy, left_x, gx_right)
        return gx_left, gx_right

    def backward(self, indexes, grad_outputs):
        return Correlation((self.out_H, self.out_W)).apply(grad_outputs)


def correlational_layer(x, y, max_displacement=40):
    return Correlation(max_displacement).apply((x, y))[0]
