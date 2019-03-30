#ifndef FORWARD_WARP_H
#define FORWARD_WARP_H

// Define GridSamplerInterpolation
namespace at { namespace native { namespace detail {
  enum class GridSamplerInterpolation {Bilinear, Nearest};
  enum class GridSamplerPadding {Zeros, Border, Reflection};
}}}

// Define CUDA_NUM_THREAS and GET_BLOCKS
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Define CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#endif
