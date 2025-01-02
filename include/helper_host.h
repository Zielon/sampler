#pragma once

#include <stdexcept>
#include <cuda_runtime.h>

constexpr uint32_t BATCH_SIZE_GRANULARITY = 256;
constexpr uint32_t N_THREADS_LINEAR = 512;
constexpr uint32_t WARP_SIZE = 32;

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor)
{
  return (val + divisor - 1) / divisor;
}

template <typename T>
__host__ __device__ T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

template <typename T>
constexpr __host__ __device__ uint32_t n_blocks_linear(T n_elements, uint32_t n_threads = N_THREADS_LINEAR)
{
  return (uint32_t)div_round_up(n_elements, (T)n_threads);
}

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
template <typename K, typename T, typename... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types... args)
{
  if (n_elements <= 0)
  {
    return;
  }
  kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);
}
#endif
