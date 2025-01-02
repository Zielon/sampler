#pragma once

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
        CHECK_CUDA(x); \
        CHECK_CONTIGUOUS(x)

#define CUDA_CHECK_THROW(x)                                 \
  do                                                        \
  {                                                         \
    cudaError_t result = x;                                 \
    if (result != cudaSuccess)                              \
      throw std::runtime_error{cudaGetErrorString(result)}; \
  } while (0)
