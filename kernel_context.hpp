#ifndef KERNEL_CONTEXT_HPP
#define KERNEL_CONTEXT_HPP 1

#ifdef USE_GPU
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#define GLOBAL_FUNCTION  __global__ 
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__

#ifdef __NVCC__
#define WARPSIZE 32
#else
#define WARPSIZE 64
#endif

#else
#define GLOBAL_FUNCTION
#define SYNCTHREADS 
#define SHARED_MEMORY 
#define DEVICE_FUNCTION
#define HOST_FUNCTION
#endif


template <typename T>
HOST_FUNCTION DEVICE_FUNCTION size_t get_max_shmem(int max_dim);

#endif
