#include "hip/hip_runtime.h"
#include "kronmult4_batched.hpp"
#include "kernel_context.hpp"

void kronmult4_batched(
                       int const n,
                       double const Aarray_[],
                       double Xarray_[],
                       double Yarray_[],
                       double Warray_[],
                       int const batchCount )
{
#ifdef USE_GPU
        int constexpr warpsize = WARPSIZE;
        int constexpr nwarps = 4;
        int constexpr nthreads = nwarps * warpsize;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult4_batched<double>), dim3(batchCount), dim3(nthreads ), 
                           get_max_shmem<double>(n), 0,  n, Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#else
        kronmult4_batched<double>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#endif

}


