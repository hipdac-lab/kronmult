#include "hip/hip_runtime.h"
#include "kronmult3_xbatched.hpp"
#include "kernel_context.hpp"

void kronmult3_xbatched(
                       int const n,
                       double const * const Aarray_[],
		       int const lda,
                       double* Xarray_[],
                       double* Yarray_[],
                       double* Warray_[],
                       int const batchCount,
                       int const subbatchCount = 0 )
{
        auto max = [](int const x,
                      int const y) -> int {
                return( (x > y) ? x : y );
        };

#ifdef USE_GPU
        int constexpr warpsize = WARPSIZE;
        int constexpr nwarps = 4;
        int constexpr nthreads = nwarps * warpsize;

        if (subbatchCount == 0) {
                subbatchCount = max(1, batchCount/2);
        };
        int const nblocks = subbatchCount;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult3_xbatched<double>), dim3(nblocks), dim3(nthreads ), 
                           get_max_shmem<double>(n), 0,  n, 
                           Aarray_, lda,
                           Xarray_, Yarray_, Warray_, batchCount,subbatchCount);
#else
        kronmult3_xbatched<double>( n, 
           Aarray_, lda,
	   Xarray_, Yarray_, Warray_, batchCount,subbatchCount);
#endif

}


