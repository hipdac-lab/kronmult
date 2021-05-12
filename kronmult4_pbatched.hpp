#ifndef KRONMULT4_PBATCHED_HPP
#define KRONMULT4_PBATCHED_HPP 1

#include "kroncommon.hpp"

#include "kronmult4.hpp"



// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A4(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult4_pbatched(
                       int const n,
                       T const Aarray_[],
                       T* pX_[],
                       T* pY_[],
                       T* pW_[],
                       int const batchCount)
//
// conceptual shape of Aarray is  (n,n,6,batchCount)
//
// pX_[] is array of pointers to X[], each of size n^4
// pY_[] is array of pointers to Y[], each of size n^4
// pW_[] is array of pointers to Z[], each of size n^4
//
// Y is the output
// X is the input (but may be modified)
// W is workspace
//
//
{
#ifdef USE_GPU
        // -------------------------------------------
        // note 1-based matlab convention for indexing
        // -------------------------------------------
        int const iz_start = blockIdx.x + 1;
        int const iz_size =  gridDim.x;
        assert( gridDim.y == 1);
        assert( gridDim.z == 1);
        extern __shared__ char shmem[];
#else
        int const iz_start = 1;
        int const iz_size = 1;
        char* shmem = NULL;
#endif


        auto Aarray = [=] (int const i1,
                           int const i2,
                           int const i3,
                           int const i4) -> T const & {
                return( Aarray_[ indx4f(i1,i2,i3,i4, n,n,4 ) ] );
        };



#ifndef USE_GPU
#ifdef _OPENMP
#pragma omp parallel for
#endif
#endif
        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T* const Xp =  pX_[ (ibatch-1) ];
                T* const Yp =  pY_[ (ibatch-1) ];
                T* const Wp =  pW_[ (ibatch-1) ];

                T const * const A1 = &(Aarray(1,1,1,ibatch));
                T const * const A2 = &(Aarray(1,1,2,ibatch));
                T const * const A3 = &(Aarray(1,1,3,ibatch));
                T const * const A4 = &(Aarray(1,1,4,ibatch));
                int const nvec = 1;
                kronmult4( n, nvec, A1,A2,A3,A4,  Xp, Yp, Wp, 0, shmem );
        };

}



#endif
