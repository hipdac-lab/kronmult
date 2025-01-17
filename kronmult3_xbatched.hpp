#ifndef KRONMULT3_XBATCHED_HPP
#define KRONMULT3_XBATCHED_HPP 1


#include "kronmult_xbatched.hpp"




// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A3(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult3_xbatched(
                       int const n,
                       T const * const Aarray_[],
		       int const lda,
                       T* pX_[],
                       T* pY_[],
                       T* pW_[],
                       int const batchCount,
		       int const subbatchCount = 0)
//
// conceptual shape of Aarray is  (ndim,batchCount)
//
// pX_[] is array of pointers to X[], each of size n^ndim
// pY_[] is array of pointers to Y[], each of size n^ndim
// pW_[] is array of pointers to Z[], each of size n^ndim
//
// Y is the output
// X is the input (but may be modified)
// W is workspace
//
//
{
#ifdef USE_GPU
        extern __shared__ char shmem[];
#else
        char* shmem = NULL;
#endif
        int constexpr ndim = 3;
	kronmult_xbatched<T,ndim>(
                n, Aarray_, lda, pX_, pY_, pW_, batchCount, subbatchCount, shmem );
}



#endif
