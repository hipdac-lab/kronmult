#ifndef KGEMM_NN_HPP
#define KGEMM_NN_HPP 1

#include "kroncommon.hpp"

//  -----------------------
//  NotransA and TransB case
//  C = alpha*A*(B) + beta *C
//  -----------------------
//

#ifdef USE_GPU
#ifdef PROFILE
extern __device__ int* sigCount, * totalCount;
extern __device__ long long sigTime, totalTime;

extern __device__ long long sigTemp, totalTemp;
#endif
#endif

// Second version of kgemm_nn routine -- uses shared memory and a TSM2L-based approach
template<typename T,typename Tc, int t1, int t2, int t3>
DEVICE_FUNCTION
void kgemm_nn_v2(volatile T* currB_, int const ldCurrB, int const padCurrB,
                 int const m, int const n, int const k,
                 T const alpha_in,
                 T const * const A_, int const ldA,
                 T const * const B_, int const ldB,
                 T const beta_in,
                 T * C_, int const ldC) 
{

#ifdef USE_LAMBDA
        auto A = [=] (int const ia,
                      int const ja) -> T const & {
                return( A_[ indx2z(ia,ja,ldA) ] );
        };

        auto B = [=] (int const ib,
                      int const jb) -> T const & {
                return( B_[ indx2z(ib,jb,ldB) ] );
        };

        auto C = [=] (int const ic,
                      int const jc) -> T& {
                return( C_[ indx2z(ic,jc,ldC) ] );
        };
        auto currB = [=] (int const icb,
                          int const jcb) -> T& {
                return( currB_[ (icb) + ((jcb) * ldCurrB) + ((jcb) * padCurrB) ] );
        };
#else
#define A(ia,ja)  A_[indx2z(ia,ja,ldA)]
#define B(ib,jb)  B_[indx2z(ib,jb,ldB)]
#define C(ic,jc)  C_[indx2z(ic,jc,ldC)]
#define currB(icb, jcb) currB_[(icb) + ((jcb) * ldCurrB) + ((jcb) * padCurrB)]
#endif
#ifndef DIV_ROUNDUP
#define DIV_ROUNDUP(n, d) ((n) - 1) / (d) + 1
#endif
        
        // Number of rows each thread contributes to
        int const nr = DIV_ROUNDUP(t2, t3);
        
        T oldC[nr];
        Tc currC[nr];
        T nextC[nr];
        T nextB[nr];
        
        int const tid = threadIdx.x % t1;
        int const rid = threadIdx.x / t1;
        int const threadInc = t1;

        int threadBase;

        for (int p = 0; p < m; p += t2) {
                threadBase = 0;
                // Loads element of Matrix C
                if (beta_in == 0 || beta_in == 1) {
                        #pragma unroll
                        for (int i = 0; i < nr; ++i) {
                                oldC[i] = 0;
                        }
                } else {
                        #pragma unroll
                        for (int i = 0; i < nr; ++i) {
                                int trid = (i * t3) + rid;
                                if (threadBase + tid < n && trid < t2 && p + trid < m) {
                                        oldC[i] = beta_in * C(p + trid, threadBase + tid);
                                }
                        }
                }

                // Loads element of Matrix B
                #pragma unroll
                for (int i = 0; i < nr; ++i) {
                        int trid = (i * t3) + rid;
                        if (threadBase + tid < n && trid < t2 && trid < k) {
                                currB(tid, trid) = B(trid, threadBase + tid);
                        }
                }

                for (; threadBase < n; threadBase += threadInc) {
                        // thread = threadBase + tid;
                        // Load loops have extra conditionals to ensure
                        // they do not make bad memory accesses

                        // Prefetch next tile of Matrix C
                        if (threadBase + threadInc < n) {
                                if (beta_in == 0 || beta_in == 1) {
                                        #pragma unroll
                                        for (int i = 0; i < nr; ++i) {
                                                nextC[i] = 0;
                                        }
                                } else {
                                        #pragma unroll
                                        for (int i = 0; i < nr; ++i) {
                                                int trid = (i * t3) + rid;
                                                if (threadBase + threadInc + tid < n && trid < t2 && p + trid < m) {
                                                        nextC[i] = C(p + trid, threadBase + threadInc + tid);
                                                }
                                        }
                                }
                        }

                        #pragma unroll
                        for (int i = 0; i < nr; ++i) {
                                currC[i] = 0;
                        }
                        // Outer product loop
                        for (int j = 0; j < k; j += t2) {
                                __syncthreads();
                
                                // Prefetch next tile of B
                                if (j + t2 < k) {
                                        #pragma unroll
                                        for (int i = 0; i < nr; ++i) {
                                                int trid = (i * t3) + rid;
                                                if (threadBase + tid < n && trid < t2 && j + t2 + trid < k) {
                                                        nextB[i] = B(j + t2 + trid, threadBase + tid);
                                                }
                                        }
                                } else if (threadBase + threadInc < n) {
                                        #pragma unroll
                                        for (int i = 0; i < nr; ++i) {
                                                int trid = (i * t3) + rid;
                                                if (threadBase + threadInc + tid < n && trid < t2 && trid < k) {
                                                        nextB[i] = B(trid, threadBase + threadInc + tid);
                                                }
                                        }
                                }

                                // Floating Point Operations
                                // Each thread does t2 mults
     
                                // Either dispatch guarded or unguarded instructions based
                                // on position in matrix A
                                if (j + t2 <= k) {
                                        #pragma unroll
                                        for (int i = 0; i < nr; ++i) {
                                                int trid = (i * t3) + rid;
                                                if (trid < t2) {
                                                        #pragma unroll
                                                        for (int b = 0; b < t2; ++b) {
                                                                currC[i] +=
                                                                        A(p + trid, j + b) * currB(tid, b);
                                                        }
                                                }
                                        }
                                } else {
                                        #pragma unroll
                                        for (int i = 0; i < nr; ++i) {
                                                int trid = (i * t3) + rid;
                                                if (trid < t2) {
                                                        #pragma unroll
                                                        for (int b = 0; b < t2; ++b) {
                                                                if (j + b < k) {
                                                                        currC[i] += 
                                                                                A(p + trid, j + b) * currB(tid, b);
                                                                }
                                                        }
                                                }
                                        }
                                }
                                __syncthreads();

                                // Load prefetched B
                                #pragma unroll
                                for (int i = 0; i < nr; ++i) {
                                        int trid = (i * t3) + rid;
                                        if (trid < t2) {
                                                currB(tid, trid) = nextB[i];
                                        }
                                }
                        }

                        // Stores and reloads C
                        #pragma unroll
                        for (int i = 0; i < nr; ++i) {
                                int trid = (i * t3) + rid;
                                if (threadBase + tid < n && trid < t2 && p + trid < m) {
                                        if (beta_in == 1) {
                                                atomicAdd(&(C(p + trid, threadBase + tid)), alpha_in * currC[i]);
                                        } else if (beta_in == 0) {
                                                C(p + trid, threadBase + tid) = alpha_in * currC[i];
                                        } else {
                                                C(p + trid, threadBase + tid) = (alpha_in * currC[i]) + oldC[i];
                                        }
                                }
                        }
            
                        #pragma unroll
                        for (int i = 0; i < nr; ++i) {
                                oldC[i] = beta_in * nextC[i];
                        }
                }
        }
#ifndef USE_LAMBDA
#undef A
#undef B
#undef C
#endif
}


// First version of kgemm_nn routine -- No shared memory used
template<typename T, typename Tc>
DEVICE_FUNCTION
void kgemm_nn_v1( int const mm, int const nn, int const kk, 
                  T const alpha_in,
                  T const * const A_,  int const ldA,
                  T const * const B_,  int const ldB,
                  T const beta_in,
                  T * C_,  int const ldC,
                  volatile char* shmem)
{
#ifdef USE_LAMBDA
        auto min = []( int const x, int const y) {
                return(  (x < y) ? x : y );
        };
        auto max = []( int const x, int const y) {
                return(  (x > y) ? x : y );
        };
#else

#ifndef min
#define min(x,y)  (((x) < (y)) ? (x) : (y) )
#endif

#ifndef max
#define max(x,y)  (((x) > (y)) ? (x) : (y) )
#endif

#endif

        Tc const alpha = alpha_in;
        Tc const beta = beta_in;

	int constexpr nb = 2*32;
#ifdef USE_GPU
        // ---------------------------
        // use matlab 1 based indexing
        // ---------------------------

	int constexpr warpsize = WARPSIZE;
        int const nthreads = blockDim.x; 

        assert( blockDim.y == 1);
        assert( blockDim.z == 1);
        assert( (nthreads % warpsize) == 0);

        // -----------------------------------------
        // -----------------------------------------



	int const ij_start = threadIdx.x + 1;
	int const ij_size = nthreads;
#else


	int const ij_start = 1;
	int const ij_size = 1;
#endif



        //  ------------------------------------
        //  commonly  nn is large, but kk, nn are small
        //
        //  consider increasing nb for more effective
        //  use of shared cache
        //
        //  ------------------------------------


#ifdef USE_LAMBDA
        auto A = [=] (int const ia,
                      int const ja) -> T const & {
                return( A_[ indx2f(ia,ja,ldA) ] );
        };

        auto B = [=] (int const ib,
                      int const jb) -> T const & {
                return( B_[ indx2f(ib,jb,ldB) ] );
        };

        auto C = [=] (int const ic,
                      int const jc) -> T& {
                return( C_[ indx2f(ic,jc,ldC) ] );
        };

#else

#define A(ia,ja)  A_[indx2f(ia,ja,ldA)]
#define B(ib,jb)  B_[indx2f(ib,jb,ldB)]
#define C(ic,jc)  C_[indx2f(ic,jc,ldC)]

#endif

        for(int istart=1; istart <= mm;  istart += nb) {

          int const iend = min( mm, istart + nb-1);
          int const isize = iend - istart + 1;

         for(int jstart=1; jstart <= nn; jstart += nb) {
            int const jend = min(nn, jstart + nb-1);
            int const jsize = jend  - jstart + 1;


                SYNCTHREADS;

                    // ---------------------------
                    // perform matrix calculations
                    // ---------------------------

		    for(int ij0 = ij_start-1; ij0 < (isize*jsize); ij0 += ij_size ) {
			    int const i = (ij0 % isize) + 1;
			    int const j = ((ij0 - (i-1))/isize) + 1;
			    int const ia = (istart-1) + i;
			    int const jb = (jstart-1) + j;

			    auto const inc_A = ldA;
			    auto const inc_B = 1;
			    Tc cij = 0;
			    bool constexpr use_pointer = true;
			    if (use_pointer) {
				    
                                    int const k = 1;
				    T const * Ap = &(A(ia,k));
				    T const * Bp = &(B(k,jb));



#define case_code(kk)  { \
				       for(int k=0; k < kk; k++) { \
					    Tc const aik = (*Ap); \
					    Tc const bkj = (*Bp); \
					    cij += aik*bkj; \
					    Ap += inc_A; \
					    Bp += inc_B; \
				            }; \
				       break; \
				       }

				    switch(kk) {
				    case 1: case_code(1)
				    case 2: case_code(2)
				    case 3: case_code(3)
				    case 4: case_code(4)
				    case 5: case_code(5)
				    case 6: case_code(6)
				    case 7: case_code(7)
				    case 8: case_code(8)
			            default:
                                    case_code(kk);
				    };
			    }
			    else {
			      for(int k=1; k <= kk; k++) {
				Tc const aik = A( ia, k);
				Tc const bkj = B( k, jb);
				cij += aik * bkj;
			      };
			    };
                           // ------------------
                           // store results to C
                           // ------------------
                           int const ic = ia;
                           int const jc = jb;
			   T alpha_cij = alpha * cij;
			   if (beta == 1) {
                             atomicAdd( &(C(ic,jc)), alpha_cij );
			     }
			   else if (beta == 0) {
		              C(ic,jc) = alpha_cij;
			      }
			   else {
			      C(ic,jc)  =  beta * C(ic,jc) + alpha_cij;
			   };

		    };

            }; // end istart
        }; // end jstart
}



#ifdef USE_GPU 
// Select parameters
template<typename T, typename Tc>
DEVICE_FUNCTION
void kgemm_nn2( int const mm, int const nn, int const kk, 
                T const alpha_in,
                T const * const A_,  int const ldA,
                T const * const B_,  int const ldB,
                T const beta_in,
                T * C_,  int const ldC,
                volatile char* shmem)
{
#ifdef PROFILE
        if (blockIdx.x == 0 && threadIdx.x == 0) {
                atomicAdd(totalCount, 1);
                totalTemp = clock64();
                if (nn >= 128) {
                        sigTemp = totalTemp;
                        atomicAdd(sigCount, 1);
                }
        }
#endif
        // V100 tuning
        int const GPU_NN_T1 = 32;
        // Avoid using technique where it is not worth it
        if (nn <= 128) {
                kgemm_nn_v1<T, Tc>(mm, nn, kk, alpha_in, A_, ldA, B_, ldB,
                                   beta_in, C_, ldC, shmem);
        } else {
                if (sizeof(T) == sizeof(double)) {
                        switch (mm) {
                        case 1:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 1, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 2:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 2, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 3:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 3, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 4:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 4, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 5:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 5, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 6:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 6, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 7:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 7, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 8:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 8, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 9:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 9, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 10:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 10, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                     mm, nn, kk,
                                                                     alpha_in,
                                                                     A_, ldA,
                                                                     B_, ldB,
                                                                     beta_in,
                                                                     C_, ldC);
                                break;
                        default:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 10, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                     mm, nn, kk,
                                                                     alpha_in,
                                                                     A_, ldA,
                                                                     B_, ldB,
                                                                     beta_in,
                                                                     C_, ldC);
                                break;
                        }
                } else {
                        switch (mm) {
                        case 1:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 1, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 2:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 2, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 3:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 3, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 4:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 4, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 5:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 5, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 6:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 6, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 7:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 7, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 8:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 8, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 9:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 9, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                    mm, nn, kk,
                                                                    alpha_in,
                                                                    A_, ldA,
                                                                    B_, ldB,
                                                                    beta_in,
                                                                    C_, ldC);
                                break;
                        case 10:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 10, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                     mm, nn, kk,
                                                                     alpha_in,
                                                                     A_, ldA,
                                                                     B_, ldB,
                                                                     beta_in,
                                                                     C_, ldC);
                                break;
                        default:
                                kgemm_nn_v2<T, Tc, GPU_NN_T1, 10, 4>((volatile T*) shmem, GPU_NN_T1, 2,
                                                                     mm, nn, kk,
                                                                     alpha_in,
                                                                     A_, ldA,
                                                                     B_, ldB,
                                                                     beta_in,
                                                                     C_, ldC);
                                break;
                        }
                }
        }
#ifdef PROFILE
        if (blockIdx.x == 0 && threadIdx.x == 0) {
                totalTime += (clock64() - totalTemp);
                if (nn >= 128) sigTime += clock64() - sigTemp;
        }
#endif
}
#else
template<typename T, typename Tc>
DEVICE_FUNCTION
void kgemm_nn2( int const mm, int const nn, int const kk, 
                T const alpha_in,
                T const * const A_,  int const ldA,
                T const * const B_,  int const ldB,
                T const beta_in,
                T * C_,  int const ldC,
                volatile char* shmem)
{ 
        kgemm_nn_v1<T, Tc>(mm, nn, kk, alpha_in, A_, ldA, B_, ldB,
                           beta_in, C_, ldC, shmem);
}
#endif


template<typename T>
DEVICE_FUNCTION
void kgemm_nn( int const mm, int const nn, int const kk, 
               T const alpha,
               T const * const A_,  int const ldA,
               T const * const B_,  int const ldB,
               T const beta,
               T * C_,  int const ldC,
               volatile char* shmem)
{
 kgemm_nn2<T,T>(
           mm,nn,kk, alpha, A_,ldA, B_,ldB,
           beta, C_,ldC,shmem);
}





template<>
DEVICE_FUNCTION
void kgemm_nn( int const mm, int const nn, int const kk, 
               float const alpha,
               float const * const A_,  int const ldA,
               float const * const B_,  int const ldB,
               float const beta,
               float * C_,  int const ldC,
               volatile char* shmem)
{
 kgemm_nn2<float,double>(
           mm,nn,kk, alpha, A_,ldA, B_,ldB,
           beta, C_,ldC,shmem);
}

#undef min
#undef max
#undef A
#undef B
#undef C
#undef case_code

#endif
