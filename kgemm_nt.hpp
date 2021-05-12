#ifndef KGEMM_NT_HPP
#define KGEMM_NT_HPP 1

#include "kroncommon.hpp"

//  -----------------------
//  NotransA and TransB case
//  C = alpha*A*transpose(B) + beta *C
//  -----------------------

#ifdef USE_GPU
extern __device__ int* sigCount, * totalCount;
extern __device__ long long int sigTime, totalTime;

extern __device__ long long int sigTemp, totalTemp;
#endif

// Second implementation of kgemm_nt using TSM2L and shared memory
template<typename T,typename Tc, int t1, int t2, int t3>
DEVICE_FUNCTION
void kgemm_nt_v2(volatile T* currB_, int const ldCurrB, int const padCurrB,
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

        T currA[t3];
        T nextA[t3];
        T nextB[t2];
        Tc currC[t2];
        
        const int tid = threadIdx.x;
        int threadBase, thread;

        threadBase = 0;

        for (; threadBase < m; threadBase += blockDim.x) {
                thread = threadBase + tid;
                for (int p = 0; p < n; p += t2) {
                        // Load loops have extra conditionals to ensure
                        // they do not make bad memory accesses

                        // Loads first tile of output registers and A
                        if (thread < m) {
                                #pragma unroll
                                for (int i = 0; i < t2; ++i) {
                                        if (p + i < n) {
                                                currC[i] = 0;
                                        }
                                }
                                // Loads currA
                                #pragma unroll
                                for (int i = 0; i < t3; ++i) {
                                        if (i < k) {
                                                currA[i] = A(thread, i);
                                        }
                                }
                        }
                        // Loads tile of B
                        if (tid < k) {
                                #pragma unroll
                                for (int i = 0; i < t2; ++i) {
                                        if (p + i < n) {
                                                currB(tid, i) = B(p + i, tid); // transposed
                                        }
                                }
                        }

                        // Outer product loop
                        for (int j = 0; j < k; j += t1) {
                                __syncthreads();
                                // Loads next tile of B
                                if (j + t1 + tid < k) {
                                        #pragma unroll
                                        for (int i = 0; i < t2; ++i) {
                                                if (p + i < n) {
                                                    nextB[i] = B(p + i, j + t1 + tid); // trans
                                                }
                                        }
                                }

                                const int t3mod = t1 % t3;

                                // Loop over A's columns
                                for (int l = j; l < j + (t1 - t3mod) && l < k; l += t3) {
                                        // Loads next A
                                        #pragma unroll
                                        for (int i = 0; i < t3; ++i) {
                                                if (l + t3 + i < k && thread < m) {
                                                        nextA[i] = A(thread, l + t3 + i);
                                                }
                                        }

                                        // Floating Point Operations (lines 32-34)
                                        // Each thread does t2 * t3 mults

                                        // Either dispatch guarded or unguarded instructions based
                                        // on position in matrix A
                                        if (l + t3 <= k) {
                                                // It is assumed that B[(l - j) .. (l - j) + t3 - 1, _]
                                                // exist
                                                #pragma unroll
                                                for (int a = 0; a < t2; ++a) {
                                                        #pragma unroll
                                                        for (int b = 0; b < t3; ++b) {
                                                                currC[a] +=
                                                                        currA[b] * currB((l - j) + b, a);
                                                        }
                                                }
                                        } else {
                                                #pragma unroll
                                                for (int a = 0; a < t2; ++a) {
                                                        #pragma unroll
                                                        for (int b = 0; b < t3; ++b) {
                                                                if (l + b < k) {
                                                                        currC[a] += currA[b] *
                                                                                currB((l - j) + b, a);
                                                                }
                                                        }
                                                }
                                        }

                                        // Stores next A in curr A
                                        #pragma unroll
                                        for (int i = 0; i < t3; ++i) {
                                                currA[i] = nextA[i];
                                        }
                                }
                                // Accommodates t3 that do not divide t1.
                                #pragma unroll
                                for (int a = 0; a < t2; ++a) {
                                        #pragma unroll
                                        for (int b = 0; b < t3mod; ++b) {
                                                if (j + t1 - t3mod + b < k) {
                                                        currC[a] +=
                                                                currA[b] * currB((t1 - t3mod) + b, a);
                                                }
                                        }
                                }

                                __syncthreads();

                                // Loads currB from each thread's nextB
                                #pragma unroll
                                for (int i = 0; i < t2; ++i) {
                                        currB(tid, i) = nextB[i];
                                }

                                // Loads next currA
                                if (t3mod != 0) {
                                        #pragma unroll
                                        for (int i = 0; i < t3; ++i) {
                                                if (j + t1 + i < k && thread < m) {
                                                        currA[i] = A(thread, j + t1 + i);
                                                }
                                        }
                                }
                        }
                        // Stores C
                        if (thread < m) {
                                #pragma unroll
                                for (int i = 0; i < t2; ++i) {
                                        if (p + i < n) {
                                                if (beta_in == 1) {
                                                        atomicAdd(&(C(thread, p + i)), alpha_in * currC[i]);
                                                } else if (beta_in == 0) {
                                                        C(thread, p + i) = alpha_in * currC[i];
                                                } else {
                                                        C(thread, p + i) =
                                                                (alpha_in * currC[i]) + (beta_in * C(thread, p + i));
                                                }
                                        }
                                }
                        }
                }
        }
#ifndef USE_LAMBDA
#undef A
#undef B
#undef C
#endif
}

// First implementation of kgemm_nt, without shared memory
template<typename T,typename Tc>
DEVICE_FUNCTION
void kgemm_nt_v1( int const mm, int const nn, int const kk, 
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

        // -----------------------------------------
        // -----------------------------------------
        assert( (nthreads % warpsize) == 0);



	int const ij_start = threadIdx.x + 1;
	int const ij_size = nthreads;

#else


	int const ij_start = 1;
	int const ij_size = 1;
#endif



        //  ------------------------------------
        //  commonly  mm is large, but kk, nn are small
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


        for(int jstart=1; jstart <= nn; jstart += nb) {
            int const jend = min(nn, jstart + nb-1);
            int const jsize = jend  - jstart + 1;

            for(int istart=1; istart <= mm;  istart += nb) {
                int const iend = min( mm, istart + nb-1);
                int const isize = iend - istart + 1;

                 SYNCTHREADS;

                    // ---------------------------
                    // perform matrix calculations
                    // ---------------------------

		    auto const inc_A = ldA;
		    auto const inc_B = ldB;

		    for(int ij0=ij_start-1; ij0 < (isize*jsize); ij0 += ij_size) {
			    int const  i = (ij0 % isize) + 1;
			    int const  j = (ij0 - (i-1))/isize + 1;
			    Tc cij = 0;
			    bool constexpr use_pointer = true;
			    if (use_pointer) {
				    int k = 1;
				    int ia = (istart-1) + i;
				    int ib = (jstart-1) + j;
				    T const * Ap = &(A(ia,k));
				    T const * Bp = &(B(ib,k));

#define case_code(kk)  { \
				       for(int k=0; k < kk; k++) { \
					    Tc const aik = (*Ap); \
					    Tc const bjk = (*Bp); \
					    cij += aik * bjk; \
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
				Tc const aik = A( (istart-1) + i, k);  
				Tc const bjk = B( (jstart-1) + j, k);
				cij += aik * bjk;
			      };
			    };
                           // ------------------
                           // store results to C
                           // ------------------
                           int const ic = (istart-1) + i;
                           int const jc = (jstart-1) + j;
			   if (beta == 1) {
                             atomicAdd( &(C(ic,jc)), alpha*cij );
			     }
			   else if (beta == 0) {
		              C(ic,jc) = alpha * cij;
			      }
			   else {
			      C(ic,jc)  =  beta * C(ic,jc) + alpha*cij;
			   };

		    };

            }; // end istart
        }; // end jstart
}

#ifdef USE_GPU
// Select parameters
template<typename T,typename Tc>
DEVICE_FUNCTION
void kgemm_nt2(int const mm, int const nn, int const kk, 
               T const alpha_in,
               T const * const A_, int const ldA,
               T const * const B_, int const ldB,
               T const beta_in,
               T * C_, int const ldC,
               volatile char* shmem)
{
        if (blockIdx.x == 0 && threadIdx.x == 0) {
                atomicAdd(totalCount, 1);
                totalTemp = clock64();
                if (mm >= 128) {
                        sigTemp = totalTemp;
                        atomicAdd(sigCount, 1);
                }
        }
        // V100 tuning
        if (true/*mm <= 128*/) {
                kgemm_nt_v1<T, Tc>(mm, nn, kk, alpha_in, A_, ldA,
                           B_, ldB, beta_in, C_, ldC, shmem);
        } else {
                int const GPU_NT_T1 = 128;
                switch (nn) {
                case 1:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 1, 1>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 2:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 2, 2>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 3:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 3, 3>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 4:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 4, 4>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 5:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 5, 5>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 6:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 6, 6>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 7:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 7, 7>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 8:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 8, 8>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 9:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 9, 9>((volatile T*) shmem, GPU_NT_T1, 0,
                                                            mm, nn, kk,
                                                            alpha_in,
                                                            A_, ldA,
                                                            B_, ldB,
                                                            beta_in,
                                                            C_, ldC);
                        break;
                case 10:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 10, 10>((volatile T*) shmem, GPU_NT_T1, 0,
                                                              mm, nn, kk,
                                                              alpha_in,
                                                              A_, ldA,
                                                              B_, ldB,
                                                              beta_in,
                                                              C_, ldC);
                        break;
                default:
                        kgemm_nt_v2<T, Tc, GPU_NT_T1, 10, 10>((volatile T*) shmem, GPU_NT_T1, 0,
                                                              mm, nn, kk,
                                                              alpha_in,
                                                              A_, ldA,
                                                              B_, ldB,
                                                              beta_in,
                                                              C_, ldC);
                        break;
                }
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
                totalTime += (clock64() - totalTemp);
                if (mm >= 128) sigTime += clock64() - sigTemp;
        }
}
#else
template<typename T,typename Tc>
DEVICE_FUNCTION
void kgemm_nt2( int const mm, int const nn, int const kk, 
                T const alpha_in,
                T const * const A_,  int const ldA,
                T const * const B_,  int const ldB,
                T const beta_in,
                T * C_,  int const ldC,
                volatile char* shmem)
{
        kgemm_nt_v1<T, Tc>(mm, nn, kk, alpha_in, A_, ldA,
                           B_, ldB, beta_in, C_, ldC, shmem);
}
#endif


template<typename T>
DEVICE_FUNCTION
void kgemm_nt( int const mm, int const nn, int const kk, 
               T const alpha_in,
               T const * const A_,  int const ldA,
               T const * const B_,  int const ldB,
               T const beta_in,
               T * C_,  int const ldC,
               volatile char* shmem)
{
   kgemm_nt2<T,T>(
       mm, nn, kk,
       alpha_in, A_, ldA, B_, ldB,
       beta_in,  C_, ldC, shmem );
}

template<>
DEVICE_FUNCTION
void kgemm_nt( int const mm, int const nn, int const kk, 
               float const alpha_in,
               float const * const A_,  int const ldA,
               float const * const B_,  int const ldB,
               float const beta_in,
               float * C_,  int const ldC,
               volatile char* shmem)
{
   kgemm_nt2<float,double>(
       mm, nn, kk,
       alpha_in, A_, ldA, B_, ldB,
       beta_in,  C_, ldC, shmem );
}



#undef min
#undef max
#undef A
#undef B
#undef C
#undef case_code

#endif
