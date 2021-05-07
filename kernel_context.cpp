#include "kernel_context.hpp"

#define MAX(a,b) (((a)>(b))?(a):(b))

// V100 specific tuning parameters
#define GPU_NT_T1 128
#define GPU_NN_T1 32
#define MAX_SHMEM_PAD 2
 
// Currently only tuned for V100
template <typename T>
HOST_FUNCTION DEVICE_FUNCTION size_t get_max_shmem(int max_dim) {
        int nt_max;
        if (max_dim > 0 && max_dim <= 10) {
                nt_max = GPU_NT_T1 * max_dim * sizeof(T);
        } else {
                nt_max = GPU_NT_T1 * 10 * sizeof(T);
        }
        int nn_max;
        if (max_dim > 0 && max_dim <= 10) {
                nn_max = (GPU_NN_T1 * max_dim + MAX_SHMEM_PAD * GPU_NN_T1) * sizeof(T);
        } else {
                nn_max = (GPU_NN_T1 * 10 + MAX_SHMEM_PAD * GPU_NN_T1) * sizeof(T);
        }

        return MAX(nt_max, nn_max);
}

template
HOST_FUNCTION DEVICE_FUNCTION size_t get_max_shmem<float>(int max_dim);

template
HOST_FUNCTION DEVICE_FUNCTION size_t get_max_shmem<double>(int max_dim);

