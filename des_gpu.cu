#include "des_gpu.cuh"
#include "des_kernels.cuh"
#include "des_cpu.h"
#include <cstdint>

#include <helper_cuda.h>

uint64_t des_gpu_crack(uint64_t message, uint64_t cipher, uint64_t start)
{
    bool h_done, *d_done;
    checkCudaErrors(cudaMalloc((void**)&d_done,sizeof(bool)));
    checkCudaErrors(cudaMemset(d_done, 0, sizeof(bool)));
    h_done = false;
    uint64_t h_key = 0, *d_key;
    checkCudaErrors(cudaMalloc((void**)&d_key,sizeof(uint64_t)));

    const size_t num_of_blocks = 1024;
    const size_t block_size = 512;
    //while(!h_done)
    //{
        des_gpu_crack_kernel<<<num_of_blocks, block_size>>>(message, cipher,
                start,1000000000, d_done, d_key);
        checkCudaErrors(cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));
    //    start += num_of_blocks * block_size;
    //}
    if(!h_done) return 0;
    checkCudaErrors(cudaMemcpy(&h_key, d_key, sizeof(h_key), cudaMemcpyDeviceToHost));
	return des_cpu::rev_permute_add_parity(h_key);
}
