#include "des_gpu.cuh"
#include "des_kernels.cuh"
#include "des_cpu.h"

#include <cstdint>

#include <thrust/device_vector.h>
//#include <thrust/device_ptr.h>
#include <helper_cuda.h>

uint64_t des_gpu_crack(uint64_t message, uint64_t cipher, uint64_t begin, uint64_t limit)
{
    bool h_done, *d_done;
    checkCudaErrors(cudaMalloc((void**)&d_done,sizeof(bool)));
    checkCudaErrors(cudaMemset(d_done, 0, sizeof(bool)));
    h_done = false;
    uint64_t h_key = 0, *d_key;
    checkCudaErrors(cudaMalloc((void**)&d_key,sizeof(uint64_t)));

    const size_t num_of_blocks = 1024;
    const size_t block_size = 512;

    thrust::device_vector<int> d_vec(num_of_blocks * block_size / 32);
    int* d_counters = thrust::raw_pointer_cast(d_vec.data());


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    des_gpu_crack_kernel<<<num_of_blocks, block_size>>>(message, cipher,
		    begin,limit, d_done, d_key, d_counters);
    checkCudaErrors(cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));

    if(!h_done) return 0;
    checkCudaErrors(cudaMemcpy(&h_key, d_key, sizeof(h_key), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float miliseconds = 0;
    cudaEventElapsedTime(&miliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto x = thrust::reduce(d_vec.begin(), d_vec.end(), (unsigned long long)(0));

    std::cout << std::dec << x << std::endl;
    std::cout << miliseconds << std::endl;
    std::cout << (x / ((double)miliseconds / 1000)) / 1000000 << std::endl;

    return des_cpu::rev_permute_add_parity(h_key);
}
