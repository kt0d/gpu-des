#include "des_gpu.cuh"
#include "common.h"
#include "des_kernels.cuh"
#include "des_cpu.h"

#include <cstdint>

#include <thrust/device_vector.h>
#include <helper_cuda.h>

des_result des_gpu_crack(uint64_t message, uint64_t cipher, uint64_t begin, 
        uint64_t limit, size_t num_of_blocks, size_t block_size)
{
    bool h_done, *d_done;
    checkCudaErrors(cudaMalloc((void**)&d_done,sizeof(bool)));
    checkCudaErrors(cudaMemset(d_done, 0, sizeof(bool)));
    h_done = false;
    uint64_t h_key = 0, *d_key;
    checkCudaErrors(cudaMalloc((void**)&d_key,sizeof(uint64_t)));
    checkCudaErrors(cudaMemset(d_key, 0, sizeof(h_key)));

    // Vector for counting checked keys.
    thrust::device_vector<int> d_vec(num_of_blocks * block_size / 32);
    int* d_counters = thrust::raw_pointer_cast(d_vec.data());

    // Start time measurment.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    // Run kernel.
    des_gpu_crack_kernel<<<num_of_blocks, block_size>>>(message, cipher,
		    begin,limit, d_done, d_key, d_counters);

    // Stop time measurment.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Calculate elapsed time.
    float miliseconds = 0;
    // Clean up.
    cudaEventElapsedTime(&miliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compute total checked keys.
    auto sum = thrust::reduce(d_vec.begin(), d_vec.end(), (unsigned long long)(0));

    checkCudaErrors(cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));

    des_result result;
    result.time = miliseconds;
    result.found = h_done;
    result.checked = sum;
    if(result.found)
    {
        checkCudaErrors(cudaMemcpy(&h_key, d_key, sizeof(h_key), cudaMemcpyDeviceToHost));
        result.key = des_cpu::rev_permute_add_parity(h_key);
    }

    checkCudaErrors(cudaFree(d_done));
    checkCudaErrors(cudaFree(d_key));


    return result;
}
