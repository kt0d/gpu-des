#include "des_gpu.cuh"
#include "common.h"
#include "des_kernels.cuh"
#include "des_cpu.h"

#include <vector>
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

des_result multi_gpu_crack(uint64_t message, uint64_t cipher, uint64_t begin, 
        uint64_t limit, size_t num_of_blocks, size_t block_size)
{
    bool h_done, *d_done;
    checkCudaErrors(cudaMallocManaged((void**)&d_done,sizeof(bool)));
    checkCudaErrors(cudaMemset(d_done, 0, sizeof(bool)));
    h_done = false;
    uint64_t h_key = 0, *d_key;
    checkCudaErrors(cudaMallocManaged((void**)&d_key,sizeof(uint64_t)));
    checkCudaErrors(cudaMemset(d_key, 0, sizeof(h_key)));

    int gpu_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&gpu_count));

    // Vector for counting checked keys, for every GPU.
    std::vector<thrust::device_vector<int>> counters;
    std::vector<int*> d_counters_ptr;
    for(int i = 0; i < gpu_count; i++)
    {
	checkCudaErrors(cudaSetDevice(i));
	counters.emplace_back(num_of_blocks * block_size / 32);
	d_counters_ptr.push_back(thrust::raw_pointer_cast(counters[i].data()));
    }

    std::vector<cudaEvent_t> start(gpu_count), stop(gpu_count);
    for(int i = 0; i < gpu_count; i++)
    {
    	// Start time measurment.
	checkCudaErrors(cudaSetDevice(i));
    	cudaEventCreate(&start[i]);
    	cudaEventCreate(&stop[i]);
    	cudaEventRecord(start[i]);
    	cudaEventSynchronize(start[i]);
    }

    // Run kernels.
    for(int i = 0; i < gpu_count; i++)
    {
	checkCudaErrors(cudaSetDevice(i));
    	multi_gpu_crack_kernel<<<num_of_blocks, block_size>>>(message, cipher,
		    begin,limit, d_done, d_key, d_counters_ptr[i], gpu_count);
    }

    float max_time = 0;
    for(int i = 0; i < gpu_count; i++)
    {
	checkCudaErrors(cudaSetDevice(i));
    	// Stop time measurment.
	cudaEventRecord(stop[i]);
	cudaEventSynchronize(stop[i]);

    	// Calculate elapsed time.
	float time = 0;
    	cudaEventElapsedTime(&time, start[i], stop[i]);
	max_time = time > max_time ? time : max_time;
    	// Clean up.
    	cudaEventDestroy(start[i]);
    	cudaEventDestroy(stop[i]);
    }

    // Compute total checked keys.
    unsigned long long total_sum = 0;
    for(int i = 0; i < gpu_count; i++)
    {
	checkCudaErrors(cudaSetDevice(i));
    	auto sum = thrust::reduce(counters[i].begin(), counters[i].end(), 
			(unsigned long long)(0));
	total_sum += sum;

    }

    checkCudaErrors(cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));

    des_result result;
    result.time = max_time;
    result.found = h_done;
    result.checked = total_sum;
    if(result.found)
    {
	h_key = *d_key;    
        result.key = des_cpu::rev_permute_add_parity(h_key);
    }

    checkCudaErrors(cudaFree(d_done));
    checkCudaErrors(cudaFree(d_key));

    return result;
}
