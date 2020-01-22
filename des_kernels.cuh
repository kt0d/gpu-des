#pragma once
#include <cstdint>

__global__ void des_gpu_crack_kernel(uint64_t message, uint64_t cipher, 
    uint64_t begin, uint64_t limit, bool* d_done, uint64_t *d_key, int* d_counters);

__global__ void multi_gpu_crack(uint64_t message, uint64_t cipher, 
    uint64_t begin, uint64_t limit, bool* d_done, uint64_t *d_key, int* d_counters);
