#pragma once
#include <cstdint>

__global__ void des_gpu_crack_kernel(uint64_t message, uint64_t cipher, 
    uint64_t start, uint64_t limit, bool* d_done, uint64_t *d_key);
