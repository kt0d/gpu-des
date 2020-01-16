#pragma once
#include "common.h"

#include <cstdint>

des_result des_gpu_crack(uint64_t message, uint64_t cipher, 
    uint64_t begin, uint64_t limit, 
    size_t num_of_blocks = 2048, size_t block_size = 1024);
