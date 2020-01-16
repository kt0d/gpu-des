#pragma once
#include "common.h"

#include <cstdint>

des_result des_gpu_crack(uint64_t message, uint64_t cipher, 
    uint64_t start, uint64_t limit);
