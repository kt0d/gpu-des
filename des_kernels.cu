#include "des_kernels.cuh"
#include <cstdio>
#include <cstdint>

template<size_t FROM, size_t TO>
__device__ __forceinline__ 
auto permute_bits_64(uint64_t source, const unsigned char* table) -> uint64_t
{
	uint64_t  p = 0;
	for(size_t i = 0; i < TO; i++)
		p |= ( (source >> (FROM-table[i])) & 1) << (TO-1-i);
	return p;
}

template<size_t FROM, size_t TO>
__device__ __forceinline__ 
auto permute_bits_32(uint32_t source, const unsigned char* table) -> uint32_t
{
	uint32_t  p = 0;
	for(size_t i = 0; i < TO; i++)
		p |= ( (source >> (FROM-table[i])) & 1) << (TO-1-i);
	return p;
}

static __constant__ unsigned char h_PC_2[48] = {
	14, 17, 11, 24, 1, 5,
	3, 28, 15, 6, 21, 10,
	23, 19, 12, 4, 26, 8,
	16, 7, 27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

static __constant__ unsigned char h_IP[64] = {
	58, 50, 42, 34, 26, 18, 10, 2,
	60, 52, 44, 36, 28, 20, 12, 4,
	62, 54, 46, 38, 30, 22, 14, 6,
	64, 56, 48, 40, 32, 24, 16, 8,
	57, 49, 41, 33, 25, 17, 9, 1,
	59, 51, 43, 35, 27, 19, 11, 3,
	61, 53, 45, 37, 29, 21, 13, 5,
	63, 55, 47, 39, 31, 23, 15, 7
};

static __constant__ unsigned char h_E_BIT[48] = {
	32, 1, 2, 3, 4, 5,
	4, 5, 6, 7, 8, 9,
	8, 9, 10, 11, 12, 13,
	12, 13, 14, 15, 16, 17,
	16, 17, 18, 19, 20, 21,
	20, 21, 22, 23, 24, 25,
	24, 25, 26, 27, 28, 29,
	28, 29, 30, 31, 32, 1
};

static __constant__ unsigned char h_S1[64] = {
	14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
	0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
	4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
	15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13
};

static __constant__ unsigned char h_S2[64] = {
	15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
	3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
	0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
	13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9
};

static __constant__ unsigned char h_S3[64] = {
	10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
	13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
	13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
	1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12
};

static __constant__ unsigned char h_S4[64] = {
	7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
	13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
	10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
	3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14
};

static __constant__ unsigned char h_S5[64] = {
	2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
	14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
	4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
	11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3
};

static __constant__ unsigned char h_S6[64] = {
	12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
	10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
	9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
	4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13,
};

static __constant__ unsigned char h_S7[64] = {
	4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
	13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
	1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
	6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12,
};

static __constant__ unsigned char h_S8[64] = {
	13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
	1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
	7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
	2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11,
};

static __constant__ unsigned char *h_S[8] = {
	h_S1, h_S2, h_S3, h_S4, h_S5, h_S6, h_S7, h_S8
};

static __constant__ unsigned char h_P[32] = {
	16, 7, 20, 21,
	29, 12, 28, 17,
	1, 15, 23, 26,
	5, 18, 31, 10,
	2, 8, 24, 14,
	32, 27, 3, 9,
	19, 13, 30, 6,
	22, 11, 4, 25
};

static __constant__ unsigned char h_IP_REV[64] = {
	40, 8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41, 9, 49, 17, 57, 25
};

static __constant__ unsigned char h_SHIFTS[16] = {
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};


__device__ __forceinline__
uint64_t feistel_function(uint64_t k, uint64_t r)
{
	// Expansion
	uint64_t e = permute_bits_64<32,48>(r, h_E_BIT);
	// Key mixing
	k = k ^ e;
	// Substitution
	e = 0;
	for(int j = 8-1; j >= 0; j--)
	{
		uint8_t block = (k >> (j)*6);
		auto row = ((block & 0b100000) >> 4) | (block & 1);
		auto col = (block & 0b011110) >> 1;
		e |= uint32_t(h_S[8-1-j][row*16+col]) << ((j)*4);
	}
	// Permutation
	uint64_t f = permute_bits_32<32,32>(e, h_P);
	return f;
}

__device__ __forceinline__
uint64_t des_encrypt_56(uint64_t key56, uint64_t message)
{
	// Initial permutation
	uint64_t ip = permute_bits_64<64,64>(message, h_IP);
	uint32_t l = (ip >> 32),
		 r = ip;

	// Rounds, with subkey generation
	uint32_t c = (key56 >> 28) & 0xfffffff;
	uint32_t d = (key56) & 0xfffffff;
	for(int i = 0; i < 16; i++)
	{
		if(h_SHIFTS[i] == 2)
		{
			c = (c << 2) | (c >> 26) ;
			d = (d << 2) | (d >> 26) ;
		}
		else
		{
			c = (c << 1) | (c >> 27) ;
			d = (d << 1) | (d >> 27) ;
		}
		c &= 0xfffffff;
		d &= 0xfffffff;

		
		uint64_t k = (uint64_t(c) << 28) | d;
		k = permute_bits_64<56,48>(k, h_PC_2);

		uint64_t f = feistel_function(k, r);

		auto old_l = l;
		l = r;
		r = old_l ^ f;
	}

	message = (uint64_t(r) << 32) | l;
	// Final permutation
	ip = permute_bits_64<64, 64>(message, h_IP_REV);
	return ip;
}

__global__ void des_gpu_crack_kernel(uint64_t message, uint64_t cipher, 
        uint64_t begin, uint64_t limit, bool* d_done, uint64_t *d_key, int* d_counters)
{
    uint64_t key = begin + blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = 0;
    while(key < limit && !(*d_done))
    {
        //printf("%lld\n",key);
        uint64_t encrypted = des_encrypt_56(key, message);
        count++;
        if(encrypted == cipher)
        {
            *d_key = key;
            *d_done = true;
            //printf("%d\n", *d_done);
            break;
        }
        key += gridDim.x * blockDim.x;
    }
    if(threadIdx.x % warpSize == 0)
    {
        size_t index = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
        d_counters[index] = count * warpSize;
    }
}

__global__ void multi_gpu_crack_kernel(uint64_t message, uint64_t cipher, 
        uint64_t begin, uint64_t limit, bool* d_done, uint64_t *d_key, int* d_counters,
	unsigned short gpu_count, unsigned short current_gpu)
{
    uint64_t key = begin + blockIdx.x * blockDim.x + threadIdx.x 
        + current_gpu * gridDim.x * blockDim.x;
    uint32_t count = 0;
    while(key < limit && !(*d_done))
    {
        //printf("%lld\n",key);
        uint64_t encrypted = des_encrypt_56(key, message);
        count++;
        if(encrypted == cipher)
        {
            *d_key = key;
            *d_done = true;
            //printf("%d\n", *d_done);
            break;
        }
        key += gpu_count * gridDim.x * blockDim.x;
    }
    if(threadIdx.x % warpSize == 0)
    {
        size_t index = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
        d_counters[index] = count * warpSize;
    }
}
