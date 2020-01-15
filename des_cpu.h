#pragma once
#include <cstdint>
namespace des_cpu
{
	uint64_t decrypt(uint64_t key, uint64_t message);
	uint64_t encrypt(uint64_t key, uint64_t message);
	uint64_t crack(uint64_t message, uint64_t cipher,
            uint64_t start, uint64_t limit);

	uint64_t permute_drop_parity(uint64_t key);
	uint64_t rev_permute_add_parity(uint64_t kplus);
	void generate_subkeys(uint64_t subkeys[16], uint64_t key56);
	uint64_t feistel_function(uint64_t k, uint64_t r);
	uint64_t des_decrypt_56(uint64_t key56, uint64_t message);
	uint64_t des_encrypt_56(uint64_t key56, uint64_t message);
};
