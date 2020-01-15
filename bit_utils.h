#include <cstddef>

template<size_t FROM, size_t TO>
inline auto permute_bits_64(uint64_t source, const unsigned char* table) -> uint64_t
{
	uint64_t  p = 0;
	for(size_t i = 0; i < TO; i++)
		p |= ( (source >> (FROM-table[i])) & 1) << (TO-1-i);
	return p;
}

template<size_t FROM, size_t TO>
inline auto permute_bits_32(uint32_t source, const unsigned char* table) -> uint32_t
{
	uint32_t  p = 0;
	for(size_t i = 0; i < TO; i++)
		p |= ( (source >> (FROM-table[i])) & 1) << (TO-1-i);
	return p;
}
