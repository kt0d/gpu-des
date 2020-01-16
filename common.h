#pragma once
#include <cstdint>

struct des_result
{
	bool found;
	uint64_t key;
	unsigned long long checked;
	float time;
};
