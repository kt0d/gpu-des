#include <iostream>
#include <cstdint>
#include "des_cpu.h"
#include "des_gpu.cuh"

int main(int argc, char **argv)
{
	using namespace std;
	uint64_t message = 0x0123456789ABCDEF;
	uint64_t key = 0x133457799BBCDFF1;
	key = des_cpu::rev_permute_add_parity(100000000);
	cout << hex << key << endl;
	cout << hex << message << endl;

	auto encrypted = des_cpu::encrypt(key, message);
	cout << hex << encrypted << endl;
	cout << hex << des_cpu::decrypt(key, encrypted) << endl;
	//cout << hex << des_cpu::crack(message, encrypted) << endl;
    cout << hex << des_gpu_crack(message, encrypted, 0) << endl;
	return 0;
}

/*
 * OpenSSL DES
//#include <openssl/des.h>
//#include <byteswap.h>
//#include <cstring>
 int err = 0;
 DES_cblock des_key, des_message, des_encrypted;
 bswap_64(key);
 bswap_64(message);
 memcpy(&des_key, &key, 8);
 memcpy(&des_message, &message, 8);
 for(int i = 0; i < 4; i++)     {
 swap(des_key[i], des_key[7-i]);
 swap(des_message[i], des_message[7-i]);
 }
 DES_key_schedule schedule;
 DES_set_key_unchecked(&des_key, &schedule);
 if(err) exit(-1);
 DES_ecb_encrypt(&des_message, &des_encrypted, &schedule, DES_ENCRYPT);
 uint64_t encrypted = 0;
 for(int i = 0; i < 4; i++)
 {
 swap(des_encrypted[i], des_encrypted[7-i]);
 }
 memcpy(&encrypted, &des_encrypted, 8);
 cout << hex << message << endl;
 cout << hex << encrypted << endl;
 DES_ecb_encrypt(&des_encrypted, &des_message, &schedule, DES_DECRYPT);
 memcpy(&encrypted, &des_message, 8);
 cout << hex << encrypted << endl;

*/
