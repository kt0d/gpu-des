#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <argp.h>

#include "des_cpu.h"
#include "des_gpu.cuh"

struct arguments
{
    bool set_key = false;
    uint64_t key = 0;
    bool set_limit = false;
    uint64_t limit = 0;
    bool run_gpu = false;
    bool run_cpu = false;
};

static char doc[] = "gpu-des - DES cracking tool for CUDA\v"
    "Numeric arguments can be provided as either decimals or hexadecimals prefixed with \"0x\".";

static struct argp_option options[] =
{
    {"limit",       'l', "L", 0,
        "Exclusive upper bound for key space search. Interpreted as 56 bit wide key after PC-1 permutation.", 0},
    {"key",         'k', "K", 0,
        "Key to crack. Interpreted as 56 bit wide key after PC-1 permutation.", 0},
    {"gpu",         'g', 0, 0,
        "Run DES cracking on GPU", 1},
    {"cpu",         'c', 0, 0,
        "Run DES cracking on CPU", 1},
	{0, 0, 0, 0, 0, 0}
};

static error_t parse_opt(int key, char* arg, struct argp_state *state);

static struct argp argp = { options, parse_opt, nullptr, doc, nullptr, nullptr, nullptr};

int main(int argc, char **argv)
{
	arguments args;
	argp_parse(&argp, argc, argv, 0, 0, &args);

	using namespace std;
	uint64_t message = 0x0123456789ABCDEF;
    uint64_t key56 = args.set_key ?
        args.key : 1000000;
	uint64_t key = des_cpu::rev_permute_add_parity(key56);
    uint64_t limit = args.set_limit ?
        args.limit : (key56 + 1);

	cout << left << setw(16) << "Key:"
        << "0x" << setfill('0') << setw(16) << hex << key << endl;
	cout << left << setfill(' ') <<  setw(16) << "Message:"
        << "0x" << setfill('0') << setw(16) << hex << message << endl;
	auto encrypted = des_cpu::encrypt(key, message);
	cout << left << setfill(' ') << setw(16) << "Encrypted:"
        << "0x" << setfill('0') << setw(16) << hex << encrypted << endl;
	cout << left << setfill(' ') << setw(16) << "Decrypted:" 
        << "0x" << setfill('0') << setw(16) << hex << des_cpu::decrypt(key, encrypted) << endl;
    if(args.run_cpu)
    {
        auto cracked = des_cpu::crack(message, encrypted, 0, limit);
	    cout << left << setfill(' ') << setw(16) << "Cracked (CPU):"
            << "0x" << setfill('0') << setw(16) << hex << cracked << endl;
    }
    if(args.run_gpu)
    {
        auto cracked = des_gpu_crack(message, encrypted, 0, limit);
	    cout << left << setfill(' ') << setw(16) << "Cracked (GPU):"
            << "0x" << setfill('0') << setw(16) << hex << cracked << endl;
    }
	return 0;
}

static error_t parse_opt(int key, char* arg, struct argp_state *state)
{
	arguments *args = (arguments*)state->input;

	switch(key)
	{
		case 'l':
            if(arg[1] == 'x')
                args->limit = strtoull(arg, nullptr, 16);
            else
                args->limit = strtoull(arg, nullptr, 10);
            args->set_limit = true;
			break;
		case 'k':
            if(arg[1] == 'x')
                args->key = strtoull(arg, nullptr, 16);
            else
                args->key = strtoull(arg, nullptr, 10);
            args->set_key = true;
			break;
		case 'g':
            args->run_gpu = true;
			break;
		case 'c':
            args->run_cpu = true;
			break;
        case ARGP_KEY_END:
            break;
		default:
			return ARGP_ERR_UNKNOWN;
	}
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
