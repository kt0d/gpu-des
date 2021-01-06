# gpu-des
DES cracker for CUDA with multi GPU support. 

You can provide it a message and a key, where key is supposed to be 56 bit wide, so after PC-1 permutation. 
It will then encrypt your message with given key, then try to find this key again by brute-force.
It attempts to decrypt the message with every possible key from key space in turn, and compares it to unencrypted message.

You can specify if you want to run the cracker on CPU, GPU or multiple GPUs. If you use GPU, you can specify number of CUDA thread blocks to run and number of threads in a thread block. Use `--help` to see all available options.

I based implementation of the DES algorithm on the following sources:

[The DES Algorithm Illustrated by J. Orlin Grabbe](http://page.math.tu-berlin.de/~kant/teaching/hess/krypto-ws2006/des.htm)

[Data Encryption Standard - Wikipedia](https://en.wikipedia.org/wiki/Data_Encryption_Standard#Description)

## Building
gpu-des depends on CUDA SDK, thrust, and Argp library. 
It also includes `helper_cuda.h` header file which is provided with NVIDIA's CUDA samples whose location depends on where CUDA toolkit is installed (see Makefile).
