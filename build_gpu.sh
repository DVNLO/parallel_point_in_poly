#! /bin/bash
nvc++ -Wall --pedantic -Wextra -O3 -march=native -std=c++20 ./main_gpu.cu -o gpu.out ;
