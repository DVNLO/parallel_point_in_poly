#! /bin/bash
g++-10 -march=native -Wall --pedantic -Wextra -O3 -std=c++20 ./main_cpu.cpp -ltbb -o cpu.out ;