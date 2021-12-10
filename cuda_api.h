#ifndef CUDA_API_H
#define CUDA_API_H

#include <cstddef>

void * cuda_malloc(std::size_t const bytes);
void cuda_free(void const * const ptr);
void * cuda_memcpy(void * const dest, void const * const src, std::size_t const bytes, cudaMemcpyKind const kind);
void * cuda_push(void const * const from_host, void * const to_device, std::size_t const bytes);
void * cuda_pull(void const * const from_device, void * const to_host, std::size_t const bytes);

#endif  // CUDA_API_H
