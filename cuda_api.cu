#include "cuda_api.h"

void *
cuda_malloc(std::size_t const bytes)
{
    void * ret{ nullptr };
    cudaError_t const rc{ cudaMalloc(&ret, bytes) };
    if(rc != cudaSuccess)
    {
        return nullptr;
    }
    return ret;
}

void
cuda_free(void * ptr)
{
    cudaFree(ptr);
}

void *
cuda_memcpy(void * const dest, void const * const src, std::size_t const bytes,
            cudaMemcpyKind const kind)
{
    cudaError_t const rc{ cudaMemcpy(
        const_cast<void *>(dest), const_cast<void const *>(src), bytes, kind) };
    if(rc != cudaSuccess)
    {
        return nullptr;
    }
    return dest;
}

void *
cuda_push(void const * const from_host, void * const to_device,
          std::size_t const bytes)
{
    return cuda_memcpy(to_device, from_host, bytes, cudaMemcpyHostToDevice);
}

void *
cuda_pull(void const * const from_device, void * const to_host,
          std::size_t const bytes)
{
    return cuda_memcpy(to_host, from_device, bytes, cudaMemcpyDeviceToHost);
}
