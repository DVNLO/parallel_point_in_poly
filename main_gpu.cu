#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cuda_api.cu"
#include "support.cu"

#define BLOCK_SZ 512U

std::size_t
div_ceil(std::size_t const val, std::size_t divisor)
// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
{
    return val ? 1 + ((val - 1) / divisor) : val;
}

unsigned
is_point_in_polygon(float const pq_x, float const pq_y,
                    float const * const polygon_x,
                    float const * const polygon_y,
                    unsigned long long polygon_vertex_count)
{
    unsigned intersect_count{ 0U };
    for(unsigned long long i{ 0U }, j{ polygon_vertex_count - 1U };
        i < polygon_vertex_count; ++i)
    {
        float const pj_x{ polygon_x[j] };
        float const pj_y{ polygon_y[j] };
        float const pi_x{ polygon_x[i] };
        float const pi_y{ polygon_y[i] };
        // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
        intersect_count
            += ((pi_y > pq_y) != (pj_y > pq_y))
               && (pq_x < (pj_x - pi_x) * (pq_y - pi_y) / (pj_y - pi_y) + pi_x);
        j = i;
    }
    return intersect_count & 0x01U;
}

__global__ void
are_points_in_polygon_kernel(float const * const points_x,
                             float const * const points_y,
                             unsigned long long const point_count,
                             float const * const polygon_x,
                             float const * const polygon_y,
                             unsigned long long polygon_vertex_count,
                             unsigned * const are_points_in_polygon_out)
{
    // uses dynamic shared memory
    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
    extern __shared__ float dynamic_shmem[];
    float * const local_polygon_x = dynamic_shmem;
    float * const local_polygon_y = local_polygon_x + polygon_vertex_count;

    unsigned int const block_idx_x = blockIdx.x;
    unsigned int const block_dim_x = blockDim.x;
    unsigned int const grid_dim_x = gridDim.x;
    unsigned int const thread_count = block_dim_x * grid_dim_x;
    unsigned int const t_idx_x = threadIdx.x;
    for(unsigned int i = t_idx_x; i < polygon_vertex_count; i += block_dim_x)
    {
        local_polygon_x[i] = polygon_x[i];
        local_polygon_y[i] = polygon_y[i];
    }
    __syncthreads();
    unsigned int const begin_idx = t_idx_x + block_idx_x * block_dim_x;
    for(unsigned int i = begin_idx; i < point_count; i += thread_count)
    {
        float const pq_x = points_x[i];
        float const pq_y = points_y[i];
        are_points_in_polygon_out[t_idx_x] = is_point_in_polygon(
            pq_x, pq_y, local_polygon_x, local_polygon_y, polygon_vertex_count);
    }
}

// issue changed interface since cpp stl types were not easily
// transferable using the cuda memory library primitives. Concern that
// c-style code will propigate into cpp application when using cuda.
void
are_points_in_polygon(float const * const points_x_h,
                      float const * const points_y_h,
                      unsigned long long const point_count,
                      float const * const polygon_x_h,
                      float const * const polygon_y_h,
                      unsigned long long polygon_vertex_count,
                      unsigned * const are_points_in_polygon_out_h)
{
    if(polygon_vertex_count < 3U)
    {
        return;
    }
    if(!point_count)
    {
        return;
    }
    unsigned long long const point_count_bytes = point_count * sizeof(float);
    float * points_x_d = (float *)(cuda_malloc(point_count_bytes));
    float * points_y_d = (float *)(cuda_malloc(point_count_bytes));

    unsigned long long const polygon_vertex_count_bytes
        = polygon_vertex_count * sizeof(float);
    float * polygon_x_d = (float *)(cuda_malloc(polygon_vertex_count_bytes));
    float * polygon_y_d = (float *)(cuda_malloc(polygon_vertex_count_bytes));

    unsigned long long const out_bytes = point_count * sizeof(unsigned);
    unsigned * const are_points_in_polygon_out_d
        = (unsigned * const)(cuda_malloc(out_bytes));

    cuda_push(points_x_h, points_x_d, point_count_bytes);
    cuda_push(points_y_h, points_y_d, point_count_bytes);
    cuda_push(polygon_x_h, polygon_x_d, polygon_vertex_count_bytes);
    cuda_push(polygon_y_h, polygon_y_d, polygon_vertex_count_bytes);

    std::size_t dynamic_shared_memory_sz = 2 * point_count_bytes;
    std::size_t const grid_sz{ div_ceil(point_count, BLOCK_SZ) };
    dim3 dim_grid(grid_sz, 1, 1);
    dim3 dim_block(BLOCK_SZ, 1, 1);

    cudaDeviceSynchronize();

    are_points_in_polygon_kernel<<<dim_grid, dim_block,
                                   dynamic_shared_memory_sz>>>(
        points_x_d, points_y_d, point_count, polygon_x_d, polygon_y_d,
        polygon_vertex_count, are_points_in_polygon_out_d);

    cudaError_t cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess)
    {
        // bad but works for now
        FATAL("Unable to launch kernel");
    }

    cuda_pull(are_points_in_polygon_out_d, are_points_in_polygon_out_h,
              out_bytes);
    cuda_free(points_x_d);
    cuda_free(points_y_d);
    cuda_free(polygon_x_d);
    cuda_free(polygon_y_d);
    cuda_free(are_points_in_polygon_out_d);
}

bool
test_unit_square()
{
    puts("begin test_unit_square");
    unsigned long long point_count = 1000000000;
    puts("begin point allocation");
    unsigned long long const point_count_bytes = point_count * sizeof(float);
    float * points_x = (float *)(malloc(point_count_bytes));
    float * points_y = (float *)(malloc(point_count_bytes));

    unsigned long long const out_bytes = point_count * sizeof(unsigned);
    unsigned * const are_points_in_polygon_out
        = (unsigned * const)(malloc(out_bytes));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 2.0);
    puts("begin point generation");
    for(std::size_t i{ 0U }; i < point_count; ++i)
    {
        points_x[i] = dis(gen);
        points_y[i] = dis(gen);
        are_points_in_polygon_out[i] = 0;
    }
    puts("init unit square");
    float unit_square_x[] = { 0.0, 0.0, 1.0, 1.0 };
    float unit_square_y[] = { 0.0, 1.0, 1.0, 0.0 };
    unsigned long long polygon_vertex_count = 4;

    puts("begin point testing");
    Timer timer;
    startTime(&timer);
    are_points_in_polygon(points_x, points_y, point_count, unit_square_x,
                          unit_square_y, polygon_vertex_count,
                          are_points_in_polygon_out);
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));
    puts("begin point verification");
    for(std::size_t i{ 0U }; i < point_count; ++i)
    {
        unsigned const s0{ (0.0 < points_x[i] && points_x[i] < 1.0
                            && 0.0 < points_y[i] && points_y[i] < 1.0) };
        if(s0 != are_points_in_polygon_out[i])
        {
            printf("%lu : (%f,%f) : %u =\\= %u\n", i, points_x[i], points_y[i],
                   s0, are_points_in_polygon_out[i]);
            return false;
        }
    }
    return true;
}

int
main()
{
    // run test trials
    for(int i{ 0 }; i < 32; ++i)
    {
        printf("################# TEST : %d\n", i);
        bool rc = test_unit_square();
        if(rc)
        {
            puts("test_unit_square passed");
        }
        else
        {
            puts("test_unit_square failed");
        }
    }
}