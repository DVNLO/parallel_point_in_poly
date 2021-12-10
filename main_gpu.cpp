#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <random>
#include <string>
#include <utility>
#include <vector>

uint_fast8_t
is_point_in_polygon(float const pq_x, float const pq_y,
                    float const * const polygon_x,
                    float const * const polygon_y,
                    std::size_t polygon_vertex_count)
{
    uint_fast8_t intersect_count{ 0U };
    for(std::size_t i{ 0U }, j{ polygon_vertex_count - 1U };
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

// issue compiling for cpu
// https://www.reddit.com/r/cpp_questions/comments/n4xg3h/does_execution_policy_in_stdtransform_in_gcc_have/

// issue using bool
/*
begin test_unit_square
end test_unit_square
184921 : (0.215338,0.423970) : 1 =\= 0
185357 : (0.891123,0.461615) : 1 =\= 0
185522 : (0.336506,0.730225) : 1 =\= 0
311530 : (0.440207,0.813861) : 1 =\= 0
311535 : (0.835318,0.045202) : 1 =\= 0
311542 : (0.684667,0.986953) : 1 =\= 0
373625 : (0.324988,0.555293) : 1 =\= 0
test_unit_square passed

*/
std::vector<uint_fast8_t>
are_points_in_polygon(float const * const points_x,
                      float const * const points_y,
                      std::size_t const point_count,
                      float const * const polygon_x,
                      float const * const polygon_y,
                      std::size_t polygon_vertex_count)
{
    if(polygon_vertex_count < 3U)
    {
        return {};
    }
    if(!point_count)
    {
        return {};
    }
    std::vector<uint_fast8_t> ret(point_count, false);
    std::transform(std::execution::par_unseq, points.cbegin(), points.cend(),
                   ret.begin(),
                   [&](std::pair<float, float> const & point) -> uint_fast8_t
                   { return is_point_in_polygon(point, polygon); });
    return ret;
}
/* nvc++ refuses to compile with -stdpar=multicore there is an issue with
 * "thrust" library. "no instance of function template
 * thrust::detail::unary_transform_functor<UnaryFunction>::operator()"
 * https://github.com/NVIDIA/thrust/issues/624
 */

bool
test_unit_square()
{
    puts("begin test_unit_square");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 2.0);
    std::size_t point_count{ 1'000'000'000 };
    puts("begin point allocation");
    std::vector<std::pair<float, float>> points(point_count);
    puts("begin point generation");
    for(std::size_t i{ 0U }; i < point_count; ++i)
    {
        points[i] = { dis(gen), dis(gen) };
    }
    puts("begin unit square allocation");
    std::vector<std::pair<float, float>> unit_square{ { 0.0, 0.0 },
                                                      { 0.0, 1.0 },
                                                      { 1.0, 1.0 },
                                                      { 1.0, 0.0 } };
    puts("begin point testing");
    auto const begin{ std::chrono::high_resolution_clock::now() };
    // push points
    // push polygon
    std::vector<uint_fast8_t> are_points_in_polygon_out{ are_points_in_polygon(
        points, unit_square) };
    // pull result
    auto const end{ std::chrono::high_resolution_clock::now() };
    std::chrono::duration<double> elapsed_seconds{ end - begin };
    printf("elapsed seconds : %f\n", elapsed_seconds.count());
    puts("begin point verification");
    for(std::size_t i{ 0U }; i < point_count; ++i)
    {
        uint_fast8_t const s0{ (0.0 < points[i].first && points[i].first < 1.0
                                && 0.0 < points[i].second
                                && points[i].second < 1.0) };
        if(s0 != are_points_in_polygon_out[i])
        {
            printf("%lu : (%f,%f) : %d =\\= %d\n", i, points[i].first,
                   points[i].second, (int)(s0),
                   (int)(are_points_in_polygon_out[i]));
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
