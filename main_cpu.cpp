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
is_point_in_polygon(std::pair<float, float> const & point,
                    std::vector<std::pair<float, float>> const & polygon)
{
    std::size_t const poly_vert_count{ polygon.size() };
    uint_fast8_t intersect_count{ 0U };
    float const pq_x{ point.first };  // x coordinate of query point
    float const pq_y{ point.second }; // y coordinate of query point
    for(std::size_t i{ 0U }, j{ poly_vert_count - 1U }; i < poly_vert_count;
        ++i)
    {
        float const pj_x{ polygon[j].first };
        float const pj_y{ polygon[j].second };
        float const pi_x{ polygon[i].first };
        float const pi_y{ polygon[i].second };
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
are_points_in_polygon(std::vector<std::pair<float, float>> const & points,
                      std::vector<std::pair<float, float>> const & polygon)
{
    auto const poly_vert_count{ polygon.size() };
    if(poly_vert_count < 3U)
    {
        return {};
    }
    auto const point_count{ points.size() };
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

void
edge_cases()
{
    std::vector<std::pair<float, float>> points{
        { 0.114946, 0.569086 }, { 0.492721, 0.805343 }, { 0.912403, 0.057812 },
        { 0.358540, 0.062636 }, { 0.898605, 0.162964 }, { 0.108513, 0.376477 },
        { 0.455804, 0.770584 }, { 0.715986, 0.874619 }, { 0.163008, 0.220783 },
        { 0.556412, 0.333519 }, { 0.784624, 0.206073 }, { 0.020554, 0.809417 },
        { 0.133199, 0.098011 }, { 0.652125, 0.476056 }, { 0.354246, 0.785854 },
        { 0.694551, 0.193726 }, { 0.669862, 0.658918 }, { 0.607459, 0.286411 },
        { 0.380459, 0.302769 }, { 0.872723, 0.624909 }, { 0.871701, 0.205591 },
        { 0.045234, 0.820016 }, { 0.974947, 0.243933 }, { 0.994142, 0.018648 },
        { 0.745300, 0.616132 }
    };
    std::vector<std::pair<float, float>> unit_square{ { 0.0, 0.0 },
                                                      { 0.0, 1.0 },
                                                      { 1.0, 1.0 },
                                                      { 1.0, 0.0 } };
    int i = 0;
    for(auto const point : points)
    {
        uint_fast8_t const res{ is_point_in_polygon(point, unit_square) };
        if(!res)
        {
            ++i;
            puts("failure");
        }
    }
    printf("%d of %lu incorrect\n", i, points.size());
}

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
    std::vector<uint_fast8_t> are_points_in_polygon_out{ are_points_in_polygon(
        points, unit_square) };
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
