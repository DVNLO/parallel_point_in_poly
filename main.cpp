#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

bool
is_point_in_polygon(std::pair<float, float> const & point,
                    std::vector<std::pair<float, float>> const & poly_vertices)
{
    auto const poly_vert_count = poly_vertices.size();
    if(poly_vert_count < 3U)
    {
        return 0U;
    }
    uint_fast8_t intersect_count = 0U;
    auto const point_x = point.first;
    auto const point_y = point.second;
    for(size_t i = 0U, j = poly_vert_count - 1U; i < poly_vert_count; ++i)
    {
        bool const s0{ point_x < poly_vertices[j].first
                       || point_x < poly_vertices[i].first };
        bool const s1{ poly_vertices[j].second < point_y
                       && point_y < poly_vertices[i].second };
        bool const s2{ poly_vertices[i].second < point_y
                       && point_y < poly_vertices[j].second };
        intersect_count += s0 && (s1 || s2);
        j = i;
    }
    return intersect_count & 0x01U;
}

void
are_points_in_polygon(
    std::vector<std::pair<float, float>> const & points,
    std::vector<std::pair<float, float>> const & poly_vertices,
    std::vector<bool> & are_points_in_polygon_out)
{
    auto const poly_vert_count = poly_vertices.size();
    if(poly_vert_count < 3U)
    {
        return;
    }
    auto const point_count = points.size();
    if(!point_count)
    {
        return;
    }
    are_points_in_polygon_out.resize(point_count);
    std::transform(
        std::execution::par_unseq, points.cbegin(), points.cend(),
        are_points_in_polygon_out.begin(),
        [&poly_vertices](std::pair<float, float> const & point) -> bool
        { return is_point_in_polygon(point, poly_vertices); });
}

int
main(int const argc, char const * const * const argv)
{
    std::vector<std::pair<float, float>> points;
    std::vector<std::pair<float, float>> polygon{ { 0.0, 0.0 },
                                                  { 0.0, 1.0 },
                                                  { 1.0, 1.0 },
                                                  { 1.0, 0.0 } };
    float delta = 0.0001;
    for(float x = 0.0 + delta; x < 1.0; x += delta)
    {
        for(float y = 0.0 + delta; y < 1.0; y += delta)
        {
            points.push_back({ x, y });
        }
    }
    std::vector<bool> are_points_in_polygon_out;
    puts("begin");
    are_points_in_polygon(points, polygon, are_points_in_polygon_out);
    puts("end");
    /*
    for(auto const is_point_in_polygon : are_points_in_polygon_out)
    {
        if(!is_point_in_polygon)
        {
            puts("f");
        }
    }
    */
}
