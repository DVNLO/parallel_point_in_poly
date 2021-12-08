#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* returns 0x01 if the point at position (point_x, point_y) is inside the
 * polygon represented by the parallel arrays poly_vertices_x and
 * poly_vertices_y.  The i-th point is represented by the tuple
 * (poly_vertices_x[i], poly_vertices_y[i]). Otherwise, returns 0x00.
 */
uint_fast8_t
is_point_in_polygon(float const * const restrict poly_vertices_x,
                    float const * const restrict poly_vertices_y,
                    size_t const poly_vertex_count, float const point_x,
                    float const point_y)
{
    if(poly_vertex_count < 3U)
    {
        return 0U;
    }
    uint_fast8_t intersect_count = 0U;
    for(size_t i = 0U, j = poly_vertex_count - 1U; i < poly_vertex_count; ++i)
    {
        uint_fast8_t const s0
            = point_x < poly_vertices_x[j] || point_x < poly_vertices_x[i];
        uint_fast8_t const s1
            = poly_vertices_y[j] < point_y && point_y < poly_vertices_y[i];
        uint_fast8_t const s2
            = poly_vertices_y[i] < point_y && point_y < poly_vertices_y[j];
        intersect_count += s0 && (s1 || s2);
        j = i;
    }
    return intersect_count & 0x01U;
}

// the real question
void
are_points_in_polygon();

int
main(int const argc, char const * const * const argv)
{
    size_t const n = 4U;
    float arr_x[] = { 0.0, 0.0, 1.0, 1.0 };
    float arr_y[] = { 0.0, 1.0, 1.0, 0.0 };
    float delta = 0.00001;
    size_t i = 0;
    for(float x = 0.0 + delta; x < 1.0; x += delta)
    {
        for(float y = 0.0 + delta; y < 1.0; y += delta)
        {
            uint_fast8_t is_in_polygon
                = is_point_in_polygon(arr_x, arr_y, n, x, y);
            if(!is_in_polygon)
            {
                printf("%ld : %f %f\n", i, x, y);
            }
            ++i;
        }
    }
}
