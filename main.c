#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint_fast8_t
is_odd(uint_fast8_t const val)
{
    return val & 0x01;
}

/* returns 0x01 if the point at position (x, y) is inside the polygon
 * represented by the parallel arrays p_x and p_y.
 */
uint_fast8_t
is_point_in_polygon(float const * const restrict p_x,
                    float const * const restrict p_y, size_t const n,
                    float const x, float const y)
{
    if(n < 3U)
    {
        return 0U;
    }
    uint_fast8_t intersect_count = 0U;
    for(size_t i = 0U, j = n - 1; i < n; ++i)
    {
        if(p_x[j] < x && p_x[i] < x)
        {
            continue;
        }
        if((p_y[j] < y && y < p_y[i]) || (p_y[i] < y && y < p_y[j]))
        {
            ++intersect_count;
        }
        j = i;
    }

    return is_odd(intersect_count);
}

int
main(int const argc, char const * const * const argv)
{
    size_t const n = 5U;
    float arr_x[] = { 0, 2, 3, 4, 7 };
    float arr_y[] = { 1, 6, 5, 4, -3 };
    float x = 0;
    float y = 7;
    uint_fast8_t is_in_polygon = is_point_in_polygon(arr_x, arr_y, n, x, y);
    printf("%d\n", (int)(is_in_polygon));
}
