#include <algorithm>
#include <execution>
#include <string>

int
main()
{
    std::string s("hel0000000000000lo");
    std::transform(std::execution::par, s.begin(), s.end(), s.begin(),
                   [](unsigned char c) -> unsigned char
                   { return std::toupper(c); });
}