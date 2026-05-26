// Emit OneDayCounter reference values.
//
// OneDayCounter is the "1/1" day counter — dayCount returns sign (+1/-1)
// and yearFraction = dayCount. Probe a handful of (d1, d2) pairs to verify
// the sign convention and name.

#include <ql/time/daycounters/one.hpp>
#include <ql/time/date.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    OneDayCounter dc;

    std::cout << "{\n";
    std::cout << "  \"name\": \"" << dc.name() << "\",\n";

    struct DC { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2; };
    const DC dc_cases[] = {
        {2024, January,  1,  2024, January,  1},   // same day → +1
        {2024, January,  1,  2024, January,  2},   // forward → +1
        {2024, January,  2,  2024, January,  1},   // backward → -1
        {2024, January,  1,  2025, January,  1},   // 1 year forward → +1
        {2025, January,  1,  2024, January,  1},   // 1 year backward → -1
    };
    std::cout << "  \"cases\": [\n";
    bool first = true;
    for (const auto& c : dc_cases) {
        if (!first) std::cout << ",\n";
        Date a(c.d1, c.m1, c.y1);
        Date b(c.d2, c.m2, c.y2);
        std::cout << "    {"
                  << "\"y1\": " << c.y1 << ", \"m1\": " << static_cast<int>(c.m1) << ", \"d1\": " << c.d1
                  << ", \"y2\": " << c.y2 << ", \"m2\": " << static_cast<int>(c.m2) << ", \"d2\": " << c.d2
                  << ", \"day_count\": " << dc.dayCount(a, b)
                  << ", \"year_fraction\": " << dc.yearFraction(a, b)
                  << "}";
        first = false;
    }
    std::cout << "\n  ]\n";
    std::cout << "}\n";
    return 0;
}
