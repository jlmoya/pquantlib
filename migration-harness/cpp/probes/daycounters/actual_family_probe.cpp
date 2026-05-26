// Emit Actual360 / Actual364 / Actual36525 / Actual365Fixed / Actual366
// reference values across a shared (d1, d2) battery.

#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual364.hpp>
#include <ql/time/daycounters/actual36525.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actual366.hpp>
#include <ql/time/date.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

struct DC { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2; };

// Shared (d1, d2) battery. Covers same-day, basic forward, leap-day boundary,
// year boundary, month-end, multi-year spans, negative (d1 > d2).
const DC g_cases[] = {
    {2024, January,   1, 2024, January,   1},   // same day
    {2024, January,   1, 2024, February,  1},   // 31 days
    {2024, February, 28, 2024, March,     1},   // 2 days (leap)
    {2024, February, 29, 2024, March,     1},   // 1 day (leap)
    {2023, February, 28, 2023, March,     1},   // 1 day (non-leap)
    {2024, January,   1, 2025, January,   1},   // 366 days (leap year)
    {2023, January,   1, 2024, January,   1},   // 365 days (non-leap)
    {2024, March,    15, 2024, September,15},   // 184 days (mid-year)
    {2024, December, 31, 2025, January,   1},   // year boundary
    {2024, January,  31, 2024, February, 29},   // 29 days (leap Feb)
    {2024, July,      1, 2024, June,     30},   // -1 (backward)
};

void emit_simple(const char* key, const DayCounter& dc) {
    std::cout << "  \"" << key << "\": {\n";
    std::cout << "    \"name\": \"" << dc.name() << "\",\n";
    std::cout << "    \"cases\": [\n";
    bool first = true;
    for (const auto& c : g_cases) {
        if (!first) std::cout << ",\n";
        Date a(c.d1, c.m1, c.y1);
        Date b(c.d2, c.m2, c.y2);
        std::cout << "      {"
                  << "\"y1\": " << c.y1 << ", \"m1\": " << static_cast<int>(c.m1) << ", \"d1\": " << c.d1
                  << ", \"y2\": " << c.y2 << ", \"m2\": " << static_cast<int>(c.m2) << ", \"d2\": " << c.d2
                  << ", \"day_count\": " << dc.dayCount(a, b)
                  << ", \"year_fraction\": " << dc.yearFraction(a, b)
                  << "}";
        first = false;
    }
    std::cout << "\n    ]\n  }";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    emit_simple("actual360", Actual360());
    std::cout << ",\n";

    emit_simple("actual360_inc", Actual360(true));
    std::cout << ",\n";

    emit_simple("actual364", Actual364());
    std::cout << ",\n";

    emit_simple("actual36525", Actual36525());
    std::cout << ",\n";

    emit_simple("actual36525_inc", Actual36525(true));
    std::cout << ",\n";

    emit_simple("actual365fixed_standard", Actual365Fixed(Actual365Fixed::Standard));
    std::cout << ",\n";

    emit_simple("actual365fixed_no_leap", Actual365Fixed(Actual365Fixed::NoLeap));
    std::cout << ",\n";

    emit_simple("actual366", Actual366());
    std::cout << ",\n";

    emit_simple("actual366_inc", Actual366(true));
    std::cout << "\n";

    // --- Actual365Fixed Canadian — needs explicit reference period --------
    {
        Actual365Fixed dc(Actual365Fixed::Canadian);
        std::cout << ",\n  \"actual365fixed_canadian\": {\n";
        std::cout << "    \"name\": \"" << dc.name() << "\",\n";
        std::cout << "    \"cases\": [\n";
        // (d1, d2, refStart, refEnd) — semi-annual periods
        struct CA { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2;
                    Year rs_y; Month rs_m; Day rs_d; Year re_y; Month re_m; Day re_d; };
        const CA ca[] = {
            // Sub-period within regular semi-annual period (Mar 15 → Apr 15 inside Mar 1–Sep 1).
            {2024, March,    15, 2024, April,    15,
             2024, March,     1, 2024, September, 1},
            // Long sub-period within a quarter (Mar 1 → Apr 15 inside Mar 1–Jun 1).
            {2024, March,     1, 2024, April,    15,
             2024, March,     1, 2024, June,      1},
            // Full semi-annual period (Mar 1 → Sep 1).
            {2024, March,     1, 2024, September, 1,
             2024, March,     1, 2024, September, 1},
            // Same day → 0.
            {2024, March,     1, 2024, March,     1,
             2024, March,     1, 2024, September, 1},
        };
        bool first = true;
        for (const auto& c : ca) {
            if (!first) std::cout << ",\n";
            Date a(c.d1, c.m1, c.y1);
            Date b(c.d2, c.m2, c.y2);
            Date rs(c.rs_d, c.rs_m, c.rs_y);
            Date re(c.re_d, c.re_m, c.re_y);
            std::cout << "      {"
                      << "\"y1\": " << c.y1 << ", \"m1\": " << static_cast<int>(c.m1) << ", \"d1\": " << c.d1
                      << ", \"y2\": " << c.y2 << ", \"m2\": " << static_cast<int>(c.m2) << ", \"d2\": " << c.d2
                      << ", \"rs_y\": " << c.rs_y << ", \"rs_m\": " << static_cast<int>(c.rs_m) << ", \"rs_d\": " << c.rs_d
                      << ", \"re_y\": " << c.re_y << ", \"re_m\": " << static_cast<int>(c.re_m) << ", \"re_d\": " << c.re_d
                      << ", \"year_fraction\": " << dc.yearFraction(a, b, rs, re)
                      << "}";
            first = false;
        }
        std::cout << "\n    ]\n  }\n";
    }

    std::cout << "}\n";
    return 0;
}
