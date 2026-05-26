// Emit Thirty360 (all 9 convention aliases) + Thirty365 reference values.

#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/daycounters/thirty365.hpp>
#include <ql/time/date.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

struct DC { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2; };

// (d1, d2) battery covering all the 30/360 corner cases:
//   - Day 31 / day 30 / day 28 / day 29 / Feb end (leap + non-leap)
//   - Forward + backward + same-day
//   - Year boundary
const DC g_cases[] = {
    {2024, January,   1, 2024, January,   1},   // same day
    {2024, January,  31, 2024, March,    31},   // both 31 (US, ISMA differ)
    {2024, January,  30, 2024, March,    31},   // d2 = 31 with d1 = 30
    {2024, January,  15, 2024, March,    31},   // d2 = 31 with d1 < 30
    {2024, February, 28, 2024, March,    31},   // Feb last (non-leap year would be 28; 2024 is leap)
    {2024, February, 29, 2024, March,    31},   // Feb last (leap)
    {2023, February, 28, 2023, March,    31},   // Feb last (non-leap)
    {2024, February, 29, 2025, February, 28},   // Feb-end leap → Feb-end non-leap
    {2024, March,    15, 2024, September,15},   // mid-year, no edge cases
    {2024, December, 31, 2025, January,  31},   // year boundary, both 31
    {2024, July,      1, 2024, June,     30},   // backward
};

void emit_section(const char* key, const DayCounter& dc) {
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

    // Thirty360 — every convention (BondBasis/ISMA share impl, European/
    // EurobondBasis share impl, ISDA/German share impl; we still probe
    // the alias name to confirm dispatch).
    emit_section("usa",            Thirty360(Thirty360::USA));            std::cout << ",\n";
    emit_section("bond_basis",     Thirty360(Thirty360::BondBasis));      std::cout << ",\n";
    emit_section("european",       Thirty360(Thirty360::European));       std::cout << ",\n";
    emit_section("eurobond_basis", Thirty360(Thirty360::EurobondBasis));  std::cout << ",\n";
    emit_section("italian",        Thirty360(Thirty360::Italian));        std::cout << ",\n";
    emit_section("german",         Thirty360(Thirty360::German));         std::cout << ",\n";
    emit_section("isma",           Thirty360(Thirty360::ISMA));           std::cout << ",\n";
    emit_section("isda",           Thirty360(Thirty360::ISDA));           std::cout << ",\n";
    emit_section("nasd",           Thirty360(Thirty360::NASD));           std::cout << ",\n";

    // Thirty365.
    emit_section("thirty365",      Thirty365());                          std::cout << ",\n";

    // ISDA convention with a non-default terminationDate. Probe one case
    // where d2 == terminationDate so isLastOfFebruary(d2) check is skipped.
    {
        Date term(29, February, 2024);
        Thirty360 dc(Thirty360::ISDA, term);
        std::cout << "  \"isda_term_2024_02_29\": {\n";
        std::cout << "    \"name\": \"" << dc.name() << "\",\n";
        std::cout << "    \"termination\": {\"y\": 2024, \"m\": 2, \"d\": 29},\n";
        std::cout << "    \"cases\": [\n";
        // d1 in Jan, d2 == termination (Feb 29 leap) — the d2 last-of-Feb
        // rule is SKIPPED because d2 == terminationDate, so dd2 stays 29.
        Date a(15, January, 2024);
        std::cout << "      {"
                  << "\"y1\": 2024, \"m1\": 1, \"d1\": 15"
                  << ", \"y2\": 2024, \"m2\": 2, \"d2\": 29"
                  << ", \"day_count\": " << dc.dayCount(a, term)
                  << ", \"year_fraction\": " << dc.yearFraction(a, term)
                  << "}\n    ]\n  }\n";
    }

    std::cout << "}\n";
    return 0;
}
