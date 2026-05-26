// Emit SimpleDayCounter + Business252 reference values.

#include <ql/time/calendars/weekendsonly.hpp>
#include <ql/time/daycounters/business252.hpp>
#include <ql/time/daycounters/simpledaycounter.hpp>
#include <ql/time/date.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

struct DC { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2; };

const DC g_cases[] = {
    {2024, January,   1, 2024, January,   1},   // same day
    {2024, January,  15, 2024, July,     15},   // 6 months (whole-month for Simple)
    {2024, January,   1, 2025, January,   1},   // 1 year
    {2024, February, 29, 2025, February, 28},   // leap-end → non-leap-end
    {2024, January,  15, 2024, March,    31},   // partial month (Simple falls back to 30/360 BondBasis)
    {2024, March,    15, 2024, September,15},   // 6 months
    {2024, March,    15, 2024, March,    20},   // 5 days within same month (Business252 quick path)
    {2024, March,    15, 2024, April,    15},   // 1 month
    {2024, December, 31, 2025, January,  31},   // cross year
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

    emit_section("simple", SimpleDayCounter());
    std::cout << ",\n";

    emit_section("business252_weekends", Business252(WeekendsOnly()));
    std::cout << "\n";

    std::cout << "}\n";
    return 0;
}
