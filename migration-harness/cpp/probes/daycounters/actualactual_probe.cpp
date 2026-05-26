// Emit ActualActual reference values across the 4 distinct impls.
//
// Covers ISDA/Historical/Actual365 (share ISDA_Impl), AFB/Euro (share
// AFB_Impl), and Old_ISMA (Bond/ISMA without schedule). Schedule-based
// ISMA is exercised with a 6-month semi-annual Schedule.

#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/date.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

struct DC { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2; };

const DC g_cases[] = {
    {2024, January,  1, 2024, January,  1},   // same day
    {2024, January,  1, 2024, July,     1},   // 6 months
    {2024, January,  1, 2025, January,  1},   // 1 year (leap)
    {2023, January,  1, 2024, January,  1},   // 1 year (non-leap)
    {2024, February, 1, 2024, March,    1},   // 29 days (leap Feb)
    {2023, February, 1, 2023, March,    1},   // 28 days (non-leap Feb)
    {2023, July,     1, 2024, July,     1},   // 1 year crossing leap (366 days)
    {2024, December, 1, 2025, March,    1},   // 90 days cross-year
    {2024, July,     1, 2024, June,    30},   // backward
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

    // ISDA / Historical / Actual365 share ISDA_Impl — same name + values.
    emit_section("isda",       ActualActual(ActualActual::ISDA));        std::cout << ",\n";
    emit_section("historical", ActualActual(ActualActual::Historical));  std::cout << ",\n";
    emit_section("actual365",  ActualActual(ActualActual::Actual365));   std::cout << ",\n";

    // AFB / Euro share AFB_Impl.
    emit_section("afb",        ActualActual(ActualActual::AFB));         std::cout << ",\n";
    emit_section("euro",       ActualActual(ActualActual::Euro));        std::cout << ",\n";

    // ISMA / Bond without schedule → Old_ISMA_Impl.
    emit_section("isma_no_schedule",  ActualActual(ActualActual::ISMA)); std::cout << ",\n";
    emit_section("bond_no_schedule",  ActualActual(ActualActual::Bond)); std::cout << "\n";

    // ISMA with a schedule — must produce different name? No, same name "Actual/Actual (ISMA)".
    // We probe ISMA-with-schedule against a 6-month semi-annual Schedule
    // covering 2024-01-01..2025-01-01.
    {
        Schedule s(Date(1, January, 2024), Date(1, January, 2025), Period(6, Months),
                   NullCalendar(), Unadjusted, Unadjusted,
                   DateGeneration::Forward, false);
        ActualActual dc(ActualActual::ISMA, s);
        std::cout << ",\n  \"isma_with_schedule\": {\n";
        std::cout << "    \"name\": \"" << dc.name() << "\",\n";
        std::cout << "    \"schedule_dates_serials\": [";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << s[i].serialNumber();
        }
        std::cout << "],\n";
        std::cout << "    \"cases\": [\n";
        // Date pairs within the schedule's range.
        struct SC { Year y1; Month m1; Day d1; Year y2; Month m2; Day d2; };
        const SC sc[] = {
            {2024, January,   1, 2024, July,      1},   // exactly the first half
            {2024, July,      1, 2025, January,   1},   // exactly the second half
            {2024, January,   1, 2025, January,   1},   // full schedule
            {2024, February,  1, 2024, May,       1},   // sub-period
        };
        bool first = true;
        for (const auto& c : sc) {
            if (!first) std::cout << ",\n";
            Date a(c.d1, c.m1, c.y1);
            Date b(c.d2, c.m2, c.y2);
            std::cout << "      {"
                      << "\"y1\": " << c.y1 << ", \"m1\": " << static_cast<int>(c.m1) << ", \"d1\": " << c.d1
                      << ", \"y2\": " << c.y2 << ", \"m2\": " << static_cast<int>(c.m2) << ", \"d2\": " << c.d2
                      << ", \"year_fraction\": " << dc.yearFraction(a, b)
                      << "}";
            first = false;
        }
        std::cout << "\n    ]\n  }\n";
    }

    std::cout << "}\n";
    return 0;
}
