// Emit ASX + ECB reference values in one probe (small enough to share).

#include <ql/time/asx.hpp>
#include <ql/time/date.hpp>
#include <ql/time/ecb.hpp>

#include <iostream>
#include <set>
#include <string>

using namespace QuantLib;

int main() {
    std::cout << "{\n";

    // ===== ASX ==============================================================
    std::cout << "  \"asx\": {\n";

    // is_asx_date — 2nd Friday of each month is ASX; main cycle is HMUZ
    struct DC { Year y; Month m; Day d; bool main_cycle; };
    const DC asx_dc[] = {
        {2024, March,    8,  true},  // 2nd Fri Mar 2024 → main ASX
        {2024, June,    14,  true},  // 2nd Fri Jun 2024 (14th not 8th — let me check)
        {2024, September,13, true},
        {2024, December, 13, true},
        {2024, January,  12, true},  // 2nd Fri Jan → not main
        {2024, January,  12, false}, // 2nd Fri Jan → non-main ASX
        {2024, March,     9, true},  // Sat → no
        {2024, March,    15, true},  // Fri but day > 14 → no
        {2024, March,     1, true},  // Fri but day < 8 → no
    };
    std::cout << "    \"is_asx_date\": [\n";
    {
        bool first = true;
        for (const auto& c : asx_dc) {
            if (!first) std::cout << ",\n";
            Date d(c.d, c.m, c.y);
            std::cout << "      {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"main_cycle\": " << (c.main_cycle ? "true" : "false")
                      << ", \"result\": " << (ASX::isASXdate(d, c.main_cycle) ? "true" : "false")
                      << "}";
            first = false;
        }
        std::cout << "\n    ],\n";
    }

    // is_asx_code
    struct CC { const char* code; bool main_cycle; };
    const CC asx_cc[] = {
        {"H4", true},  {"M4", true},  {"U4", true},  {"Z4", true},
        {"H4", false}, {"F4", false}, {"K4", false},
        {"F4", true},     // F (Jan) not in main cycle
        {"AB", false},    // not a letter
    };
    std::cout << "    \"is_asx_code\": [\n";
    {
        bool first = true;
        for (const auto& c : asx_cc) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"code\": \"" << c.code
                      << "\", \"main_cycle\": " << (c.main_cycle ? "true" : "false")
                      << ", \"result\": " << (ASX::isASXcode(c.code, c.main_cycle) ? "true" : "false")
                      << "}";
            first = false;
        }
        std::cout << "\n    ],\n";
    }

    // code() for known ASX dates
    struct CD { Year y; Month m; Day d; };
    const CD asx_cd[] = {
        {2024, January,  12},   // F4
        {2024, March,     8},   // H4
        {2024, June,     14},   // M4
        {2024, September,13},   // U4
        {2024, December, 13},   // Z4
    };
    std::cout << "    \"code\": [\n";
    {
        bool first = true;
        for (const auto& c : asx_cd) {
            if (!first) std::cout << ",\n";
            Date d(c.d, c.m, c.y);
            std::cout << "      {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"code\": \"" << ASX::code(d) << "\"}";
            first = false;
        }
        std::cout << "\n    ],\n";
    }

    // date(code, ref)
    struct DR { const char* code; Year ry; };
    const DR asx_dr[] = {
        {"H4", 2024}, {"M4", 2024}, {"Z4", 2024},
    };
    std::cout << "    \"date\": [\n";
    {
        bool first = true;
        for (const auto& r : asx_dr) {
            if (!first) std::cout << ",\n";
            Date ref(1, January, r.ry);
            Date d = ASX::date(r.code, ref);
            std::cout << "      {\"code\": \"" << r.code
                      << "\", \"ref_y\": " << r.ry
                      << ", \"out_y\": " << d.year()
                      << ", \"out_m\": " << static_cast<int>(d.month())
                      << ", \"out_d\": " << d.dayOfMonth() << "}";
            first = false;
        }
        std::cout << "\n    ],\n";
    }

    // next_date
    struct ND { Year y; Month m; Day d; bool main_cycle; };
    const ND asx_nd[] = {
        {2024, January,   1, true},     // next main ASX = Mar 8
        {2024, March,     7, true},     // before Mar 8 → Mar 8
        {2024, March,     8, true},     // ON Mar 8 → next is Jun 14
        {2024, January,   1, false},    // next non-main = Jan 12 (2nd Fri)
    };
    std::cout << "    \"next_date\": [\n";
    {
        bool first = true;
        for (const auto& n : asx_nd) {
            if (!first) std::cout << ",\n";
            Date d(n.d, n.m, n.y);
            Date nx = ASX::nextDate(d, n.main_cycle);
            std::cout << "      {\"y\": " << n.y
                      << ", \"m\": " << static_cast<int>(n.m)
                      << ", \"d\": " << n.d
                      << ", \"main_cycle\": " << (n.main_cycle ? "true" : "false")
                      << ", \"out_y\": " << nx.year()
                      << ", \"out_m\": " << static_cast<int>(nx.month())
                      << ", \"out_d\": " << nx.dayOfMonth() << "}";
            first = false;
        }
        std::cout << "\n    ]\n";
    }
    std::cout << "  },\n";

    // ===== ECB ==============================================================
    std::cout << "  \"ecb\": {\n";

    // is_ecb_code
    const char* ecb_codes[] = {
        "MAR10", "DEC23", "JUN15",
        "XYZ10",  // invalid month
        "MAR1",   // wrong length
        "MARAB",  // digits expected
    };
    std::cout << "    \"is_ecb_code\": [\n";
    {
        bool first = true;
        for (const auto& c : ecb_codes) {
            if (!first) std::cout << ",\n";
            std::cout << "      {\"code\": \"" << c
                      << "\", \"result\": " << (ECB::isECBcode(c) ? "true" : "false")
                      << "}";
            first = false;
        }
        std::cout << "\n    ],\n";
    }

    // known_dates_count + first / last
    {
        const std::set<Date>& known = ECB::knownDates();
        std::cout << "    \"known_dates_count\": " << known.size() << ",\n";
        std::cout << "    \"known_dates_first_serial\": " << known.begin()->serialNumber() << ",\n";
        std::cout << "    \"known_dates_last_serial\": " << known.rbegin()->serialNumber() << ",\n";
    }

    // next_date — first known date is 2005-01-12 (serial 38371). Probe a few.
    struct EN { Date input; };
    Date inputs[] = {
        Date(38370),  // day before first known
        Date(38371),  // exactly first known
        Date(43859),  // first 2020 date
        Date(45000),  // mid-table
    };
    std::cout << "    \"next_date\": [\n";
    {
        bool first = true;
        for (const auto& i : inputs) {
            if (!first) std::cout << ",\n";
            Date n = ECB::nextDate(i);
            std::cout << "      {\"input_serial\": " << i.serialNumber()
                      << ", \"out_serial\": " << n.serialNumber() << "}";
            first = false;
        }
        std::cout << "\n    ],\n";
    }

    // code() for first known date (2005-01-12 → JAN05)
    {
        Date first_known = *ECB::knownDates().begin();
        std::cout << "    \"first_known_code\": \"" << ECB::code(first_known) << "\",\n";
        std::cout << "    \"first_known_serial\": " << first_known.serialNumber() << ",\n";
    }

    // next_code for "MAR10" -> APR10; "DEC09" -> JAN10
    std::cout << "    \"next_code\": [\n";
    std::cout << "      {\"in\": \"MAR10\", \"out\": \"" << ECB::nextCode("MAR10") << "\"},\n";
    std::cout << "      {\"in\": \"DEC09\", \"out\": \"" << ECB::nextCode("DEC09") << "\"},\n";
    std::cout << "      {\"in\": \"DEC19\", \"out\": \"" << ECB::nextCode("DEC19") << "\"}\n";
    std::cout << "    ]\n";

    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
