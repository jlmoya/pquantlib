// Emit IMM reference values: isIMMdate, isIMMcode, code(d), date(code, ref),
// nextDate (date and code variants).

#include <ql/time/imm.hpp>
#include <ql/time/date.hpp>

#include <iostream>
#include <string>

using namespace QuantLib;

int main() {
    std::cout << "{\n";

    // --- is_imm_date ----------------------------------------------------
    struct DC { Year y; Month m; Day d; bool main_cycle; };
    const DC dc[] = {
        {2013, March,    20, true},   // 3rd Wed Mar → main IMM
        {2013, June,     19, true},
        {2013, September,18, true},
        {2013, December, 18, true},
        {2013, January,  16, true},   // 3rd Wed but Jan → not main
        {2013, January,  16, false},  // 3rd Wed Jan → non-main IMM
        {2013, March,    21, true},   // Thu → no
        {2013, March,    13, true},   // Wed but day < 15 → no
        {2013, March,    27, true},   // Wed but day > 21 → no
    };
    std::cout << "  \"is_imm_date\": [\n";
    {
        bool first = true;
        for (const auto& c : dc) {
            if (!first) std::cout << ",\n";
            Date d(c.d, c.m, c.y);
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"main_cycle\": " << (c.main_cycle ? "true" : "false")
                      << ", \"result\": " << (IMM::isIMMdate(d, c.main_cycle) ? "true" : "false")
                      << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- is_imm_code ----------------------------------------------------
    struct CC { const char* code; bool main_cycle; };
    const CC cc[] = {
        {"H3", true},  {"M3", true},  {"U3", true},  {"Z3", true},
        {"H3", false}, {"F3", false}, {"K3", false},
        {"F3", true},     // F (Jan) not in main cycle
        {"AB", false},    // not a letter
        {"H",  true},     // wrong length
        {"3H", true},     // wrong order
    };
    std::cout << "  \"is_imm_code\": [\n";
    {
        bool first = true;
        for (const auto& c : cc) {
            if (!first) std::cout << ",\n";
            std::cout << "    {\"code\": \"" << c.code
                      << "\", \"main_cycle\": " << (c.main_cycle ? "true" : "false")
                      << ", \"result\": " << (IMM::isIMMcode(c.code, c.main_cycle) ? "true" : "false")
                      << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- code(d) for known IMM dates ------------------------------------
    struct CD { Year y; Month m; Day d; };
    const CD cd[] = {
        {2013, January,  16},   // F3
        {2013, March,    20},   // H3
        {2013, June,     19},   // M3
        {2013, September,18},   // U3
        {2013, December, 18},   // Z3
        {2024, March,    20},   // H4
    };
    std::cout << "  \"code\": [\n";
    {
        bool first = true;
        for (const auto& c : cd) {
            if (!first) std::cout << ",\n";
            Date d(c.d, c.m, c.y);
            std::cout << "    {\"y\": " << c.y
                      << ", \"m\": " << static_cast<int>(c.m)
                      << ", \"d\": " << c.d
                      << ", \"code\": \"" << IMM::code(d) << "\"}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- date(code, ref) ------------------------------------------------
    struct DR { const char* code; Year ry; Month rm; Day rd; };
    const DR dr[] = {
        {"H3", 2013, January, 1},
        {"M3", 2013, January, 1},
        {"Z3", 2013, January, 1},
        {"H4", 2024, January, 1},
    };
    std::cout << "  \"date\": [\n";
    {
        bool first = true;
        for (const auto& r : dr) {
            if (!first) std::cout << ",\n";
            Date ref(r.rd, r.rm, r.ry);
            Date d = IMM::date(r.code, ref);
            std::cout << "    {\"code\": \"" << r.code << "\""
                      << ", \"ref_y\": " << r.ry
                      << ", \"out_y\": " << d.year()
                      << ", \"out_m\": " << static_cast<int>(d.month())
                      << ", \"out_d\": " << d.dayOfMonth() << "}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- next_date ------------------------------------------------------
    struct ND { Year y; Month m; Day d; bool main_cycle; };
    const ND nd[] = {
        {2024, January, 15, true},    // next main IMM = Mar 20, 2024
        {2024, March,   19, true},    // day before Mar 20 → Mar 20
        {2024, March,   20, true},    // ON Mar 20 → next is Jun 19
        {2024, March,   21, true},    // after Mar 20 → Jun 19
        {2024, January, 15, false},   // next any IMM = Jan 17 (3rd Wed)
    };
    std::cout << "  \"next_date\": [\n";
    {
        bool first = true;
        for (const auto& n : nd) {
            if (!first) std::cout << ",\n";
            Date d(n.d, n.m, n.y);
            Date nx = IMM::nextDate(d, n.main_cycle);
            std::cout << "    {\"y\": " << n.y
                      << ", \"m\": " << static_cast<int>(n.m)
                      << ", \"d\": " << n.d
                      << ", \"main_cycle\": " << (n.main_cycle ? "true" : "false")
                      << ", \"out_y\": " << nx.year()
                      << ", \"out_m\": " << static_cast<int>(nx.month())
                      << ", \"out_d\": " << nx.dayOfMonth() << "}";
            first = false;
        }
        std::cout << "\n  ]\n";
    }

    std::cout << "}\n";
    return 0;
}
