// Emit Period algebra reference values: Frequency↔Period conversion,
// normalize(), arithmetic +/-/*/, comparisons, and years/months/weeks/days
// extractors.

#include <ql/time/period.hpp>
#include <ql/time/frequency.hpp>
#include <ql/time/timeunit.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

const char* unit_name(TimeUnit u) {
    switch (u) {
      case Days:         return "Days";
      case Weeks:        return "Weeks";
      case Months:       return "Months";
      case Years:        return "Years";
      case Hours:        return "Hours";
      case Minutes:      return "Minutes";
      case Seconds:      return "Seconds";
      case Milliseconds: return "Milliseconds";
      case Microseconds: return "Microseconds";
    }
    return "?";
}

const char* freq_name(Frequency f) {
    switch (f) {
      case NoFrequency:      return "NoFrequency";
      case Once:             return "Once";
      case Annual:           return "Annual";
      case Semiannual:       return "Semiannual";
      case EveryFourthMonth: return "EveryFourthMonth";
      case Quarterly:        return "Quarterly";
      case Bimonthly:        return "Bimonthly";
      case Monthly:          return "Monthly";
      case EveryFourthWeek:  return "EveryFourthWeek";
      case Biweekly:         return "Biweekly";
      case Weekly:           return "Weekly";
      case Daily:            return "Daily";
      case OtherFrequency:   return "OtherFrequency";
    }
    return "?";
}

void emit_period(const char* key, const Period& p, const char* trailing) {
    std::cout << "    \"" << key << "\": [" << p.length()
              << ", \"" << unit_name(p.units()) << "\"]" << trailing;
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- Period(Frequency) ----------------------------------------------------
    std::cout << "  \"frequency_to_period\": [\n";
    const std::vector<Frequency> freqs_ok = {
        NoFrequency, Once, Annual, Semiannual, EveryFourthMonth,
        Quarterly, Bimonthly, Monthly,
        EveryFourthWeek, Biweekly, Weekly, Daily
    };
    for (size_t i = 0; i < freqs_ok.size(); ++i) {
        Period p(freqs_ok[i]);
        std::cout << "    {\"frequency\": \"" << freq_name(freqs_ok[i])
                  << "\", \"length\": " << p.length()
                  << ", \"units\": \"" << unit_name(p.units()) << "\"}"
                  << (i + 1 < freqs_ok.size() ? "," : "") << "\n";
    }
    std::cout << "  ],\n";

    // --- Period.frequency() --------------------------------------------------
    struct PFCase { int n; TimeUnit u; };
    const std::vector<PFCase> pf_cases = {
        {0, Years}, {0, Days},
        {1, Years}, {3, Years},
        {1, Months}, {2, Months}, {3, Months}, {4, Months}, {6, Months}, {12, Months},
        {5, Months},
        {1, Weeks}, {2, Weeks}, {4, Weeks}, {3, Weeks},
        {1, Days}, {2, Days}
    };
    std::cout << "  \"period_to_frequency\": [\n";
    for (size_t i = 0; i < pf_cases.size(); ++i) {
        Period p(pf_cases[i].n, pf_cases[i].u);
        std::cout << "    {\"length\": " << pf_cases[i].n
                  << ", \"units\": \"" << unit_name(pf_cases[i].u)
                  << "\", \"frequency\": \"" << freq_name(p.frequency()) << "\"}"
                  << (i + 1 < pf_cases.size() ? "," : "") << "\n";
    }
    std::cout << "  ],\n";

    // --- normalize() ---------------------------------------------------------
    struct NCase { int n; TimeUnit u; };
    const std::vector<NCase> n_cases = {
        {0, Months}, {12, Months}, {24, Months}, {6, Months}, {18, Months},
        {7, Days}, {14, Days}, {3, Days},
        {1, Years}, {1, Weeks}
    };
    std::cout << "  \"normalize\": [\n";
    for (size_t i = 0; i < n_cases.size(); ++i) {
        Period p(n_cases[i].n, n_cases[i].u);
        p.normalize();
        std::cout << "    {\"in\": [" << n_cases[i].n
                  << ", \"" << unit_name(n_cases[i].u) << "\"], "
                  << "\"out\": [" << p.length()
                  << ", \"" << unit_name(p.units()) << "\"]}"
                  << (i + 1 < n_cases.size() ? "," : "") << "\n";
    }
    std::cout << "  ],\n";

    // --- arithmetic +/-/*/, scalar --- (only well-defined cases) -------------
    struct AddCase { int n1; TimeUnit u1; int n2; TimeUnit u2; };
    const std::vector<AddCase> add_cases = {
        {1, Years, 1, Years},
        {1, Years, 6, Months},
        {6, Months, 6, Months},
        {6, Months, 1, Years},
        {3, Weeks, 7, Days},
        {7, Days, 1, Weeks},
        {0, Years, 5, Days},   // zero LHS adopts RHS
        {5, Days, 0, Years},   // zero RHS no-op
    };
    std::cout << "  \"addition\": [\n";
    for (size_t i = 0; i < add_cases.size(); ++i) {
        Period a(add_cases[i].n1, add_cases[i].u1);
        Period b(add_cases[i].n2, add_cases[i].u2);
        Period r = a + b;
        std::cout << "    {"
                  << "\"a\": [" << add_cases[i].n1 << ", \"" << unit_name(add_cases[i].u1) << "\"], "
                  << "\"b\": [" << add_cases[i].n2 << ", \"" << unit_name(add_cases[i].u2) << "\"], "
                  << "\"result\": [" << r.length() << ", \"" << unit_name(r.units()) << "\"]}"
                  << (i + 1 < add_cases.size() ? "," : "") << "\n";
    }
    std::cout << "  ],\n";

    // --- *scalar ---
    std::cout << "  \"mul\": [\n";
    Period p1(2, Months); Period r1 = p1 * 3;
    Period p2(1, Years);  Period r2 = p2 * 2;
    Period p3(7, Days);   Period r3 = p3 * (-1);  // negation
    std::cout << "    {\"a\": [2, \"Months\"], \"k\": 3, \"result\": ["
              << r1.length() << ", \"" << unit_name(r1.units()) << "\"]},\n";
    std::cout << "    {\"a\": [1, \"Years\"], \"k\": 2, \"result\": ["
              << r2.length() << ", \"" << unit_name(r2.units()) << "\"]},\n";
    std::cout << "    {\"a\": [7, \"Days\"], \"k\": -1, \"result\": ["
              << r3.length() << ", \"" << unit_name(r3.units()) << "\"]}\n";
    std::cout << "  ],\n";

    // --- years/months/weeks/days ---
    std::cout << "  \"years_of\": [\n";
    std::cout << "    {\"in\": [0, \"Days\"], \"out\": " << years(Period(0, Days)) << "},\n";
    std::cout << "    {\"in\": [12, \"Months\"], \"out\": " << years(Period(12, Months)) << "},\n";
    std::cout << "    {\"in\": [3, \"Years\"], \"out\": " << years(Period(3, Years)) << "}\n";
    std::cout << "  ],\n";

    std::cout << "  \"months_of\": [\n";
    std::cout << "    {\"in\": [0, \"Days\"], \"out\": " << months(Period(0, Days)) << "},\n";
    std::cout << "    {\"in\": [6, \"Months\"], \"out\": " << months(Period(6, Months)) << "},\n";
    std::cout << "    {\"in\": [1, \"Years\"], \"out\": " << months(Period(1, Years)) << "}\n";
    std::cout << "  ],\n";

    std::cout << "  \"weeks_of\": [\n";
    std::cout << "    {\"in\": [0, \"Days\"], \"out\": " << weeks(Period(0, Days)) << "},\n";
    std::cout << "    {\"in\": [14, \"Days\"], \"out\": " << weeks(Period(14, Days)) << "},\n";
    std::cout << "    {\"in\": [3, \"Weeks\"], \"out\": " << weeks(Period(3, Weeks)) << "}\n";
    std::cout << "  ],\n";

    std::cout << "  \"days_of\": [\n";
    std::cout << "    {\"in\": [0, \"Days\"], \"out\": " << days(Period(0, Days)) << "},\n";
    std::cout << "    {\"in\": [5, \"Days\"], \"out\": " << days(Period(5, Days)) << "},\n";
    std::cout << "    {\"in\": [2, \"Weeks\"], \"out\": " << days(Period(2, Weeks)) << "}\n";
    std::cout << "  ],\n";

    // --- comparisons ---
    struct CmpCase { int n1; TimeUnit u1; int n2; TimeUnit u2; };
    const std::vector<CmpCase> cmp_cases = {
        {1, Years, 12, Months},       // equal lengths
        {1, Years, 13, Months},       // years < 13mo
        {7, Days, 1, Weeks},          // equal
        {6, Days, 1, Weeks},          // <
        {2, Years, 1, Years},         // >
        {0, Days, 1, Days},           // zero specials
    };
    std::cout << "  \"compare\": [\n";
    for (size_t i = 0; i < cmp_cases.size(); ++i) {
        Period a(cmp_cases[i].n1, cmp_cases[i].u1);
        Period b(cmp_cases[i].n2, cmp_cases[i].u2);
        std::cout << "    {"
                  << "\"a\": [" << cmp_cases[i].n1 << ", \"" << unit_name(cmp_cases[i].u1) << "\"], "
                  << "\"b\": [" << cmp_cases[i].n2 << ", \"" << unit_name(cmp_cases[i].u2) << "\"], "
                  << "\"a_lt_b\": " << (a < b ? "true" : "false") << ", "
                  << "\"b_lt_a\": " << (b < a ? "true" : "false") << "}"
                  << (i + 1 < cmp_cases.size() ? "," : "") << "\n";
    }
    std::cout << "  ]\n";

    std::cout << "}\n";
    return 0;
}
