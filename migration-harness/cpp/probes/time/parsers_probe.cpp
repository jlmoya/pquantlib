// Emit reference outputs for PeriodParser.parse and DateParser.parseISO.
// DateParser.parseFormatted is not probed: the C++ implementation delegates
// to boost::date_time's locale-driven facets, which we deliberately diverge
// from in the Python port (Python uses strptime-style format codes).

#include <ql/utilities/dataparsers.hpp>
#include <ql/time/period.hpp>
#include <ql/time/date.hpp>
#include <ql/time/timeunit.hpp>

#include <iostream>

using namespace QuantLib;

namespace {

const char* uname(TimeUnit u) {
    switch (u) {
      case Days:   return "Days";
      case Weeks:  return "Weeks";
      case Months: return "Months";
      case Years:  return "Years";
      default:     return "?";
    }
}

}  // namespace

int main() {
    std::cout << "{\n";

    // --- PeriodParser.parse: single + composite + signed + case-insensitive
    const std::string period_inputs[] = {
        "1D", "1d",       // case-insensitive
        "7D",  "30D",
        "1W",  "2W",
        "1M",  "6M", "12M",
        "1Y",  "2Y",
        "1Y6M",           // composite: 1Y + 6M = 18M
        "2Y3M",           // 2Y + 3M = 27M
        "1W3D",           // 1W + 3D = 10D
        "-3M",            // negative
        "+5D",            // explicit positive
    };
    std::cout << "  \"period_parse\": [\n";
    {
        bool first = true;
        for (const auto& s : period_inputs) {
            if (!first) std::cout << ",\n";
            Period p = PeriodParser::parse(s);
            std::cout << "    {\"input\": \"" << s
                      << "\", \"length\": " << p.length()
                      << ", \"units\": \"" << uname(p.units()) << "\"}";
            first = false;
        }
        std::cout << "\n  ],\n";
    }

    // --- DateParser.parseISO
    const std::string date_inputs[] = {
        "1901-01-01",
        "2024-01-15",
        "2024-02-29",     // leap day
        "2024-12-31",
        "2025-05-23",     // today
        "2199-12-31",     // max
    };
    std::cout << "  \"date_parse_iso\": [\n";
    {
        bool first = true;
        for (const auto& s : date_inputs) {
            if (!first) std::cout << ",\n";
            Date d = DateParser::parseISO(s);
            std::cout << "    {\"input\": \"" << s
                      << "\", \"serial\": " << d.serialNumber()
                      << ", \"y\": " << d.year()
                      << ", \"m\": " << static_cast<int>(d.month())
                      << ", \"d\": " << d.dayOfMonth() << "}";
            first = false;
        }
        std::cout << "\n  ]\n";
    }

    std::cout << "}\n";
    return 0;
}
