// Emit the six time-layer enums' integer values for cross-validation.
//
// QuantLib overloads operator<< on every enum to print the human name
// (e.g. std::cout << Sunday emits "Sunday"), so each value is wrapped in
// static_cast<int>(...) to force the integral value into the stream.
//
// Output JSON shape (each section is {name: int}):
//   {
//     "Weekday": {"Sunday": 1, ..., "Sun": 1, ...},
//     "Month":   {"January": 1, ..., "Jan": 1, ...},
//     "TimeUnit": {"Days": 0, ..., "Microseconds": 8},
//     "Frequency": {"NoFrequency": -1, "Once": 0, ..., "OtherFrequency": 999},
//     "BusinessDayConvention": {"Following": 0, ..., "Nearest": 6},
//     "DateGenerationRule": {"Backward": 0, ..., "CDS2015": 9}
//   }

#include <ql/time/businessdayconvention.hpp>
#include <ql/time/dategenerationrule.hpp>
#include <ql/time/date.hpp>
#include <ql/time/frequency.hpp>
#include <ql/time/timeunit.hpp>
#include <ql/time/weekday.hpp>

#include <iostream>

using namespace QuantLib;

#define EMIT(name) "    \"" #name "\": " << static_cast<int>(name)
#define EMIT_DG(name) "    \"" #name "\": " << static_cast<int>(DateGeneration::name)

int main() {
    std::cout << "{\n";

    std::cout << "  \"Weekday\": {\n";
    std::cout << EMIT(Sunday)    << ",\n";
    std::cout << EMIT(Monday)    << ",\n";
    std::cout << EMIT(Tuesday)   << ",\n";
    std::cout << EMIT(Wednesday) << ",\n";
    std::cout << EMIT(Thursday)  << ",\n";
    std::cout << EMIT(Friday)    << ",\n";
    std::cout << EMIT(Saturday)  << ",\n";
    std::cout << EMIT(Sun) << ",\n";
    std::cout << EMIT(Mon) << ",\n";
    std::cout << EMIT(Tue) << ",\n";
    std::cout << EMIT(Wed) << ",\n";
    std::cout << EMIT(Thu) << ",\n";
    std::cout << EMIT(Fri) << ",\n";
    std::cout << EMIT(Sat) << "\n";
    std::cout << "  },\n";

    std::cout << "  \"Month\": {\n";
    std::cout << EMIT(January)   << ",\n";
    std::cout << EMIT(February)  << ",\n";
    std::cout << EMIT(March)     << ",\n";
    std::cout << EMIT(April)     << ",\n";
    std::cout << EMIT(May)       << ",\n";
    std::cout << EMIT(June)      << ",\n";
    std::cout << EMIT(July)      << ",\n";
    std::cout << EMIT(August)    << ",\n";
    std::cout << EMIT(September) << ",\n";
    std::cout << EMIT(October)   << ",\n";
    std::cout << EMIT(November)  << ",\n";
    std::cout << EMIT(December)  << ",\n";
    std::cout << EMIT(Jan) << ",\n";
    std::cout << EMIT(Feb) << ",\n";
    std::cout << EMIT(Mar) << ",\n";
    std::cout << EMIT(Apr) << ",\n";
    std::cout << EMIT(Jun) << ",\n";
    std::cout << EMIT(Jul) << ",\n";
    std::cout << EMIT(Aug) << ",\n";
    std::cout << EMIT(Sep) << ",\n";
    std::cout << EMIT(Oct) << ",\n";
    std::cout << EMIT(Nov) << ",\n";
    std::cout << EMIT(Dec) << "\n";
    std::cout << "  },\n";

    std::cout << "  \"TimeUnit\": {\n";
    std::cout << EMIT(Days)         << ",\n";
    std::cout << EMIT(Weeks)        << ",\n";
    std::cout << EMIT(Months)       << ",\n";
    std::cout << EMIT(Years)        << ",\n";
    std::cout << EMIT(Hours)        << ",\n";
    std::cout << EMIT(Minutes)      << ",\n";
    std::cout << EMIT(Seconds)      << ",\n";
    std::cout << EMIT(Milliseconds) << ",\n";
    std::cout << EMIT(Microseconds) << "\n";
    std::cout << "  },\n";

    std::cout << "  \"Frequency\": {\n";
    std::cout << EMIT(NoFrequency)      << ",\n";
    std::cout << EMIT(Once)             << ",\n";
    std::cout << EMIT(Annual)           << ",\n";
    std::cout << EMIT(Semiannual)       << ",\n";
    std::cout << EMIT(EveryFourthMonth) << ",\n";
    std::cout << EMIT(Quarterly)        << ",\n";
    std::cout << EMIT(Bimonthly)        << ",\n";
    std::cout << EMIT(Monthly)          << ",\n";
    std::cout << EMIT(EveryFourthWeek)  << ",\n";
    std::cout << EMIT(Biweekly)         << ",\n";
    std::cout << EMIT(Weekly)           << ",\n";
    std::cout << EMIT(Daily)            << ",\n";
    std::cout << EMIT(OtherFrequency)   << "\n";
    std::cout << "  },\n";

    std::cout << "  \"BusinessDayConvention\": {\n";
    std::cout << EMIT(Following)                  << ",\n";
    std::cout << EMIT(ModifiedFollowing)          << ",\n";
    std::cout << EMIT(Preceding)                  << ",\n";
    std::cout << EMIT(ModifiedPreceding)          << ",\n";
    std::cout << EMIT(Unadjusted)                 << ",\n";
    std::cout << EMIT(HalfMonthModifiedFollowing) << ",\n";
    std::cout << EMIT(Nearest)                    << "\n";
    std::cout << "  },\n";

    std::cout << "  \"DateGenerationRule\": {\n";
    std::cout << EMIT_DG(Backward)                << ",\n";
    std::cout << EMIT_DG(Forward)                 << ",\n";
    std::cout << EMIT_DG(Zero)                    << ",\n";
    std::cout << EMIT_DG(ThirdWednesday)          << ",\n";
    std::cout << EMIT_DG(ThirdWednesdayInclusive) << ",\n";
    std::cout << EMIT_DG(Twentieth)               << ",\n";
    std::cout << EMIT_DG(TwentiethIMM)            << ",\n";
    std::cout << EMIT_DG(OldCDS)                  << ",\n";
    std::cout << EMIT_DG(CDS)                     << ",\n";
    std::cout << EMIT_DG(CDS2015)                 << "\n";
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
