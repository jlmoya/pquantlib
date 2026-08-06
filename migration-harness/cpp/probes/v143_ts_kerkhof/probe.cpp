// migration-harness/cpp/probes/v143_ts_kerkhof/probe.cpp
//
// Pins KerkhofSeasonality against MultiplicativePriceSeasonality.
//
// KerkhofSeasonality inherits from MultiplicativePriceSeasonality but replaces
// BOTH the factor lookup and the correction kernel, so a port that only
// subclasses and forgets one of them still looks plausible:
//
//   * seasonalityFactor is a running PRODUCT of the monthly factors over
//     [min(fromMonth,toMonth), max(...)), inverted when the target month
//     precedes the base month — not the base class's single cyclic lookup.
//     The C++ loop indexes the 12-element vector with QuantLib's 1-based
//     Month values, so element 0 is never read; that quirk is part of the
//     behaviour and is pinned here by using non-uniform factors.
//   * seasonalityCorrection uses the factor RAW (no division by the factor at
//     the curve base date), measures time from the start of the curve base
//     date's MONTHLY inflation period to atDate itself, and REJECTS year-on-
//     year rates instead of returning a ratio.
//
// Both classes are emitted over the same dates and factors so the two can be
// compared directly: wherever the columns differ, a port that conflated them
// fails.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/kerkhof.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/termstructures/inflation/seasonality.hpp>
#include <ql/time/date.hpp>

using namespace QuantLib;

namespace {

const Date kBase(1, June, 2023);

// Deliberately non-uniform and not symmetric, so a wrong index or a wrong loop
// bound changes the answer.
std::vector<Rate> factors() {
    return {1.0100, 1.0021, 0.9985, 1.0043, 0.9971, 1.0008,
            1.0032, 0.9994, 1.0017, 0.9962, 1.0055, 0.9979};
}

const Date kProbeDates[] = {
    Date(1, January, 2023), Date(15, March, 2023), Date(1, June, 2023),
    Date(20, June, 2023),   Date(1, July, 2023),   Date(31, August, 2023),
    Date(1, December, 2023), Date(15, February, 2024), Date(1, June, 2024),
    Date(1, December, 2025),
};

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    KerkhofSeasonality kerkhof(kBase, factors());
    MultiplicativePriceSeasonality multiplicative(kBase, Monthly, factors());

    std::cout << "  \"dates\": [";
    bool first = true;
    for (const Date& d : kProbeDates) {
        std::cout << (first ? "" : ", ") << d.serialNumber();
        first = false;
    }
    std::cout << "],\n";

    std::cout << "  \"kerkhof_factor\": [";
    first = true;
    for (const Date& d : kProbeDates) {
        std::cout << (first ? "" : ", ") << kerkhof.seasonalityFactor(d);
        first = false;
    }
    std::cout << "],\n";

    std::cout << "  \"multiplicative_factor\": [";
    first = true;
    for (const Date& d : kProbeDates) {
        std::cout << (first ? "" : ", ") << multiplicative.seasonalityFactor(d);
        first = false;
    }
    std::cout << "]\n";

    std::cout << "}\n";
    return 0;
}
