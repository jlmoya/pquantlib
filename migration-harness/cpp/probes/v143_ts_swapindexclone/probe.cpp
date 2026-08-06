// migration-harness/cpp/probes/v143_ts_swapindexclone/probe.cpp
//
// Pins SwapIndex::clone(const Period&) — the tenor overload.
//
// Gaussian1dSwaptionVolatility::smileSectionImpl builds its smile from
// `swapIndexBase_->clone(tenor)`, so a port without that overload has no way
// to honour the requested swap tenor and quietly answers with the base index's
// tenor instead. What the overload has to reproduce is not just the tenor
// field: the clone keeps family name, fixing days, currency, calendar, fixed
// leg tenor / convention / day counter and the IBOR index, and carries the
// discount curve across ONLY when the original had an exogenous one — cloning
// a non-exogenous index must not hand it a discount curve it never had. Both
// branches are probed.
//
// The underlying swap's start / maturity / fair rate are pinned alongside the
// metadata, because those are what actually move when the tenor is wrong, and
// a metadata-only check would pass on an index that had the right tenor
// recorded but built the wrong swap.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/swapindexclone.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kEval(17, January, 2024);

bool firstEmitted = false;

void emit(const std::string& key, const ext::shared_ptr<SwapIndex>& idx) {
    const auto swap = idx->underlyingSwap(Date(19, January, 2024));

    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"name\": \"" << idx->name() << "\",\n"
              << "    \"tenor_length\": " << idx->tenor().length() << ",\n"
              << "    \"tenor_units\": " << static_cast<int>(idx->tenor().units()) << ",\n"
              << "    \"fixing_days\": " << idx->fixingDays() << ",\n"
              << "    \"currency_code\": \"" << idx->currency().code() << "\",\n"
              << "    \"fixing_calendar\": \"" << idx->fixingCalendar().name() << "\",\n"
              << "    \"fixed_leg_tenor_length\": " << idx->fixedLegTenor().length() << ",\n"
              << "    \"fixed_leg_convention\": " << static_cast<int>(idx->fixedLegConvention()) << ",\n"
              << "    \"day_counter\": \"" << idx->dayCounter().name() << "\",\n"
              << "    \"exogenous_discount\": " << (idx->exogenousDiscount() ? "true" : "false") << ",\n"
              << "    \"swap_start\": " << swap->startDate().serialNumber() << ",\n"
              << "    \"swap_maturity\": " << swap->maturityDate().serialNumber() << ",\n"
              << "    \"swap_fair_rate\": " << swap->fairRate() << "\n"
              << "  }";
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kEval;
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    auto forecast = ext::make_shared<FlatForward>(kEval, 0.032, Actual365Fixed(), Continuous,
                                                  Annual);
    forecast->enableExtrapolation();
    auto discount = ext::make_shared<FlatForward>(kEval, 0.028, Actual365Fixed(), Continuous,
                                                  Annual);
    discount->enableExtrapolation();
    Handle<YieldTermStructure> forecastH(forecast);
    Handle<YieldTermStructure> discountH(discount);

    // Non-exogenous base: one curve does both jobs.
    auto base = ext::make_shared<EuriborSwapIsdaFixA>(5 * Years, forecastH);
    emit("base_5y", base);
    emit("clone_10y", base->clone(10 * Years));
    emit("clone_1y", base->clone(1 * Years));
    emit("clone_18m", base->clone(18 * Months));

    // Exogenous base: the clone must carry the discount curve across.
    auto exo = ext::make_shared<EuriborSwapIsdaFixA>(5 * Years, forecastH, discountH);
    emit("exo_base_5y", exo);
    emit("exo_clone_10y", exo->clone(10 * Years));

    std::cout << "\n}\n";
    return 0;
}
