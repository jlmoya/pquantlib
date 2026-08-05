// migration-harness/cpp/probes/v143_indexes_swap/probe.cpp
//
// Reference values for the ql/indexes/swap/* families PQuantLib was missing
// against C++ QuantLib v1.43, plus OvernightIndexedSwapIndex from
// ql/indexes/swapindex.hpp.
//
// A swap index carries more wiring than an ibor index — on top of name,
// fixing days, currency, fixing calendar and day counter it fixes the fixed
// leg's tenor and roll convention and picks an underlying ibor index whose
// tenor is itself a function of the swap tenor (3M at or below 1Y, 6M above
// it, in most of the ISDA-fix families but not all). Each of those is probed
// for a short and a long tenor so the ternary cannot be missed.
//
// maturityDate is deliberately included: C++ derives it from the underlying
// vanilla swap's schedule rather than by advancing the index tenor on the
// fixing calendar, and the two disagree whenever the fixed leg's roll differs
// from a plain tenor advance.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/indexes/swap.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/currencies/america.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/ibor/eonia.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/indexes/swap/chfliborswap.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/indexes/swap/eurliborswap.hpp>
#include <ql/indexes/swap/gbpliborswap.hpp>
#include <ql/indexes/swap/jpyliborswap.hpp>
#include <ql/indexes/swap/usdliborswap.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/instruments/overnightindexedswap.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/time/period.hpp>

using namespace QuantLib;

namespace {

// Mid-week (a Monday) so the value/maturity roll is driven by the index's own
// calendar and convention rather than by a weekend.
const Date kSeed(15, June, 2026);

bool first = true;

void emitSwap(const std::string& key, const SwapIndex& idx) {
    const Date fixing = idx.fixingCalendar().adjust(kSeed);
    const Date value = idx.valueDate(fixing);
    if (!first)
        std::cout << ",\n";
    first = false;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"name\": \"" << idx.name() << "\",\n"
              << "    \"fixing_days\": " << idx.fixingDays() << ",\n"
              << "    \"currency_code\": \"" << idx.currency().code() << "\",\n"
              << "    \"fixing_calendar\": \"" << idx.fixingCalendar().name() << "\",\n"
              << "    \"day_counter\": \"" << idx.dayCounter().name() << "\",\n"
              << "    \"tenor_length\": " << idx.tenor().length() << ",\n"
              << "    \"tenor_units\": " << static_cast<int>(idx.tenor().units()) << ",\n"
              << "    \"fixed_leg_tenor_length\": " << idx.fixedLegTenor().length() << ",\n"
              << "    \"fixed_leg_tenor_units\": "
              << static_cast<int>(idx.fixedLegTenor().units()) << ",\n"
              << "    \"fixed_leg_convention\": " << static_cast<int>(idx.fixedLegConvention())
              << ",\n"
              << "    \"ibor_index_name\": \"" << idx.iborIndex()->name() << "\",\n"
              << "    \"exogenous_discount\": " << (idx.exogenousDiscount() ? "true" : "false")
              << ",\n"
              << "    \"fixing_serial\": " << fixing.serialNumber() << ",\n"
              << "    \"value_date_serial\": " << value.serialNumber() << ",\n"
              << "    \"fixing_date_serial\": " << idx.fixingDate(value).serialNumber() << ",\n"
              << "    \"maturity_date_serial\": " << idx.maturityDate(value).serialNumber() << "\n"
              << "  }";
}

// Every ISDA/IFR-fix family, probed at a tenor at or below 1Y and one above it
// so the "3M underlying below, 6M above" ternary is pinned on both sides.
template <class Index>
void emitFamily(const std::string& key) {
    emitSwap(key + "_1y", Index(Period(1, Years)));
    emitSwap(key + "_10y", Index(Period(10, Years)));
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    emitFamily<EuriborSwapIsdaFixA>("euribor_swap_isda_fix_a");
    emitFamily<EuriborSwapIsdaFixB>("euribor_swap_isda_fix_b");
    emitFamily<EuriborSwapIfrFix>("euribor_swap_ifr_fix");
    emitFamily<EurLiborSwapIsdaFixA>("eur_libor_swap_isda_fix_a");
    emitFamily<EurLiborSwapIsdaFixB>("eur_libor_swap_isda_fix_b");
    emitFamily<EurLiborSwapIfrFix>("eur_libor_swap_ifr_fix");
    emitFamily<ChfLiborSwapIsdaFix>("chf_libor_swap_isda_fix");
    emitFamily<GbpLiborSwapIsdaFix>("gbp_libor_swap_isda_fix");
    emitFamily<JpyLiborSwapIsdaFixAm>("jpy_libor_swap_isda_fix_am");
    emitFamily<JpyLiborSwapIsdaFixPm>("jpy_libor_swap_isda_fix_pm");
    emitFamily<UsdLiborSwapIsdaFixAm>("usd_libor_swap_isda_fix_am");
    emitFamily<UsdLiborSwapIsdaFixPm>("usd_libor_swap_isda_fix_pm");

    // OvernightIndexedSwapIndex — fixed leg is always 1Y / ModifiedFollowing,
    // and the fixing calendar and day counter are inherited from the overnight
    // index rather than declared.
    emitSwap("ois_index_eonia_5y",
             OvernightIndexedSwapIndex("EoniaSwapIsdaFix", Period(5, Years), 2, EURCurrency(),
                                       ext::make_shared<Eonia>()));
    emitSwap("ois_index_sofr_10y",
             OvernightIndexedSwapIndex("SofrSwap", Period(10, Years), 2, USDCurrency(),
                                       ext::make_shared<Sofr>()));

    std::cout << "\n}\n";
    return 0;
}
