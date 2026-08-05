// migration-harness/cpp/probes/currencies_exchangerate/probe.cpp
//
// Behavioural reference for ExchangeRate (ql/exchangerate.hpp) and
// ExchangeRateManager (ql/currencies/exchangeratemanager.hpp) at C++ QuantLib
// v1.43, plus the two downstream users those two unlock: the Money conversion
// modes (Money::BaseCurrencyConversion / Money::AutomatedConversion) and
// CommodityPricingHelper::calculateFxConversionFactor.
//
// Unlike the currency data probe, none of this is a data sheet: the manager is
// a keyed store of date-ranged entries with a lookup that prefers direct rates,
// falls back to the source's or target's triangulation currency, and finally
// walks the whole store depth-first ("smart lookup") avoiding cycles. Each of
// those branches gets a case, in both the succeeding and the failing
// direction, because a port that only ever exercises the happy path will
// silently return a rate through the wrong chain.
//
// Two ordering constraints are baked into the case sequence:
//   * every known-rate case runs before any user rate is added, since
//     smart lookup iterates the whole store and a new entry can change which
//     chain it finds first;
//   * ExchangeRateManager is a singleton, so clear() is called explicitly
//     between phases rather than relying on construction order.
//
// Failing cases are pinned as {"raises": true} rather than by message: QL_FAIL
// text carries file/line context that is not portable across ports.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/currencies/exchangerate.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/currencies/america.hpp>
#include <ql/currencies/asia.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/currencies/exchangeratemanager.hpp>
#include <ql/errors.hpp>
#include <ql/exchangerate.hpp>
#include <ql/experimental/commodities/commoditypricinghelpers.hpp>
#include <ql/money.hpp>
#include <ql/settings.hpp>
#include <ql/version.hpp>

using namespace QuantLib;

namespace {

void emitRate(const char* key, const ExchangeRate& r, bool trailingComma) {
    std::cout << "    \"" << key << "\": {\n"
              << "      \"source\": \"" << r.source().code() << "\",\n"
              << "      \"target\": \"" << r.target().code() << "\",\n"
              << "      \"rate\": " << r.rate() << ",\n"
              << "      \"type\": " << static_cast<int>(r.type()) << "\n"
              << "    }" << (trailingComma ? "," : "") << "\n";
}

void emitMoney(const char* key, const Money& m, bool trailingComma) {
    std::cout << "    \"" << key << "\": {\n"
              << "      \"value\": " << m.value() << ",\n"
              << "      \"currency\": \"" << m.currency().code() << "\"\n"
              << "    }" << (trailingComma ? "," : "") << "\n";
}

void emitRaises(const char* key, bool trailingComma) {
    std::cout << "    \"" << key << "\": { \"raises\": true }" << (trailingComma ? "," : "") << "\n";
}

// Runs `f` and emits either its rate or {"raises": true}, so that the probe
// records what C++ actually does instead of what the probe author expects.
template <typename F>
void emitRateOrRaises(const char* key, F f, bool trailingComma) {
    try {
        ExchangeRate r = f();
        emitRate(key, r, trailingComma);
    } catch (Error&) {
        emitRaises(key, trailingComma);
    }
}

const Date kInRange(15, June, 2010);

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"quantlib_version\": \"" << QL_VERSION << "\",\n";

    // ================= ExchangeRate =================
    std::cout << "  \"exchange_rate\": {\n";

    const ExchangeRate eurUsd(EURCurrency(), USDCurrency(), 1.1);
    emitRate("direct", eurUsd, true);

    // A Direct rate applies in both directions: multiply from source, divide
    // from target.
    emitMoney("exchange_from_source", eurUsd.exchange(Money(100.0, EURCurrency())), true);
    emitMoney("exchange_from_target", eurUsd.exchange(Money(110.0, USDCurrency())), true);
    try {
        [[maybe_unused]] Money unused = eurUsd.exchange(Money(100.0, JPYCurrency()));
        std::cout << "    \"exchange_unrelated_currency\": { \"raises\": false },\n";
    } catch (Error&) {
        emitRaises("exchange_unrelated_currency", true);
    }

    // The four orientations of ExchangeRate::chain, which decide the resulting
    // source/target pair and whether the rate is a ratio, a product, or a
    // reciprocal.
    const ExchangeRate eurGbp(EURCurrency(), GBPCurrency(), 0.85);
    const ExchangeRate gbpEur(GBPCurrency(), EURCurrency(), 1.2);
    const ExchangeRate usdJpy(USDCurrency(), JPYCurrency(), 150.0);
    const ExchangeRate gbpUsd(GBPCurrency(), USDCurrency(), 1.3);

    const ExchangeRate chainSS = ExchangeRate::chain(eurUsd, eurGbp);
    const ExchangeRate chainST = ExchangeRate::chain(eurUsd, gbpEur);
    const ExchangeRate chainTS = ExchangeRate::chain(eurUsd, usdJpy);
    const ExchangeRate chainTT = ExchangeRate::chain(eurUsd, gbpUsd);
    emitRate("chain_source_eq_source", chainSS, true);
    emitRate("chain_source_eq_target", chainST, true);
    emitRate("chain_target_eq_source", chainTS, true);
    emitRate("chain_target_eq_target", chainTT, true);

    try {
        [[maybe_unused]] ExchangeRate unused = ExchangeRate::chain(eurUsd, ExchangeRate(GBPCurrency(), JPYCurrency(), 2.0));
        std::cout << "    \"chain_not_chainable\": { \"raises\": false },\n";
    } catch (Error&) {
        emitRaises("chain_not_chainable", true);
    }

    // A Derived rate exchanges by walking its chain, in either direction.
    emitMoney("chain_exchange_from_source", chainTS.exchange(Money(100.0, EURCurrency())), true);
    emitMoney("chain_exchange_from_target", chainTS.exchange(Money(16500.0, JPYCurrency())), false);

    std::cout << "  },\n";

    // ================= ExchangeRateManager (known rates) =================
    ExchangeRateManager& mgr = ExchangeRateManager::instance();
    mgr.clear();

    std::cout << "  \"manager\": {\n";

    emitRateOrRaises("same_currency",
                     [&] { return mgr.lookup(EURCurrency(), EURCurrency(), kInRange); }, true);

    // Direct: EUR -> ATS is one of the euro-legacy rates seeded at construction.
    emitRateOrRaises("direct_eur_ats",
                     [&] { return mgr.lookup(EURCurrency(), ATSCurrency(), kInRange,
                                             ExchangeRate::Direct); }, true);

    // Reversed: asking ATS -> EUR returns the STORED orientation (EUR -> ATS);
    // it is exchange() that handles the direction, not lookup().
    emitRateOrRaises("reversed_ats_eur",
                     [&] { return mgr.lookup(ATSCurrency(), EURCurrency(), kInRange); }, true);

    // Date-ranged: the EUR/GRD rate only starts on 1-Jan-2001.
    emitRateOrRaises("dated_grd_before_start",
                     [&] { return mgr.lookup(EURCurrency(), GRDCurrency(),
                                             Date(1, June, 2000)); }, true);
    emitRateOrRaises("dated_grd_after_start",
                     [&] { return mgr.lookup(EURCurrency(), GRDCurrency(),
                                             Date(1, June, 2001)); }, true);

    // Triangulated: ATS and DEM both carry EUR as triangulation currency, so
    // the chain is ATS -> EUR -> DEM and the result is Derived.
    emitRateOrRaises("triangulated_ats_dem",
                     [&] { return mgr.lookup(ATSCurrency(), DEMCurrency(), kInRange); }, true);

    // ...and Direct refuses to triangulate.
    emitRateOrRaises("direct_only_ats_dem",
                     [&] { return mgr.lookup(ATSCurrency(), DEMCurrency(), kInRange,
                                             ExchangeRate::Direct); }, true);

    // Smart lookup: neither PEH nor PEN triangulates, so the manager walks its
    // store and finds PEH -> PEI -> PEN.
    emitRateOrRaises("smart_peh_pen",
                     [&] { return mgr.lookup(PEHCurrency(), PENCurrency(), kInRange); }, true);

    // No chain at all exists between USD and JPY among the seeded rates.
    emitRateOrRaises("unreachable_usd_jpy",
                     [&] { return mgr.lookup(USDCurrency(), JPYCurrency(), kInRange); }, true);

    // A null date falls back to Settings::evaluationDate(), so the same lookup
    // succeeds or fails depending only on the global evaluation date.
    Settings::instance().evaluationDate() = Date(1, June, 2001);
    emitRateOrRaises("default_date_after_start",
                     [&] { return mgr.lookup(EURCurrency(), GRDCurrency()); }, true);
    Settings::instance().evaluationDate() = Date(1, June, 2000);
    emitRateOrRaises("default_date_before_start",
                     [&] { return mgr.lookup(EURCurrency(), GRDCurrency()); }, true);
    Settings::instance().evaluationDate() = Date(15, June, 2010);

    // ================= ExchangeRateManager (user rates) =================
    mgr.clear();
    mgr.add(ExchangeRate(USDCurrency(), EURCurrency(), 1.25),
            Date(1, January, 2020), Date(31, December, 2020));

    emitRateOrRaises("user_added_in_range",
                     [&] { return mgr.lookup(USDCurrency(), EURCurrency(),
                                             Date(15, June, 2020), ExchangeRate::Direct); }, true);
    emitRateOrRaises("user_added_before_range",
                     [&] { return mgr.lookup(USDCurrency(), EURCurrency(),
                                             Date(15, June, 2019), ExchangeRate::Direct); }, true);
    emitRateOrRaises("user_added_after_range",
                     [&] { return mgr.lookup(USDCurrency(), EURCurrency(),
                                             Date(15, June, 2021), ExchangeRate::Direct); }, true);

    // Overlapping ranges: the LAST rate added wins.
    mgr.add(ExchangeRate(USDCurrency(), EURCurrency(), 1.5),
            Date(1, January, 2020), Date(31, December, 2020));
    emitRateOrRaises("user_added_latest_wins",
                     [&] { return mgr.lookup(USDCurrency(), EURCurrency(),
                                             Date(15, June, 2020), ExchangeRate::Direct); }, true);

    // clear() drops user rates and re-seeds the known ones.
    mgr.clear();
    emitRateOrRaises("cleared_drops_user_rate",
                     [&] { return mgr.lookup(USDCurrency(), EURCurrency(),
                                             Date(15, June, 2020), ExchangeRate::Direct); }, true);
    emitRateOrRaises("cleared_restores_known",
                     [&] { return mgr.lookup(EURCurrency(), ATSCurrency(), kInRange,
                                             ExchangeRate::Direct); }, false);

    std::cout << "  },\n";

    // ================= Money conversion =================
    // Both conversion modes go through ExchangeRateManager::lookup and then
    // round in the *target* currency, which is what makes the EUR case
    // interesting: EUR is the only currency with a non-default rounding
    // (ClosestRounding(2)).
    std::cout << "  \"money\": {\n";

    Money::Settings::instance().conversionType() = Money::AutomatedConversion;
    Money sumAutomated = Money(100.0, ATSCurrency()) + Money(1.0, EURCurrency());
    emitMoney("automated_conversion_sum", sumAutomated, true);

    Money::Settings::instance().conversionType() = Money::BaseCurrencyConversion;
    Money::Settings::instance().baseCurrency() = EURCurrency();
    Money sumBase = Money(100.0, ATSCurrency()) + Money(1.0, EURCurrency());
    emitMoney("base_currency_conversion_sum", sumBase, true);

    // Comparison goes through the same conversion machinery.
    bool lessBase = Money(100.0, ATSCurrency()) < Money(100.0, EURCurrency());
    std::cout << "    \"base_currency_conversion_less\": " << (lessBase ? "true" : "false") << ",\n";

    Money::Settings::instance().conversionType() = Money::NoConversion;
    try {
        [[maybe_unused]] Money unused = Money(100.0, ATSCurrency()) + Money(1.0, EURCurrency());
        std::cout << "    \"no_conversion_sum\": { \"raises\": false }\n";
    } catch (Error&) {
        emitRaises("no_conversion_sum", false);
    }

    std::cout << "  },\n";

    // ================= CommodityPricingHelper FX factor =================
    // The one other caller of ExchangeRateManager in the library. It asks for
    // a Direct rate and then inverts it if the stored orientation runs the
    // other way, which is the interesting half.
    std::cout << "  \"fx_conversion_factor\": {\n";
    std::cout << "    \"same_currency\": "
              << CommodityPricingHelper::calculateFxConversionFactor(
                     EURCurrency(), EURCurrency(), kInRange)
              << ",\n";
    std::cout << "    \"stored_orientation\": "
              << CommodityPricingHelper::calculateFxConversionFactor(
                     EURCurrency(), ATSCurrency(), kInRange)
              << ",\n";
    std::cout << "    \"inverted_orientation\": "
              << CommodityPricingHelper::calculateFxConversionFactor(
                     ATSCurrency(), EURCurrency(), kInRange)
              << "\n";
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
