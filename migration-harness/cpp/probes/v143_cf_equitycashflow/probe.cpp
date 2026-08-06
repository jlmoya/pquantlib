// migration-harness/cpp/probes/v143_cf_equitycashflow/probe.cpp
//
// Reference values for the equity cash-flow family of C++ QuantLib v1.43:
//
//   * EquityIndex               (ql/indexes/equityindex.{hpp,cpp})
//   * QuantoTermStructure       (ql/termstructures/yield/quantotermstructure.hpp)
//   * EquityCashFlow            (ql/cashflows/equitycashflow.{hpp,cpp})
//   * EquityCashFlowPricer      (abstract; exercised through the concrete pricer)
//   * EquityQuantoCashFlowPricer
//   * setCouponPricer(Leg, ext::shared_ptr<EquityCashFlowPricer>)
//
// WHAT IS PINNED AND WHY
//
// 1. EquityIndex wiring — name / fixing calendar / currency. Cheap to get
//    subtly wrong, and a wrong fixing calendar silently changes which dates
//    are valid fixing dates (and hence which forecasts are even reachable).
//
// 2. All THREE branches of EquityIndex::forecastFixing (equityindex.cpp:73-89):
//      (a) interest + dividend curves  -> spot * P_div(T) / P_int(T)
//      (b) interest only (empty div)   -> spot / P_int(T)   [equity-forward curve]
//      (c) empty spot handle           -> last historical fixing used as spot
//    A port that drops the dividend curve, or that ignores the empty-spot
//    fallback, only differs on one of these three, so all three are pinned
//    at several future dates.
//
// 3. fixing() dispatch (equityindex.cpp:52-71): past fixing, today's fixing
//    from history, today's fixing with forecastTodaysFixing=true (-> spot),
//    today's fixing missing but spot present (-> spot as proxy).
//
// 4. clone() relinked to DIFFERENT curves and a DIFFERENT spot. The forecast
//    of the clone is pinned so that a clone that silently keeps the original
//    curves fails.
//
// 5. QuantoTermStructure zero rates / discounts, plus its delegation of
//    dayCounter() and referenceDate() to the *dividend* curve. The delegation
//    is made observable by giving the dividend curve a day counter and a
//    reference date that differ from every other curve in that sub-section.
//    Its `strike` and `exchRateATMlevel` arguments are invisible against a
//    constant vol, so a further sub-section drives it with strike-dependent
//    BlackVarianceSurfaces and varies exactly those two arguments; one grid
//    point of the pricer uses the same surfaces so that the strike the pricer
//    threads through (index->fixing(fixingDate)) is pinned too.
//
// 6. EquityCashFlow: date() (payment date, deliberately LATER than the fixing
//    date so a port that returns the fixing date fails), baseDate(),
//    fixingDate(), notional(), and amount() for BOTH growthOnly values --
//    with and without a pricer, since growthOnly changes the payoff in both
//    code paths (IndexedCashFlow::amount and EquityQuantoCashFlowPricer::price).
//
// 7. EquityQuantoCashFlowPricer::price() and the resulting amount() over a
//    grid of (equity vol, fx vol, correlation), including rho = 0 and a
//    NEGATIVE rho, so that a dropped correlation or a sign error fails; plus
//    the with-dividend / without-dividend variants of the index.
//
// 8. setCouponPricer over a Leg that mixes EquityCashFlows with a
//    SimpleCashFlow: the equity flows must pick the pricer up, the simple
//    flow must be left alone.
//
// 9. Every QL_REQUIRE / QL_FAIL failure branch reachable from this family,
//    pinned as {"raises": true} rather than by message text (messages are not
//    part of the contract; the raise is).
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/equitycashflow.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/version.hpp>

#include <ql/cashflows/equitycashflow.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/equityindex.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/math/matrix.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancesurface.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/quantotermstructure.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------- market data

const Calendar kCalendar = TARGET();
const DayCounter kDayCount = Actual365Fixed();

// 27 January 2023 is a Friday and a TARGET business day, so calendar.adjust
// leaves it alone -- same anchor as the C++ test suite (test-suite/equityindex.cpp).
const Date kToday(27, January, 2023);

const Real kNotional = 1.0e7;
const Real kSpot = 8700.0;
const Real kCorrelation = 0.4;

const Rate kLocalRate = 0.0375;
const Rate kDividendRate = 0.005;
const Rate kQuantoRate = 0.001;
const Volatility kEquityVol = 0.4;
const Volatility kFxVol = 0.2;

// Historical fixings.
const Date kBaseDate(5, January, 2023);
const Real kBaseFixing = 9010.0;
const Real kTodaysFixing = 8690.0;

// Cash-flow schedule. The payment date is deliberately three business days
// AFTER the fixing date so that date() cannot be confused with fixingDate().
const Date kFixingDate(5, April, 2023);
const Date kPaymentDate(12, April, 2023);

// Future dates used for the forecast tables. All three are TARGET business days.
const Date kForecast1(5, April, 2023);
const Date kForecast2(31, December, 2024);
const Date kForecast3(20, May, 2030);

Handle<YieldTermStructure> flatCurve(Rate r, const DayCounter& dc = kDayCount,
                                     const Date& ref = kToday) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(ref, r, dc));
}

Handle<BlackVolTermStructure> flatVolCurve(Volatility v, const Date& ref = kToday) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(ref, kCalendar, v, kDayCount));
}

Handle<Quote> quoteHandle(Real v) {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(v));
}

// --- strike-dependent vol surfaces -------------------------------------------
//
// Three strike pillars x three date pillars, bilinear in variance. All query
// points below stay strictly inside both grids so that no extrapolation mode
// is exercised -- the point of these surfaces is only that vol depends on
// strike, which is what makes QuantoTermStructure's `strike` and
// `exchRateATMlevel` arguments observable at all.

const Date kVolDates[] = {Date(27, July, 2023), Date(27, January, 2024),
                          Date(27, January, 2025)};
const Real kEquityStrikes[] = {7000.0, 8500.0, 10000.0};
const Real kFxStrikes[] = {0.5, 1.0, 1.5};

Handle<BlackVolTermStructure> surfaceCurve(const Real* strikes, const Real* vols) {
    std::vector<Date> dates(kVolDates, kVolDates + 3);
    std::vector<Real> strikeVec(strikes, strikes + 3);
    Matrix vol(3, 3);
    for (Size i = 0; i < 3; ++i)
        for (Size j = 0; j < 3; ++j)
            vol[i][j] = vols[i * 3 + j];
    return Handle<BlackVolTermStructure>(ext::make_shared<BlackVarianceSurface>(
        kToday, kCalendar, dates, strikeVec, vol, kDayCount));
}

// rows = strikes (7000 / 8500 / 10000), cols = the three dates.
const Real kEquityVols[9] = {0.50, 0.46, 0.42, 0.40, 0.38, 0.36, 0.34, 0.33, 0.32};
// rows = strikes (0.5 / 1.0 / 1.5), cols = the three dates.
const Real kFxVols[9] = {0.28, 0.26, 0.24, 0.20, 0.19, 0.18, 0.16, 0.155, 0.15};

// ---------------------------------------------------------------- JSON output

std::string gIndent = "  ";

void openObject(const std::string& key) {
    std::cout << gIndent << "\"" << key << "\": {\n";
    gIndent += "  ";
}

void closeObject(bool trailingComma) {
    gIndent.resize(gIndent.size() - 2);
    std::cout << gIndent << "}" << (trailingComma ? "," : "") << "\n";
}

void emitReal(const std::string& key, Real v, bool trailingComma = true) {
    std::cout << gIndent << "\"" << key << "\": " << v << (trailingComma ? "," : "") << "\n";
}

void emitInt(const std::string& key, long v, bool trailingComma = true) {
    std::cout << gIndent << "\"" << key << "\": " << v << (trailingComma ? "," : "") << "\n";
}

void emitString(const std::string& key, const std::string& v, bool trailingComma = true) {
    std::cout << gIndent << "\"" << key << "\": \"" << v << "\"" << (trailingComma ? "," : "")
              << "\n";
}

void emitBool(const std::string& key, bool v, bool trailingComma = true) {
    std::cout << gIndent << "\"" << key << "\": " << (v ? "true" : "false")
              << (trailingComma ? "," : "") << "\n";
}

// Runs `f` and emits either {"value": <real>} or {"raises": true}, so the probe
// records what C++ actually does rather than what the probe author expects.
template <typename F>
void emitRealOrRaises(const std::string& key, F f, bool trailingComma = true) {
    Real value = 0.0;
    bool raised = false;
    try {
        value = f();
    } catch (Error&) {
        raised = true;
    }
    if (raised) {
        std::cout << gIndent << "\"" << key << "\": { \"raises\": true }"
                  << (trailingComma ? "," : "") << "\n";
    } else {
        std::cout << gIndent << "\"" << key << "\": { \"value\": " << value << " }"
                  << (trailingComma ? "," : "") << "\n";
    }
}

// ------------------------------------------------------------------ scenarios

// Emits fixing()/forecastFixing() at the three future dates for `index`.
void emitForecastTable(const std::string& key, const ext::shared_ptr<EquityIndex>& index,
                       bool trailingComma) {
    openObject(key);
    const Date dates[] = {kForecast1, kForecast2, kForecast3};
    for (int i = 0; i < 3; ++i) {
        const std::string suffix = std::to_string(i + 1);
        emitInt("date_serial_" + suffix, dates[i].serialNumber());
        emitReal("fixing_" + suffix, index->fixing(dates[i]));
        emitReal("forecast_fixing_" + suffix, index->forecastFixing(dates[i]),
                 i != 2);
    }
    closeObject(trailingComma);
}

// Builds an EquityCashFlow (optionally with a quanto pricer) and emits its
// full observable surface: dates, notional, price and amount.
void emitCashFlow(const std::string& key,
                  const ext::shared_ptr<EquityIndex>& index,
                  const Date& baseDate,
                  const Date& fixingDate,
                  const Date& paymentDate,
                  bool growthOnly,
                  const ext::shared_ptr<EquityQuantoCashFlowPricer>& pricer,
                  bool trailingComma) {
    auto cf = ext::make_shared<EquityCashFlow>(kNotional, index, baseDate, fixingDate,
                                               paymentDate, growthOnly);
    if (pricer)
        cf->setPricer(pricer);

    openObject(key);
    emitInt("date_serial", cf->date().serialNumber());
    emitInt("base_date_serial", cf->baseDate().serialNumber());
    emitInt("fixing_date_serial", cf->fixingDate().serialNumber());
    emitReal("notional", cf->notional());
    emitBool("growth_only", cf->growthOnly());
    emitBool("has_pricer", static_cast<bool>(cf->pricer()));
    emitRealOrRaises("base_fixing", [&] { return cf->baseFixing(); });
    emitRealOrRaises("index_fixing", [&] { return cf->indexFixing(); });
    if (pricer) {
        pricer->initialize(*cf);
        emitRealOrRaises("price", [&] { return pricer->price(); });
    }
    emitRealOrRaises("amount", [&] { return cf->amount(); }, false);
    closeObject(trailingComma);
}

// One grid point, driven by arbitrary equity / FX vol structures.
void emitQuantoPointWithVols(const std::string& key,
                             const ext::shared_ptr<EquityIndex>& index,
                             const Handle<BlackVolTermStructure>& equityVol,
                             const Handle<BlackVolTermStructure>& fxVol,
                             Real correlation,
                             bool growthOnly,
                             bool trailingComma) {
    auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(
        flatCurve(kQuantoRate), equityVol, fxVol, quoteHandle(correlation));

    auto cf = ext::make_shared<EquityCashFlow>(kNotional, index, kBaseDate, kFixingDate,
                                               kPaymentDate, growthOnly);
    cf->setPricer(pricer);
    pricer->initialize(*cf);

    openObject(key);
    emitReal("correlation", correlation);
    emitBool("growth_only", growthOnly);
    emitReal("price", pricer->price());
    emitReal("amount", cf->amount(), false);
    closeObject(trailingComma);
}

// One (equity vol, fx vol, correlation) grid point over flat vols.
void emitQuantoPoint(const std::string& key,
                     const ext::shared_ptr<EquityIndex>& index,
                     Volatility equityVol,
                     Volatility fxVol,
                     Real correlation,
                     bool growthOnly,
                     bool trailingComma) {
    auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(
        flatCurve(kQuantoRate), flatVolCurve(equityVol), flatVolCurve(fxVol),
        quoteHandle(correlation));

    auto cf = ext::make_shared<EquityCashFlow>(kNotional, index, kBaseDate, kFixingDate,
                                               kPaymentDate, growthOnly);
    cf->setPricer(pricer);
    pricer->initialize(*cf);

    openObject(key);
    emitReal("equity_vol", equityVol);
    emitReal("fx_vol", fxVol);
    emitReal("correlation", correlation);
    emitBool("growth_only", growthOnly);
    emitReal("price", pricer->price());
    emitReal("amount", cf->amount(), false);
    closeObject(trailingComma);
}

// Runs cf->amount() and records only whether it raised: used for the pricer's
// QL_REQUIRE branches, which are all reached through amount() -> initialize().
void emitAmountRaises(const std::string& key,
                      const ext::shared_ptr<EquityCashFlow>& cf,
                      bool trailingComma) {
    bool raised = false;
    try {
        (void)cf->amount();
    } catch (Error&) {
        raised = true;
    }
    std::cout << gIndent << "\"" << key << "\": { \"raises\": " << (raised ? "true" : "false")
              << " }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);

    Settings::instance().evaluationDate() = kToday;

    Handle<YieldTermStructure> localCcyInterest = flatCurve(kLocalRate);
    Handle<YieldTermStructure> dividend = flatCurve(kDividendRate);
    Handle<YieldTermStructure> quantoCcyInterest = flatCurve(kQuantoRate);
    Handle<BlackVolTermStructure> equityVol = flatVolCurve(kEquityVol);
    Handle<BlackVolTermStructure> fxVol = flatVolCurve(kFxVol);
    Handle<Quote> spot = quoteHandle(kSpot);
    Handle<Quote> correlation = quoteHandle(kCorrelation);

    // The main index: interest + dividend + explicit spot.
    auto equityIndex = ext::make_shared<EquityIndex>("eqIndex", kCalendar, EURCurrency(),
                                                     localCcyInterest, dividend, spot);
    equityIndex->addFixing(kBaseDate, kBaseFixing);
    equityIndex->addFixing(kToday, kTodaysFixing);

    std::cout << "{\n";
    emitString("quantlib_version", QL_VERSION);

    // =========================================================== setup echo
    openObject("setup");
    emitInt("today_serial", kToday.serialNumber());
    emitInt("base_date_serial", kBaseDate.serialNumber());
    emitInt("fixing_date_serial", kFixingDate.serialNumber());
    emitInt("payment_date_serial", kPaymentDate.serialNumber());
    emitReal("notional", kNotional);
    emitReal("spot", kSpot);
    emitReal("correlation", kCorrelation);
    emitReal("local_rate", kLocalRate);
    emitReal("dividend_rate", kDividendRate);
    emitReal("quanto_rate", kQuantoRate);
    emitReal("equity_vol", kEquityVol);
    emitReal("fx_vol", kFxVol);
    emitReal("base_fixing", kBaseFixing);
    emitReal("todays_fixing", kTodaysFixing, false);
    closeObject(true);

    // ======================================================== EquityIndex
    openObject("equity_index");
    emitString("name", equityIndex->name());
    emitString("fixing_calendar", equityIndex->fixingCalendar().name());
    emitString("currency_code", equityIndex->currency().code());

    // isValidFixingDate: a business day, a TARGET holiday (New Year's Day,
    // 1 Jan 2023, a Sunday) and a plain weekend day (28 Jan 2023, Saturday).
    emitInt("valid_business_day_serial", kToday.serialNumber());
    emitBool("is_valid_business_day", equityIndex->isValidFixingDate(kToday));
    emitInt("holiday_serial", Date(1, January, 2023).serialNumber());
    emitBool("is_valid_holiday", equityIndex->isValidFixingDate(Date(1, January, 2023)));
    emitInt("weekend_serial", Date(28, January, 2023).serialNumber());
    emitBool("is_valid_weekend", equityIndex->isValidFixingDate(Date(28, January, 2023)));

    // Past fixings, added through the IndexManager by Index::addFixing.
    emitBool("has_historical_fixing_base",
             equityIndex->hasHistoricalFixing(kBaseDate));
    emitReal("past_fixing_base", equityIndex->pastFixing(kBaseDate));
    emitReal("fixing_base_date", equityIndex->fixing(kBaseDate));
    emitReal("past_fixing_today", equityIndex->pastFixing(kToday));

    // Today's fixing: history wins; forecastTodaysFixing=true forces the
    // forward (which at t=0 collapses to the spot quote).
    emitReal("fixing_today", equityIndex->fixing(kToday));
    emitReal("fixing_today_forecast", equityIndex->fixing(kToday, true));
    emitReal("spot_value", equityIndex->spot()->value());
    emitBool("has_dividend_curve", !equityIndex->equityDividendCurve().empty());
    emitBool("has_interest_curve", !equityIndex->equityInterestRateCurve().empty());

    // (a) interest + dividend + spot
    emitForecastTable("forecast_with_dividend", equityIndex, true);

    // (b) equity-forward curve: empty dividend handle, so the interest curve
    //     alone carries the forward.
    auto indexNoDividend =
        equityIndex->clone(localCcyInterest, Handle<YieldTermStructure>(), spot);
    emitForecastTable("forecast_no_dividend", indexNoDividend, true);

    // (c) empty spot handle: the last historical fixing (today's, 8690) is
    //     used as the spot proxy.
    auto indexNoSpot = equityIndex->clone(localCcyInterest, dividend, Handle<Quote>());
    emitForecastTable("forecast_no_spot", indexNoSpot, true);

    // clone() relinked to genuinely different curves AND a different spot:
    // every number below must move.
    auto indexRelinked =
        equityIndex->clone(flatCurve(0.06), flatCurve(0.02), quoteHandle(9000.0));
    emitString("clone_name", indexRelinked->name());
    emitString("clone_calendar", indexRelinked->fixingCalendar().name());
    emitString("clone_currency_code", indexRelinked->currency().code());
    emitReal("clone_spot_value", indexRelinked->spot()->value());
    emitForecastTable("forecast_clone_relinked", indexRelinked, true);

    // Today's fixing missing but spot present -> spot used as proxy. Needs a
    // pristine index name so the IndexManager history is empty.
    auto indexNoHistory = ext::make_shared<EquityIndex>("eqIndexNoHistory", kCalendar,
                                                        EURCurrency(), localCcyInterest,
                                                        dividend, spot);
    emitReal("spot_proxy_today", indexNoHistory->fixing(kToday));

    // Failure branches.
    openObject("raises");
    // Invalid fixing date (TARGET holiday).
    emitRealOrRaises("fixing_on_holiday",
                     [&] { return equityIndex->fixing(Date(1, January, 2023)); });
    // Valid business day in the past, but no fixing stored.
    emitRealOrRaises("fixing_missing_past",
                     [&] { return equityIndex->fixing(Date(2, January, 2023)); });
    // Null interest rate term structure.
    auto indexNoCurves = equityIndex->clone(Handle<YieldTermStructure>(),
                                            Handle<YieldTermStructure>(), Handle<Quote>());
    emitRealOrRaises("forecast_without_interest_curve",
                     [&] { return indexNoCurves->fixing(kForecast3); });
    // Neither spot nor historical fixing.
    auto indexNoSpotNoHistory = ext::make_shared<EquityIndex>(
        "eqIndexNoSpotNoHistory", kCalendar, EURCurrency(), localCcyInterest, dividend,
        Handle<Quote>());
    emitRealOrRaises("forecast_without_spot_and_history",
                     [&] { return indexNoSpotNoHistory->fixing(kForecast3); }, false);
    closeObject(false);
    closeObject(true);

    // ================================================== QuantoTermStructure
    // Numeric section: every curve on Actual365Fixed with reference date today,
    // matching the way EquityQuantoCashFlowPricer builds it.
    {
        const Real strike = equityIndex->fixing(kFixingDate);
        QuantoTermStructure qts(dividend, quantoCcyInterest, localCcyInterest, equityVol,
                                strike, fxVol, 1.0, kCorrelation);
        openObject("quanto_term_structure");
        emitString("day_counter", qts.dayCounter().name());
        emitInt("reference_date_serial", qts.referenceDate().serialNumber());
        emitInt("max_date_serial", qts.maxDate().serialNumber());
        emitReal("strike", strike);
        const Time times[] = {0.25, 1.0, 7.5};
        for (int i = 0; i < 3; ++i) {
            const std::string suffix = std::to_string(i + 1);
            emitReal("time_" + suffix, times[i]);
            emitReal("zero_rate_" + suffix,
                     qts.zeroRate(times[i], Continuous, NoFrequency, true).rate());
            emitReal("discount_" + suffix, qts.discount(times[i], true));
        }
        emitInt("discount_date_serial", kForecast3.serialNumber());
        emitReal("discount_at_date", qts.discount(kForecast3, true), false);
        closeObject(true);
    }

    // Delegation section: the dividend curve gets a DIFFERENT day counter and
    // a DIFFERENT reference date from everything else, so that a port which
    // delegates to the wrong curve (or to itself) fails.
    {
        const Date divRef = kCalendar.advance(kToday, -5, Days);
        Handle<YieldTermStructure> oddDividend = flatCurve(kDividendRate, Actual360(), divRef);
        QuantoTermStructure qts(oddDividend, quantoCcyInterest, localCcyInterest, equityVol,
                                8000.0, fxVol, 1.0, kCorrelation);
        openObject("quanto_term_structure_delegation");
        emitInt("dividend_reference_date_serial", divRef.serialNumber());
        emitString("day_counter", qts.dayCounter().name());
        emitInt("reference_date_serial", qts.referenceDate().serialNumber());
        emitReal("zero_rate_1y", qts.zeroRate(1.0, Continuous, NoFrequency, true).rate(), false);
        closeObject(true);
    }

    // Strike sensitivity: same curves, strike-dependent vol surfaces, and only
    // the `strike` / `exchRateATMlevel` arguments varying. A port that accepts
    // either argument and drops it produces four identical zero rates here.
    {
        Handle<BlackVolTermStructure> eqSurface = surfaceCurve(kEquityStrikes, kEquityVols);
        Handle<BlackVolTermStructure> fxSurface = surfaceCurve(kFxStrikes, kFxVols);

        openObject("quanto_term_structure_strike_sensitivity");
        for (int i = 0; i < 3; ++i) {
            const std::string suffix = std::to_string(i + 1);
            emitInt("vol_date_serial_" + suffix, kVolDates[i].serialNumber());
        }
        const Time t = 0.75;
        emitReal("time", t);
        const Real strikes[] = {8000.0, 9000.0};
        const Real atmLevels[] = {1.0, 1.25};
        for (int s = 0; s < 2; ++s) {
            for (int a = 0; a < 2; ++a) {
                const std::string key =
                    "k" + std::to_string(s + 1) + "_atm" + std::to_string(a + 1);
                QuantoTermStructure qts(dividend, quantoCcyInterest, localCcyInterest, eqSurface,
                                        strikes[s], fxSurface, atmLevels[a], kCorrelation);
                const bool last = (s == 1 && a == 1);
                openObject(key);
                emitReal("strike", strikes[s]);
                emitReal("exch_rate_atm_level", atmLevels[a]);
                emitReal("equity_vol", eqSurface->blackVol(t, strikes[s], true));
                emitReal("fx_vol", fxSurface->blackVol(t, atmLevels[a], true));
                emitReal("zero_rate", qts.zeroRate(t, Continuous, NoFrequency, true).rate(), false);
                closeObject(!last);
            }
        }
        closeObject(true);
    }

    // ====================================================== EquityCashFlow
    openObject("equity_cash_flow");
    // No pricer: falls through to IndexedCashFlow::amount().
    emitCashFlow("no_pricer_growth_only", equityIndex, kBaseDate, kFixingDate, kPaymentDate,
                 true, nullptr, true);
    emitCashFlow("no_pricer_total_return", equityIndex, kBaseDate, kFixingDate, kPaymentDate,
                 false, nullptr, true);

    // With the quanto pricer, both growthOnly values.
    {
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(quantoCcyInterest, equityVol,
                                                                   fxVol, correlation);
        emitCashFlow("quanto_growth_only", equityIndex, kBaseDate, kFixingDate, kPaymentDate,
                     true, pricer, true);
    }
    {
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(quantoCcyInterest, equityVol,
                                                                   fxVol, correlation);
        emitCashFlow("quanto_total_return", equityIndex, kBaseDate, kFixingDate, kPaymentDate,
                     false, pricer, true);
    }
    // Same index, no dividend curve (equity-forward flavour).
    {
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(quantoCcyInterest, equityVol,
                                                                   fxVol, correlation);
        emitCashFlow("quanto_no_dividend", indexNoDividend, kBaseDate, kFixingDate,
                     kPaymentDate, true, pricer, false);
    }
    closeObject(true);

    // ============================================ EquityQuantoCashFlowPricer
    openObject("quanto_grid");
    emitQuantoPoint("base", equityIndex, kEquityVol, kFxVol, kCorrelation, true, true);
    emitQuantoPoint("zero_correlation", equityIndex, kEquityVol, kFxVol, 0.0, true, true);
    emitQuantoPoint("negative_correlation", equityIndex, kEquityVol, kFxVol, -0.5, true, true);
    emitQuantoPoint("unit_correlation", equityIndex, kEquityVol, kFxVol, 1.0, true, true);
    emitQuantoPoint("swapped_vols", equityIndex, 0.25, 0.35, kCorrelation, true, true);
    emitQuantoPoint("zero_equity_vol", equityIndex, 0.0, kFxVol, kCorrelation, true, true);
    emitQuantoPoint("zero_fx_vol", equityIndex, kEquityVol, 0.0, kCorrelation, true, true);
    emitQuantoPoint("total_return", equityIndex, kEquityVol, kFxVol, kCorrelation, false, true);
    emitQuantoPoint("no_dividend", indexNoDividend, kEquityVol, kFxVol, kCorrelation, true, true);
    // Strike-dependent surfaces: pins the strike the pricer itself threads
    // through, namely index->fixing(fixingDate).
    emitQuantoPointWithVols("strike_dependent_surfaces", equityIndex,
                            surfaceCurve(kEquityStrikes, kEquityVols),
                            surfaceCurve(kFxStrikes, kFxVols), kCorrelation, true, false);
    closeObject(true);

    // ================================================ pricer failure branches
    openObject("pricer_raises");
    {
        // Fixing date before base date.
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(quantoCcyInterest, equityVol,
                                                                   fxVol, correlation);
        auto cf = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kFixingDate,
                                                   kBaseDate, kPaymentDate, true);
        cf->setPricer(pricer);
        emitAmountRaises("fixing_before_base_date", cf, true);
    }
    {
        // Empty quanto currency term structure handle.
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(
            Handle<YieldTermStructure>(), equityVol, fxVol, correlation);
        auto cf = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kBaseDate,
                                                   kFixingDate, kPaymentDate, true);
        cf->setPricer(pricer);
        emitAmountRaises("empty_quanto_curve", cf, true);
    }
    {
        // Empty equity volatility handle.
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(
            quantoCcyInterest, Handle<BlackVolTermStructure>(), fxVol, correlation);
        auto cf = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kBaseDate,
                                                   kFixingDate, kPaymentDate, true);
        cf->setPricer(pricer);
        emitAmountRaises("empty_equity_vol", cf, true);
    }
    {
        // Empty FX volatility handle.
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(
            quantoCcyInterest, equityVol, Handle<BlackVolTermStructure>(), correlation);
        auto cf = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kBaseDate,
                                                   kFixingDate, kPaymentDate, true);
        cf->setPricer(pricer);
        emitAmountRaises("empty_fx_vol", cf, true);
    }
    {
        // Empty correlation handle.
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(quantoCcyInterest, equityVol,
                                                                   fxVol, Handle<Quote>());
        auto cf = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kBaseDate,
                                                   kFixingDate, kPaymentDate, true);
        cf->setPricer(pricer);
        emitAmountRaises("empty_correlation", cf, true);
    }
    {
        // Inconsistent reference dates between quanto curve and the vols.
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(
            flatCurve(0.02, kDayCount, Date(26, January, 2023)), equityVol, fxVol, correlation);
        auto cf = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kBaseDate,
                                                   kFixingDate, kPaymentDate, true);
        cf->setPricer(pricer);
        emitAmountRaises("inconsistent_reference_dates", cf, false);
    }
    closeObject(true);

    // ==================================================== setCouponPricer
    {
        auto pricer = ext::make_shared<EquityQuantoCashFlowPricer>(quantoCcyInterest, equityVol,
                                                                   fxVol, correlation);
        auto cf1 = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kBaseDate,
                                                    kFixingDate, kPaymentDate, true);
        auto cf2 = ext::make_shared<EquityCashFlow>(kNotional, equityIndex, kBaseDate,
                                                    kForecast2, kForecast2, true);
        auto simple = ext::make_shared<SimpleCashFlow>(1234.5, kPaymentDate);

        Leg leg = {cf1, cf2, simple};
        const Real amountBefore1 = cf1->amount();
        const Real amountBefore2 = cf2->amount();

        setCouponPricer(leg, pricer);

        openObject("set_coupon_pricer");
        emitReal("amount_before_1", amountBefore1);
        emitReal("amount_before_2", amountBefore2);
        emitBool("cf1_has_pricer", static_cast<bool>(cf1->pricer()));
        emitBool("cf2_has_pricer", static_cast<bool>(cf2->pricer()));
        emitReal("amount_after_1", cf1->amount());
        emitReal("amount_after_2", cf2->amount());
        emitReal("simple_amount", simple->amount());
        emitInt("simple_date_serial", simple->date().serialNumber(), false);
        closeObject(false);
    }

    std::cout << "}\n";
    return 0;
}
