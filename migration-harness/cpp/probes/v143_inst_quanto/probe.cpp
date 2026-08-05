// migration-harness/cpp/probes/v143_inst_quanto/probe.cpp
//
// Reference values for the C++ QuantLib v1.43 quanto instrument family:
//
//   * QuantoVanillaOption        (ql/instruments/quantovanillaoption.hpp)
//   * QuantoForwardVanillaOption (ql/instruments/quantoforwardvanillaoption.hpp)
//   * QuantoBarrierOption        (ql/instruments/quantobarrieroption.hpp)
//
// together with the engine that actually produces the quanto greeks,
// QuantoEngine<Instr, Engine> (ql/pricingengines/quanto/quantoengine.hpp), in
// its three concrete instantiations:
//
//   QuantoEngine<VanillaOption,        AnalyticEuropeanEngine>
//   QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>
//   QuantoEngine<BarrierOption,        AnalyticBarrierEngine>
//
// WHAT IS PINNED AND WHY
// ----------------------
// A quanto option is an option on a foreign asset settled in domestic currency
// at a fixed FX rate. QuantoEngine implements that by *replacing the dividend
// curve* of the underlying Black-Scholes process with a QuantoTermStructure
//
//     q_quanto(t) = q(t) + r_dom(t) - r_for(t) + rho * sigma_S(t,K) * sigma_X(t,1)
//
// and then delegating to an ordinary engine. Three inputs feed only that
// quanto adjustment and nothing else:
//
//   * the FOREIGN risk-free curve      (fx_rate below),
//   * the EXCHANGE-RATE volatility     (fx_vol below),
//   * the underlying/FX CORRELATION    (correlation below).
//
// Each of them is trivially droppable in a port — a handle that is stored and
// never read produces a perfectly plausible-looking price. So the grid below
// sweeps all three INDEPENDENTLY (including correlation = 0 and a negative
// correlation), and every case pins the full result set, not just the NPV:
// value + delta/gamma/theta/vega/rho/dividendRho + the three quanto greeks
// qvega/qrho/qlambda. A dropped correlation or a dropped fx-vol handle cannot
// survive this table.
//
// Any accessor that throws (because the engine left the field at Null<Real>())
// is emitted as JSON `null`. That is itself load-bearing: AnalyticBarrierEngine
// fills ONLY results_.value, so for QuantoBarrierOption every greek — standard
// and quanto — must be null, and the port must reproduce that rather than
// inventing zeros.
//
// Also pinned:
//   * "expired"      — after setupExpired() (evaluation date past maturity)
//                      C++ sets every greek, including the three quanto
//                      greeks, to 0.0 (quantovanillaoption.cpp:52-55). Note
//                      this is NOT Null: see "results_reset" below.
//   * "results_reset"— QuantoOptionResults<>::reset() sets qvega/qrho/qlambda
//                      to Null<Real>() (quantovanillaoption.hpp:37-40). The
//                      two are deliberately different sentinels.
//   * "raises"       — the QL_REQUIRE failure branches.
//   * the barrier grid includes barrier == spot, which C++ does NOT consider
//     triggered (BarrierOption::engine::triggered uses strict < / >,
//     barrieroption.cpp:126-136).
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/quanto.json.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/forwardvanillaoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/quantobarrieroption.hpp>
#include <ql/instruments/quantoforwardvanillaoption.hpp>
#include <ql/instruments/quantovanillaoption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/barrier/analyticbarrierengine.hpp>
#include <ql/pricingengines/forward/forwardengine.hpp>
#include <ql/pricingengines/quanto/quantoengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Market, built inline from literal values (no Settings-dependent helpers).
// ---------------------------------------------------------------------------

const Date kToday(15, May, 2026);
const Date kMaturity(11, November, 2026); // kToday + 180 days
const Real kSpot = 100.0;
const Rate kQ = 0.04;   // domestic dividend yield of the underlying
const Rate kR = 0.08;   // domestic risk-free rate
const Volatility kVol = 0.20;

DayCounter dc() { return Actual360(); }

Handle<Quote> spotHandle() {
    static ext::shared_ptr<SimpleQuote> q = ext::make_shared<SimpleQuote>(kSpot);
    return Handle<Quote>(q);
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, r, dc()));
}

Handle<BlackVolTermStructure> flatVol(Volatility v) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(kToday, NullCalendar(), v, dc()));
}

ext::shared_ptr<GeneralizedBlackScholesProcess> makeProcess(Real spot = kSpot) {
    Handle<Quote> s(ext::make_shared<SimpleQuote>(spot));
    return ext::make_shared<GeneralizedBlackScholesProcess>(s, flatCurve(kQ), flatCurve(kR),
                                                            flatVol(kVol));
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

std::string fmt(Real x) {
    std::ostringstream os;
    os << std::setprecision(17) << x;
    return os.str();
}

// Evaluate an accessor; emit `null` when it throws, which is exactly the
// "engine left this field at Null<Real>()" case (the accessors QL_REQUIRE on it).
template <typename F>
std::string tryReal(F f) {
    try {
        return fmt(f());
    } catch (const std::exception&) {
        return "null";
    }
}

// True when the callable throws — used for the QL_REQUIRE branches.
template <typename F>
bool raises(F f) {
    try {
        f();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

// The result block shared by all three instruments. `Opt` is any of the three
// quanto option classes: each exposes the OneAssetOption greeks plus qvega /
// qrho / qlambda.
template <class Opt>
std::string resultBlock(Opt& o) {
    std::ostringstream os;
    os << "\"npv\": " << tryReal([&] { return o.NPV(); })
       << ", \"delta\": " << tryReal([&] { return o.delta(); })
       << ", \"gamma\": " << tryReal([&] { return o.gamma(); })
       << ", \"theta\": " << tryReal([&] { return o.theta(); })
       << ", \"vega\": " << tryReal([&] { return o.vega(); })
       << ", \"rho\": " << tryReal([&] { return o.rho(); })
       << ", \"dividend_rho\": " << tryReal([&] { return o.dividendRho(); })
       << ", \"qvega\": " << tryReal([&] { return o.qvega(); })
       << ", \"qrho\": " << tryReal([&] { return o.qrho(); })
       << ", \"qlambda\": " << tryReal([&] { return o.qlambda(); });
    return os.str();
}

// ---------------------------------------------------------------------------
// Sweep axes. The three quanto-only inputs are varied independently; the
// correlation axis deliberately contains 0.0 and a negative value.
// ---------------------------------------------------------------------------

const Rate kFxRates[] = {0.02, 0.05, 0.09};
const Volatility kFxVols[] = {0.10, 0.25};
const Real kCorrelations[] = {-0.4, 0.0, 0.6};

struct Market {
    Rate fxRate;
    Volatility fxVol;
    Real correlation;
};

// Base market + one-axis-at-a-time perturbations, used where a full cross
// product would blow up the case count (forward option).
const Market kMarkets[] = {
    {0.05, 0.20, 0.30},  // base
    {0.09, 0.20, 0.30},  // foreign rate moved
    {0.05, 0.35, 0.30},  // fx vol moved
    {0.05, 0.20, 0.00},  // correlation switched off
    {0.05, 0.20, -0.45}, // correlation negative
};

const Market kBaseMarket = kMarkets[0];

ext::shared_ptr<PricingEngine> vanillaQuantoEngine(
    const ext::shared_ptr<GeneralizedBlackScholesProcess>& process, const Market& m) {
    return ext::make_shared<QuantoEngine<VanillaOption, AnalyticEuropeanEngine>>(
        process, flatCurve(m.fxRate), flatVol(m.fxVol),
        Handle<Quote>(ext::make_shared<SimpleQuote>(m.correlation)));
}

ext::shared_ptr<PricingEngine> forwardQuantoEngine(
    const ext::shared_ptr<GeneralizedBlackScholesProcess>& process, const Market& m) {
    return ext::make_shared<
        QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>>(
        process, flatCurve(m.fxRate), flatVol(m.fxVol),
        Handle<Quote>(ext::make_shared<SimpleQuote>(m.correlation)));
}

ext::shared_ptr<PricingEngine> barrierQuantoEngine(
    const ext::shared_ptr<GeneralizedBlackScholesProcess>& process, const Market& m) {
    return ext::make_shared<QuantoEngine<BarrierOption, AnalyticBarrierEngine>>(
        process, flatCurve(m.fxRate), flatVol(m.fxVol),
        Handle<Quote>(ext::make_shared<SimpleQuote>(m.correlation)));
}

std::string marketFields(const Market& m) {
    std::ostringstream os;
    os << "\"fx_rate\": " << fmt(m.fxRate) << ", \"fx_vol\": " << fmt(m.fxVol)
       << ", \"correlation\": " << fmt(m.correlation);
    return os.str();
}

// ---------------------------------------------------------------------------
// Case emitters
// ---------------------------------------------------------------------------

void emitVanillaCases() {
    const Option::Type types[] = {Option::Call, Option::Put};
    const Real strikes[] = {95.0, 105.0};

    std::cout << "  \"vanilla\": [\n";
    bool first = true;
    for (auto type : types) {
        for (Real strike : strikes) {
            for (Rate fxr : kFxRates) {
                for (Volatility fxv : kFxVols) {
                    for (Real corr : kCorrelations) {
                        const Market m{fxr, fxv, corr};
                        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
                        auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);
                        QuantoVanillaOption option(payoff, exercise);
                        option.setPricingEngine(vanillaQuantoEngine(makeProcess(), m));

                        if (!first)
                            std::cout << ",\n";
                        first = false;
                        std::cout << "    {\"option_type\": " << int(type)
                                  << ", \"strike\": " << fmt(strike) << ", " << marketFields(m)
                                  << ", " << resultBlock(option) << "}";
                    }
                }
            }
        }
    }
    std::cout << "\n  ],\n";
}

void emitForwardCases() {
    const Option::Type types[] = {Option::Call, Option::Put};
    const Real moneynesses[] = {0.9, 1.0, 1.1};
    const int resetDays[] = {30, 90};

    std::cout << "  \"forward\": [\n";
    bool first = true;
    for (auto type : types) {
        for (Real moneyness : moneynesses) {
            for (int days : resetDays) {
                for (const Market& m : kMarkets) {
                    const Date reset = kToday + days;
                    // The payoff strike is irrelevant for a forward-start option
                    // (ForwardVanillaEngine rebuilds the payoff at moneyness *
                    // spot) but it is still carried through the arguments, so
                    // pin it at a value that is NOT the effective strike.
                    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, 60.0);
                    auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);
                    QuantoForwardVanillaOption option(moneyness, reset, payoff, exercise);
                    option.setPricingEngine(forwardQuantoEngine(makeProcess(), m));

                    if (!first)
                        std::cout << ",\n";
                    first = false;
                    std::cout << "    {\"option_type\": " << int(type)
                              << ", \"moneyness\": " << fmt(moneyness)
                              << ", \"reset_days\": " << days
                              << ", \"reset_serial\": " << reset.serialNumber() << ", "
                              << marketFields(m) << ", " << resultBlock(option) << "}";
                }
            }
        }
    }
    std::cout << "\n  ],\n";
}

struct BarrierSpec {
    Barrier::Type type;
    Real level;
};

void emitBarrierCase(const BarrierSpec& spec, Option::Type type, Real strike, Real rebate,
                     const Market& m, bool& first) {
    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
    auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);
    QuantoBarrierOption option(spec.type, spec.level, rebate, payoff, exercise);
    option.setPricingEngine(barrierQuantoEngine(makeProcess(), m));

    if (!first)
        std::cout << ",\n";
    first = false;
    std::cout << "    {\"barrier_type\": " << int(spec.type)
              << ", \"barrier\": " << fmt(spec.level) << ", \"rebate\": " << fmt(rebate)
              << ", \"option_type\": " << int(type) << ", \"strike\": " << fmt(strike) << ", "
              << marketFields(m) << ", " << resultBlock(option) << "}";
}

// Structural grid: all four barrier types x both option types x two barrier
// levels (one of them equal to spot, which C++ does not treat as triggered)
// x rebate on/off, all at the base market.
void emitBarrierStructuralCases() {
    // Down barriers must sit at or below spot, up barriers at or above it.
    const BarrierSpec specs[] = {
        {Barrier::DownIn, 90.0},  {Barrier::DownIn, 100.0},  {Barrier::UpIn, 110.0},
        {Barrier::UpIn, 100.0},   {Barrier::DownOut, 90.0},  {Barrier::DownOut, 100.0},
        {Barrier::UpOut, 110.0},  {Barrier::UpOut, 100.0},
    };
    const Option::Type types[] = {Option::Call, Option::Put};
    const Real rebates[] = {0.0, 3.0};

    std::cout << "  \"barrier_structural\": [\n";
    bool first = true;
    for (const BarrierSpec& spec : specs)
        for (auto type : types)
            for (Real rebate : rebates)
                emitBarrierCase(spec, type, 105.0, rebate, kBaseMarket, first);
    std::cout << "\n  ],\n";
}

// Market grid: two fixed structures swept over the full (foreign rate x fx vol
// x correlation) cross product. The barrier engine fills only the NPV, so the
// NPV is the only place the three quanto inputs can show up here.
void emitBarrierMarketCases() {
    const BarrierSpec specs[] = {{Barrier::DownOut, 95.0}, {Barrier::UpIn, 110.0}};
    const Option::Type types[] = {Option::Call, Option::Put};

    std::cout << "  \"barrier_market\": [\n";
    bool first = true;
    for (std::size_t i = 0; i < 2; ++i)
        for (Rate fxr : kFxRates)
            for (Volatility fxv : kFxVols)
                for (Real corr : kCorrelations)
                    emitBarrierCase(specs[i], types[i], 105.0, 3.0, Market{fxr, fxv, corr}, first);
    std::cout << "\n  ],\n";
}

// ---------------------------------------------------------------------------
// setupExpired(): C++ sets every greek (including the quanto ones) to 0.0.
// ---------------------------------------------------------------------------

void emitExpiredCases() {
    Settings::instance().evaluationDate() = kMaturity + 1;

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 105.0);
    auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    QuantoVanillaOption vanilla(payoff, exercise);
    vanilla.setPricingEngine(vanillaQuantoEngine(makeProcess(), kBaseMarket));

    QuantoForwardVanillaOption forward(1.0, kToday + 90, payoff, exercise);
    forward.setPricingEngine(forwardQuantoEngine(makeProcess(), kBaseMarket));

    QuantoBarrierOption barrier(Barrier::DownOut, 90.0, 3.0, payoff, exercise);
    barrier.setPricingEngine(barrierQuantoEngine(makeProcess(), kBaseMarket));

    std::cout << "  \"expired\": {\n"
              << "    \"vanilla\": {" << resultBlock(vanilla) << "},\n"
              << "    \"forward\": {" << resultBlock(forward) << "},\n"
              << "    \"barrier\": {" << resultBlock(barrier) << "}\n"
              << "  },\n";

    Settings::instance().evaluationDate() = kToday;
}

// ---------------------------------------------------------------------------
// QuantoOptionResults<>::reset() — Null<Real>(), NOT 0.0.
// ---------------------------------------------------------------------------

void emitResultsReset() {
    QuantoOptionResults<OneAssetOption::results> results;
    results.value = 1.0;
    results.delta = 2.0;
    results.qvega = 3.0;
    results.qrho = 4.0;
    results.qlambda = 5.0;
    results.reset();

    auto nullOr = [](Real x) { return x == Null<Real>() ? std::string("null") : fmt(x); };

    std::cout << "  \"results_reset\": {"
              << "\"value\": " << nullOr(results.value) << ", \"delta\": " << nullOr(results.delta)
              << ", \"qvega\": " << nullOr(results.qvega)
              << ", \"qrho\": " << nullOr(results.qrho)
              << ", \"qlambda\": " << nullOr(results.qlambda) << "},\n";
}

// ---------------------------------------------------------------------------
// QL_REQUIRE branches
// ---------------------------------------------------------------------------

void emitRaises() {
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 105.0);
    auto exercise = ext::make_shared<EuropeanExercise>(kMaturity);

    // QuantoEngine::calculate — "negative or null underlying".
    const bool nullUnderlying = raises([&] {
        QuantoVanillaOption o(payoff, exercise);
        o.setPricingEngine(vanillaQuantoEngine(makeProcess(0.0), kBaseMarket));
        return o.NPV();
    });

    // ForwardOptionArguments::validate — "negative or zero moneyness given".
    const bool zeroMoneyness = raises([&] {
        QuantoForwardVanillaOption o(0.0, kToday + 90, payoff, exercise);
        o.setPricingEngine(forwardQuantoEngine(makeProcess(), kBaseMarket));
        return o.NPV();
    });

    // ForwardOptionArguments::validate — "reset date later or equal to maturity".
    const bool resetAfterMaturity = raises([&] {
        QuantoForwardVanillaOption o(1.0, kMaturity, payoff, exercise);
        o.setPricingEngine(forwardQuantoEngine(makeProcess(), kBaseMarket));
        return o.NPV();
    });

    // ForwardOptionArguments::validate — "reset date in the past".
    const bool resetInThePast = raises([&] {
        QuantoForwardVanillaOption o(1.0, kToday - 1, payoff, exercise);
        o.setPricingEngine(forwardQuantoEngine(makeProcess(), kBaseMarket));
        return o.NPV();
    });

    // AnalyticBarrierEngine::calculate — "barrier touched".
    const bool barrierTouched = raises([&] {
        QuantoBarrierOption o(Barrier::DownOut, 110.0, 3.0, payoff, exercise);
        o.setPricingEngine(barrierQuantoEngine(makeProcess(), kBaseMarket));
        return o.NPV();
    });

    // QuantoBarrierOption greeks — AnalyticBarrierEngine leaves everything but
    // the value at Null<Real>(), so every greek accessor must throw.
    QuantoBarrierOption priced(Barrier::DownOut, 90.0, 3.0, payoff, exercise);
    priced.setPricingEngine(barrierQuantoEngine(makeProcess(), kBaseMarket));
    const bool barrierDelta = raises([&] { return priced.delta(); });
    const bool barrierQvega = raises([&] { return priced.qvega(); });
    const bool barrierQrho = raises([&] { return priced.qrho(); });
    const bool barrierQlambda = raises([&] { return priced.qlambda(); });

    auto b = [](bool v) { return v ? "true" : "false"; };
    std::cout << "  \"raises\": {\n"
              << "    \"quanto_engine_null_underlying\": {\"raises\": " << b(nullUnderlying)
              << "},\n"
              << "    \"forward_zero_moneyness\": {\"raises\": " << b(zeroMoneyness) << "},\n"
              << "    \"forward_reset_after_maturity\": {\"raises\": " << b(resetAfterMaturity)
              << "},\n"
              << "    \"forward_reset_in_the_past\": {\"raises\": " << b(resetInThePast) << "},\n"
              << "    \"barrier_touched\": {\"raises\": " << b(barrierTouched) << "},\n"
              << "    \"barrier_delta_not_provided\": {\"raises\": " << b(barrierDelta) << "},\n"
              << "    \"barrier_qvega_not_provided\": {\"raises\": " << b(barrierQvega) << "},\n"
              << "    \"barrier_qrho_not_provided\": {\"raises\": " << b(barrierQrho) << "},\n"
              << "    \"barrier_qlambda_not_provided\": {\"raises\": " << b(barrierQlambda) << "}\n"
              << "  }\n";
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"market\": {\"today_serial\": " << kToday.serialNumber()
              << ", \"maturity_serial\": " << kMaturity.serialNumber()
              << ", \"day_counter\": \"" << dc().name() << "\", \"spot\": " << fmt(kSpot)
              << ", \"dividend_rate\": " << fmt(kQ) << ", \"risk_free_rate\": " << fmt(kR)
              << ", \"volatility\": " << fmt(kVol) << "},\n";
    emitVanillaCases();
    emitForwardCases();
    emitBarrierStructuralCases();
    emitBarrierMarketCases();
    emitExpiredCases();
    emitResultsReset();
    emitRaises();
    std::cout << "}\n";
    return 0;
}
