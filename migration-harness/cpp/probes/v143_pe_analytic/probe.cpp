// migration-harness/cpp/probes/v143_pe_analytic/probe.cpp
//
// Reference values for the nine "analytic" pricing engines of the wave-8
// `analytic` cluster, all read off C++ QuantLib v1.43 (tag v1.43, 6b57206e0):
//
//   * AnalyticDividendEuropeanEngine  ql/pricingengines/vanilla/analyticdividendeuropeanengine
//   * CashDividendEuropeanEngine      ql/pricingengines/vanilla/cashdividendeuropeanengine
//   * AnalyticCliquetEngine           ql/pricingengines/cliquet/analyticcliquetengine
//   * AnalyticPerformanceEngine       ql/pricingengines/cliquet/analyticperformanceengine
//   * AnalyticContinuousFixedLookbackEngine
//                                     ql/pricingengines/lookback/analyticcontinuousfixedlookback
//   * AnalyticDiscreteGeometricAverageStrikeAsianEngine
//                                     ql/pricingengines/asian/analytic_discr_geom_av_strike
//   * TurnbullWakemanAsianEngine      ql/pricingengines/asian/turnbullwakemanasianengine
//   * AnalyticDoubleBarrierBinaryEngine
//                                     ql/pricingengines/barrier/analyticdoublebarrierbinaryengine
//   * ForwardPerformanceVanillaEngine ql/pricingengines/forward/forwardperformanceengine
//     (plus its base ForwardVanillaEngine, re-verified here on purpose)
//
// Emits JSON on stdout and nothing else; the harness redirects it to
// references/v143/pe/analytic.json.
//
// ---------------------------------------------------------------------------
// What is pinned, and why
// ---------------------------------------------------------------------------
// Every case carries its whole market description under "inputs" so the Python
// test reconstructs the setup instead of restating constants. Every result
// field is pinned together with a `<name>_provided` boolean, because in this
// cluster *which greeks are missing* is as much a part of the contract as the
// numbers: several of these engines assign only `results_.value`, and a port
// that invents a delta where C++ raises "delta not provided" is a defect.
//
// Non-finite results are emitted as the JSON strings "nan" / "inf" / "-inf"
// (JSON has no literal for them); two cases below genuinely produce NaN in
// C++ and reproducing that is part of the contract.
//
// ---------------------------------------------------------------------------
// Guard clauses / early returns that the cases deliberately hit
// ---------------------------------------------------------------------------
// AnalyticDividendEuropeanEngine
//  1. The dividend filter is `date >= settlementDate && date <= exercise
//     ->lastDate()`, BOTH bounds inclusive. So a dividend ON today counts and a
//     dividend ON the expiry date counts, while one day earlier / later does
//     not. Cases `adiv_*_div_today`, `*_div_on_expiry`, `*_div_before_today`,
//     `*_div_after_expiry` pin all four sides.
//  2. `spot = x0 - riskless` must stay > 0 -> `adiv_huge_dividend_throws`.
//  3. Only value/delta/gamma/vega/theta/rho are assigned; `dividendRho` is
//     never touched, so `option.dividendRho()` throws. Pinned everywhere.
//  4. The greeks use THREE different day counters -- rfdc for the zero rate in
//     delta_theta, dydc for the dividend zero rate, voldc for the vega time --
//     and `process_->time()` (which is the risk-free day counter) for theta and
//     rho. The `*_mixed_dc` cases give the three curves different day counters
//     so a port that reuses one of them everywhere fails.
//  5. `results_.theta` is wrapped in try/catch and set to Null on error.
//
// CashDividendEuropeanEngine
//  6. Four distinct code paths, three of which are reachable without a basket
//     engine and are pinned here:
//       (a) `cashDividendModel_ == Escrowed` OR exactly one dividend falling on
//           the settlement date -> delegates to AnalyticDividendEuropeanEngine
//           and copies ONLY `results_.value`; every greek stays unset even
//           though the inner engine computed them.
//       (b) after filtering to (settlement <= date <= maturity && amount > 0)
//           nothing is left -> `underlyings` is just the strike -> plain
//           AnalyticEuropeanEngine.
//       (c) a single dividend falling exactly ON the maturity date is *merged*
//           into the strike (`amount + strike`) and again collapses to a single
//           underlying -> AnalyticEuropeanEngine at the bumped strike. This is
//           the subtlest branch in the file and is pinned for Call and Put.
//       (d) anything else -> ChoiBasketEngine. Pinned as well, so the values
//           are on record for whoever ports ChoiBasketEngine.
//     The amount > 0 filter is pinned by `cashdiv_spot_zero_amount_dividend`.
//  7. The payoff must be a PlainVanillaPayoff (the dynamic_pointer_cast targets
//     PlainVanillaPayoff even though the variable is a StrikedTypePayoff), so a
//     CashOrNothingPayoff is rejected -> `cashdiv_cash_or_nothing_throws`.
//
// AnalyticCliquetEngine / AnalyticPerformanceEngine
//  8. Both refuse a started option (accruedCoupon / lastFixing non-null) and a
//     capped/floored one. `CliquetOption::setupArguments` never populates those
//     six fields, so the guards are unreachable through the stock instrument;
//     the probe subclasses CliquetOption to set them and pin the throws.
//  9. The reset grid is `arguments_.resetDates` with the exercise date appended,
//     and the sum runs over CONSECUTIVE pairs. A one-reset option therefore has
//     two intervals, not one.
// 10. Cliquet: `weight = q->discount(resetDates[i-1])` but `discount =
//     r->discount(t_i)/r->discount(t_{i-1})` -- weight is a dividend discount,
//     the black discount is a *forward* risk-free discount. Performance: the
//     outer weight is a risk-free discount `r->discount(resetDates[i-1])`
//     instead, and the strike enters as `1/moneyness` inside the forward with
//     an outer `* moneyness`. Swapping the two curves is the obvious porting
//     error, so the `*_mixed_dc` and non-zero-q cases separate them.
// 11. Cliquet sets gamma to exactly 0.0 by fiat; Performance sets both delta
//     and gamma to exactly 0.0. Pinned as numbers, not as "unset".
// 12. Cliquet's theta uses the DIVIDEND curve's forward rate; Performance's
//     theta uses the RISK-FREE curve's. Also pinned.
//
// AnalyticContinuousFixedLookbackEngine
// 13. Branch selection is `strike <= minmax -> A+C` for calls and
//     `strike >= minmax -> A+C` for puts, else B. Equality goes to A+C in both
//     cases; pinned by the `k100_minmax100` cases.
// 14. Calls allow strike == 0 (`>= 0.0`), puts do not (`> 0.0`).
// 15. `lambda = 2*(r-q)/vol^2` sits in a denominator. With r == q it is
//     exactly 0 and C++ divides by zero: the case `fixedlb_r_equals_q` pins
//     whatever C++ actually produces there (NaN / +-inf), because a port that
//     "helpfully" special-cases it diverges.
// 16. `volatility()` is read at `blackVol(residualTime(), strike())` where
//     residualTime comes from the RISK-FREE day counter via process_->time(),
//     not from the vol surface's own day counter -- the `*_mixed_dc` case
//     separates the two.
// 17. Only `results_.value` is assigned: every greek is unset.
//
// AnalyticDiscreteGeometricAverageStrikeAsianEngine
// 18. Fixing times are measured from `fixingDates[0]`, NOT from the curve
//     reference date, using the VOL day counter; `residualTime` is measured
//     from `fixingDates[pastFixings]` to expiry using the RISK-FREE one.
// 19. `pastFixings != 0` is rejected outright ("past fixings currently not
//     managed") even though the surrounding arithmetic is written for it.
// 20. The payoff STRIKE is never used -- only its option type. Two cases with
//     different strikes and identical expected values pin that.
// 21. A single fixing that coincides with expiry makes sigmaSum_2 exactly 0
//     and C++ divides by zero -> NaN. Pinned.
// 22. Only `results_.value` is assigned.
//
// TurnbullWakemanAsianEngine
// 23. `accruedAverage = runningAccumulator / (pastFixings + futureFixings)`
//     divides by the TOTAL number of fixings, not by the past ones.
// 24. `effectiveStrike = strike - accruedAverage <= 0` is a separate closed
//     form: calls get `discount * (S_A_hat - strike)` and a delta, puts get a
//     hard zero; both get gamma 0 and NO forward/sigma/tte additional results.
// 25. The volatility is read at the EFFECTIVE strike, and `tn` is the time of
//     the LAST fixing date, which is not the exercise date in general.
// 26. vega / rho / theta / dividendRho are never assigned.
//
// AnalyticDoubleBarrierBinaryEngine
// 27. KnockIn/KnockOut require European exercise; KIKO/KOKI require American
//     exercise whose first date is <= the vol reference date. Both rejections
//     pinned.
// 28. Four degenerate early returns (spot outside the barriers) which set
//     value/delta/gamma/vega/rho but NOT theta or dividendRho. Every one of the
//     eight sides is pinned.
// 29. The C.H. Hui series is truncated at maxIteration = 100 (payoffAtExpiry)
//     or 1000 (payoffKIKO) with a hard `QL_REQUIRE(|last term| < 1e-8)`. Wide
//     barriers with a tiny volatility blow the check up ->
//     `dbbin_series_does_not_converge_throws`. The iteration count is a real
//     knob: the loop runs i = 1 .. maxIteration-1 (99 resp. 999 terms).
// 30. On the non-degenerate path ONLY `results_.value` is assigned.
//
// ForwardVanillaEngine / ForwardPerformanceVanillaEngine
// 31. Both build the reset-date view with ImpliedTermStructure /
//     ImpliedVolTermStructure and price a plain vanilla at strike
//     `moneyness * S(0)`.
// 32. ForwardVanillaEngine discounts with the DIVIDEND curve
//     (`discQ = q->discount(resetDate)`) and its theta uses the dividend
//     curve's zero rate; ForwardPerformanceVanillaEngine discounts with the
//     RISK-FREE curve divided by spot (`discR = r->discount(resetDate)/S`)
//     and its theta uses the risk-free zero rate. Both are pinned side by side
//     on the same market so a port cannot swap them.
// 33. ForwardPerformanceVanillaEngine hard-zeroes delta and gamma and does NOT
//     guard the inner results against Null the way the base class does.

#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/cashflows/dividend.hpp>
#include <ql/exercise.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/averagetype.hpp>
#include <ql/instruments/cliquetoption.hpp>
#include <ql/instruments/doublebarrieroption.hpp>
#include <ql/instruments/forwardvanillaoption.hpp>
#include <ql/instruments/lookbackoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/asian/analytic_discr_geom_av_strike.hpp>
#include <ql/pricingengines/asian/turnbullwakemanasianengine.hpp>
#include <ql/pricingengines/barrier/analyticdoublebarrierbinaryengine.hpp>
#include <ql/pricingengines/cliquet/analyticcliquetengine.hpp>
#include <ql/pricingengines/cliquet/analyticperformanceengine.hpp>
#include <ql/pricingengines/forward/forwardengine.hpp>
#include <ql/pricingengines/forward/forwardperformanceengine.hpp>
#include <ql/pricingengines/lookback/analyticcontinuousfixedlookback.hpp>
#include <ql/pricingengines/vanilla/analyticdividendeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/cashdividendeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// Non-finite doubles become the strings "nan" / "inf" / "-inf".
// ---------------------------------------------------------------------------
std::string num(Real v) {
    if (std::isnan(v))
        return "\"nan\"";
    if (std::isinf(v))
        return v > 0 ? "\"inf\"" : "\"-inf\"";
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, "\"" + v + "\""); }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
    Obj& raw(const std::string& k, const std::string& v) { return put(k, v); }

    Obj& dates(const std::string& k, const std::vector<Date>& ds, const Date& ref) {
        std::string a = "[";
        for (std::size_t j = 0; j < ds.size(); ++j)
            a += (j ? ", " : "") + std::to_string(static_cast<long long>(ds[j] - ref));
        return put(k, a + "]");
    }
    Obj& reals(const std::string& k, const std::vector<Real>& xs) {
        std::string a = "[";
        for (std::size_t j = 0; j < xs.size(); ++j)
            a += (j ? ", " : "") + num(xs[j]);
        return put(k, a + "]");
    }

    std::string str() const { return "{" + body_ + "}"; }

  private:
    Obj& put(const std::string& k, const std::string& v) {
        if (!body_.empty())
            body_ += ", ";
        body_ += "\"" + k + "\": " + v;
        return *this;
    }
    std::string body_;
};

std::vector<std::pair<std::string, std::string>> gCases;

void addCase(const std::string& name, const Obj& inputs, const Obj& expected) {
    gCases.emplace_back(name, "{\"inputs\": " + inputs.str() +
                                  ", \"expected\": " + expected.str() + "}");
}

void emitDocument() {
    std::cout << "{\n";
    for (std::size_t i = 0; i < gCases.size(); ++i)
        std::cout << "  \"" << gCases[i].first << "\": " << gCases[i].second
                  << (i + 1 < gCases.size() ? "," : "") << "\n";
    std::cout << "}\n";
}

// Pin a quantity that the engine may legitimately have left unset. C++ signals
// "not provided" by throwing out of the accessor, so the probe records both the
// availability flag and (when available) the number.
void putOpt(Obj& o, const std::string& key, const std::function<Real()>& f) {
    Real v = 0.0;
    bool ok = true;
    try {
        v = f();
    } catch (const std::exception&) {
        ok = false;
    }
    o.b(key + "_provided", ok);
    if (ok)
        o.n(key, v);
}

bool throwsOn(const std::function<void()>& f) {
    try {
        f();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

// Record NPV + the full one-asset greek block, availability flags included.
void putOneAssetResults(Obj& o, OneAssetOption& option) {
    putOpt(o, "value", [&] { return option.NPV(); });
    putOpt(o, "delta", [&] { return option.delta(); });
    putOpt(o, "gamma", [&] { return option.gamma(); });
    putOpt(o, "theta", [&] { return option.theta(); });
    putOpt(o, "vega", [&] { return option.vega(); });
    putOpt(o, "rho", [&] { return option.rho(); });
    putOpt(o, "dividendRho", [&] { return option.dividendRho(); });
}

// ---------------------------------------------------------------------------
// Market. 1 March 2025 is the single evaluation date for every case; the Python
// test must pin the same date in an autouse fixture.
// ---------------------------------------------------------------------------
const Date kToday(1, March, 2025);

const DayCounter& dc365() {
    static const DayCounter d = Actual365Fixed();
    return d;
}
const DayCounter& dc360() {
    static const DayCounter d = Actual360();
    return d;
}

Handle<Quote> quote(Real v) {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(v));
}

Handle<YieldTermStructure> flatCurve(Rate r, const DayCounter& dc) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(kToday, r, dc));
}

Handle<BlackVolTermStructure> flatVol(Volatility v, const DayCounter& dc) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(kToday, NullCalendar(), v, dc));
}

using Process = ext::shared_ptr<GeneralizedBlackScholesProcess>;

// Uniform-day-count market: every curve on Actual365Fixed.
Process bsm(Real spot, Rate q, Rate r, Volatility vol) {
    return ext::make_shared<BlackScholesMertonProcess>(
        quote(spot), flatCurve(q, dc365()), flatCurve(r, dc365()), flatVol(vol, dc365()));
}

// Mixed-day-count market: risk-free on Actual365Fixed, dividend on Actual360,
// vol on Actual360. Every engine in this cluster reads at least two of the
// three day counters separately; a port that reuses one of them everywhere
// reproduces the uniform cases and fails these.
Process bsmMixedDc(Real spot, Rate q, Rate r, Volatility vol) {
    return ext::make_shared<BlackScholesMertonProcess>(
        quote(spot), flatCurve(q, dc360()), flatCurve(r, dc365()), flatVol(vol, dc360()));
}

// Common "inputs" preamble for a flat market.
void putFlatMarket(Obj& in, Real spot, Rate q, Rate r, Volatility vol, bool mixedDc) {
    in.n("spot", spot);
    in.n("q", q);
    in.n("r", r);
    in.n("vol", vol);
    in.s("day_counters", mixedDc ? "mixed" : "act365f");
}

// ===========================================================================
// 1. AnalyticDividendEuropeanEngine
// ===========================================================================

void adivCase(const std::string& name, Real spot, Rate q, Rate r, Volatility vol, bool mixedDc,
              Option::Type type, Real strike, const std::vector<Date>& divDates,
              const std::vector<Real>& divAmounts, Integer maturityDays, bool american) {
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", "AnalyticDividendEuropeanEngine");
    putFlatMarket(in, spot, q, r, vol, mixedDc);
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.i("maturity_days", maturityDays);
    in.dates("dividend_days", divDates, kToday);
    in.reals("dividend_amounts", divAmounts);
    in.s("exercise", american ? "American" : "European");

    const Process p = mixedDc ? bsmMixedDc(spot, q, r, vol) : bsm(spot, q, r, vol);
    const ext::shared_ptr<Exercise> ex =
        american ? ext::shared_ptr<Exercise>(
                       ext::make_shared<AmericanExercise>(kToday, maturity))
                 : ext::shared_ptr<Exercise>(ext::make_shared<EuropeanExercise>(maturity));

    VanillaOption option(ext::make_shared<PlainVanillaPayoff>(type, strike), ex);
    option.setPricingEngine(ext::make_shared<AnalyticDividendEuropeanEngine>(
        p, DividendVector(divDates, divAmounts)));

    Obj out;
    const bool threw = throwsOn([&] { option.NPV(); });
    out.b("throws", threw);
    if (!threw)
        putOneAssetResults(out, option);
    addCase(name, in, out);
}

void emitAnalyticDividendEuropean() {
    // 1 March 2025 -> 1 March 2026 is 365 days, so with Actual365Fixed the
    // exercise time is exactly 1.0 and no day-count rounding noise leaks in.
    const Integer T = 365;
    const std::vector<Date> mid{Date(1, September, 2025)};
    const std::vector<Real> five{5.0};

    // No dividends at all: the engine must degenerate onto the plain
    // Black-Scholes European price (riskless == 0, spot untouched).
    adivCase("adiv_call_k100_no_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0,
             {}, {}, T, false);
    adivCase("adiv_put_k100_no_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Put, 100.0, {},
             {}, T, false);

    // One dividend halfway through: ITM / ATM / OTM, both types.
    adivCase("adiv_call_k80_mid_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 80.0, mid,
             five, T, false);
    adivCase("adiv_call_k100_mid_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0,
             mid, five, T, false);
    adivCase("adiv_call_k130_mid_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 130.0,
             mid, five, T, false);
    adivCase("adiv_put_k80_mid_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Put, 80.0, mid,
             five, T, false);
    adivCase("adiv_put_k100_mid_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Put, 100.0, mid,
             five, T, false);
    adivCase("adiv_put_k130_mid_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Put, 130.0, mid,
             five, T, false);

    // Zero strike: BlackCalculator's close(strike, 0) branch, reached through
    // the dividend engine (forward is the dividend-adjusted one).
    adivCase("adiv_call_k0_mid_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 0.0, mid,
             five, T, false);

    // The four sides of the inclusive [settlementDate, lastDate] filter.
    adivCase("adiv_call_k100_div_today", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0,
             {kToday}, five, T, false);
    adivCase("adiv_call_k100_div_yesterday", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0,
             {kToday - 1}, five, T, false);
    adivCase("adiv_call_k100_div_on_expiry", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0,
             {kToday + T}, five, T, false);
    adivCase("adiv_call_k100_div_after_expiry", 100.0, 0.03, 0.06, 0.25, false, Option::Call,
             100.0, {kToday + T + 1}, five, T, false);

    // Several dividends, one of which is outside the window.
    adivCase("adiv_call_k100_multi_dividend", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0,
             {Date(1, May, 2025), Date(1, September, 2025), Date(1, January, 2026),
              Date(1, June, 2026)},
             {3.0, 4.0, 2.0, 7.0}, T, false);

    // Zero dividend yield: dividendDiscount == 1 everywhere, so the
    // riskless/dividendDiscount division in the accumulation is a no-op and a
    // port that dropped it still passes -- which is why the cases above use
    // q = 3%. Kept as the control.
    adivCase("adiv_call_k100_zero_q", 100.0, 0.0, 0.06, 0.25, false, Option::Call, 100.0, mid, five,
             T, false);

    // Mixed day counters: rfdc = Act/365F, dydc = Act/360, voldc = Act/360.
    adivCase("adiv_call_k100_mid_dividend_mixed_dc", 100.0, 0.03, 0.06, 0.25, true, Option::Call,
             100.0, mid, five, T, false);
    adivCase("adiv_put_k100_multi_dividend_mixed_dc", 100.0, 0.03, 0.06, 0.25, true, Option::Put,
             100.0, {Date(1, May, 2025), Date(1, September, 2025), Date(1, January, 2026)},
             {3.0, 4.0, 2.0}, T, false);

    // Guards.
    adivCase("adiv_huge_dividend_throws", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0, mid,
             {150.0}, T, false);
    adivCase("adiv_american_exercise_throws", 100.0, 0.03, 0.06, 0.25, false, Option::Call, 100.0,
             mid, five, T, true);
}

// ===========================================================================
// 2. CashDividendEuropeanEngine
// ===========================================================================

// `branch` is documentation only -- it names which of the four code paths the
// case is meant to exercise, so the Python test can assert coverage of each.
void cashdivCase(const std::string& name, const std::string& branch, Real spot, Rate q, Rate r,
                 Volatility vol, bool mixedDc, Option::Type type, Real strike,
                 const std::vector<Date>& divDates, const std::vector<Real>& divAmounts,
                 Integer maturityDays, CashDividendEuropeanEngine::CashDividendModel model,
                 bool american, bool cashOrNothingPayoff) {
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", "CashDividendEuropeanEngine");
    in.s("branch", branch);
    putFlatMarket(in, spot, q, r, vol, mixedDc);
    in.s("model", model == CashDividendEuropeanEngine::Spot ? "Spot" : "Escrowed");
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.i("maturity_days", maturityDays);
    in.dates("dividend_days", divDates, kToday);
    in.reals("dividend_amounts", divAmounts);
    in.s("exercise", american ? "American" : "European");
    in.s("payoff", cashOrNothingPayoff ? "CashOrNothing" : "PlainVanilla");

    const Process p = mixedDc ? bsmMixedDc(spot, q, r, vol) : bsm(spot, q, r, vol);
    const ext::shared_ptr<Exercise> ex =
        american ? ext::shared_ptr<Exercise>(
                       ext::make_shared<AmericanExercise>(kToday, maturity))
                 : ext::shared_ptr<Exercise>(ext::make_shared<EuropeanExercise>(maturity));
    const ext::shared_ptr<StrikedTypePayoff> payoff =
        cashOrNothingPayoff
            ? ext::shared_ptr<StrikedTypePayoff>(
                  ext::make_shared<CashOrNothingPayoff>(type, strike, 10.0))
            : ext::shared_ptr<StrikedTypePayoff>(
                  ext::make_shared<PlainVanillaPayoff>(type, strike));

    VanillaOption option(payoff, ex);
    option.setPricingEngine(ext::make_shared<CashDividendEuropeanEngine>(
        p, DividendVector(divDates, divAmounts), model));

    Obj out;
    const bool threw = throwsOn([&] { option.NPV(); });
    out.b("throws", threw);
    if (!threw)
        putOneAssetResults(out, option);
    addCase(name, in, out);
}

void emitCashDividendEuropean() {
    const Integer T = 548;  // ~18 months, 1 March 2025 -> 30 August 2026
    const auto Spot = CashDividendEuropeanEngine::Spot;
    const auto Escrowed = CashDividendEuropeanEngine::Escrowed;
    const std::vector<Date> mid{Date(1, September, 2025)};
    const std::vector<Real> five{5.0};

    // (a) Escrowed always delegates to AnalyticDividendEuropeanEngine and
    //     copies ONLY the value -- the delegate's greeks are discarded.
    cashdivCase("cashdiv_escrowed_call_k95_mid", "escrowed_delegate", 100.0, 0.025, 0.05, 0.30,
                false, Option::Call, 95.0, mid, five, T, Escrowed, false, false);
    cashdivCase("cashdiv_escrowed_put_k95_mid", "escrowed_delegate", 100.0, 0.025, 0.05, 0.30,
                false, Option::Put, 95.0, mid, five, T, Escrowed, false, false);
    cashdivCase("cashdiv_escrowed_call_k95_multi", "escrowed_delegate", 100.0, 0.025, 0.05, 0.30,
                false, Option::Call, 95.0,
                {Date(1, June, 2025), Date(1, December, 2025), Date(1, June, 2026)},
                {2.0, 3.0, 2.5}, T, Escrowed, false, false);
    cashdivCase("cashdiv_escrowed_call_k95_no_dividend", "escrowed_delegate", 100.0, 0.025, 0.05,
                0.30, false, Option::Call, 95.0, {}, {}, T, Escrowed, false, false);
    cashdivCase("cashdiv_escrowed_call_k95_mixed_dc", "escrowed_delegate", 100.0, 0.025, 0.05, 0.30,
                true, Option::Call, 95.0, mid, five, T, Escrowed, false, false);

    // Escrowed passes `dividends_`, the UNFILTERED schedule, to the delegate,
    // so a dividend outside the window is dropped by the delegate's own filter
    // rather than by the Spot filter above it. Same value either way, pinned so
    // a port that filters first still agrees.
    cashdivCase("cashdiv_escrowed_call_k95_div_after_maturity", "escrowed_delegate", 100.0, 0.025,
                0.05, 0.30, false, Option::Call, 95.0, {kToday + T + 30}, five, T, Escrowed, false,
                false);

    // (a') Spot with exactly one dividend ON the settlement date takes the same
    //      delegating branch, even though the model is Spot.
    cashdivCase("cashdiv_spot_call_k95_single_dividend_today", "escrowed_delegate", 100.0, 0.025,
                0.05, 0.30, false, Option::Call, 95.0, {kToday}, five, T, Spot, false, false);
    cashdivCase("cashdiv_spot_put_k95_single_dividend_today", "escrowed_delegate", 100.0, 0.025,
                0.05, 0.30, false, Option::Put, 95.0, {kToday}, five, T, Spot, false, false);
    // ...but TWO dividends, one of which is today, does not: the size == 1 test
    // is on the filtered schedule, so this falls through to the basket branch.
    cashdivCase("cashdiv_spot_call_k95_two_dividends_one_today", "choi_basket", 100.0, 0.025, 0.05,
                0.30, false, Option::Call, 95.0, {kToday, Date(1, September, 2025)}, {5.0, 4.0}, T,
                Spot, false, false);

    // (b) Spot with nothing left after filtering -> AnalyticEuropeanEngine at
    //     the plain strike. Three different reasons for the schedule to empty.
    cashdivCase("cashdiv_spot_call_k95_no_dividend", "single_underlying", 100.0, 0.025, 0.05, 0.30,
                false, Option::Call, 95.0, {}, {}, T, Spot, false, false);
    cashdivCase("cashdiv_spot_put_k95_no_dividend", "single_underlying", 100.0, 0.025, 0.05, 0.30,
                false, Option::Put, 95.0, {}, {}, T, Spot, false, false);
    cashdivCase("cashdiv_spot_call_k95_dividend_after_maturity", "single_underlying", 100.0, 0.025,
                0.05, 0.30, false, Option::Call, 95.0, {kToday + T + 1}, five, T, Spot, false,
                false);
    cashdivCase("cashdiv_spot_call_k95_dividend_before_today", "single_underlying", 100.0, 0.025,
                0.05, 0.30, false, Option::Call, 95.0, {kToday - 1}, five, T, Spot, false, false);
    // amount > 0 is part of the filter, so a zero dividend is dropped entirely.
    cashdivCase("cashdiv_spot_call_k95_zero_amount_dividend", "single_underlying", 100.0, 0.025,
                0.05, 0.30, false, Option::Call, 95.0, mid, {0.0}, T, Spot, false, false);

    // (c) A single dividend falling exactly ON the maturity date is folded into
    //     the strike (amount + strike) and again collapses to one underlying,
    //     so this prices as a plain European at strike 100 rather than 95.
    cashdivCase("cashdiv_spot_call_k95_dividend_on_maturity", "strike_merged", 100.0, 0.025, 0.05,
                0.30, false, Option::Call, 95.0, {kToday + T}, five, T, Spot, false, false);
    cashdivCase("cashdiv_spot_put_k95_dividend_on_maturity", "strike_merged", 100.0, 0.025, 0.05,
                0.30, false, Option::Put, 95.0, {kToday + T}, five, T, Spot, false, false);
    cashdivCase("cashdiv_spot_call_k95_dividend_on_maturity_mixed_dc", "strike_merged", 100.0,
                0.025, 0.05, 0.30, true, Option::Call, 95.0, {kToday + T}, five, T, Spot, false,
                false);

    // (d) Everything else goes through ChoiBasketEngine.
    cashdivCase("cashdiv_spot_call_k95_mid", "choi_basket", 100.0, 0.025, 0.05, 0.30, false,
                Option::Call, 95.0, mid, five, T, Spot, false, false);
    cashdivCase("cashdiv_spot_put_k95_mid", "choi_basket", 100.0, 0.025, 0.05, 0.30, false,
                Option::Put, 95.0, mid, five, T, Spot, false, false);
    cashdivCase("cashdiv_spot_call_k95_multi", "choi_basket", 100.0, 0.025, 0.05, 0.30, false,
                Option::Call, 95.0,
                {Date(1, June, 2025), Date(1, December, 2025), Date(1, June, 2026)},
                {2.0, 3.0, 2.5}, T, Spot, false, false);
    cashdivCase("cashdiv_spot_put_k95_multi", "choi_basket", 100.0, 0.025, 0.05, 0.30, false,
                Option::Put, 95.0,
                {Date(1, June, 2025), Date(1, December, 2025), Date(1, June, 2026)},
                {2.0, 3.0, 2.5}, T, Spot, false, false);

    // Guards.
    cashdivCase("cashdiv_american_exercise_throws", "guard", 100.0, 0.025, 0.05, 0.30, false,
                Option::Call, 95.0, mid, five, T, Escrowed, true, false);
    cashdivCase("cashdiv_cash_or_nothing_payoff_throws", "guard", 100.0, 0.025, 0.05, 0.30, false,
                Option::Call, 95.0, mid, five, T, Escrowed, false, true);
}

// ===========================================================================
// 3./4. AnalyticCliquetEngine and AnalyticPerformanceEngine
// ===========================================================================

// CliquetOption::setupArguments never populates accruedCoupon / lastFixing /
// the four cap+floor fields, so the two "cannot price ..." guards in both
// engines are unreachable through the stock instrument. This subclass sets
// them, which is the only way to pin those throws.
class SeasonedCliquetOption : public CliquetOption {
  public:
    SeasonedCliquetOption(const ext::shared_ptr<PercentageStrikePayoff>& payoff,
                          const ext::shared_ptr<EuropeanExercise>& maturity,
                          std::vector<Date> resetDates, Real accruedCoupon, Real lastFixing,
                          Real localCap)
    : CliquetOption(payoff, maturity, std::move(resetDates)), accruedCoupon_(accruedCoupon),
      lastFixing_(lastFixing), localCap_(localCap) {}

    void setupArguments(PricingEngine::arguments* args) const override {
        CliquetOption::setupArguments(args);
        auto* more = dynamic_cast<CliquetOption::arguments*>(args);
        QL_REQUIRE(more != nullptr, "wrong engine type");
        more->accruedCoupon = accruedCoupon_;
        more->lastFixing = lastFixing_;
        more->localCap = localCap_;
    }

  private:
    Real accruedCoupon_, lastFixing_, localCap_;
};

void cliquetCase(const std::string& name, bool performance, Real spot, Rate q, Rate r,
                 Volatility vol, bool mixedDc, Option::Type type, Real moneyness,
                 const std::vector<Integer>& resetDays, Integer maturityDays, Real accruedCoupon,
                 Real lastFixing, Real localCap) {
    std::vector<Date> resetDates;
    resetDates.reserve(resetDays.size());
    for (Integer d : resetDays)
        resetDates.push_back(kToday + d);
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", performance ? "AnalyticPerformanceEngine" : "AnalyticCliquetEngine");
    putFlatMarket(in, spot, q, r, vol, mixedDc);
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("moneyness", moneyness);
    in.dates("reset_days", resetDates, kToday);
    in.i("maturity_days", maturityDays);
    in.b("accrued_coupon_set", accruedCoupon != Null<Real>());
    if (accruedCoupon != Null<Real>())
        in.n("accrued_coupon", accruedCoupon);
    in.b("last_fixing_set", lastFixing != Null<Real>());
    if (lastFixing != Null<Real>())
        in.n("last_fixing", lastFixing);
    in.b("local_cap_set", localCap != Null<Real>());
    if (localCap != Null<Real>())
        in.n("local_cap", localCap);

    const Process p = mixedDc ? bsmMixedDc(spot, q, r, vol) : bsm(spot, q, r, vol);
    const auto payoff = ext::make_shared<PercentageStrikePayoff>(type, moneyness);
    const auto exercise = ext::make_shared<EuropeanExercise>(maturity);

    const bool seasoned = accruedCoupon != Null<Real>() || lastFixing != Null<Real>() ||
                          localCap != Null<Real>();
    ext::shared_ptr<CliquetOption> option =
        seasoned ? ext::shared_ptr<CliquetOption>(ext::make_shared<SeasonedCliquetOption>(
                       payoff, exercise, resetDates, accruedCoupon, lastFixing, localCap))
                 : ext::make_shared<CliquetOption>(payoff, exercise, resetDates);

    if (performance)
        option->setPricingEngine(ext::make_shared<AnalyticPerformanceEngine>(p));
    else
        option->setPricingEngine(ext::make_shared<AnalyticCliquetEngine>(p));

    Obj out;
    const bool threw = throwsOn([&] { option->NPV(); });
    out.b("throws", threw);
    if (!threw)
        putOneAssetResults(out, *option);
    addCase(name, in, out);
}

void emitCliquetEngines() {
    const Real N = Null<Real>();
    // Haug p.37 market: S = 60, q = 4%, r = 8%, sigma = 30%, one reset at
    // ~3 months, maturity at ~1 year.
    const std::vector<Integer> oneReset{90};
    const std::vector<Integer> quarterly{90, 180, 270};
    const std::vector<Integer> semiannual{183};
    const Integer T = 360;

    for (int pass = 0; pass < 2; ++pass) {
        const bool perf = pass == 1;
        const std::string pre = perf ? "perf_" : "cliquet_";

        cliquetCase(pre + "call_m110_one_reset", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.1, oneReset, T, N, N, N);
        cliquetCase(pre + "put_m110_one_reset", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Put,
                    1.1, oneReset, T, N, N, N);
        cliquetCase(pre + "call_m100_one_reset", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.0, oneReset, T, N, N, N);
        cliquetCase(pre + "call_m090_one_reset", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    0.9, oneReset, T, N, N, N);
        cliquetCase(pre + "put_m090_one_reset", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Put,
                    0.9, oneReset, T, N, N, N);

        // Several reset periods: the running sums over consecutive pairs are
        // where a port that forgets to append the exercise date, or that
        // indexes resetDates from 0 instead of 1, breaks.
        cliquetCase(pre + "call_m100_quarterly", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.0, quarterly, T, N, N, N);
        cliquetCase(pre + "put_m100_quarterly", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Put,
                    1.0, quarterly, T, N, N, N);
        cliquetCase(pre + "call_m110_semiannual", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.1, semiannual, T, N, N, N);

        // q == 0 kills the dividend-discount weight in the cliquet engine and
        // the dividendRho term in both; the non-zero-q cases above are what
        // actually separate the two curves.
        cliquetCase(pre + "call_m100_zero_q", perf, 60.0, 0.0, 0.08, 0.30, false, Option::Call, 1.0,
                    quarterly, T, N, N, N);
        // r == 0 does the mirror job for the risk-free discount.
        cliquetCase(pre + "call_m100_zero_r", perf, 60.0, 0.04, 0.0, 0.30, false, Option::Call, 1.0,
                    quarterly, T, N, N, N);

        cliquetCase(pre + "call_m110_mixed_dc", perf, 60.0, 0.04, 0.08, 0.30, true, Option::Call,
                    1.1, oneReset, T, N, N, N);
        cliquetCase(pre + "put_m100_quarterly_mixed_dc", perf, 60.0, 0.04, 0.08, 0.30, true,
                    Option::Put, 1.0, quarterly, T, N, N, N);

        // Guards: a started option and a capped one.
        cliquetCase(pre + "accrued_coupon_throws", perf, 60.0, 0.04, 0.08, 0.30, false,
                    Option::Call, 1.1, oneReset, T, 1.5, N, N);
        cliquetCase(pre + "last_fixing_throws", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.1, oneReset, T, N, 62.0, N);
        cliquetCase(pre + "local_cap_throws", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.1, oneReset, T, N, N, 0.05);
    }
}

// ===========================================================================
// 5. AnalyticContinuousFixedLookbackEngine
// ===========================================================================

void fixedLookbackCase(const std::string& name, Real spot, Rate q, Rate r, Volatility vol,
                       bool mixedDc, Option::Type type, Real strike, Real minmax,
                       Integer maturityDays, bool floatingPayoff) {
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", "AnalyticContinuousFixedLookbackEngine");
    putFlatMarket(in, spot, q, r, vol, mixedDc);
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.n("minmax", minmax);
    in.i("maturity_days", maturityDays);
    in.s("payoff", floatingPayoff ? "FloatingType" : "PlainVanilla");
    // Which branch the case is meant to reach, given C++'s selection rule.
    const bool aPlusC = (type == Option::Call) ? (strike <= minmax) : (strike >= minmax);
    in.s("branch", aPlusC ? "A_plus_C" : "B");

    const Process p = mixedDc ? bsmMixedDc(spot, q, r, vol) : bsm(spot, q, r, vol);
    const ext::shared_ptr<StrikedTypePayoff> payoff =
        ext::make_shared<PlainVanillaPayoff>(type, strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(maturity);

    // The floating-payoff case is the "non-plain payoff" rejection; the
    // instrument's ctor takes a StrikedTypePayoff, so the only way to reach it
    // is a striked-but-not-plain payoff.
    const ext::shared_ptr<StrikedTypePayoff> used =
        floatingPayoff ? ext::shared_ptr<StrikedTypePayoff>(
                             ext::make_shared<CashOrNothingPayoff>(type, strike, 10.0))
                       : payoff;

    ContinuousFixedLookbackOption option(minmax, used, exercise);
    option.setPricingEngine(ext::make_shared<AnalyticContinuousFixedLookbackEngine>(p));

    Obj out;
    const bool threw = throwsOn([&] { option.NPV(); });
    out.b("throws", threw);
    if (!threw)
        putOneAssetResults(out, option);
    addCase(name, in, out);
}

void emitFixedLookback() {
    // Haug pp.63-64 market: S = 100, running extremum 100, r = 10%, q = 0.
    const Integer T = 365;

    // Calls: strike <= minmax -> A+C (including exact equality); strike >
    // minmax -> B.
    fixedLookbackCase("fixedlb_call_k95_minmax100", 100.0, 0.0, 0.10, 0.20, false, Option::Call,
                      95.0, 100.0, T, false);
    fixedLookbackCase("fixedlb_call_k100_minmax100", 100.0, 0.0, 0.10, 0.20, false, Option::Call,
                      100.0, 100.0, T, false);
    fixedLookbackCase("fixedlb_call_k105_minmax100", 100.0, 0.0, 0.10, 0.20, false, Option::Call,
                      105.0, 100.0, T, false);
    // Puts: strike >= minmax -> A+C; strike < minmax -> B.
    fixedLookbackCase("fixedlb_put_k105_minmax100", 100.0, 0.0, 0.10, 0.20, false, Option::Put,
                      105.0, 100.0, T, false);
    fixedLookbackCase("fixedlb_put_k100_minmax100", 100.0, 0.0, 0.10, 0.20, false, Option::Put,
                      100.0, 100.0, T, false);
    fixedLookbackCase("fixedlb_put_k95_minmax100", 100.0, 0.0, 0.10, 0.20, false, Option::Put, 95.0,
                      100.0, T, false);

    // Volatility sweep at the two branch sides; the lambda exponent
    // 2(r-q)/vol^2 changes by an order of magnitude across these.
    fixedLookbackCase("fixedlb_call_k95_vol10", 100.0, 0.0, 0.10, 0.10, false, Option::Call, 95.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_call_k95_vol30", 100.0, 0.0, 0.10, 0.30, false, Option::Call, 95.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_call_k105_vol30", 100.0, 0.0, 0.10, 0.30, false, Option::Call, 105.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_put_k105_vol30", 100.0, 0.0, 0.10, 0.30, false, Option::Put, 105.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_call_k95_half_year", 100.0, 0.0, 0.10, 0.20, false, Option::Call,
                      95.0, 100.0, 182, false);

    // Non-zero dividend yield: separates riskFreeDiscount from
    // dividendDiscount inside A / B.
    fixedLookbackCase("fixedlb_call_k95_q4", 100.0, 0.04, 0.10, 0.20, false, Option::Call, 95.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_put_k105_q4", 100.0, 0.04, 0.10, 0.20, false, Option::Put, 105.0,
                      100.0, T, false);
    // q > r flips the sign of lambda.
    fixedLookbackCase("fixedlb_call_k95_q_above_r", 100.0, 0.12, 0.10, 0.20, false, Option::Call,
                      95.0, 100.0, T, false);

    // A running extremum away from spot, on both sides.
    fixedLookbackCase("fixedlb_call_k95_minmax120", 100.0, 0.0, 0.10, 0.20, false, Option::Call,
                      95.0, 120.0, T, false);
    fixedLookbackCase("fixedlb_put_k105_minmax80", 100.0, 0.0, 0.10, 0.20, false, Option::Put,
                      105.0, 80.0, T, false);

    // Zero strike: allowed for calls (`strike >= 0`), rejected for puts
    // (`strike > 0`).
    fixedLookbackCase("fixedlb_call_k0_minmax100", 100.0, 0.0, 0.10, 0.20, false, Option::Call, 0.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_put_k0_throws", 100.0, 0.0, 0.10, 0.20, false, Option::Put, 0.0,
                      100.0, T, false);

    // r == q makes lambda exactly 0 and C++ divides by it. Whatever comes out
    // (NaN / +-inf) is the contract: a port must not special-case it.
    fixedLookbackCase("fixedlb_r_equals_q_call", 100.0, 0.10, 0.10, 0.20, false, Option::Call, 95.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_r_equals_q_put", 100.0, 0.10, 0.10, 0.20, false, Option::Put, 105.0,
                      100.0, T, false);

    fixedLookbackCase("fixedlb_call_k95_mixed_dc", 100.0, 0.04, 0.10, 0.20, true, Option::Call, 95.0,
                      100.0, T, false);
    fixedLookbackCase("fixedlb_put_k95_mixed_dc", 100.0, 0.04, 0.10, 0.20, true, Option::Put, 95.0,
                      100.0, T, false);

    fixedLookbackCase("fixedlb_non_plain_payoff_throws", 100.0, 0.0, 0.10, 0.20, false,
                      Option::Call, 95.0, 100.0, T, true);
}

// ===========================================================================
// 6. AnalyticDiscreteGeometricAverageStrikeAsianEngine
// ===========================================================================

void geomStrikeCase(const std::string& name, Real spot, Rate q, Rate r, Volatility vol,
                    bool mixedDc, Option::Type type, Real strike,
                    const std::vector<Integer>& fixingDays, Integer maturityDays,
                    Average::Type averageType, Size pastFixings, Real runningAccumulator,
                    bool american) {
    std::vector<Date> fixingDates;
    fixingDates.reserve(fixingDays.size());
    for (Integer d : fixingDays)
        fixingDates.push_back(kToday + d);
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", "AnalyticDiscreteGeometricAverageStrikeAsianEngine");
    putFlatMarket(in, spot, q, r, vol, mixedDc);
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.dates("fixing_days", fixingDates, kToday);
    in.i("maturity_days", maturityDays);
    in.s("average_type", averageType == Average::Geometric ? "Geometric" : "Arithmetic");
    in.i("past_fixings", static_cast<long long>(pastFixings));
    in.n("running_accumulator", runningAccumulator);
    in.s("exercise", american ? "American" : "European");

    const Process p = mixedDc ? bsmMixedDc(spot, q, r, vol) : bsm(spot, q, r, vol);
    const ext::shared_ptr<Exercise> ex =
        american ? ext::shared_ptr<Exercise>(
                       ext::make_shared<AmericanExercise>(kToday, maturity))
                 : ext::shared_ptr<Exercise>(ext::make_shared<EuropeanExercise>(maturity));

    DiscreteAveragingAsianOption option(averageType, runningAccumulator, pastFixings, fixingDates,
                                        ext::make_shared<PlainVanillaPayoff>(type, strike), ex);
    option.setPricingEngine(
        ext::make_shared<AnalyticDiscreteGeometricAverageStrikeAsianEngine>(p));

    Obj out;
    const bool threw = throwsOn([&] { option.NPV(); });
    out.b("throws", threw);
    if (!threw)
        putOneAssetResults(out, option);
    addCase(name, in, out);
}

void emitGeometricAverageStrike() {
    // Upstream market (test-suite/asianoptions.cpp testAnalyticDiscrete
    // GeometricAverageStrike): S = 100, q = 3%, r = 6%, sigma = 20%, 10 fixings
    // 36 days apart, expiry at day 360.
    std::vector<Integer> tenFixings;
    for (int j = 1; j <= 10; ++j)
        tenFixings.push_back(36 * j);
    const std::vector<Integer> threeFixings{120, 240, 360};
    const Integer T = 360;
    const Average::Type G = Average::Geometric;

    geomStrikeCase("geomstrike_call_k100_10fix", 100.0, 0.03, 0.06, 0.20, false, Option::Call,
                   100.0, tenFixings, T, G, 0, 1.0, false);
    geomStrikeCase("geomstrike_put_k100_10fix", 100.0, 0.03, 0.06, 0.20, false, Option::Put, 100.0,
                   tenFixings, T, G, 0, 1.0, false);

    // The payoff STRIKE never enters the formula -- only the option type does.
    // These two must reproduce `geomstrike_call_k100_10fix` exactly.
    geomStrikeCase("geomstrike_call_k50_10fix_strike_ignored", 100.0, 0.03, 0.06, 0.20, false,
                   Option::Call, 50.0, tenFixings, T, G, 0, 1.0, false);
    geomStrikeCase("geomstrike_call_k200_10fix_strike_ignored", 100.0, 0.03, 0.06, 0.20, false,
                   Option::Call, 200.0, tenFixings, T, G, 0, 1.0, false);

    geomStrikeCase("geomstrike_call_k100_3fix", 100.0, 0.03, 0.06, 0.20, false, Option::Call, 100.0,
                   threeFixings, T, G, 0, 1.0, false);
    geomStrikeCase("geomstrike_put_k100_3fix", 100.0, 0.03, 0.06, 0.20, false, Option::Put, 100.0,
                   threeFixings, T, G, 0, 1.0, false);

    // Fixing grid anchored at today: fixingTimes are measured from
    // fixingDates[0], so this is the case where that origin coincides with the
    // curve reference date and a port that used the reference date agrees. The
    // cases above, whose first fixing is at day 36 or 120, are what separate
    // them.
    std::vector<Integer> fromToday{0, 90, 180, 270};
    geomStrikeCase("geomstrike_call_k100_fixings_from_today", 100.0, 0.03, 0.06, 0.20, false,
                   Option::Call, 100.0, fromToday, T, G, 0, 1.0, false);

    // A single fixing well before expiry: N == 1, timeSum == 0, so variance
    // collapses to 0 and the whole price is driven by sigmaSum_2 =
    // vol^2 * residualTime.
    geomStrikeCase("geomstrike_call_k100_single_fixing", 100.0, 0.03, 0.06, 0.20, false,
                   Option::Call, 100.0, {120}, T, G, 0, 1.0, false);
    geomStrikeCase("geomstrike_put_k100_single_fixing", 100.0, 0.03, 0.06, 0.20, false, Option::Put,
                   100.0, {120}, T, G, 0, 1.0, false);
    // ...and a single fixing ON the expiry date: residualTime is 0 too, so
    // sigmaSum_2 is exactly 0 and C++ divides by zero.
    geomStrikeCase("geomstrike_call_single_fixing_at_expiry_nan", 100.0, 0.03, 0.06, 0.20, false,
                   Option::Call, 100.0, {T}, T, G, 0, 1.0, false);

    geomStrikeCase("geomstrike_call_k100_zero_q", 100.0, 0.0, 0.06, 0.20, false, Option::Call,
                   100.0, tenFixings, T, G, 0, 1.0, false);
    geomStrikeCase("geomstrike_call_k100_q_above_r", 100.0, 0.09, 0.06, 0.20, false, Option::Call,
                   100.0, tenFixings, T, G, 0, 1.0, false);
    geomStrikeCase("geomstrike_call_k100_mixed_dc", 100.0, 0.03, 0.06, 0.20, true, Option::Call,
                   100.0, tenFixings, T, G, 0, 1.0, false);
    geomStrikeCase("geomstrike_put_k100_mixed_dc", 100.0, 0.03, 0.06, 0.20, true, Option::Put,
                   100.0, tenFixings, T, G, 0, 1.0, false);

    // Guards.
    geomStrikeCase("geomstrike_arithmetic_throws", 100.0, 0.03, 0.06, 0.20, false, Option::Call,
                   100.0, tenFixings, T, Average::Arithmetic, 0, 1.0, false);
    geomStrikeCase("geomstrike_past_fixings_throws", 100.0, 0.03, 0.06, 0.20, false, Option::Call,
                   100.0, tenFixings, T, G, 2, 9800.0, false);
    geomStrikeCase("geomstrike_american_throws", 100.0, 0.03, 0.06, 0.20, false, Option::Call,
                   100.0, tenFixings, T, G, 0, 1.0, true);
}

// ===========================================================================
// 7. TurnbullWakemanAsianEngine
// ===========================================================================

// "flat" | "up" | "down" -- the three vol-term-structure shapes of the upstream
// test. "up" and "down" build a BlackVarianceCurve through the fixing dates.
Handle<BlackVolTermStructure> twVol(const std::string& slope, Volatility base,
                                    const std::vector<Date>& fixingDates) {
    if (slope == "flat")
        return flatVol(base, dc365());
    const Real volSlope = 0.005;
    const std::size_t n = fixingDates.size();
    std::vector<Volatility> vols(n);
    for (std::size_t j = 0; j < n; ++j)
        vols[j] = (slope == "up")
                      ? base - Real(n - 1) * volSlope + Real(j) * volSlope
                      : base + Real(n - 1) * volSlope - Real(j) * volSlope;
    return Handle<BlackVolTermStructure>(ext::make_shared<BlackVarianceCurve>(
        kToday, fixingDates, vols, dc365(), slope == "up"));
}

void putRealVector(Obj& o, const std::string& key, OneAssetOption& option,
                   const std::string& tag) {
    try {
        const std::vector<Real> v = option.result<std::vector<Real>>(tag);
        o.reals(key, v);
    } catch (const std::exception&) {
        o.b(key + "_provided", false);
    }
}

void twCase(const std::string& name, Real spot, Rate q, Rate r, Volatility vol,
            const std::string& slope, Option::Type type, Real strike,
            const std::vector<Integer>& fixingDays, Integer maturityDays, Size pastFixings,
            Real runningAccumulator, Average::Type averageType, bool american,
            bool cashOrNothingPayoff) {
    std::vector<Date> fixingDates;
    fixingDates.reserve(fixingDays.size());
    for (Integer d : fixingDays)
        fixingDates.push_back(kToday + d);
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", "TurnbullWakemanAsianEngine");
    putFlatMarket(in, spot, q, r, vol, false);
    in.s("vol_slope", slope);
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.dates("fixing_days", fixingDates, kToday);
    in.i("maturity_days", maturityDays);
    in.i("past_fixings", static_cast<long long>(pastFixings));
    in.n("running_accumulator", runningAccumulator);
    in.s("average_type", averageType == Average::Geometric ? "Geometric" : "Arithmetic");
    in.s("exercise", american ? "American" : "European");
    in.s("payoff", cashOrNothingPayoff ? "CashOrNothing" : "PlainVanilla");

    const auto p = ext::make_shared<BlackScholesMertonProcess>(
        quote(spot), flatCurve(q, dc365()), flatCurve(r, dc365()),
        twVol(slope, vol, fixingDates));

    const ext::shared_ptr<Exercise> ex =
        american ? ext::shared_ptr<Exercise>(
                       ext::make_shared<AmericanExercise>(kToday, maturity))
                 : ext::shared_ptr<Exercise>(ext::make_shared<EuropeanExercise>(maturity));
    const ext::shared_ptr<StrikedTypePayoff> payoff =
        cashOrNothingPayoff
            ? ext::shared_ptr<StrikedTypePayoff>(
                  ext::make_shared<CashOrNothingPayoff>(type, strike, 10.0))
            : ext::shared_ptr<StrikedTypePayoff>(
                  ext::make_shared<PlainVanillaPayoff>(type, strike));

    DiscreteAveragingAsianOption option(averageType, runningAccumulator, pastFixings, fixingDates,
                                        payoff, ex);
    option.setPricingEngine(ext::make_shared<TurnbullWakemanAsianEngine>(p));

    Obj out;
    const bool threw = throwsOn([&] { option.NPV(); });
    out.b("throws", threw);
    if (!threw) {
        putOneAssetResults(out, option);
        for (const char* tag : {"accrued", "discount", "strike", "effective_strike", "forward",
                                "exp_A_2", "tte", "sigma"})
            putOpt(out, std::string("ar_") + tag,
                   [&] { return option.result<Real>(tag); });
        putRealVector(out, "ar_times", option, "times");
        putRealVector(out, "ar_spotVols", option, "spotVols");
        putRealVector(out, "ar_forwards", option, "forwards");
    }
    addCase(name, in, out);
}

void emitTurnbullWakeman() {
    // Upstream market (test-suite/asianoptions.cpp testTurnbullWakemanAsian
    // Engine): S = 100, cost of carry b = 0 so q == r = 5%, sigma = 20%, 26
    // fixings from t = 1/52 to t = 0.5 and expiry at t = 0.5.
    const Size fixings = 26;
    const Real first = 1.0 / 52.0, expiry = 0.5;
    const Real dt = (expiry - first) / Real(fixings - 1);
    std::vector<Integer> fixingDays;
    for (Size j = 0; j < fixings; ++j)
        fixingDays.push_back(static_cast<Integer>(std::lround(365.0 * (first + Real(j) * dt))));
    const Integer T = static_cast<Integer>(std::lround(365.0 * expiry));
    const Average::Type A = Average::Arithmetic;

    for (Real strike : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        const std::string k = std::to_string(static_cast<int>(strike));
        twCase("tw_call_k" + k + "_flat", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call, strike,
               fixingDays, T, 0, 0.0, A, false, false);
        twCase("tw_put_k" + k + "_flat", 100.0, 0.05, 0.05, 0.20, "flat", Option::Put, strike,
               fixingDays, T, 0, 0.0, A, false, false);
    }

    // Sloping vol term structures: the engine reads blackVariance PER FIXING
    // DATE at the effective strike, so a port that reads a single vol at expiry
    // reproduces the flat cases and fails these.
    twCase("tw_call_k100_vol_up", 100.0, 0.05, 0.05, 0.20, "up", Option::Call, 100.0, fixingDays, T,
           0, 0.0, A, false, false);
    twCase("tw_call_k100_vol_down", 100.0, 0.05, 0.05, 0.20, "down", Option::Call, 100.0,
           fixingDays, T, 0, 0.0, A, false, false);
    twCase("tw_put_k100_vol_up", 100.0, 0.05, 0.05, 0.20, "up", Option::Put, 100.0, fixingDays, T,
           0, 0.0, A, false, false);
    twCase("tw_put_k100_vol_down", 100.0, 0.05, 0.05, 0.20, "down", Option::Put, 100.0, fixingDays,
           T, 0, 0.0, A, false, false);

    // b != 0: q and r differ, so `forward = spot * qDF / rDF` is exercised.
    twCase("tw_call_k100_carry", 100.0, 0.02, 0.05, 0.20, "flat", Option::Call, 100.0, fixingDays,
           T, 0, 0.0, A, false, false);
    twCase("tw_put_k100_carry", 100.0, 0.02, 0.05, 0.20, "flat", Option::Put, 100.0, fixingDays, T,
           0, 0.0, A, false, false);

    // Seasoned: 6 past fixings summing to 594 (average 99), 26 future ones.
    // accrued = 594 / (6 + 26), i.e. divided by the TOTAL count, not by 6.
    twCase("tw_call_k100_past_fixings", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call, 100.0,
           fixingDays, T, 6, 594.0, A, false, false);
    twCase("tw_put_k100_past_fixings", 100.0, 0.05, 0.05, 0.20, "flat", Option::Put, 100.0,
           fixingDays, T, 6, 594.0, A, false, false);
    twCase("tw_call_k100_one_past_fixing", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call, 100.0,
           fixingDays, T, 1, 99.0, A, false, false);

    // effectiveStrike == strike - accrued <= 0 -> the closed-form guaranteed
    // exercise / permanent OTM branch. 26 past fixings averaging 100 give an
    // accrued of 100*26/52 = 50, so a strike of 40 is already covered.
    twCase("tw_call_guaranteed_exercise", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call, 40.0,
           fixingDays, T, 26, 2600.0, A, false, false);
    twCase("tw_put_guaranteed_otm", 100.0, 0.05, 0.05, 0.20, "flat", Option::Put, 40.0, fixingDays,
           T, 26, 2600.0, A, false, false);
    // ...and exactly at the boundary, effectiveStrike == 0, which the `<= 0`
    // test sends down the same branch.
    twCase("tw_call_effective_strike_exactly_zero", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call,
           50.0, fixingDays, T, 26, 2600.0, A, false, false);

    // Guards.
    twCase("tw_geometric_throws", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call, 100.0, fixingDays,
           T, 0, 1.0, Average::Geometric, false, false);
    twCase("tw_american_throws", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call, 100.0, fixingDays,
           T, 0, 0.0, A, true, false);
    twCase("tw_cash_or_nothing_throws", 100.0, 0.05, 0.05, 0.20, "flat", Option::Call, 100.0,
           fixingDays, T, 0, 0.0, A, false, true);
}

// ===========================================================================
// 8. AnalyticDoubleBarrierBinaryEngine
// ===========================================================================

std::string barrierName(DoubleBarrier::Type t) {
    switch (t) {
        case DoubleBarrier::KnockIn:
            return "KnockIn";
        case DoubleBarrier::KnockOut:
            return "KnockOut";
        case DoubleBarrier::KIKO:
            return "KIKO";
        case DoubleBarrier::KOKI:
            return "KOKI";
    }
    return "?";
}

void dbBinaryCase(const std::string& name, DoubleBarrier::Type barrierType, Real barrierLo,
                  Real barrierHi, Real cash, Real spot, Rate q, Rate r, Volatility vol,
                  Integer maturityDays, Option::Type type, Real strike, const std::string& exType,
                  bool plainPayoff) {
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", "AnalyticDoubleBarrierBinaryEngine");
    putFlatMarket(in, spot, q, r, vol, false);
    in.s("barrier_type", barrierName(barrierType));
    in.n("barrier_lo", barrierLo);
    in.n("barrier_hi", barrierHi);
    in.n("cash", cash);
    in.n("rebate", 0.0);
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.i("maturity_days", maturityDays);
    in.s("exercise", exType);
    in.s("payoff", plainPayoff ? "PlainVanilla" : "CashOrNothing");

    const Process p = bsm(spot, q, r, vol);
    ext::shared_ptr<Exercise> ex;
    if (exType == "European")
        ex = ext::make_shared<EuropeanExercise>(maturity);
    else if (exType == "American")
        ex = ext::make_shared<AmericanExercise>(kToday, maturity);
    else  // "AmericanWindow": first exercise date after the vol reference date
        ex = ext::make_shared<AmericanExercise>(kToday + 30, maturity);

    const ext::shared_ptr<StrikedTypePayoff> payoff =
        plainPayoff ? ext::shared_ptr<StrikedTypePayoff>(
                          ext::make_shared<PlainVanillaPayoff>(type, strike))
                    : ext::shared_ptr<StrikedTypePayoff>(
                          ext::make_shared<CashOrNothingPayoff>(type, strike, cash));

    DoubleBarrierOption option(barrierType, barrierLo, barrierHi, 0.0, payoff, ex);
    option.setPricingEngine(ext::make_shared<AnalyticDoubleBarrierBinaryEngine>(p));

    Obj out;
    const bool threw = throwsOn([&] { option.NPV(); });
    out.b("throws", threw);
    if (!threw)
        putOneAssetResults(out, option);
    addCase(name, in, out);
}

void emitDoubleBarrierBinary() {
    // Haug p.181 market: S = 100, q = 2%, r = 5%, T = 0.25, cash = 10.
    // 91 days on Actual365Fixed is 0.24931507 -- close enough to the book's
    // 0.25 that the reproduced values are recognisable, and exact under the
    // day counter, which is what matters for cross-validation.
    const Integer T = 91;
    const auto KO = DoubleBarrier::KnockOut;
    const auto KI = DoubleBarrier::KnockIn;

    // Barrier-width sweep at four volatilities: the Hui series converges much
    // more slowly for narrow barriers and high vol.
    for (auto lohi : {std::make_pair(80.0, 120.0), std::make_pair(85.0, 115.0),
                      std::make_pair(90.0, 110.0), std::make_pair(95.0, 105.0)}) {
        const std::string w = std::to_string(static_cast<int>(lohi.first)) + "_" +
                              std::to_string(static_cast<int>(lohi.second));
        for (Volatility v : {0.10, 0.20, 0.30, 0.50}) {
            const std::string vs = std::to_string(static_cast<int>(v * 100));
            dbBinaryCase("dbbin_ko_" + w + "_vol" + vs, KO, lohi.first, lohi.second, 10.0, 100.0,
                         0.02, 0.05, v, T, Option::Call, 100.0, "European", false);
        }
    }
    // KnockIn on the same grid: the KI branch is `max(cash * discount - tot, 0)`
    // and needs the discount factor, which the KO branch never touches.
    for (Volatility v : {0.10, 0.20, 0.30, 0.50}) {
        const std::string vs = std::to_string(static_cast<int>(v * 100));
        dbBinaryCase("dbbin_ki_80_120_vol" + vs, KI, 80.0, 120.0, 10.0, 100.0, 0.02, 0.05, v, T,
                     Option::Call, 100.0, "European", false);
    }
    dbBinaryCase("dbbin_ki_90_110_vol30", KI, 90.0, 110.0, 10.0, 100.0, 0.02, 0.05, 0.30, T,
                 Option::Call, 100.0, "European", false);

    // KIKO / KOKI use the *other* series (payoffKIKO, 1000 iterations) and
    // require American exercise. KOKI swaps the two barriers before the series.
    for (Volatility v : {0.10, 0.20, 0.30, 0.50}) {
        const std::string vs = std::to_string(static_cast<int>(v * 100));
        dbBinaryCase("dbbin_kiko_80_120_vol" + vs, DoubleBarrier::KIKO, 80.0, 120.0, 10.0, 100.0,
                     0.02, 0.05, v, T, Option::Call, 100.0, "American", false);
        dbBinaryCase("dbbin_koki_80_120_vol" + vs, DoubleBarrier::KOKI, 80.0, 120.0, 10.0, 100.0,
                     0.02, 0.05, v, T, Option::Call, 100.0, "American", false);
    }
    dbBinaryCase("dbbin_kiko_90_110_vol20", DoubleBarrier::KIKO, 90.0, 110.0, 10.0, 100.0, 0.02,
                 0.05, 0.20, T, Option::Call, 100.0, "American", false);
    dbBinaryCase("dbbin_koki_90_110_vol20", DoubleBarrier::KOKI, 90.0, 110.0, 10.0, 100.0, 0.02,
                 0.05, 0.20, T, Option::Call, 100.0, "American", false);

    // The Put option type is IGNORED by the engine (the commented-out
    // `Option::Type type = payoff_->optionType();` in the C++ source): only the
    // cash payoff matters. This must reproduce dbbin_ko_80_120_vol20 exactly.
    dbBinaryCase("dbbin_ko_80_120_vol20_put_type_ignored", KO, 80.0, 120.0, 10.0, 100.0, 0.02, 0.05,
                 0.20, T, Option::Put, 100.0, "European", false);
    // ...and so is the strike.
    dbBinaryCase("dbbin_ko_80_120_vol20_strike_ignored", KO, 80.0, 120.0, 10.0, 100.0, 0.02, 0.05,
                 0.20, T, Option::Call, 130.0, "European", false);

    // Non-zero carry: b = r - q enters alpha and beta.
    dbBinaryCase("dbbin_ko_80_120_zero_q", KO, 80.0, 120.0, 10.0, 100.0, 0.0, 0.05, 0.20, T,
                 Option::Call, 100.0, "European", false);
    dbBinaryCase("dbbin_ko_80_120_q_above_r", KO, 80.0, 120.0, 10.0, 100.0, 0.09, 0.05, 0.20, T,
                 Option::Call, 100.0, "European", false);

    // All eight degenerate early returns (spot outside the barriers). These are
    // the ONLY paths on which the engine fills delta / gamma / vega / rho --
    // all with hard zeros -- while still leaving theta and dividendRho unset.
    dbBinaryCase("dbbin_ko_spot_at_lo_barrier", KO, 100.0, 120.0, 10.0, 100.0, 0.02, 0.05, 0.20, T,
                 Option::Call, 100.0, "European", false);
    dbBinaryCase("dbbin_ko_spot_above_hi_barrier", KO, 80.0, 100.0, 10.0, 100.0, 0.02, 0.05, 0.20,
                 T, Option::Call, 100.0, "European", false);
    dbBinaryCase("dbbin_ki_spot_at_lo_barrier", KI, 100.0, 120.0, 10.0, 100.0, 0.02, 0.05, 0.20, T,
                 Option::Call, 100.0, "European", false);
    dbBinaryCase("dbbin_ki_spot_above_hi_barrier", KI, 80.0, 100.0, 10.0, 100.0, 0.02, 0.05, 0.20,
                 T, Option::Call, 100.0, "European", false);
    dbBinaryCase("dbbin_kiko_spot_above_hi_barrier", DoubleBarrier::KIKO, 80.0, 100.0, 10.0, 100.0,
                 0.02, 0.05, 0.20, T, Option::Call, 100.0, "American", false);
    dbBinaryCase("dbbin_kiko_spot_at_lo_barrier", DoubleBarrier::KIKO, 100.0, 120.0, 10.0, 100.0,
                 0.02, 0.05, 0.20, T, Option::Call, 100.0, "American", false);
    dbBinaryCase("dbbin_koki_spot_at_lo_barrier", DoubleBarrier::KOKI, 100.0, 120.0, 10.0, 100.0,
                 0.02, 0.05, 0.20, T, Option::Call, 100.0, "American", false);
    dbBinaryCase("dbbin_koki_spot_above_hi_barrier", DoubleBarrier::KOKI, 80.0, 100.0, 10.0, 100.0,
                 0.02, 0.05, 0.20, T, Option::Call, 100.0, "American", false);

    // Guards.
    dbBinaryCase("dbbin_kiko_european_throws", DoubleBarrier::KIKO, 80.0, 120.0, 10.0, 100.0, 0.02,
                 0.05, 0.20, T, Option::Call, 100.0, "European", false);
    dbBinaryCase("dbbin_koki_european_throws", DoubleBarrier::KOKI, 80.0, 120.0, 10.0, 100.0, 0.02,
                 0.05, 0.20, T, Option::Call, 100.0, "European", false);
    dbBinaryCase("dbbin_kiko_american_window_throws", DoubleBarrier::KIKO, 80.0, 120.0, 10.0, 100.0,
                 0.02, 0.05, 0.20, T, Option::Call, 100.0, "AmericanWindow", false);
    dbBinaryCase("dbbin_ko_american_throws", KO, 80.0, 120.0, 10.0, 100.0, 0.02, 0.05, 0.20, T,
                 Option::Call, 100.0, "American", false);
    dbBinaryCase("dbbin_plain_vanilla_payoff_throws", KO, 80.0, 120.0, 10.0, 100.0, 0.02, 0.05,
                 0.20, T, Option::Call, 100.0, "European", true);
    dbBinaryCase("dbbin_lo_above_hi_throws", KO, 120.0, 80.0, 10.0, 100.0, 0.02, 0.05, 0.20, T,
                 Option::Call, 100.0, "European", false);

    // Very wide barriers with a tiny volatility: alpha ~ -74 makes
    // (spot/barrier_hi)^alpha astronomically large, the 99-term truncation
    // leaves a term far above 1e-8 and the QL_REQUIRE fires.
    dbBinaryCase("dbbin_series_does_not_converge_throws", KO, 50.0, 200.0, 10.0, 100.0, 0.02, 0.05,
                 0.02, T, Option::Call, 100.0, "European", false);
}

// ===========================================================================
// 9. ForwardPerformanceVanillaEngine (and its base, ForwardVanillaEngine)
// ===========================================================================

void forwardCase(const std::string& name, bool performance, Real spot, Rate q, Rate r,
                 Volatility vol, bool mixedDc, Option::Type type, Real moneyness,
                 Integer resetDays, Integer maturityDays) {
    const Date reset = kToday + resetDays;
    const Date maturity = kToday + maturityDays;

    Obj in;
    in.s("engine", performance ? "ForwardPerformanceVanillaEngine" : "ForwardVanillaEngine");
    putFlatMarket(in, spot, q, r, vol, mixedDc);
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("moneyness", moneyness);
    in.i("reset_days", resetDays);
    in.i("maturity_days", maturityDays);

    const Process p = mixedDc ? bsmMixedDc(spot, q, r, vol) : bsm(spot, q, r, vol);
    // C++ passes a strike-0 PlainVanillaPayoff; the engine replaces the strike
    // with `moneyness * process->x0()` before delegating, so the payoff strike
    // carried by the instrument is irrelevant.
    ForwardVanillaOption option(moneyness, reset,
                                ext::make_shared<PlainVanillaPayoff>(type, 0.0),
                                ext::make_shared<EuropeanExercise>(maturity));
    if (performance)
        option.setPricingEngine(
            ext::make_shared<ForwardPerformanceVanillaEngine<AnalyticEuropeanEngine>>(p));
    else
        option.setPricingEngine(
            ext::make_shared<ForwardVanillaEngine<AnalyticEuropeanEngine>>(p));

    Obj out;
    const bool threw = throwsOn([&] { option.NPV(); });
    out.b("throws", threw);
    if (!threw)
        putOneAssetResults(out, option);
    addCase(name, in, out);
}

void emitForwardEngines() {
    // Haug p.37 market: S = 60, q = 4%, r = 8%, sigma = 30%, reset at 0.25y,
    // expiry at 1y.
    const Integer reset = 91, T = 365;

    for (int pass = 0; pass < 2; ++pass) {
        const bool perf = pass == 1;
        const std::string pre = perf ? "fwdperf_" : "fwd_";

        for (Real m : {0.9, 1.0, 1.1}) {
            const std::string ms = std::to_string(static_cast<int>(m * 100));
            forwardCase(pre + "call_m" + ms, perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call, m,
                        reset, T);
            forwardCase(pre + "put_m" + ms, perf, 60.0, 0.04, 0.08, 0.30, false, Option::Put, m,
                        reset, T);
        }

        // q == 0 collapses ForwardVanillaEngine's discQ to 1 and its theta to 0;
        // r == 0 does the same to ForwardPerformanceVanillaEngine's discR. Both
        // are pinned for both engines so the two discounting rules cannot be
        // swapped and still pass.
        forwardCase(pre + "call_m110_zero_q", perf, 60.0, 0.0, 0.08, 0.30, false, Option::Call, 1.1,
                    reset, T);
        forwardCase(pre + "call_m110_zero_r", perf, 60.0, 0.04, 0.0, 0.30, false, Option::Call, 1.1,
                    reset, T);
        // A spot far from 1 matters only for the performance engine, whose
        // discR is divided by the spot.
        forwardCase(pre + "call_m110_spot1", perf, 1.0, 0.04, 0.08, 0.30, false, Option::Call, 1.1,
                    reset, T);
        forwardCase(pre + "call_m110_spot500", perf, 500.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.1, reset, T);

        // Reset near inception and near expiry.
        forwardCase(pre + "call_m100_early_reset", perf, 60.0, 0.04, 0.08, 0.30, false,
                    Option::Call, 1.0, 7, T);
        forwardCase(pre + "call_m100_late_reset", perf, 60.0, 0.04, 0.08, 0.30, false, Option::Call,
                    1.0, 350, T);

        forwardCase(pre + "call_m110_mixed_dc", perf, 60.0, 0.04, 0.08, 0.30, true, Option::Call,
                    1.1, reset, T);
        forwardCase(pre + "put_m090_mixed_dc", perf, 60.0, 0.04, 0.08, 0.30, true, Option::Put, 0.9,
                    reset, T);
    }
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    emitAnalyticDividendEuropean();
    emitCashDividendEuropean();
    emitCliquetEngines();
    emitFixedLookback();
    emitGeometricAverageStrike();
    emitTurnbullWakeman();
    emitDoubleBarrierBinary();
    emitForwardEngines();

    emitDocument();
    return 0;
}
