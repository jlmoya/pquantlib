// migration-harness/cpp/probes/v143_inst_lookbackvarswap/probe.cpp
//
// Reference values for the two partial-time lookback instruments and the
// variance swap of C++ QuantLib v1.43, together with the three engines
// without which they are inert:
//
//   * ContinuousPartialFloatingLookbackOption  (ql/instruments/lookbackoption.{hpp,cpp})
//     + AnalyticContinuousPartialFloatingLookbackEngine
//       (ql/pricingengines/lookback/analyticcontinuouspartialfloatinglookback.{hpp,cpp})
//       Heynen-Kat (1994) partial-time floating-strike lookback, Haug 2nd ed. p.146.
//
//   * ContinuousPartialFixedLookbackOption     (same header)
//     + AnalyticContinuousPartialFixedLookbackEngine
//       (ql/pricingengines/lookback/analyticcontinuouspartialfixedlookback.{hpp,cpp})
//       Heynen-Kat (1994) partial-time fixed-strike lookback, Haug 2nd ed. p.148.
//
//   * VarianceSwap                             (ql/instruments/varianceswap.{hpp,cpp})
//     + ReplicatingVarianceSwapEngine
//       (ql/pricingengines/forward/replicatingvarianceswapengine.hpp -- header only)
//       Demeterfi-Derman-Kamal-Zou (1999) static replication of a log contract.
//
// What has to be pinned, and why
// ------------------------------
// The two partial-time lookbacks differ from the plain lookbacks PQuantLib
// already has by exactly two extra arguments: `lambda` + `lookbackPeriodEnd`
// (floating) and `lookbackPeriodStart` (fixed). A port that accepts them and
// drops them still compiles, still prices, and silently returns something close
// to the non-partial price. So every case below is part of a sweep that moves
// ONE of those arguments while holding the rest fixed:
//
//   * `pflt_*_lbend*`  moves lookbackPeriodEnd across T1/T2/T3 and onto the
//     exercise date, which flips `fullLookbackPeriod` and takes the engine
//     down its OTHER branch (the 3-term "simpler calculation" instead of the
//     7-term one). Both branches are therefore covered.
//   * `pflt_*_lambda*` moves lambda away from 1.0 in the direction each option
//     type permits (>= 1 for calls, <= 1 for puts -- the arguments::validate
//     constraint). At lambda == 1 and lookbackPeriodEnd == exercise the price
//     collapses onto the plain floating lookback, which is precisely the value
//     a port that ignored both arguments would return; `pflt_*_degenerate_*`
//     pins that collapse so the ignoring port passes there and fails
//     everywhere else, making the diagnosis unambiguous.
//   * `pfix_*_lbstart*` likewise moves lookbackPeriodStart and flips
//     `differentStartOfLookback`.
//
// Around those, spot / running extremum / strike / volatility / risk-free rate
// / dividend yield are each swept one at a time off a common base case, plus
// one all-varied case per option type so that a compensating pair of errors in
// two separate arguments cannot survive.
//
// Neither lookback engine writes a single greek: `calculate()` assigns only
// `results_.value`, so OneAssetOption::results keeps every Greeks/MoreGreeks
// field at Null<Real>() and `option.delta()` throws "delta not provided".
// NPV is the entire observable surface, and `*_greeks_unavailable` pins that
// emptiness explicitly so a port does not invent greeks C++ does not produce.
// The market-wiring quantities the engines feed on (residual time, lookback
// time, both discount factors, and the volatility AT THE STRIKE THE ENGINE
// LOOKS IT UP WITH) are pinned alongside each NPV, because an NPV can match
// while two errors cancel. The floating engine looks the vol up at `minmax`
// and the fixed engine at `strike`; the `*_smile` cases put a genuine skew
// under the option with minmax/strike away from spot, which is the only
// configuration where using the wrong one of those shows up at all.
//
// For the variance swap the arguments that get dropped are `notional` and
// `position` (pure multipliers on NPV, invisible in `variance()`) and the
// engine's `dk` / `callStrikes` / `putStrikes`. So:
//
//   * every market is priced Long AND Short at the same strike and notional,
//     and `vs_*_variance_invariant_*` pins that `variance()` is IDENTICAL
//     across position, notional and strike while NPV is not;
//   * `vs_a_notional*` scales the notional by 1 / 50000 / 1e6;
//   * `vs_a_dk*` moves dk off its 5.0 default, which changes the end strike
//     used for the last slope in the piecewise-log-payoff approximation and
//     hence the whole weight ladder;
//   * `vs_a_ladder*` swaps the strike ladders for shorter ones, for
//     single-point ones, for an UNSORTED permutation of the base ones (which
//     must reproduce the base answer exactly -- that pins the internal
//     std::sort) and for one with duplicate strikes (which pins the
//     std::unique that follows the sort);
//   * `option_weights` is emitted in full for representative cases -- payoff
//     type, strike and weight for every leg of the replicating strip, in the
//     engine's own emission order (all calls ascending, then all puts
//     descending). That is `results_.additionalResults["optionWeights"]`, so
//     it is genuinely observable through the public API.
//   * `vs_*_startdate*` pins that `startDate` reaches the arguments and the
//     inspector but does NOT reach the price (the engine only reads
//     maturityDate) -- an argument that is stored and unused is still an
//     argument a port must not silently reorder with maturityDate.
//
// isExpired
// ---------
// VarianceSwap::isExpired() is `detail::simple_event(maturityDate_).hasOccurred()`,
// i.e. Settings::evaluationDate + Settings::includeReferenceDateEvents drive it:
// includeReferenceDateEvents == false (the C++ default) gives `maturity <= ref`,
// true gives `maturity < ref`. `vs_expired_*` pins all six combinations of
// {ref < maturity, ref == maturity, ref > maturity} x {false, true}, because
// the boundary is the only place the flag is observable.
//
// QL_REQUIRE branches
// -------------------
// Emitted as {"raises": true} after being verified to actually throw:
// VarianceSwap::arguments::validate (null/non-positive strike, null/non-positive
// notional, null start date, null maturity date -- note that the null MATURITY
// branch is unreachable through NPV(), because isExpired() short-circuits on it
// first and returns 0; see vs_null_maturity_*); the ReplicatingVarianceSwapEngine
// constructor (empty call ladder, empty put ladder, non-positive min put strike,
// min-call != max-put); ContinuousPartialFloatingLookbackOption::arguments::validate
// (lookback end after exercise, lambda < 1 for a call, lambda > 1 for a put,
// negative prior extremum); ContinuousPartialFixedLookbackOption::arguments::validate
// (lookback start after exercise); and both engines' payoff / underlying / strike
// guards.
//
// Calendar
// --------
// today = 1 March 2025, exercise = 1 March 2026 is exactly 365 days, so with
// Actual365Fixed the residual time is exactly 1.0 and no day-count rounding
// noise leaks into the reference values.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/lookbackvarswap.json. Nothing else may be printed.

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/lookbackoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/varianceswap.hpp>
#include <ql/math/distributions/bivariatenormaldistribution.hpp>
#include <ql/math/matrix.hpp>
#include <ql/pricingengines/forward/replicatingvarianceswapengine.hpp>
#include <ql/pricingengines/lookback/analyticcontinuouspartialfixedlookback.hpp>
#include <ql/pricingengines/lookback/analyticcontinuouspartialfloatinglookback.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancesurface.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// ---------------------------------------------------------------------------
std::string num(Real v) {
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

    Obj& nums(const std::string& k, const std::vector<Real>& v) {
        std::string a = "[";
        for (std::size_t i = 0; i < v.size(); ++i)
            a += (i != 0U ? ", " : "") + num(v[i]);
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

// Runs `f`, records {"raises": true} when it throws a QuantLib error and
// {"raises": false} when it does not, so a branch that stopped throwing
// upstream shows up as a reference mismatch rather than as silence.
template <class F>
void addRaises(const std::string& name, const Obj& inputs, F f) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception&) {
        threw = true;
    }
    addCase(name, inputs, Obj().b("raises", threw));
}

void emitDocument() {
    std::cout << "{\n";
    for (std::size_t i = 0; i < gCases.size(); ++i)
        std::cout << "  \"" << gCases[i].first << "\": " << gCases[i].second
                  << (i + 1 < gCases.size() ? "," : "") << "\n";
    std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// Market scaffolding
// ---------------------------------------------------------------------------
const Date kToday(1, March, 2025);
const Date kExpiry(1, March, 2026);  // exactly 365 days -> T == 1.0
const Date kT1(1, June, 2025);       // 92 days
const Date kT2(1, September, 2025);  // 184 days
const Date kT3(1, December, 2025);   // 275 days

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

Handle<Quote> quote(Real v) {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(v));
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kToday, r, dayCounter()));
}

Handle<BlackVolTermStructure> flatVol(Volatility v) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(kToday, NullCalendar(), v, dayCounter()));
}

// A genuine skew for the lookback cases: 5 strikes at the single exercise
// tenor. The engines look the volatility up at `minmax` (floating) or at
// `strike` (fixed) -- never at spot -- and only a strike-dependent surface can
// tell those apart. Every smile case keeps its lookup strike inside [80, 120]
// so BlackVolTermStructure::checkStrike never has to extrapolate.
const std::vector<Real> kSmileStrikes = {80.0, 90.0, 100.0, 110.0, 120.0};
const std::vector<Volatility> kSmileVols = {0.34, 0.29, 0.25, 0.22, 0.20};

Handle<BlackVolTermStructure> smileVol() {
    Matrix m(kSmileStrikes.size(), 1);
    for (std::size_t i = 0; i < kSmileStrikes.size(); ++i)
        m[i][0] = kSmileVols[i];
    const std::vector<Date> dates = {kExpiry};
    return Handle<BlackVolTermStructure>(ext::make_shared<BlackVarianceSurface>(
        kToday, NullCalendar(), dates, kSmileStrikes, m, dayCounter()));
}

ext::shared_ptr<GeneralizedBlackScholesProcess>
process(Real spot, Rate r, Rate q, Volatility vol, bool smile) {
    return ext::make_shared<BlackScholesMertonProcess>(
        quote(spot), flatCurve(q), flatCurve(r), smile ? smileVol() : flatVol(vol));
}

std::string typeName(Option::Type t) { return t == Option::Call ? "Call" : "Put"; }

// ---------------------------------------------------------------------------
// Partial-time FLOATING strike lookback
// ---------------------------------------------------------------------------
struct FloatCase {
    std::string name;
    Option::Type type = Option::Call;
    Real spot = 100.0;
    Rate r = 0.05;
    Rate q = 0.02;
    Volatility vol = 0.25;
    Real minmax = 100.0;
    Real lambda = 1.0;
    Date lbEnd = kT2;
    bool smile = false;
};

void runFloat(const FloatCase& c) {
    const auto proc = process(c.spot, c.r, c.q, c.vol, c.smile);
    const auto payoff = ext::make_shared<FloatingTypePayoff>(c.type);
    const auto exercise = ext::make_shared<EuropeanExercise>(kExpiry);

    ContinuousPartialFloatingLookbackOption option(c.minmax, c.lambda, c.lbEnd,
                                                   payoff, exercise);
    option.setPricingEngine(
        ext::make_shared<AnalyticContinuousPartialFloatingLookbackEngine>(proc));

    const Time t = proc->time(kExpiry);
    const Time tLb = proc->time(c.lbEnd);

    Obj in;
    in.s("kind", "partial_floating")
        .s("option_type", typeName(c.type))
        .n("spot", c.spot)
        .n("r", c.r)
        .n("q", c.q)
        .n("vol", c.vol)
        .b("smile", c.smile)
        .n("minmax", c.minmax)
        .n("lambda", c.lambda)
        .i("lookback_end_serial", c.lbEnd.serialNumber())
        .i("expiry_serial", kExpiry.serialNumber());

    Obj out;
    out.n("npv", option.NPV())
        .n("residual_time", t)
        .n("lookback_time", tLb)
        .n("risk_free_discount", proc->riskFreeRate()->discount(t))
        .n("dividend_discount", proc->dividendYield()->discount(t))
        // The engine looks the volatility up at `minmax`, not at spot.
        .n("vol_used", proc->blackVolatility()->blackVol(t, c.minmax))
        // Pinned so a port cannot quietly renumber the arguments carrier.
        .n("args_minmax", c.minmax)
        .n("args_lambda", c.lambda)
        .i("args_lookback_end_serial", c.lbEnd.serialNumber());

    addCase(c.name, in, out);
}

// ---------------------------------------------------------------------------
// Partial-time FIXED strike lookback
// ---------------------------------------------------------------------------
struct FixedCase {
    std::string name;
    Option::Type type = Option::Call;
    Real spot = 100.0;
    Rate r = 0.05;
    Rate q = 0.02;
    Volatility vol = 0.25;
    Real strike = 100.0;
    Date lbStart = kT2;
    bool smile = false;
};

void runFixed(const FixedCase& c) {
    const auto proc = process(c.spot, c.r, c.q, c.vol, c.smile);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(c.type, c.strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(kExpiry);

    ContinuousPartialFixedLookbackOption option(c.lbStart, payoff, exercise);
    option.setPricingEngine(
        ext::make_shared<AnalyticContinuousPartialFixedLookbackEngine>(proc));

    const Time t = proc->time(kExpiry);
    const Time tLb = proc->time(c.lbStart);

    Obj in;
    in.s("kind", "partial_fixed")
        .s("option_type", typeName(c.type))
        .n("spot", c.spot)
        .n("r", c.r)
        .n("q", c.q)
        .n("vol", c.vol)
        .b("smile", c.smile)
        .n("strike", c.strike)
        .i("lookback_start_serial", c.lbStart.serialNumber())
        .i("expiry_serial", kExpiry.serialNumber());

    Obj out;
    out.n("npv", option.NPV())
        .n("residual_time", t)
        .n("lookback_time", tLb)
        .n("risk_free_discount", proc->riskFreeRate()->discount(t))
        .n("dividend_discount", proc->dividendYield()->discount(t))
        // The engine looks the volatility up at `strike`, not at spot.
        .n("vol_used", proc->blackVolatility()->blackVol(t, c.strike))
        // ContinuousPartialFixedLookbackOption's constructor forwards a HARD
        // ZERO as the base class's prior extremum (lookbackoption.cpp:121);
        // it takes no running-extremum argument at all.
        .n("args_minmax", 0.0)
        .i("args_lookback_start_serial", c.lbStart.serialNumber());

    addCase(c.name, in, out);
}

// ---------------------------------------------------------------------------
// Variance swap
// ---------------------------------------------------------------------------
// The Demeterfi-Derman-Kamal-Zou (1999) skew, verbatim from the v1.43
// test-suite case testReplicatingVarianceSwap (test-suite/varianceswaps.cpp).
const std::vector<Real> kDdkzStrikes = {50.0,  55.0,  60.0,  65.0,  70.0,  75.0,
                                        80.0,  85.0,  90.0,  95.0,  100.0, 105.0,
                                        110.0, 115.0, 120.0, 125.0, 130.0, 135.0};
const std::vector<Volatility> kDdkzVols = {0.30, 0.29, 0.28, 0.27, 0.26, 0.25,
                                           0.24, 0.23, 0.22, 0.21, 0.20, 0.19,
                                           0.18, 0.17, 0.16, 0.15, 0.14, 0.13};
const std::vector<Real> kDdkzCalls = {100.0, 105.0, 110.0, 115.0,
                                      120.0, 125.0, 130.0, 135.0};
const std::vector<Real> kDdkzPuts = {50.0, 55.0, 60.0, 65.0,  70.0, 75.0,
                                     80.0, 85.0, 90.0, 95.0, 100.0};

// Market A: the DDKZ example. maturity = today + 90 days.
const Date kVsMatA = kToday + 90;
// Market B: flat vol, different spot / rates / maturity, so a wiring error
// that happens to cancel under the DDKZ skew has nowhere to hide.
const Date kVsMatB = kToday + 270;

Handle<BlackVolTermStructure> ddkzVol() {
    Matrix m(kDdkzStrikes.size(), 1);
    for (std::size_t i = 0; i < kDdkzStrikes.size(); ++i)
        m[i][0] = kDdkzVols[i];
    const std::vector<Date> dates = {kVsMatA};
    return Handle<BlackVolTermStructure>(ext::make_shared<BlackVarianceSurface>(
        kToday, NullCalendar(), dates, kDdkzStrikes, m, dayCounter()));
}

struct VarCase {
    std::string name;
    std::string market = "A";  // "A" = DDKZ skew, "B" = flat vol
    Position::Type position = Position::Long;
    Real strike = 0.04;
    Real notional = 50000.0;
    Date start = kToday;
    Date maturity = kVsMatA;
    Real dk = 5.0;
    std::vector<Real> calls = kDdkzCalls;
    std::vector<Real> puts = kDdkzPuts;
    bool emitWeights = false;
};

ext::shared_ptr<GeneralizedBlackScholesProcess> varProcess(const std::string& market) {
    if (market == "A")
        return ext::make_shared<BlackScholesMertonProcess>(
            quote(100.0), flatCurve(0.00), flatCurve(0.05), ddkzVol());
    return ext::make_shared<BlackScholesMertonProcess>(
        quote(120.0), flatCurve(0.03), flatCurve(0.02), flatVol(0.22));
}

void runVar(const VarCase& c) {
    const auto proc = varProcess(c.market);
    const auto engine = ext::make_shared<ReplicatingVarianceSwapEngine>(
        proc, c.dk, c.calls, c.puts);

    VarianceSwap swap(c.position, c.strike, c.notional, c.start, c.maturity);
    swap.setPricingEngine(engine);

    Obj in;
    in.s("kind", "variance_swap")
        .s("market", c.market)
        .s("position", c.position == Position::Long ? "Long" : "Short")
        .n("strike", c.strike)
        .n("notional", c.notional)
        .i("start_serial", c.start.serialNumber())
        .i("maturity_serial", c.maturity.serialNumber())
        .n("dk", c.dk)
        .nums("call_strikes", c.calls)
        .nums("put_strikes", c.puts);

    Obj out;
    out.n("npv", swap.NPV())
        .n("variance", swap.variance())
        .n("inspector_strike", swap.strike())
        .n("inspector_notional", swap.notional())
        .s("inspector_position", swap.position() == Position::Long ? "Long" : "Short")
        .i("inspector_start_serial", swap.startDate().serialNumber())
        .i("inspector_maturity_serial", swap.maturityDate().serialNumber())
        .b("is_expired", swap.isExpired())
        .n("residual_time", proc->time(c.maturity))
        .n("risk_free_discount", proc->riskFreeRate()->discount(c.maturity));

    if (c.emitWeights) {
        // results_.additionalResults["optionWeights"] -- the replicating strip
        // in the engine's own emission order: calls ascending, then puts
        // descending. Payoff type + strike + weight for every leg.
        using Weights = ReplicatingVarianceSwapEngine::weights_type;
        const Weights w = swap.result<Weights>("optionWeights");
        std::string arr = "[";
        for (std::size_t i = 0; i < w.size(); ++i) {
            const auto plain = ext::dynamic_pointer_cast<PlainVanillaPayoff>(w[i].first);
            arr += (i != 0U ? ", " : "");
            arr += Obj()
                       .s("type", typeName(plain->optionType()))
                       .n("strike", plain->strike())
                       .n("weight", w[i].second)
                       .str();
        }
        out.raw("option_weights", arr + "]");
    }

    addCase(c.name, in, out);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = kToday;

    // =======================================================================
    // Partial-time FLOATING lookback
    // =======================================================================
    for (const Option::Type type : {Option::Call, Option::Put}) {
        const std::string tag = type == Option::Call ? "call" : "put";
        // lambda must be >= 1 for calls and <= 1 for puts
        // (ContinuousPartialFloatingLookbackOption::arguments::validate).
        const std::vector<Real> lambdas =
            type == Option::Call ? std::vector<Real>{1.05, 1.25, 1.60}
                                 : std::vector<Real>{0.95, 0.80, 0.60};

        FloatCase base;
        base.name = "pflt_" + tag + "_base";
        base.type = type;
        runFloat(base);

        // --- lookbackPeriodEnd, one at a time --------------------------------
        // kExpiry flips `fullLookbackPeriod` and selects the engine's OTHER
        // branch (the 3-term "simpler calculation").
        int k = 0;
        for (const Date& d : {kT1, kT3, kExpiry}) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_lbend" + std::to_string(k++);
            c.lbEnd = d;
            runFloat(c);
        }

        // --- lambda, one at a time -------------------------------------------
        k = 0;
        for (const Real lam : lambdas) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_lambda" + std::to_string(k++);
            c.lambda = lam;
            runFloat(c);
        }

        // --- lambda AND lookbackPeriodEnd together, on both branches ---------
        k = 0;
        for (const Real lam : lambdas) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_lambda_full" + std::to_string(k++);
            c.lambda = lam;
            c.lbEnd = kExpiry;  // fullLookbackPeriod branch
            runFloat(c);
        }

        // --- the degenerate corner: lambda == 1 and the lookback running to
        //     expiry. This is exactly the plain (non-partial) floating
        //     lookback, i.e. the value a port that accepted lambda and
        //     lookbackPeriodEnd and then dropped them would return for EVERY
        //     case above. Pinning it makes such a port pass here and fail
        //     everywhere else, which is an unambiguous diagnosis.
        {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_degenerate";
            c.lambda = 1.0;
            c.lbEnd = kExpiry;
            runFloat(c);
        }

        // --- running extremum ------------------------------------------------
        k = 0;
        for (const Real mm : {85.0, 115.0}) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_minmax" + std::to_string(k++);
            c.minmax = mm;
            runFloat(c);
        }

        // --- spot ------------------------------------------------------------
        k = 0;
        for (const Real s : {90.0, 110.0}) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_spot" + std::to_string(k++);
            c.spot = s;
            runFloat(c);
        }

        // --- volatility ------------------------------------------------------
        k = 0;
        for (const Volatility v : {0.15, 0.40}) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_vol" + std::to_string(k++);
            c.vol = v;
            runFloat(c);
        }

        // --- risk-free rate. carry = r - q must stay away from 0: the engine
        //     divides by x = 2*carry/vol^2 and by carry itself, so r == q
        //     (base q = 0.02) is a genuine singularity and is avoided.
        k = 0;
        for (const Rate r : {0.035, 0.09}) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_r" + std::to_string(k++);
            c.r = r;
            runFloat(c);
        }

        // --- dividend yield --------------------------------------------------
        k = 0;
        for (const Rate q : {0.00, 0.07}) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_q" + std::to_string(k++);
            c.q = q;
            runFloat(c);
        }

        // --- negative carry (q > r) flips the sign of x and of the pow_s term.
        {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_negative_carry";
            c.r = 0.02;
            c.q = 0.07;
            runFloat(c);
        }

        // --- skew: minmax away from spot, so the vol the engine picks up is
        //     the one AT minmax and not the one at spot.
        k = 0;
        for (const Real mm : {90.0, 110.0}) {
            FloatCase c = base;
            c.name = "pflt_" + tag + "_smile" + std::to_string(k++);
            c.smile = true;
            c.minmax = mm;
            runFloat(c);
        }

        // --- everything moved at once, on each branch ------------------------
        {
            FloatCase c;
            c.name = "pflt_" + tag + "_combined_partial";
            c.type = type;
            c.spot = 108.0;
            c.r = 0.037;
            c.q = 0.011;
            c.vol = 0.31;
            c.minmax = 93.0;
            c.lambda = lambdas.back();
            c.lbEnd = kT1;
            runFloat(c);

            c.name = "pflt_" + tag + "_combined_full";
            c.lbEnd = kExpiry;
            runFloat(c);
        }
    }

    // =======================================================================
    // Partial-time FIXED lookback
    // =======================================================================
    for (const Option::Type type : {Option::Call, Option::Put}) {
        const std::string tag = type == Option::Call ? "call" : "put";

        FixedCase base;
        base.name = "pfix_" + tag + "_base";
        base.type = type;
        runFixed(base);

        // --- lookbackPeriodStart, one at a time. kExpiry makes
        //     `differentStartOfLookback` false, which zeroes e1/e2 and
        //     collapses cnbn2/cnbn3 to rho == 0.
        int k = 0;
        for (const Date& d : {kT1, kT3, kExpiry}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_lbstart" + std::to_string(k++);
            c.lbStart = d;
            runFixed(c);
        }

        // --- strike ----------------------------------------------------------
        k = 0;
        for (const Real x : {85.0, 115.0}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_strike" + std::to_string(k++);
            c.strike = x;
            runFixed(c);
        }

        // --- strike AND lookbackPeriodStart together -------------------------
        k = 0;
        for (const Real x : {85.0, 115.0}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_strike_same_start" + std::to_string(k++);
            c.strike = x;
            c.lbStart = kExpiry;
            runFixed(c);
        }

        // --- spot ------------------------------------------------------------
        k = 0;
        for (const Real s : {90.0, 110.0}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_spot" + std::to_string(k++);
            c.spot = s;
            runFixed(c);
        }

        // --- volatility ------------------------------------------------------
        k = 0;
        for (const Volatility v : {0.15, 0.40}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_vol" + std::to_string(k++);
            c.vol = v;
            runFixed(c);
        }

        // --- risk-free rate. As above, r == q (base q = 0.02) makes carry
        //     zero and the engine divides by it, so 0.02 is avoided.
        k = 0;
        for (const Rate r : {0.035, 0.09}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_r" + std::to_string(k++);
            c.r = r;
            runFixed(c);
        }

        // --- dividend yield --------------------------------------------------
        k = 0;
        for (const Rate q : {0.00, 0.07}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_q" + std::to_string(k++);
            c.q = q;
            runFixed(c);
        }

        // --- negative carry --------------------------------------------------
        {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_negative_carry";
            c.r = 0.02;
            c.q = 0.07;
            runFixed(c);
        }

        // --- skew: strike away from spot, so the vol the engine picks up is
        //     the one AT the strike and not the one at spot.
        k = 0;
        for (const Real x : {90.0, 110.0}) {
            FixedCase c = base;
            c.name = "pfix_" + tag + "_smile" + std::to_string(k++);
            c.smile = true;
            c.strike = x;
            runFixed(c);
        }

        // --- everything moved at once, on each branch ------------------------
        {
            FixedCase c;
            c.name = "pfix_" + tag + "_combined_partial";
            c.type = type;
            c.spot = 108.0;
            c.r = 0.037;
            c.q = 0.011;
            c.vol = 0.31;
            c.strike = 93.0;
            c.lbStart = kT1;
            runFixed(c);

            c.name = "pfix_" + tag + "_combined_same_start";
            c.lbStart = kExpiry;
            runFixed(c);
        }
    }

    // Neither lookback engine fills any greek: calculate() assigns only
    // results_.value, so every Greeks / MoreGreeks field stays Null<Real>()
    // and the accessor throws. Pinned so a port does not invent them.
    {
        const auto proc = process(100.0, 0.05, 0.02, 0.25, false);
        const auto exercise = ext::make_shared<EuropeanExercise>(kExpiry);

        ContinuousPartialFloatingLookbackOption flt(
            100.0, 1.0, kT2, ext::make_shared<FloatingTypePayoff>(Option::Call),
            exercise);
        flt.setPricingEngine(
            ext::make_shared<AnalyticContinuousPartialFloatingLookbackEngine>(proc));
        flt.NPV();

        ContinuousPartialFixedLookbackOption fix(
            kT2, ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0), exercise);
        fix.setPricingEngine(
            ext::make_shared<AnalyticContinuousPartialFixedLookbackEngine>(proc));
        fix.NPV();

        const char* names[] = {"delta", "gamma", "theta", "vega", "rho", "dividendRho"};
        for (int g = 0; g < 6; ++g) {
            addRaises("pflt_greeks_unavailable_" + std::string(names[g]),
                      Obj().s("greek", names[g]), [&] {
                          switch (g) {
                              case 0: flt.delta(); break;
                              case 1: flt.gamma(); break;
                              case 2: flt.theta(); break;
                              case 3: flt.vega(); break;
                              case 4: flt.rho(); break;
                              default: flt.dividendRho(); break;
                          }
                      });
            addRaises("pfix_greeks_unavailable_" + std::string(names[g]),
                      Obj().s("greek", names[g]), [&] {
                          switch (g) {
                              case 0: fix.delta(); break;
                              case 1: fix.gamma(); break;
                              case 2: fix.theta(); break;
                              case 3: fix.vega(); break;
                              case 4: fix.rho(); break;
                              default: fix.dividendRho(); break;
                          }
                      });
        }
    }

    // =======================================================================
    // Variance swap
    // =======================================================================
    {
        VarCase base;
        base.name = "vs_a_long_base";
        base.emitWeights = true;
        runVar(base);

        // position: pure sign flip on NPV, no effect on variance().
        {
            VarCase c = base;
            c.name = "vs_a_short_base";
            c.position = Position::Short;
            runVar(c);
        }

        // notional: pure scale on NPV, no effect on variance().
        int k = 0;
        for (const Real n : {1.0, 1.0e6}) {
            VarCase c = base;
            c.name = "vs_a_notional" + std::to_string(k++);
            c.notional = n;
            c.emitWeights = false;
            runVar(c);
        }

        // variance strike: shifts NPV by -notional*df*dK, no effect on variance().
        k = 0;
        for (const Real x : {0.01, 0.09}) {
            VarCase c = base;
            c.name = "vs_a_strike" + std::to_string(k++);
            c.strike = x;
            c.emitWeights = false;
            runVar(c);
        }
        // Short at a strike above the fair variance -- the NPV sign flips
        // relative to vs_a_short_base, so a port that dropped the position
        // cannot compensate with a global sign.
        {
            VarCase c = base;
            c.name = "vs_a_short_strike_high";
            c.position = Position::Short;
            c.strike = 0.09;
            c.emitWeights = false;
            runVar(c);
        }

        // dk: the end-strike step of the piecewise log-payoff approximation.
        // Default is 5.0; moving it changes the LAST slope of each ladder and
        // therefore the whole weight vector.
        k = 0;
        for (const Real dk : {2.5, 10.0}) {
            VarCase c = base;
            c.name = "vs_a_dk" + std::to_string(k++);
            c.dk = dk;
            c.emitWeights = true;
            runVar(c);
        }

        // Shorter ladders. min(call) == max(put) is a constructor invariant.
        {
            VarCase c = base;
            c.name = "vs_a_ladder_short";
            c.calls = {100.0, 105.0, 110.0};
            c.puts = {90.0, 95.0, 100.0};
            c.emitWeights = true;
            runVar(c);
        }
        // Single-point ladders: each ladder is {k, k+dk} / {k, k-dk} after the
        // end-strike is appended, so exactly one option per side survives.
        {
            VarCase c = base;
            c.name = "vs_a_ladder_single";
            c.calls = {100.0};
            c.puts = {100.0};
            c.emitWeights = true;
            runVar(c);
        }
        // Unsorted input: std::sort inside computeOptionWeights must make this
        // identical to vs_a_ladder_short.
        {
            VarCase c = base;
            c.name = "vs_a_ladder_unsorted";
            c.calls = {110.0, 100.0, 105.0};
            c.puts = {95.0, 100.0, 90.0};
            c.emitWeights = true;
            runVar(c);
        }
        // Duplicates: std::unique after the sort must collapse them, again
        // reproducing vs_a_ladder_short.
        {
            VarCase c = base;
            c.name = "vs_a_ladder_duplicates";
            c.calls = {100.0, 105.0, 105.0, 110.0, 100.0};
            c.puts = {100.0, 95.0, 90.0, 95.0, 90.0};
            c.emitWeights = true;
            runVar(c);
        }

        // startDate reaches the arguments and the inspector but NOT the price:
        // the engine only reads maturityDate. Same NPV as vs_a_long_base.
        {
            VarCase c = base;
            c.name = "vs_a_startdate_shifted";
            c.start = kToday - 30;
            c.emitWeights = false;
            runVar(c);
        }

        // maturityDate drives residualTime, the option exercise and the
        // discount factor, so it moves everything.
        {
            VarCase c = base;
            c.name = "vs_a_maturity_earlier";
            c.maturity = kToday + 60;
            c.emitWeights = false;
            runVar(c);
        }

        // --- market B: flat vol, different spot / rates / maturity ----------
        VarCase b;
        b.name = "vs_b_long_base";
        b.market = "B";
        b.maturity = kVsMatB;
        b.strike = 0.05;
        b.notional = 250000.0;
        b.calls = {120.0, 125.0, 130.0};
        b.puts = {110.0, 115.0, 120.0};
        b.emitWeights = true;
        runVar(b);
        {
            VarCase c = b;
            c.name = "vs_b_short_base";
            c.position = Position::Short;
            c.emitWeights = false;
            runVar(c);
        }
        {
            VarCase c = b;
            c.name = "vs_b_dk_small";
            c.dk = 1.0;
            c.emitWeights = true;
            runVar(c);
        }
        {
            VarCase c = b;
            c.name = "vs_b_notional_unit";
            c.notional = 1.0;
            c.emitWeights = false;
            runVar(c);
        }
    }

    // isExpired: detail::simple_event(maturityDate_).hasOccurred(), driven by
    // Settings::evaluationDate and Settings::includeReferenceDateEvents. The
    // flag is only observable ON the maturity date itself, so all six
    // combinations are pinned.
    {
        const bool savedFlag = Settings::instance().includeReferenceDateEvents();
        for (const bool include : {false, true}) {
            Settings::instance().includeReferenceDateEvents() = include;
            int k = 0;
            for (const Date& ref : {kVsMatA - 1, kVsMatA, kVsMatA + 1}) {
                Settings::instance().evaluationDate() = ref;
                VarianceSwap swap(Position::Long, 0.04, 50000.0, kToday, kVsMatA);
                addCase("vs_expired_" + std::string(include ? "incl" : "excl") + "_" +
                            std::to_string(k++),
                        Obj().b("include_reference_date_events", include)
                            .i("evaluation_date_serial", ref.serialNumber())
                            .i("maturity_serial", kVsMatA.serialNumber()),
                        Obj().b("is_expired", swap.isExpired()));
            }
        }
        Settings::instance().includeReferenceDateEvents() = savedFlag;
        Settings::instance().evaluationDate() = kToday;
    }

    // =======================================================================
    // QL_REQUIRE branches
    // =======================================================================
    {
        const auto proc = varProcess("A");
        const auto exercise = ext::make_shared<EuropeanExercise>(kExpiry);

        // --- VarianceSwap::arguments::validate ------------------------------
        auto priceSwap = [&](Real strike, Real notional, const Date& start,
                             const Date& maturity) {
            VarianceSwap swap(Position::Long, strike, notional, start, maturity);
            swap.setPricingEngine(ext::make_shared<ReplicatingVarianceSwapEngine>(
                proc, 5.0, kDdkzCalls, kDdkzPuts));
            swap.NPV();
        };
        addRaises("vs_validate_null_strike", Obj().s("field", "strike"),
                  [&] { priceSwap(Null<Real>(), 50000.0, kToday, kVsMatA); });
        addRaises("vs_validate_zero_strike", Obj().n("strike", 0.0),
                  [&] { priceSwap(0.0, 50000.0, kToday, kVsMatA); });
        addRaises("vs_validate_negative_strike", Obj().n("strike", -0.04),
                  [&] { priceSwap(-0.04, 50000.0, kToday, kVsMatA); });
        addRaises("vs_validate_null_notional", Obj().s("field", "notional"),
                  [&] { priceSwap(0.04, Null<Real>(), kToday, kVsMatA); });
        addRaises("vs_validate_zero_notional", Obj().n("notional", 0.0),
                  [&] { priceSwap(0.04, 0.0, kToday, kVsMatA); });
        addRaises("vs_validate_negative_notional", Obj().n("notional", -50000.0),
                  [&] { priceSwap(0.04, -50000.0, kToday, kVsMatA); });
        addRaises("vs_validate_null_start_date", Obj().s("field", "startDate"),
                  [&] { priceSwap(0.04, 50000.0, Date(), kVsMatA); });
        // A NULL maturity date never reaches validate(): Instrument::calculate()
        // asks isExpired() first, that is simple_event(Date()).hasOccurred(),
        // and the null date's serial number 0 is <= any evaluation date, so the
        // swap reports itself expired, setupExpired() runs and NPV() returns 0
        // WITHOUT throwing. The QL_REQUIRE on maturityDate is therefore only
        // reachable by calling validate() directly -- both facts are pinned.
        addRaises("vs_null_maturity_swallowed_by_is_expired",
                  Obj().s("field", "maturityDate"),
                  [&] { priceSwap(0.04, 50000.0, kToday, Date()); });
        {
            VarianceSwap swap(Position::Long, 0.04, 50000.0, kToday, Date());
            addCase("vs_null_maturity_is_expired", Obj().s("field", "maturityDate"),
                    Obj().b("is_expired", swap.isExpired()).n("npv", swap.NPV()));
        }
        // The validate() branches themselves, reached directly.
        auto validateArgs = [](Real strike, Real notional, const Date& start,
                               const Date& maturity) {
            VarianceSwap::arguments args;
            args.position = Position::Long;
            args.strike = strike;
            args.notional = notional;
            args.startDate = start;
            args.maturityDate = maturity;
            args.validate();
        };
        addRaises("vs_args_validate_null_maturity_date",
                  Obj().s("field", "maturityDate"),
                  [&] { validateArgs(0.04, 50000.0, kToday, Date()); });
        addRaises("vs_args_validate_null_start_date", Obj().s("field", "startDate"),
                  [&] { validateArgs(0.04, 50000.0, Date(), kVsMatA); });
        addRaises("vs_args_validate_ok", Obj().s("field", "none"),
                  [&] { validateArgs(0.04, 50000.0, kToday, kVsMatA); });

        // --- ReplicatingVarianceSwapEngine constructor ----------------------
        addRaises("vs_engine_empty_call_strikes", Obj().s("field", "callStrikes"), [&] {
            ReplicatingVarianceSwapEngine(proc, 5.0, std::vector<Real>(), kDdkzPuts);
        });
        addRaises("vs_engine_empty_put_strikes", Obj().s("field", "putStrikes"), [&] {
            ReplicatingVarianceSwapEngine(proc, 5.0, kDdkzCalls, std::vector<Real>());
        });
        addRaises("vs_engine_nonpositive_put_strike",
                  Obj().nums("put_strikes", {0.0, 100.0}), [&] {
                      ReplicatingVarianceSwapEngine(proc, 5.0, kDdkzCalls,
                                                    std::vector<Real>{0.0, 100.0});
                  });
        addRaises("vs_engine_negative_put_strike",
                  Obj().nums("put_strikes", {-5.0, 100.0}), [&] {
                      ReplicatingVarianceSwapEngine(proc, 5.0, kDdkzCalls,
                                                    std::vector<Real>{-5.0, 100.0});
                  });
        addRaises("vs_engine_min_call_max_put_differ",
                  Obj().nums("call_strikes", {105.0, 110.0})
                      .nums("put_strikes", {90.0, 100.0}),
                  [&] {
                      ReplicatingVarianceSwapEngine(proc, 5.0,
                                                    std::vector<Real>{105.0, 110.0},
                                                    std::vector<Real>{90.0, 100.0});
                  });
        // Ordering inside a ladder is NOT an error -- the engine sorts. Pinned
        // as raises=false so a port does not add a constraint C++ lacks.
        addRaises("vs_engine_unsorted_ladders_ok",
                  Obj().nums("call_strikes", {110.0, 100.0})
                      .nums("put_strikes", {90.0, 100.0}),
                  [&] {
                      ReplicatingVarianceSwapEngine(proc, 5.0,
                                                    std::vector<Real>{110.0, 100.0},
                                                    std::vector<Real>{90.0, 100.0});
                  });

        // --- ContinuousPartialFloatingLookbackOption::arguments::validate ---
        auto priceFloat = [&](Real minmax, Real lambda, const Date& lbEnd,
                              Option::Type type) {
            const auto p = process(100.0, 0.05, 0.02, 0.25, false);
            ContinuousPartialFloatingLookbackOption option(
                minmax, lambda, lbEnd, ext::make_shared<FloatingTypePayoff>(type),
                exercise);
            option.setPricingEngine(
                ext::make_shared<AnalyticContinuousPartialFloatingLookbackEngine>(p));
            option.NPV();
        };
        addRaises("pflt_validate_lambda_below_one_call", Obj().n("lambda", 0.9),
                  [&] { priceFloat(100.0, 0.9, kT2, Option::Call); });
        addRaises("pflt_validate_lambda_above_one_put", Obj().n("lambda", 1.1),
                  [&] { priceFloat(100.0, 1.1, kT2, Option::Put); });
        addRaises("pflt_validate_lookback_end_after_exercise",
                  Obj().i("lookback_end_serial", (kExpiry + 1).serialNumber()),
                  [&] { priceFloat(100.0, 1.0, kExpiry + 1, Option::Call); });
        addRaises("pflt_validate_negative_minmax", Obj().n("minmax", -1.0),
                  [&] { priceFloat(-1.0, 1.0, kT2, Option::Call); });
        // lambda == 1 is admissible for BOTH types (the constraints are
        // non-strict). Pinned as raises=false.
        addRaises("pflt_validate_lambda_one_call_ok", Obj().n("lambda", 1.0),
                  [&] { priceFloat(100.0, 1.0, kT2, Option::Call); });
        addRaises("pflt_validate_lambda_one_put_ok", Obj().n("lambda", 1.0),
                  [&] { priceFloat(100.0, 1.0, kT2, Option::Put); });
        // lookbackPeriodEnd exactly ON the exercise date is admissible.
        addRaises("pflt_validate_lookback_end_on_exercise_ok",
                  Obj().i("lookback_end_serial", kExpiry.serialNumber()),
                  [&] { priceFloat(100.0, 1.0, kExpiry, Option::Call); });

        // --- ContinuousPartialFixedLookbackOption::arguments::validate ------
        auto priceFixed = [&](Real strike, const Date& lbStart, Option::Type type,
                              Real spot) {
            const auto p = process(spot, 0.05, 0.02, 0.25, false);
            ContinuousPartialFixedLookbackOption option(
                lbStart, ext::make_shared<PlainVanillaPayoff>(type, strike), exercise);
            option.setPricingEngine(
                ext::make_shared<AnalyticContinuousPartialFixedLookbackEngine>(p));
            option.NPV();
        };
        addRaises("pfix_validate_lookback_start_after_exercise",
                  Obj().i("lookback_start_serial", (kExpiry + 1).serialNumber()),
                  [&] { priceFixed(100.0, kExpiry + 1, Option::Call, 100.0); });
        addRaises("pfix_validate_lookback_start_on_exercise_ok",
                  Obj().i("lookback_start_serial", kExpiry.serialNumber()),
                  [&] { priceFixed(100.0, kExpiry, Option::Call, 100.0); });
        // Engine guards: strike >= 0 for calls, strike > 0 for puts,
        // underlying > 0. A zero strike is therefore legal for a CALL and
        // illegal for a PUT -- an asymmetry easy to lose in a port.
        addRaises("pfix_engine_negative_strike_call", Obj().n("strike", -10.0),
                  [&] { priceFixed(-10.0, kT2, Option::Call, 100.0); });
        addRaises("pfix_engine_zero_strike_put", Obj().n("strike", 0.0),
                  [&] { priceFixed(0.0, kT2, Option::Put, 100.0); });
        addRaises("pfix_engine_negative_strike_put", Obj().n("strike", -10.0),
                  [&] { priceFixed(-10.0, kT2, Option::Put, 100.0); });
        addRaises("pfix_engine_nonpositive_underlying", Obj().n("spot", 0.0),
                  [&] { priceFixed(100.0, kT2, Option::Call, 0.0); });
        addRaises("pflt_engine_nonpositive_underlying", Obj().n("spot", 0.0), [&] {
            const auto p = process(0.0, 0.05, 0.02, 0.25, false);
            ContinuousPartialFloatingLookbackOption option(
                100.0, 1.0, kT2, ext::make_shared<FloatingTypePayoff>(Option::Call),
                exercise);
            option.setPricingEngine(
                ext::make_shared<AnalyticContinuousPartialFloatingLookbackEngine>(p));
            option.NPV();
        });
        // The fixed engine requires a PLAIN vanilla payoff, not merely a
        // striked one: a CashOrNothingPayoff is a StrikedTypePayoff and is
        // still rejected.
        addRaises("pfix_engine_non_plain_payoff", Obj().s("payoff", "CashOrNothing"),
                  [&] {
                      const auto p = process(100.0, 0.05, 0.02, 0.25, false);
                      ContinuousPartialFixedLookbackOption option(
                          kT2,
                          ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0,
                                                                10.0),
                          exercise);
                      option.setPricingEngine(
                          ext::make_shared<AnalyticContinuousPartialFixedLookbackEngine>(
                              p));
                      option.NPV();
                  });
    }

    // =======================================================================
    // BivariateCumulativeNormalDistributionWe04DP at |rho| == 1
    // =======================================================================
    // Both lookback engines construct the degenerate copula: the floating one
    // at rho == +1 whenever the lookback window runs to expiry, the fixed one
    // at rho == -1 whenever the window starts at expiry. C++ admits |rho| == 1
    // in the constructor and then skips its whole Genz series block
    // (bivariatenormaldistribution.cpp:212 `if (fabs(correlation_) < 1)`),
    // leaving only the closing correction. These cases pin what that
    // correction actually evaluates to, so the port's closed-form endpoints
    // are cross-validated directly rather than by inference from the engine
    // NPVs. The (x, y) grid straddles every branch of the rho == -1 tail
    // selection: x + y below / at / above zero, and x below / above zero.
    {
        const Real xs[] = {-2.0, -0.75, -0.3, 0.0, 0.3, 0.75, 2.0};
        const Real ys[] = {-2.0, -0.75, 0.3, 0.0, 0.75, 2.0};
        for (const Real rho : {1.0, -1.0}) {
            const BivariateCumulativeNormalDistributionWe04DP bvn(rho);
            int idx = 0;
            for (const Real x : xs) {
                for (const Real y : ys) {
                    addCase("bvn_rho" + std::string(rho > 0 ? "p1_" : "m1_") +
                                std::to_string(idx++),
                            Obj().n("rho", rho).n("x", x).n("y", y),
                            Obj().n("value", bvn(x, y)));
                }
            }
        }
    }

    emitDocument();
    return 0;
}
