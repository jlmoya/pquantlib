// migration-harness/cpp/probes/v143_pe_american/probe.cpp
//
// Reference values for the "American / digital-American / jump-diffusion"
// cluster of C++ QuantLib v1.43 pricing engines:
//
//   * AmericanPayoffAtHit          (ql/pricingengines/americanpayoffathit.{hpp,cpp})
//   * AmericanPayoffAtExpiry       (ql/pricingengines/americanpayoffatexpiry.{hpp,cpp})
//   * AnalyticDigitalAmericanEngine    ) ql/pricingengines/vanilla/
//   * AnalyticDigitalAmericanKOEngine  )   analyticdigitalamericanengine.{hpp,cpp}
//   * BaroneAdesiWhaleyApproximationEngine (vanilla/baroneadesiwhaleyengine.{hpp,cpp})
//   * BjerksundStenslandApproximationEngine(vanilla/bjerksundstenslandengine.{hpp,cpp})
//   * JuQuadraticApproximationEngine       (vanilla/juquadraticengine.{hpp,cpp})
//   * IntegralEngine                       (vanilla/integralengine.{hpp,cpp})
//   * JumpDiffusionEngine + Merton76Process(vanilla/jumpdiffusionengine.{hpp,cpp},
//                                           ql/processes/merton76process.{hpp,cpp})
//   * DiscretizedVanillaOption             (vanilla/discretizedvanillaoption.{hpp,cpp})
//
// Emits JSON on stdout and nothing else; the caller redirects it to
// references/v143/pe/american.json.
//
// ===========================================================================
// What has to be pinned, and why
// ===========================================================================
//
// Every one of these engines is a branch thicket. The value of a case here is
// not "it prices a call" — it is "it takes THIS branch and no other". The
// branches, engine by engine:
//
// --- AmericanPayoffAtHit --------------------------------------------------
// value() = K * (forward*alpha + X*beta), with the four state variables set
// by a switch on option type crossed with the spot/strike ordering:
//   Call, strike >  spot : alpha = N(-d1), beta = N(-d2)  (out-of-the-money)
//   Call, strike <= spot : alpha = beta = 0.5             (barrier already hit)
//   Put,  strike <  spot : alpha = N( d1), beta = N( d2)
//   Put,  strike >= spot : alpha = beta = 0.5             (barrier already hit)
// and inTheMoney_ = (Call && strike<spot) || (Put && strike>spot) selecting
// forward_ = X_ = 1 versus the two (strike/spot)^(mu±lambda) powers.
// NOTE the asymmetry that a port will get wrong if it "tidies" the code: the
// 0.5 branch uses `>=` / `<=` (so strike == spot takes it) while inTheMoney_
// uses strict `<` / `>` (so strike == spot does NOT take it). At strike ==
// spot exactly the two disagree, and the payoff still evaluates the powers —
// harmlessly, since (1)^anything == 1, but only by luck. Cases
// `hit_*_atm_exact` pin that corner.
//   The variance < QL_EPSILON path ("not tested yet" in the C++ source) is
// reached by cases `hit_*_zero_variance`; they use r == q == 0 so that
// mu_ = log(1)/v - 0.5 = -0.5 and lambda_ = sqrt(0.25 - 0) = 0.5 stay finite
// (with r != 0 that branch divides log(discount) by ~1e-17 and produces
// +-inf / NaN — which is a genuine C++ behaviour but not a useful pin).
//   The `discount_ == 0.0 && dividendDiscount_ == 0.0` sub-branch inside the
// variance >= QL_EPSILON path is DEAD CODE: QL_REQUIRE(discount_ > 0.0) has
// already fired. A port must keep it (or drop it) without changing behaviour.
// The four QL_REQUIREs are pinned as `throws` cases.
//   delta()/gamma() divide by (-spot*stdDev); rho(T) requires T >= 0 and is
// really dV/dr divided by T. All three are pinned for every state -- EXCEPT
// gamma() on the variance < QL_EPSILON path, which reads D1_ and D2_. Those
// two members are only assigned in the variance >= QL_EPSILON arm and have no
// default member initialiser, so on the small-variance path gamma() reads an
// indeterminate value (this build returns NaN). Cases on that path carry
// "gamma_undefined": true and the Python test skips gamma for them.
//
// TWO GENUINE C++ v1.43 DEFECTS ARE DOCUMENTED IN THIS PROBE, both of the same
// shape (a data member with no initialiser being read):
//   (a) AmericanPayoffAtHit::gamma() on the small-variance path, above;
//   (b) BjerksundStenslandApproximationEngine leaving deltaForward /
//       elasticity / itmCashProbability indeterminate -- see readGreeks().
// Neither is cross-validated; both are pinned only as "do not trust this".
//
// --- AmericanPayoffAtExpiry -----------------------------------------------
// Same shape, but with a knock_in flag, and only value() is exposed. The
// (eta, phi) sign pair is chosen by type x knock_in, then a SECOND switch
// overrides cum_d1_/cum_d2_ when the barrier is already breached
// (strike <= spot for a Call, strike >= spot for a Put): 0.5/0.5 for the
// knock-in, 0.0/0.0 for the knock-out. Finally Y_ is negated for the
// knock-out. All eight (type x knock_in x breached) combinations are pinned,
// for both cash-or-nothing and asset-or-nothing payoffs. Asset-or-nothing
// additionally sets K_ = forward_ and mu_ += 1.0 — pinned separately because
// a port that forgets the mu_ shift still gets the right sign and roughly the
// right magnitude.
//   The `if (cum_d2_ == 0.0) Y_ = 0.0;` guard ("check needed on some extreme
// cases") fires for the knock-out breached cases and for the zero-variance
// cases; without it std::pow would be evaluated needlessly. Pinned.
//
// --- AnalyticDigitalAmericanEngine / ...KOEngine --------------------------
// The engine dispatches on `ex->payoffAtExpiry()`:
//   true  -> AmericanPayoffAtExpiry(..., knock_in()), and it sets ONLY
//            results_.value. delta/gamma/rho/vega/theta/dividendRho all stay
//            Null<Real>() and the instrument's accessors throw.
//   false -> AmericanPayoffAtHit, which sets value, delta, gamma and rho —
//            and NOTHING ELSE (no vega, no theta, no dividendRho).
// TRAP, pinned explicitly by the `ko_at_hit_equals_ki_at_hit` cases:
// AmericanPayoffAtHit takes no knock_in argument, so with payoffAtExpiry()
// == false the KO engine returns EXACTLY the knock-in price. The `knock_in()`
// virtual is consulted only on the at-expiry path. A port that threads
// knock_in through both paths will differ from C++.
// Guards: non-American exercise -> "non-American exercise given";
// ex->dates()[0] > volatility reference date -> "American option with window
// exercise not handled yet"; spot <= 0 -> "negative or null underlying given".
//
// --- BaroneAdesiWhaleyApproximationEngine ---------------------------------
// criticalPrice() is a PUBLIC STATIC and is pinned directly (cases
// `baw_critical_*`), not only through NPV: it is reused verbatim by
// JuQuadraticApproximationEngine, so a port that hides it inside the engine
// cannot implement Ju. Its `tolerance` argument defaults to 1e-6 and is a
// *relative-to-strike* stopping criterion on |LHS - RHS|; cases vary it so a
// port cannot hardcode the default. QL_REQUIRE(riskFreeDiscount <= 1.0)
// rejects negative rates — pinned as a throw.
//   calculate() has exactly two arms:
//     dividendDiscount >= 1.0 && type == Call  -> early exercise never
//        optimal: the FULL European greek set is filled (value, delta,
//        deltaForward, elasticity, gamma, rho, dividendRho, vega, theta,
//        thetaPerDay, strikeSensitivity, itmCashProbability).
//        Note `>= 1.0`, so q == 0 exactly takes this arm.
//     otherwise -> the quadratic approximation, which fills ONLY
//        results_.value. Every greek stays Null. Pinned.
//   Inside the approximation, `spot < Sk` (Call) / `spot > Sk` (Put) chooses
//   between the corrected European price and the intrinsic value; both sides
//   are pinned for both types, plus 1-day and 10-year maturities and a
//   zero-volatility case (variance == 0 divides by zero inside criticalPrice;
//   whatever C++ produces — including non-finite — is pinned as-is).
//
// --- JuQuadraticApproximationEngine ---------------------------------------
// Same European early-out as BAW (identical condition, identical full greek
// set). The approximation arm fills value, delta and gamma but NOT rho, vega,
// theta, dividendRho — a strictly different "which greeks exist" signature
// from BAW, and pinned as such.
//   The hA correction term hA = phi*(Sk - K) - blackFormula(...)*rfD and the
// chi = ln(S/Sk)*(b*ln(S/Sk) + c) correction are what distinguish Ju from
// BAW; the `phi*(Sk - spot) > 0` test selects them versus the intrinsic
// value (delta = phi, gamma = 0). Both sides pinned for Call and Put.
//
// --- BjerksundStenslandApproximationEngine --------------------------------
// The richest branch set here.
//   calculate(): Puts are handled by the put-call symmetry transform —
//   swap(spot, strike), swap(riskFreeDiscount, dividendDiscount), replace the
//   payoff by a Call — and the results are un-transformed afterwards by
//   swapping delta<->strikeSensitivity, gamma<->additionalResults["strikeGamma"],
//   rho<->dividendRho, and then rescaling rho *= tr/tq, dividendRho *= tq/tr.
//   That rescale is a no-op only when the two day counters agree; the
//   `bs_put_*` cases pin the post-transform values so a port must do the
//   whole dance.
//   Guards / arms:
//     dividendDiscount > 1.0 && riskFreeDiscount > dividendDiscount
//         -> QL_FAIL "double-boundary case r<q<0 for a call given". Pinned.
//     dividendDiscount >= 1.0 && dividendDiscount >= riskFreeDiscount
//         -> europeanCallResults (exerciseType == "European").
//     else americanCallApproximation, which itself has three arms:
//         S >= I               -> immediateExercise  (exerciseType "Immediate")
//         q = ln(I/fwd)/sqrt(v) > 12.5 -> europeanResults ("run-away exercise
//                                 boundary"; reached by a very low vol +
//                                 short maturity deep-OTM call)
//         otherwise            -> the full approximation ("American"), with
//                                 analytic delta/gamma/rho/dividendRho/vega
//                                 built from the phi_* derivative helpers.
//     then `if (results.value < europeanResults.value) results = europeanResults;`
//     and finally, back in calculate(),
//         `if (results_.value < (spot-strike)*(1+10*QL_EPSILON))
//              results_ = immediateExercise(spot, strike);`
//   B0 = (bT == rT) ? X : max(X, rT/(rT-bT)*X): bT == rT means dividendDiscount
//   == 1.0 exactly, i.e. (after the put transform) the original r == 0. Case
//   `bs_put_zero_rate_negative_q` drives that equality exactly.
//   B0Dq / B0Dr each have a `(dD <= rfD) ? 0 : ...` branch; both sides appear.
//   additionalResults carries "strikeGamma" (Real) and "exerciseType"
//   (std::string) — both pinned for every case, because the Put
//   post-processing *swaps* strikeGamma with gamma and a port that treats
//   additionalResults as decoration will silently return the wrong gamma.
//   europeanCallResults / immediateExercise do NOT set deltaForward,
//   elasticity or itmCashProbability: pinned as absent.
//
// --- IntegralEngine -------------------------------------------------------
// European only (QL_REQUIRE, pinned as a throw for an American exercise).
// Fills ONLY results_.value. The quadrature is a fixed SegmentIntegral(5000)
// — a midpoint rule on 5000 equal segments — over
//   [drift - 10*sqrt(variance), drift + 10*sqrt(variance)],
// with drift = ln(dividendDiscount/riskFreeDiscount) - 0.5*variance, and the
// result scaled by riskFreeDiscount / sqrt(2*pi*variance). The integrand
// applies `arguments_.payoff` (NOT the StrikedTypePayoff it validated) to
// s0*exp(x), so cash-or-nothing and asset-or-nothing payoffs work too and are
// covered here. The number of segments is part of the answer: a port using an
// adaptive integrator will not reproduce these values, which is exactly why
// they are pinned tight.
//
// --- JumpDiffusionEngine + Merton76Process --------------------------------
// The engine sums Merton's Poisson series of Black-Scholes prices, relinking
// a RelinkableHandle rate/vol pair on every term. Two tuning knobs:
// relativeAccuracy (default 1e-4) and maxIterations (default 100); both are
// varied, and maxIterations = 1 is pinned as a throw (QL_ENSURE fires).
// The loop condition
//     (lastContribution > relativeAccuracy_ && i < maxIterations_)
//        || i < Size(lambda*t)
// has an OR in it: the series always runs at least floor(lambda*t) terms
// regardless of accuracy. A port using a pure while-not-converged loop will
// differ whenever lambda*t >= 1 — cases `jd_high_intensity_*` sit there.
// The theta correction (vega and rho terms plus the lambda*value telescoping
// with p(i-1)) is idiosyncratic and pinned.
// The engine fills value, delta, gamma, theta, vega, rho, dividendRho and
// NOTHING else (no thetaPerDay, no strikeSensitivity, no itmCashProbability).
// Merton76Process itself: x0() and time() delegate to the inner
// BlackScholesMertonProcess, while drift(), diffusion() and apply() all
// QL_FAIL — pinned as throws, because they are part of the class surface.
//
// --- DiscretizedVanillaOption ---------------------------------------------
// A DiscretizedAsset, not an engine: the probe drives it exactly the way
// BinomialVanillaEngine<T> does — CoxRossRubinstein tree, BlackScholesLattice,
// TimeGrid(maturity, steps), initialize(lattice, maturity), then the Hull
// three-step rollback (grid[2] -> 3 nodes, grid[1] -> 2 nodes, 0.0 ->
// presentValue). The node arrays at each stop are pinned, not just the final
// PV, so a port cannot get the right price with the wrong intermediate slice.
//   postAdjustValuesImpl() switches on the exercise type:
//     American  : `now <= stoppingTimes_[1] && now >= stoppingTimes_[0]`
//                 (a RANGE test, and it indexes [1] — so an American exercise
//                 must carry two dates)
//     European  : `isOnTime(stoppingTimes_[0])`
//     Bermudan  : isOnTime for each stopping time
//   All three are exercised. The ctor snaps every stopping time onto the grid
//   with `grid.closestTime(...)` when a grid is supplied — the Bermudan case
//   uses dates that do NOT land on grid points so the snapping is observable.
//   reset(size) zero-fills and then calls adjustValues(), which is why a
//   European option is worth its payoff at maturity rather than zero.
//
// ===========================================================================

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/any.hpp>
#include <ql/discretizedasset.hpp>
#include <ql/exercise.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/methods/lattices/bsmlattice.hpp>
#include <ql/pricingengines/americanpayoffatexpiry.hpp>
#include <ql/pricingengines/americanpayoffathit.hpp>
#include <ql/pricingengines/vanilla/analyticdigitalamericanengine.hpp>
#include <ql/pricingengines/vanilla/baroneadesiwhaleyengine.hpp>
#include <ql/pricingengines/vanilla/bjerksundstenslandengine.hpp>
#include <ql/pricingengines/vanilla/discretizedvanillaoption.hpp>
#include <ql/pricingengines/vanilla/integralengine.hpp>
#include <ql/pricingengines/vanilla/juquadraticengine.hpp>
#include <ql/pricingengines/vanilla/jumpdiffusionengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/merton76process.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// Non-finite doubles are emitted as JSON *strings* ("nan"/"inf"/"-inf")
// because JSON has no literal for them; the Python side maps them back.
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
    //! numeric-or-Null<Real>() slot: emits JSON null when the engine left it unset
    Obj& g(const std::string& k, Real v) {
        if (v == Null<Real>())
            return put(k, "null");
        return n(k, v);
    }
    Obj& arr(const std::string& k, const std::vector<Real>& v) {
        std::string body = "[";
        for (Size j = 0; j < v.size(); ++j) {
            if (j != 0)
                body += ", ";
            body += num(v[j]);
        }
        body += "]";
        return put(k, body);
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
    gCases.emplace_back(name,
                        "{\"inputs\": " + inputs.str() + ", \"expected\": " + expected.str() + "}");
}

void emitDocument() {
    std::cout << "{\n";
    for (Size i = 0; i < gCases.size(); ++i) {
        std::cout << "  \"" << gCases[i].first << "\": " << gCases[i].second;
        if (i + 1 != gCases.size())
            std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// Shared market conventions. Every term structure is anchored at TODAY with
// an explicit reference date, and Settings::evaluationDate is pinned to the
// same day so nothing here is wall-clock dependent.
// ---------------------------------------------------------------------------
const Date TODAY(1, March, 2025);
const DayCounter DC = Actual365Fixed();
const Calendar CAL = NullCalendar();

std::string typeName(Option::Type t) { return t == Option::Call ? "Call" : "Put"; }

struct Market {
    ext::shared_ptr<SimpleQuote> spot;
    ext::shared_ptr<GeneralizedBlackScholesProcess> process;
};

Market makeMarket(Real spot, Rate q, Rate r, Volatility vol) {
    auto s = ext::make_shared<SimpleQuote>(spot);
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(TODAY, q, DC));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(TODAY, r, DC));
    Handle<BlackVolTermStructure> vTS(ext::make_shared<BlackConstantVol>(TODAY, CAL, vol, DC));
    return {s, ext::make_shared<BlackScholesMertonProcess>(Handle<Quote>(s), qTS, rTS, vTS)};
}

//! Put the whole market description into `inputs` so the Python test
//! reconstructs the case instead of restating constants.
void describeMarket(Obj& in, Real spot, Rate q, Rate r, Volatility vol, int maturityDays) {
    in.n("spot", spot).n("q", q).n("r", r).n("vol", vol).i("maturity_days", maturityDays);
}

//! Read every greek the instrument exposes.
//!
//! A greek the engine never wrote is still Null<Real>() after
//! PricingEngine::reset(), so the instrument's accessor throws and we emit the
//! string "unset". That is the normal case.
//!
//! `structAssigned` marks the engines that build a LOCAL
//! `OneAssetOption::results` and assign the whole struct into `results_`
//! (BjerksundStenslandApproximationEngine does this in europeanCallResults(),
//! immediateExercise() and americanCallApproximation()). `Greeks` and
//! `MoreGreeks` have no default member initialisers, so the local object's
//! itmCashProbability / deltaForward / elasticity are INDETERMINATE and the
//! assignment copies that indeterminacy into results_ — the accessors then
//! return garbage instead of throwing. Observed on this build:
//! deltaForward == -1.03e+80, itmCashProbability == 3.01e-314. That is a
//! genuine C++ v1.43 defect (reading an uninitialised object), so those three
//! slots are emitted as "indeterminate" and must NOT be cross-validated.
void readGreeks(Obj& ex, VanillaOption& option, bool structAssigned = false) {
    ex.n("npv", option.NPV());
    auto slot = [&](const std::string& key, Real (VanillaOption::*f)() const) {
        try {
            ex.n(key, (option.*f)());
        } catch (const std::exception&) {
            ex.s(key, "unset");
        }
    };
    auto maybeSlot = [&](const std::string& key, Real (VanillaOption::*f)() const) {
        if (structAssigned)
            ex.s(key, "indeterminate");
        else
            slot(key, f);
    };
    slot("delta", &VanillaOption::delta);
    slot("gamma", &VanillaOption::gamma);
    slot("theta", &VanillaOption::theta);
    slot("theta_per_day", &VanillaOption::thetaPerDay);
    slot("vega", &VanillaOption::vega);
    slot("rho", &VanillaOption::rho);
    slot("dividend_rho", &VanillaOption::dividendRho);
    slot("strike_sensitivity", &VanillaOption::strikeSensitivity);
    maybeSlot("itm_cash_probability", &VanillaOption::itmCashProbability);
    maybeSlot("delta_forward", &VanillaOption::deltaForward);
    maybeSlot("elasticity", &VanillaOption::elasticity);
}

// ===========================================================================
// 1. AmericanPayoffAtHit — direct
// ===========================================================================

void emitAtHit(const std::string& name,
               Real spot,
               Rate q,
               Rate r,
               Volatility vol,
               Real T,
               Option::Type type,
               Real strike,
               bool cashOrNothing,
               Real cash) {
    const DiscountFactor discount = std::exp(-r * T);
    const DiscountFactor dividendDiscount = std::exp(-q * T);
    const Real variance = vol * vol * T;

    ext::shared_ptr<StrikedTypePayoff> payoff;
    if (cashOrNothing)
        payoff = ext::make_shared<CashOrNothingPayoff>(type, strike, cash);
    else
        payoff = ext::make_shared<AssetOrNothingPayoff>(type, strike);

    Obj in;
    in.n("spot", spot)
        .n("q", q)
        .n("r", r)
        .n("vol", vol)
        .n("T", T)
        .n("discount", discount)
        .n("dividend_discount", dividendDiscount)
        .n("variance", variance)
        .s("payoff", cashOrNothing ? "CashOrNothing" : "AssetOrNothing")
        .s("type", typeName(type))
        .n("strike", strike)
        .n("cash", cash);

    Obj ex;
    try {
        const AmericanPayoffAtHit pricer(spot, discount, dividendDiscount, variance, payoff);
        ex.b("throws", false)
            .n("value", pricer.value())
            .n("delta", pricer.delta())
            .n("gamma", pricer.gamma())
            .n("rho", pricer.rho(T))
            // gamma() reads D1_ / D2_, which the variance < QL_EPSILON arm of
            // the ctor never assigns -- an indeterminate read. Flagged so the
            // Python test does not cross-validate it.
            .b("gamma_undefined", variance < QL_EPSILON);
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

//! Same, but with (discount, dividendDiscount, variance) supplied directly.
//! Needed for the sub-QL_EPSILON variance branch: a variance that small is not
//! reachable from any representable (vol, T) pair with a representable rate.
void emitAtHitRaw(const std::string& name,
                  Real spot,
                  DiscountFactor discount,
                  DiscountFactor dividendDiscount,
                  Real variance,
                  Real T,
                  Option::Type type,
                  Real strike,
                  bool cashOrNothing,
                  Real cash) {
    ext::shared_ptr<StrikedTypePayoff> payoff;
    if (cashOrNothing)
        payoff = ext::make_shared<CashOrNothingPayoff>(type, strike, cash);
    else
        payoff = ext::make_shared<AssetOrNothingPayoff>(type, strike);

    Obj in;
    in.n("spot", spot)
        .n("discount", discount)
        .n("dividend_discount", dividendDiscount)
        .n("variance", variance)
        .n("T", T)
        .s("payoff", cashOrNothing ? "CashOrNothing" : "AssetOrNothing")
        .s("type", typeName(type))
        .n("strike", strike)
        .n("cash", cash);

    Obj ex;
    try {
        const AmericanPayoffAtHit pricer(spot, discount, dividendDiscount, variance, payoff);
        ex.b("throws", false)
            .n("value", pricer.value())
            .n("delta", pricer.delta())
            .n("gamma", pricer.gamma())
            .n("rho", pricer.rho(T))
            // gamma() reads D1_ / D2_, which the variance < QL_EPSILON arm of
            // the ctor never assigns -- an indeterminate read. Flagged so the
            // Python test does not cross-validate it.
            .b("gamma_undefined", variance < QL_EPSILON);
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// ===========================================================================
// 2. AmericanPayoffAtExpiry — direct
// ===========================================================================

void emitAtExpiry(const std::string& name,
                  Real spot,
                  Rate q,
                  Rate r,
                  Volatility vol,
                  Real T,
                  Option::Type type,
                  Real strike,
                  bool cashOrNothing,
                  Real cash,
                  bool knockIn) {
    const DiscountFactor discount = std::exp(-r * T);
    const DiscountFactor dividendDiscount = std::exp(-q * T);
    const Real variance = vol * vol * T;

    ext::shared_ptr<StrikedTypePayoff> payoff;
    if (cashOrNothing)
        payoff = ext::make_shared<CashOrNothingPayoff>(type, strike, cash);
    else
        payoff = ext::make_shared<AssetOrNothingPayoff>(type, strike);

    Obj in;
    in.n("spot", spot)
        .n("q", q)
        .n("r", r)
        .n("vol", vol)
        .n("T", T)
        .n("discount", discount)
        .n("dividend_discount", dividendDiscount)
        .n("variance", variance)
        .s("payoff", cashOrNothing ? "CashOrNothing" : "AssetOrNothing")
        .s("type", typeName(type))
        .n("strike", strike)
        .n("cash", cash)
        .b("knock_in", knockIn);

    Obj ex;
    try {
        const AmericanPayoffAtExpiry pricer(spot, discount, dividendDiscount, variance, payoff,
                                            knockIn);
        ex.b("throws", false).n("value", pricer.value());
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

//! Same, but with (discount, dividendDiscount, variance) supplied directly.
void emitAtExpiryRaw(const std::string& name,
                     Real spot,
                     DiscountFactor discount,
                     DiscountFactor dividendDiscount,
                     Real variance,
                     Option::Type type,
                     Real strike,
                     bool cashOrNothing,
                     Real cash,
                     bool knockIn) {
    ext::shared_ptr<StrikedTypePayoff> payoff;
    if (cashOrNothing)
        payoff = ext::make_shared<CashOrNothingPayoff>(type, strike, cash);
    else
        payoff = ext::make_shared<AssetOrNothingPayoff>(type, strike);

    Obj in;
    in.n("spot", spot)
        .n("discount", discount)
        .n("dividend_discount", dividendDiscount)
        .n("variance", variance)
        .s("payoff", cashOrNothing ? "CashOrNothing" : "AssetOrNothing")
        .s("type", typeName(type))
        .n("strike", strike)
        .n("cash", cash)
        .b("knock_in", knockIn);

    Obj ex;
    try {
        const AmericanPayoffAtExpiry pricer(spot, discount, dividendDiscount, variance, payoff,
                                            knockIn);
        ex.b("throws", false).n("value", pricer.value());
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// ===========================================================================
// 3. AnalyticDigitalAmericanEngine / AnalyticDigitalAmericanKOEngine
// ===========================================================================

void emitDigital(const std::string& name,
                 Real spot,
                 Rate q,
                 Rate r,
                 Volatility vol,
                 int maturityDays,
                 Option::Type type,
                 Real strike,
                 bool cashOrNothing,
                 Real cash,
                 bool payoffAtExpiry,
                 bool knockOutEngine,
                 int earliestOffsetDays = 0) {
    Obj in;
    describeMarket(in, spot, q, r, vol, maturityDays);
    in.s("payoff", cashOrNothing ? "CashOrNothing" : "AssetOrNothing")
        .s("type", typeName(type))
        .n("strike", strike)
        .n("cash", cash)
        .b("payoff_at_expiry", payoffAtExpiry)
        .s("engine", knockOutEngine ? "AnalyticDigitalAmericanKOEngine"
                                    : "AnalyticDigitalAmericanEngine")
        .i("earliest_offset_days", earliestOffsetDays);

    Obj ex;
    try {
        const Market m = makeMarket(spot, q, r, vol);
        ext::shared_ptr<StrikedTypePayoff> payoff;
        if (cashOrNothing)
            payoff = ext::make_shared<CashOrNothingPayoff>(type, strike, cash);
        else
            payoff = ext::make_shared<AssetOrNothingPayoff>(type, strike);

        const Date exDate = TODAY + maturityDays;
        const Date earliest = TODAY + earliestOffsetDays;
        auto exercise = ext::make_shared<AmericanExercise>(earliest, exDate, payoffAtExpiry);

        VanillaOption option(payoff, exercise);
        if (knockOutEngine)
            option.setPricingEngine(ext::make_shared<AnalyticDigitalAmericanKOEngine>(m.process));
        else
            option.setPricingEngine(ext::make_shared<AnalyticDigitalAmericanEngine>(m.process));

        ex.b("throws", false);
        readGreeks(ex, option);
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

//! Non-American exercise -> "non-American exercise given".
void emitDigitalEuropeanExerciseThrows() {
    Obj in;
    describeMarket(in, 100.0, 0.02, 0.05, 0.25, 180);
    in.s("exercise", "European").s("type", "Call").n("strike", 100.0).n("cash", 10.0);
    Obj ex;
    try {
        const Market m = makeMarket(100.0, 0.02, 0.05, 0.25);
        auto payoff = ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0);
        auto exercise = ext::make_shared<EuropeanExercise>(TODAY + 180);
        VanillaOption option(payoff, exercise);
        option.setPricingEngine(ext::make_shared<AnalyticDigitalAmericanEngine>(m.process));
        option.NPV();
        ex.b("throws", false);
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase("digital_rejects_european_exercise", in, ex);
}

// ===========================================================================
// 4. BaroneAdesiWhaleyApproximationEngine
// ===========================================================================

void emitBawCritical(const std::string& name,
                     Option::Type type,
                     Real strike,
                     Rate q,
                     Rate r,
                     Volatility vol,
                     Real T,
                     Real tolerance) {
    const DiscountFactor riskFreeDiscount = std::exp(-r * T);
    const DiscountFactor dividendDiscount = std::exp(-q * T);
    const Real variance = vol * vol * T;

    Obj in;
    in.s("type", typeName(type))
        .n("strike", strike)
        .n("q", q)
        .n("r", r)
        .n("vol", vol)
        .n("T", T)
        .n("risk_free_discount", riskFreeDiscount)
        .n("dividend_discount", dividendDiscount)
        .n("variance", variance)
        .n("tolerance", tolerance);

    Obj ex;
    try {
        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
        const Real sk = BaroneAdesiWhaleyApproximationEngine::criticalPrice(
            payoff, riskFreeDiscount, dividendDiscount, variance, tolerance);
        ex.b("throws", false).n("critical_price", sk);
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

//! criticalPrice with the DEFAULT tolerance argument omitted at the call site.
void emitBawCriticalDefaultTolerance(const std::string& name,
                                     Option::Type type,
                                     Real strike,
                                     Rate q,
                                     Rate r,
                                     Volatility vol,
                                     Real T) {
    const DiscountFactor riskFreeDiscount = std::exp(-r * T);
    const DiscountFactor dividendDiscount = std::exp(-q * T);
    const Real variance = vol * vol * T;

    Obj in;
    in.s("type", typeName(type))
        .n("strike", strike)
        .n("q", q)
        .n("r", r)
        .n("vol", vol)
        .n("T", T)
        .n("risk_free_discount", riskFreeDiscount)
        .n("dividend_discount", dividendDiscount)
        .n("variance", variance)
        .s("tolerance", "default");

    Obj ex;
    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
    const Real sk = BaroneAdesiWhaleyApproximationEngine::criticalPrice(
        payoff, riskFreeDiscount, dividendDiscount, variance);
    ex.b("throws", false).n("critical_price", sk);
    addCase(name, in, ex);
}

enum class AmEngine { BaroneAdesiWhaley, JuQuadratic, BjerksundStensland };

std::string engineName(AmEngine e) {
    switch (e) {
        case AmEngine::BaroneAdesiWhaley:
            return "BaroneAdesiWhaleyApproximationEngine";
        case AmEngine::JuQuadratic:
            return "JuQuadraticApproximationEngine";
        default:
            return "BjerksundStenslandApproximationEngine";
    }
}

void emitAmerican(const std::string& name,
                  AmEngine which,
                  Real spot,
                  Rate q,
                  Rate r,
                  Volatility vol,
                  int maturityDays,
                  Option::Type type,
                  Real strike,
                  bool payoffAtExpiry = false,
                  bool europeanExercise = false) {
    Obj in;
    describeMarket(in, spot, q, r, vol, maturityDays);
    in.s("engine", engineName(which))
        .s("type", typeName(type))
        .n("strike", strike)
        .b("payoff_at_expiry", payoffAtExpiry)
        .b("european_exercise", europeanExercise);

    Obj ex;
    try {
        const Market m = makeMarket(spot, q, r, vol);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
        const Date exDate = TODAY + maturityDays;
        ext::shared_ptr<Exercise> exercise;
        if (europeanExercise)
            exercise = ext::make_shared<EuropeanExercise>(exDate);
        else
            exercise = ext::make_shared<AmericanExercise>(TODAY, exDate, payoffAtExpiry);

        VanillaOption option(payoff, exercise);
        switch (which) {
            case AmEngine::BaroneAdesiWhaley:
                option.setPricingEngine(
                    ext::make_shared<BaroneAdesiWhaleyApproximationEngine>(m.process));
                break;
            case AmEngine::JuQuadratic:
                option.setPricingEngine(
                    ext::make_shared<JuQuadraticApproximationEngine>(m.process));
                break;
            default:
                option.setPricingEngine(
                    ext::make_shared<BjerksundStenslandApproximationEngine>(m.process));
                break;
        }

        ex.b("throws", false);
        readGreeks(ex, option, which == AmEngine::BjerksundStensland);
        if (which == AmEngine::BjerksundStensland) {
            ex.n("strike_gamma", option.result<Real>("strikeGamma"))
                .s("exercise_type", option.result<std::string>("exerciseType"));
        }
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// ===========================================================================
// 5. IntegralEngine
// ===========================================================================

void emitIntegral(const std::string& name,
                  Real spot,
                  Rate q,
                  Rate r,
                  Volatility vol,
                  int maturityDays,
                  Option::Type type,
                  Real strike,
                  const std::string& payoffKind = "PlainVanilla",
                  Real cash = 0.0,
                  bool americanExercise = false) {
    Obj in;
    describeMarket(in, spot, q, r, vol, maturityDays);
    in.s("type", typeName(type))
        .n("strike", strike)
        .s("payoff", payoffKind)
        .n("cash", cash)
        .b("american_exercise", americanExercise);

    Obj ex;
    try {
        const Market m = makeMarket(spot, q, r, vol);
        ext::shared_ptr<StrikedTypePayoff> payoff;
        if (payoffKind == "CashOrNothing")
            payoff = ext::make_shared<CashOrNothingPayoff>(type, strike, cash);
        else if (payoffKind == "AssetOrNothing")
            payoff = ext::make_shared<AssetOrNothingPayoff>(type, strike);
        else
            payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);

        const Date exDate = TODAY + maturityDays;
        ext::shared_ptr<Exercise> exercise;
        if (americanExercise)
            exercise = ext::make_shared<AmericanExercise>(TODAY, exDate);
        else
            exercise = ext::make_shared<EuropeanExercise>(exDate);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(ext::make_shared<IntegralEngine>(m.process));
        ex.b("throws", false);
        readGreeks(ex, option);
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// ===========================================================================
// 6. JumpDiffusionEngine + Merton76Process
// ===========================================================================

void emitJumpDiffusion(const std::string& name,
                       Real spot,
                       Rate q,
                       Rate r,
                       Volatility vol,
                       int maturityDays,
                       Option::Type type,
                       Real strike,
                       Real jumpIntensity,
                       Real logMeanJump,
                       Real logJumpVol,
                       Real relativeAccuracy = 1e-4,
                       Size maxIterations = 100,
                       bool americanExercise = false) {
    Obj in;
    describeMarket(in, spot, q, r, vol, maturityDays);
    in.s("type", typeName(type))
        .n("strike", strike)
        .n("jump_intensity", jumpIntensity)
        .n("log_mean_jump", logMeanJump)
        .n("log_jump_volatility", logJumpVol)
        .n("relative_accuracy", relativeAccuracy)
        .i("max_iterations", static_cast<long long>(maxIterations))
        .b("american_exercise", americanExercise);

    Obj ex;
    try {
        auto s = ext::make_shared<SimpleQuote>(spot);
        Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(TODAY, q, DC));
        Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(TODAY, r, DC));
        Handle<BlackVolTermStructure> vTS(ext::make_shared<BlackConstantVol>(TODAY, CAL, vol, DC));
        Handle<Quote> jumpInt(ext::make_shared<SimpleQuote>(jumpIntensity));
        Handle<Quote> logJMean(ext::make_shared<SimpleQuote>(logMeanJump));
        Handle<Quote> logJVol(ext::make_shared<SimpleQuote>(logJumpVol));

        auto process = ext::make_shared<Merton76Process>(Handle<Quote>(s), qTS, rTS, vTS, jumpInt,
                                                         logJMean, logJVol);

        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
        const Date exDate = TODAY + maturityDays;
        ext::shared_ptr<Exercise> exercise;
        if (americanExercise)
            exercise = ext::make_shared<AmericanExercise>(TODAY, exDate);
        else
            exercise = ext::make_shared<EuropeanExercise>(exDate);

        VanillaOption option(payoff, exercise);
        option.setPricingEngine(
            ext::make_shared<JumpDiffusionEngine>(process, relativeAccuracy, maxIterations));

        ex.b("throws", false);
        readGreeks(ex, option);
        ex.n("process_x0", process->x0()).n("process_time_at_maturity", process->time(exDate));
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

//! Merton76Process::drift / diffusion / apply all QL_FAIL.
void emitMertonUnsupported() {
    auto s = ext::make_shared<SimpleQuote>(100.0);
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(TODAY, 0.0, DC));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(TODAY, 0.08, DC));
    Handle<BlackVolTermStructure> vTS(ext::make_shared<BlackConstantVol>(TODAY, CAL, 0.25, DC));
    Handle<Quote> jumpInt(ext::make_shared<SimpleQuote>(1.0));
    Handle<Quote> logJMean(ext::make_shared<SimpleQuote>(-0.03125));
    Handle<Quote> logJVol(ext::make_shared<SimpleQuote>(0.25));
    auto process = ext::make_shared<Merton76Process>(Handle<Quote>(s), qTS, rTS, vTS, jumpInt,
                                                     logJMean, logJVol);

    auto emit = [&](const std::string& name, const std::string& method) {
        Obj in;
        in.s("method", method).n("spot", 100.0);
        Obj ex;
        bool threw = false;
        try {
            if (method == "drift")
                static_cast<void>(process->drift(0.5, 100.0));
            else if (method == "diffusion")
                static_cast<void>(process->diffusion(0.5, 100.0));
            else
                static_cast<void>(process->apply(100.0, 0.01));
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    emit("merton76_drift_unsupported", "drift");
    emit("merton76_diffusion_unsupported", "diffusion");
    emit("merton76_apply_unsupported", "apply");

    // Inspectors delegate to the inner BlackScholesMertonProcess.
    Obj in;
    in.n("spot", 100.0).n("q", 0.0).n("r", 0.08).n("vol", 0.25).i("maturity_days", 180);
    Obj ex;
    ex.n("x0", process->x0())
        .n("time_at_maturity", process->time(TODAY + 180))
        .n("jump_intensity", process->jumpIntensity()->value())
        .n("log_mean_jump", process->logMeanJump()->value())
        .n("log_jump_volatility", process->logJumpVolatility()->value())
        .n("state_variable", process->stateVariable()->value())
        .n("dividend_discount", process->dividendYield()->discount(TODAY + 180))
        .n("risk_free_discount", process->riskFreeRate()->discount(TODAY + 180))
        .n("black_variance", process->blackVolatility()->blackVariance(TODAY + 180, 100.0));
    addCase("merton76_inspectors", in, ex);
}

// ===========================================================================
// 7. DiscretizedVanillaOption
// ===========================================================================

//! Drive the asset exactly as BinomialVanillaEngine<CoxRossRubinstein> does.
void emitDiscretizedVanilla(const std::string& name,
                            Real spot,
                            Rate q,
                            Rate r,
                            Volatility vol,
                            int maturityDays,
                            Option::Type type,
                            Real strike,
                            Size timeSteps,
                            const std::string& exerciseKind,
                            const std::vector<int>& bermudanOffsetDays = {}) {
    Obj in;
    describeMarket(in, spot, q, r, vol, maturityDays);
    in.s("type", typeName(type))
        .n("strike", strike)
        .i("time_steps", static_cast<long long>(timeSteps))
        .s("exercise", exerciseKind);
    {
        std::vector<Real> offs;
        offs.reserve(bermudanOffsetDays.size());
        for (int d : bermudanOffsetDays)
            offs.push_back(Real(d));
        in.arr("bermudan_offset_days", offs);
    }

    Obj ex;
    try {
        const Market m = makeMarket(spot, q, r, vol);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
        const Date exDate = TODAY + maturityDays;

        ext::shared_ptr<Exercise> exercise;
        if (exerciseKind == "European") {
            exercise = ext::make_shared<EuropeanExercise>(exDate);
        } else if (exerciseKind == "Bermudan") {
            std::vector<Date> dates;
            dates.reserve(bermudanOffsetDays.size());
            for (int d : bermudanOffsetDays)
                dates.push_back(TODAY + d);
            exercise = ext::make_shared<BermudanExercise>(dates);
        } else {
            exercise = ext::make_shared<AmericanExercise>(TODAY, exDate);
        }

        VanillaOption option(payoff, exercise);
        VanillaOption::arguments args;
        option.setupArguments(&args);
        args.validate();

        const Time maturity = DC.yearFraction(TODAY, exDate);
        // Same flat-coefficient reconstruction the C++ binomial engine does.
        const Rate rr = m.process->riskFreeRate()->zeroRate(exDate, DC, Continuous, NoFrequency);
        const Rate qq = m.process->dividendYield()->zeroRate(exDate, DC, Continuous, NoFrequency);
        const Volatility vv = m.process->blackVolatility()->blackVol(exDate, spot);

        Handle<YieldTermStructure> flatR(ext::make_shared<FlatForward>(TODAY, rr, DC));
        Handle<YieldTermStructure> flatQ(ext::make_shared<FlatForward>(TODAY, qq, DC));
        Handle<BlackVolTermStructure> flatV(
            ext::make_shared<BlackConstantVol>(TODAY, CAL, vv, DC));
        ext::shared_ptr<StochasticProcess1D> bs(
            new GeneralizedBlackScholesProcess(m.process->stateVariable(), flatQ, flatR, flatV));

        const TimeGrid grid(maturity, timeSteps);
        auto tree = ext::make_shared<CoxRossRubinstein>(bs, maturity, timeSteps, strike);
        auto lattice =
            ext::make_shared<BlackScholesLattice<CoxRossRubinstein>>(tree, rr, maturity, timeSteps);

        DiscretizedVanillaOption asset(args, *m.process, grid);
        asset.initialize(lattice, maturity);

        ex.b("throws", false);
        ex.n("maturity", maturity).n("flat_r", rr).n("flat_q", qq).n("flat_vol", vv);
        {
            const std::vector<Time> stopping = asset.mandatoryTimes();
            ex.arr("mandatory_times", std::vector<Real>(stopping.begin(), stopping.end()));
        }

        asset.rollback(grid[2]);
        {
            const Array& va2 = asset.values();
            ex.arr("values_at_step2", std::vector<Real>(va2.begin(), va2.end()));
            std::vector<Real> u2;
            for (Size j = 0; j < va2.size(); ++j)
                u2.push_back(lattice->underlying(2, j));
            ex.arr("underlying_at_step2", u2);
        }

        asset.rollback(grid[1]);
        {
            const Array& va1 = asset.values();
            ex.arr("values_at_step1", std::vector<Real>(va1.begin(), va1.end()));
            std::vector<Real> u1;
            for (Size j = 0; j < va1.size(); ++j)
                u1.push_back(lattice->underlying(1, j));
            ex.arr("underlying_at_step1", u1);
        }

        asset.rollback(0.0);
        ex.n("present_value", asset.presentValue());
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

// ===========================================================================
// case tables
// ===========================================================================

void emitAtHitCases() {
    // --- cash-or-nothing, both types, both sides of the barrier -----------
    // Call: barrier (== strike) above spot -> N(-d1)/N(-d2) arm, powers used.
    emitAtHit("hit_cash_call_barrier_above", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Call, 120.0,
              true, 15.0);
    // Call: barrier below spot -> already hit, alpha = beta = 0.5, forward = X = 1.
    emitAtHit("hit_cash_call_barrier_below", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Call, 80.0,
              true, 15.0);
    // Call at strike == spot exactly: the 0.5 arm fires (`>=`) but inTheMoney_
    // is false (strict `<`), so the powers are still evaluated -- at 1.0.
    emitAtHit("hit_cash_call_atm_exact", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Call, 100.0, true,
              15.0);
    // Put: barrier below spot -> N(d1)/N(d2) arm.
    emitAtHit("hit_cash_put_barrier_below", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Put, 80.0, true,
              15.0);
    // Put: barrier above spot -> already hit.
    emitAtHit("hit_cash_put_barrier_above", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Put, 120.0, true,
              15.0);
    emitAtHit("hit_cash_put_atm_exact", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Put, 100.0, true,
              15.0);
    // Haug p.176 setup used by the C++ test-suite (cash-or-nothing at hit).
    emitAtHit("hit_cash_haug_put", 100.0, 0.0, 0.05, 0.20, 0.5, Option::Put, 95.0, true, 15.0);

    // --- asset-or-nothing: K_ = spot when in the money, strike otherwise ---
    emitAtHit("hit_asset_call_barrier_above", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Call, 120.0,
              false, 0.0);
    emitAtHit("hit_asset_call_barrier_below", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Call, 80.0,
              false, 0.0);
    emitAtHit("hit_asset_put_barrier_below", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Put, 80.0,
              false, 0.0);
    emitAtHit("hit_asset_put_barrier_above", 100.0, 0.04, 0.05, 0.25, 0.75, Option::Put, 120.0,
              false, 0.0);
    emitAtHit("hit_asset_haug_put", 100.0, 0.0, 0.05, 0.20, 0.5, Option::Put, 95.0, false, 0.0);

    // --- r == q (mu_ = -0.5 exactly), and negative carry -------------------
    emitAtHit("hit_cash_zero_carry", 100.0, 0.05, 0.05, 0.30, 1.0, Option::Call, 130.0, true, 20.0);
    emitAtHit("hit_cash_negative_carry", 100.0, 0.09, 0.03, 0.30, 1.0, Option::Call, 130.0, true,
              20.0);

    // --- variance below QL_EPSILON: the "not tested yet" branch. -----------
    // Reached with a representable-but-tiny variance and a discount just below
    // 1, so mu_ = log(1/d)/v - 0.5 and lambda_ = sqrt(mu^2 - 2 log d / v) stay
    // finite (9.5 and 10.5 respectively for these numbers) and the branch's
    // real content -- cum_d1_/cum_d2_ collapsing to 0 or 1 by the sign of
    // log(strike/spot), and n_d1 = n_d2 = 0 -- is what gets pinned.
    emitAtHitRaw("hit_cash_call_subeps_variance_otm", 100.0, 0.999999999999999, 1.0, 1e-16, 1.0,
                 Option::Call, 120.0, true, 15.0);
    emitAtHitRaw("hit_cash_call_subeps_variance_itm", 100.0, 0.999999999999999, 1.0, 1e-16, 1.0,
                 Option::Call, 80.0, true, 15.0);
    emitAtHitRaw("hit_cash_put_subeps_variance_otm", 100.0, 0.999999999999999, 1.0, 1e-16, 1.0,
                 Option::Put, 80.0, true, 15.0);
    emitAtHitRaw("hit_asset_put_subeps_variance_itm", 100.0, 0.999999999999999, 1.0, 1e-16, 1.0,
                 Option::Put, 120.0, false, 0.0);
    // variance == 0 with r == q == 0: mu_ = log(1)/0 = 0/0 = NaN and the whole
    // state goes non-finite. That IS what C++ v1.43 returns; it is pinned so a
    // port cannot "helpfully" special-case a zero variance and silently
    // disagree.
    emitAtHit("hit_cash_call_zero_variance_nan", 100.0, 0.0, 0.0, 0.0, 0.0, Option::Call, 120.0,
              true, 15.0);
    emitAtHit("hit_cash_call_zero_variance_itm_nan", 100.0, 0.0, 0.0, 0.0, 0.0, Option::Call, 80.0,
              true, 15.0);
    emitAtHit("hit_cash_put_zero_variance_nan", 100.0, 0.0, 0.0, 0.0, 0.0, Option::Put, 80.0, true,
              15.0);
    emitAtHit("hit_asset_put_zero_variance_nan", 100.0, 0.0, 0.0, 0.0, 0.0, Option::Put, 120.0,
              false, 0.0);

    // --- QL_REQUIRE guards -------------------------------------------------
    auto guard = [](const std::string& name, Real spot, Real discount, Real dividendDiscount,
                    Real variance) {
        Obj in;
        in.n("spot", spot)
            .n("discount", discount)
            .n("dividend_discount", dividendDiscount)
            .n("variance", variance)
            .s("type", "Call")
            .n("strike", 100.0)
            .n("cash", 10.0)
            .s("payoff", "CashOrNothing");
        Obj ex;
        bool threw = false;
        try {
            auto payoff = ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0);
            const AmericanPayoffAtHit pricer(spot, discount, dividendDiscount, variance, payoff);
            static_cast<void>(pricer.value());
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    guard("hit_rejects_nonpositive_spot", 0.0, 0.95, 0.98, 0.04);
    guard("hit_rejects_nonpositive_discount", 100.0, 0.0, 0.98, 0.04);
    guard("hit_rejects_nonpositive_dividend_discount", 100.0, 0.95, 0.0, 0.04);
    guard("hit_rejects_negative_variance", 100.0, 0.95, 0.98, -1e-10);

    // rho(T) requires T >= 0.
    {
        Obj in;
        in.n("spot", 100.0)
            .n("discount", 0.96)
            .n("dividend_discount", 0.97)
            .n("variance", 0.05)
            .n("T", -1.0)
            .s("payoff", "CashOrNothing")
            .s("type", "Call")
            .n("strike", 120.0)
            .n("cash", 10.0)
            .s("accessor", "rho");
        Obj ex;
        bool threw = false;
        try {
            auto payoff = ext::make_shared<CashOrNothingPayoff>(Option::Call, 120.0, 10.0);
            const AmericanPayoffAtHit pricer(100.0, 0.96, 0.97, 0.05, payoff);
            static_cast<void>(pricer.rho(-1.0));
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase("hit_rho_rejects_negative_maturity", in, ex);
    }
}

void emitAtExpiryCases() {
    // knock-in x knock-out, Call x Put, breached x not-breached, both payoffs.
    for (const bool knockIn : {true, false}) {
        const std::string ki = knockIn ? "ki" : "ko";
        emitAtExpiry("expiry_cash_call_barrier_above_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75,
                     Option::Call, 120.0, true, 15.0, knockIn);
        emitAtExpiry("expiry_cash_call_barrier_below_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75,
                     Option::Call, 80.0, true, 15.0, knockIn);
        emitAtExpiry("expiry_cash_call_atm_exact_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75, Option::Call,
                     100.0, true, 15.0, knockIn);
        emitAtExpiry("expiry_cash_put_barrier_below_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75,
                     Option::Put, 80.0, true, 15.0, knockIn);
        emitAtExpiry("expiry_cash_put_barrier_above_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75,
                     Option::Put, 120.0, true, 15.0, knockIn);
        emitAtExpiry("expiry_cash_put_atm_exact_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75, Option::Put,
                     100.0, true, 15.0, knockIn);
        emitAtExpiry("expiry_asset_call_barrier_above_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75,
                     Option::Call, 120.0, false, 0.0, knockIn);
        emitAtExpiry("expiry_asset_put_barrier_below_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75,
                     Option::Put, 80.0, false, 0.0, knockIn);
        emitAtExpiry("expiry_asset_put_barrier_above_" + ki, 100.0, 0.04, 0.05, 0.25, 0.75,
                     Option::Put, 120.0, false, 0.0, knockIn);
        // sub-QL_EPSILON variance branch (cum_d1_/cum_d2_ collapse to 0/1)
        emitAtExpiryRaw("expiry_cash_call_subeps_variance_otm_" + ki, 100.0, 0.999999999999999, 1.0,
                        1e-16, Option::Call, 120.0, true, 15.0, knockIn);
        emitAtExpiryRaw("expiry_cash_put_subeps_variance_otm_" + ki, 100.0, 0.999999999999999, 1.0,
                        1e-16, Option::Put, 80.0, true, 15.0, knockIn);
        emitAtExpiryRaw("expiry_asset_call_subeps_variance_otm_" + ki, 100.0, 0.999999999999999,
                        1.0, 1e-16, Option::Call, 120.0, false, 0.0, knockIn);
        // variance == 0 with r == q == 0 -> mu_ is NaN; pinned as-is (see the
        // AmericanPayoffAtHit note).
        emitAtExpiry("expiry_cash_call_zero_variance_nan_" + ki, 100.0, 0.0, 0.0, 0.0, 0.0,
                     Option::Call, 120.0, true, 15.0, knockIn);
        emitAtExpiry("expiry_cash_put_zero_variance_nan_" + ki, 100.0, 0.0, 0.0, 0.0, 0.0,
                     Option::Put, 80.0, true, 15.0, knockIn);
    }
    // Haug at-expiry setup (C++ test-suite testCashAtExpiryOrNothingAmericanValues).
    emitAtExpiry("expiry_cash_haug_put_ki", 100.0, 0.0, 0.05, 0.20, 0.5, Option::Put, 95.0, true,
                 15.0, true);
    emitAtExpiry("expiry_asset_haug_put_ki", 100.0, 0.0, 0.05, 0.20, 0.5, Option::Put, 95.0, false,
                 0.0, true);

    // guards
    auto guard = [](const std::string& name, Real spot, Real discount, Real dividendDiscount,
                    Real variance) {
        Obj in;
        in.n("spot", spot)
            .n("discount", discount)
            .n("dividend_discount", dividendDiscount)
            .n("variance", variance)
            .s("type", "Call")
            .n("strike", 100.0)
            .n("cash", 10.0)
            .s("payoff", "CashOrNothing")
            .b("knock_in", true);
        Obj ex;
        bool threw = false;
        try {
            auto payoff = ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0);
            const AmericanPayoffAtExpiry pricer(spot, discount, dividendDiscount, variance, payoff,
                                                true);
            static_cast<void>(pricer.value());
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase(name, in, ex);
    };
    guard("expiry_rejects_nonpositive_spot", 0.0, 0.95, 0.98, 0.04);
    guard("expiry_rejects_nonpositive_discount", 100.0, 0.0, 0.98, 0.04);
    guard("expiry_rejects_nonpositive_dividend_discount", 100.0, 0.95, 0.0, 0.04);
    guard("expiry_rejects_negative_variance", 100.0, 0.95, 0.98, -1e-10);
}

void emitDigitalCases() {
    // --- at-hit: value + delta + gamma + rho, nothing else ----------------
    emitDigital("digital_hit_cash_call_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Call, 120.0, true,
                15.0, false, false);
    emitDigital("digital_hit_cash_put_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Put, 80.0, true,
                15.0, false, false);
    emitDigital("digital_hit_asset_call_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Call, 120.0,
                false, 0.0, false, false);
    emitDigital("digital_hit_asset_put_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Put, 80.0, false,
                0.0, false, false);
    // barrier already breached at t=0
    emitDigital("digital_hit_cash_call_breached", 100.0, 0.04, 0.05, 0.25, 270, Option::Call, 80.0,
                true, 15.0, false, false);
    emitDigital("digital_hit_cash_put_breached", 100.0, 0.04, 0.05, 0.25, 270, Option::Put, 120.0,
                true, 15.0, false, false);

    // THE TRAP: with payoffAtExpiry() == false the KO engine returns the
    // knock-in price, because AmericanPayoffAtHit has no knock_in argument.
    emitDigital("digital_hit_cash_call_ko_equals_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Call,
                120.0, true, 15.0, false, true);
    emitDigital("digital_hit_cash_put_ko_equals_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Put,
                80.0, true, 15.0, false, true);

    // --- at-expiry: ONLY value; every greek unset -------------------------
    emitDigital("digital_expiry_cash_call_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Call, 120.0,
                true, 15.0, true, false);
    emitDigital("digital_expiry_cash_call_ko", 100.0, 0.04, 0.05, 0.25, 270, Option::Call, 120.0,
                true, 15.0, true, true);
    emitDigital("digital_expiry_cash_put_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Put, 80.0, true,
                15.0, true, false);
    emitDigital("digital_expiry_cash_put_ko", 100.0, 0.04, 0.05, 0.25, 270, Option::Put, 80.0, true,
                15.0, true, true);
    emitDigital("digital_expiry_asset_call_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Call, 120.0,
                false, 0.0, true, false);
    emitDigital("digital_expiry_asset_call_ko", 100.0, 0.04, 0.05, 0.25, 270, Option::Call, 120.0,
                false, 0.0, true, true);
    // breached at t = 0: knock-in -> 0.5/0.5, knock-out -> 0/0
    emitDigital("digital_expiry_cash_call_breached_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Call,
                80.0, true, 15.0, true, false);
    emitDigital("digital_expiry_cash_call_breached_ko", 100.0, 0.04, 0.05, 0.25, 270, Option::Call,
                80.0, true, 15.0, true, true);
    emitDigital("digital_expiry_cash_put_breached_ki", 100.0, 0.04, 0.05, 0.25, 270, Option::Put,
                120.0, true, 15.0, true, false);
    emitDigital("digital_expiry_cash_put_breached_ko", 100.0, 0.04, 0.05, 0.25, 270, Option::Put,
                120.0, true, 15.0, true, true);

    // Haug reference setups from the C++ test-suite.
    emitDigital("digital_hit_haug_cash_put", 100.0, 0.0, 0.05, 0.20, 182, Option::Put, 95.0, true,
                15.0, false, false);
    emitDigital("digital_expiry_haug_cash_put", 100.0, 0.0, 0.05, 0.20, 182, Option::Put, 95.0,
                true, 15.0, true, false);

    // --- guards ------------------------------------------------------------
    // Window exercise (earliest date after the vol reference date) is refused.
    emitDigital("digital_rejects_window_exercise", 100.0, 0.04, 0.05, 0.25, 270, Option::Call,
                120.0, true, 15.0, false, false, 30);
    emitDigitalEuropeanExerciseThrows();
}

void emitBawCases() {
    // --- criticalPrice, pinned directly -----------------------------------
    emitBawCriticalDefaultTolerance("baw_critical_call_default_tol", Option::Call, 100.0, 0.04, 0.08,
                                    0.20, 0.5);
    emitBawCriticalDefaultTolerance("baw_critical_put_default_tol", Option::Put, 100.0, 0.04, 0.08,
                                    0.20, 0.5);
    emitBawCritical("baw_critical_call_tol_1e6", Option::Call, 100.0, 0.04, 0.08, 0.20, 0.5, 1e-6);
    emitBawCritical("baw_critical_call_tol_1e10", Option::Call, 100.0, 0.04, 0.08, 0.20, 0.5, 1e-10);
    emitBawCritical("baw_critical_put_tol_1e10", Option::Put, 100.0, 0.04, 0.08, 0.20, 0.5, 1e-10);
    // low vol / high vol, short / long maturity
    emitBawCritical("baw_critical_call_lowvol", Option::Call, 100.0, 0.10, 0.10, 0.15, 0.1, 1e-6);
    emitBawCritical("baw_critical_call_highvol", Option::Call, 100.0, 0.10, 0.10, 0.35, 0.5, 1e-6);
    emitBawCritical("baw_critical_put_lowvol", Option::Put, 100.0, 0.10, 0.10, 0.15, 0.1, 1e-6);
    emitBawCritical("baw_critical_put_longdated", Option::Put, 100.0, 0.02, 0.06, 0.30, 10.0, 1e-6);
    // riskFreeDiscount > 1 (negative rate) is refused
    emitBawCritical("baw_critical_rejects_negative_rate", Option::Call, 100.0, 0.04, -0.01, 0.20,
                    0.5, 1e-6);
    // r == 0 exactly: close(riskFreeDiscount, 1.0, 1000) fires -> K = 2/variance
    emitBawCritical("baw_critical_zero_rate_uses_K_fallback", Option::Call, 100.0, 0.04, 0.0, 0.20,
                    0.5, 1e-6);

    // --- engine: European arm (q <= 0 and Call) ----------------------------
    emitAmerican("baw_call_q_zero_european_arm", AmEngine::BaroneAdesiWhaley, 100.0, 0.0, 0.05, 0.25,
                 365, Option::Call, 100.0);
    emitAmerican("baw_call_q_negative_european_arm", AmEngine::BaroneAdesiWhaley, 100.0, -0.03, 0.05,
                 0.25, 365, Option::Call, 100.0);

    // --- engine: approximation arm, Call ----------------------------------
    // Haug p.24 grid (the C++ test-suite's testBaroneAdesiWhaleyValues).
    emitAmerican("baw_call_haug_otm", AmEngine::BaroneAdesiWhaley, 90.0, 0.10, 0.10, 0.15, 37,
                 Option::Call, 100.0);
    emitAmerican("baw_call_haug_atm", AmEngine::BaroneAdesiWhaley, 100.0, 0.10, 0.10, 0.25, 37,
                 Option::Call, 100.0);
    emitAmerican("baw_call_haug_itm", AmEngine::BaroneAdesiWhaley, 110.0, 0.10, 0.10, 0.35, 183,
                 Option::Call, 100.0);
    // spot >= Sk -> intrinsic (deep ITM with a heavy dividend yield)
    emitAmerican("baw_call_deep_itm_intrinsic", AmEngine::BaroneAdesiWhaley, 300.0, 0.30, 0.05, 0.15,
                 365, Option::Call, 100.0);

    // --- engine: approximation arm, Put -----------------------------------
    emitAmerican("baw_put_haug_otm", AmEngine::BaroneAdesiWhaley, 110.0, 0.10, 0.10, 0.15, 37,
                 Option::Put, 100.0);
    emitAmerican("baw_put_haug_atm", AmEngine::BaroneAdesiWhaley, 100.0, 0.10, 0.10, 0.25, 37,
                 Option::Put, 100.0);
    emitAmerican("baw_put_zero_dividend", AmEngine::BaroneAdesiWhaley, 100.0, 0.0, 0.05, 0.25, 365,
                 Option::Put, 100.0);
    // spot <= Sk -> intrinsic
    emitAmerican("baw_put_deep_itm_intrinsic", AmEngine::BaroneAdesiWhaley, 20.0, 0.02, 0.10, 0.15,
                 365, Option::Put, 100.0);

    // --- maturity extremes and zero volatility ----------------------------
    emitAmerican("baw_call_one_day", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05, 0.25, 1,
                 Option::Call, 100.0);
    emitAmerican("baw_put_one_day", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05, 0.25, 1,
                 Option::Put, 100.0);
    emitAmerican("baw_call_ten_years", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05, 0.25, 3650,
                 Option::Call, 100.0);
    emitAmerican("baw_put_ten_years", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05, 0.25, 3650,
                 Option::Put, 100.0);
    // variance == 0 -> criticalPrice divides by zero; whatever comes out is pinned.
    emitAmerican("baw_call_zero_vol", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05, 0.0, 365,
                 Option::Call, 100.0);
    emitAmerican("baw_put_zero_vol", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05, 0.0, 365,
                 Option::Put, 100.0);

    // --- guards ------------------------------------------------------------
    emitAmerican("baw_rejects_european_exercise", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05,
                 0.25, 365, Option::Call, 100.0, false, true);
    emitAmerican("baw_rejects_payoff_at_expiry", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, 0.05,
                 0.25, 365, Option::Call, 100.0, true, false);
    emitAmerican("baw_rejects_negative_rate", AmEngine::BaroneAdesiWhaley, 100.0, 0.06, -0.01, 0.25,
                 365, Option::Put, 100.0);
}

void emitJuCases() {
    // European arm — identical condition to BAW, full greek set.
    emitAmerican("ju_call_q_zero_european_arm", AmEngine::JuQuadratic, 100.0, 0.0, 0.05, 0.25, 365,
                 Option::Call, 100.0);
    emitAmerican("ju_call_q_negative_european_arm", AmEngine::JuQuadratic, 100.0, -0.03, 0.05, 0.25,
                 365, Option::Call, 100.0);

    // Approximation arm: value + delta + gamma, no rho/vega/theta.
    // Ju (1999) table 1 grid, as used by the C++ test-suite.
    emitAmerican("ju_call_otm", AmEngine::JuQuadratic, 90.0, 0.10, 0.10, 0.15, 37, Option::Call,
                 100.0);
    emitAmerican("ju_call_atm", AmEngine::JuQuadratic, 100.0, 0.10, 0.10, 0.25, 37, Option::Call,
                 100.0);
    emitAmerican("ju_call_itm", AmEngine::JuQuadratic, 110.0, 0.10, 0.10, 0.35, 183, Option::Call,
                 100.0);
    emitAmerican("ju_put_otm", AmEngine::JuQuadratic, 110.0, 0.10, 0.10, 0.15, 37, Option::Put,
                 100.0);
    emitAmerican("ju_put_atm", AmEngine::JuQuadratic, 100.0, 0.10, 0.10, 0.25, 37, Option::Put,
                 100.0);
    emitAmerican("ju_put_itm", AmEngine::JuQuadratic, 90.0, 0.10, 0.10, 0.35, 183, Option::Put,
                 100.0);
    emitAmerican("ju_put_zero_dividend", AmEngine::JuQuadratic, 100.0, 0.0, 0.05, 0.25, 365,
                 Option::Put, 100.0);

    // phi*(Sk - spot) <= 0 -> intrinsic value, delta = phi, gamma = 0.
    emitAmerican("ju_call_deep_itm_intrinsic", AmEngine::JuQuadratic, 300.0, 0.30, 0.05, 0.15, 365,
                 Option::Call, 100.0);
    emitAmerican("ju_put_deep_itm_intrinsic", AmEngine::JuQuadratic, 20.0, 0.02, 0.10, 0.15, 365,
                 Option::Put, 100.0);

    emitAmerican("ju_call_one_day", AmEngine::JuQuadratic, 100.0, 0.06, 0.05, 0.25, 1, Option::Call,
                 100.0);
    emitAmerican("ju_put_ten_years", AmEngine::JuQuadratic, 100.0, 0.06, 0.05, 0.25, 3650,
                 Option::Put, 100.0);

    emitAmerican("ju_rejects_european_exercise", AmEngine::JuQuadratic, 100.0, 0.06, 0.05, 0.25, 365,
                 Option::Call, 100.0, false, true);
    emitAmerican("ju_rejects_payoff_at_expiry", AmEngine::JuQuadratic, 100.0, 0.06, 0.05, 0.25, 365,
                 Option::Call, 100.0, true, false);
}

void emitBjerksundCases() {
    // --- European arm: dividendDiscount >= 1 && dividendDiscount >= rfD ----
    emitAmerican("bs_call_q_zero_european_arm", AmEngine::BjerksundStensland, 100.0, 0.0, 0.05, 0.25,
                 365, Option::Call, 100.0);
    emitAmerican("bs_call_q_negative_european_arm", AmEngine::BjerksundStensland, 100.0, -0.02, 0.05,
                 0.25, 365, Option::Call, 100.0);
    // Put with q > 0 and r > 0: the transform makes dD = exp(-rT) < 1, so this
    // is NOT the European arm -- it is the approximation on the mirrored call.
    emitAmerican("bs_put_transform_american", AmEngine::BjerksundStensland, 100.0, 0.04, 0.08, 0.25,
                 365, Option::Put, 100.0);
    // Put with r = 0: after the transform dD == 1.0 exactly, so bT == rT and
    // B0 takes the `(bT == rT) ? X` arm.
    emitAmerican("bs_put_zero_rate_negative_q", AmEngine::BjerksundStensland, 100.0, -0.04, 0.0,
                 0.25, 365, Option::Put, 100.0);
    // Put with r = 0 and q > 0: transform gives dD == 1 >= rfD -> European arm.
    emitAmerican("bs_put_zero_rate_positive_q", AmEngine::BjerksundStensland, 100.0, 0.04, 0.0, 0.25,
                 365, Option::Put, 100.0);

    // --- approximation arm, the "American" exerciseType --------------------
    emitAmerican("bs_call_atm", AmEngine::BjerksundStensland, 100.0, 0.10, 0.10, 0.25, 37,
                 Option::Call, 100.0);
    emitAmerican("bs_call_otm", AmEngine::BjerksundStensland, 90.0, 0.10, 0.10, 0.15, 37,
                 Option::Call, 100.0);
    emitAmerican("bs_call_itm", AmEngine::BjerksundStensland, 110.0, 0.10, 0.10, 0.35, 183,
                 Option::Call, 100.0);
    emitAmerican("bs_put_atm", AmEngine::BjerksundStensland, 100.0, 0.10, 0.10, 0.25, 37,
                 Option::Put, 100.0);
    emitAmerican("bs_put_otm", AmEngine::BjerksundStensland, 110.0, 0.10, 0.10, 0.15, 37,
                 Option::Put, 100.0);
    emitAmerican("bs_put_itm", AmEngine::BjerksundStensland, 90.0, 0.10, 0.10, 0.35, 183,
                 Option::Put, 100.0);
    emitAmerican("bs_call_long_dated", AmEngine::BjerksundStensland, 100.0, 0.08, 0.05, 0.30, 3650,
                 Option::Call, 100.0);

    // --- S >= I -> immediateExercise (exerciseType "Immediate") -----------
    emitAmerican("bs_call_deep_itm_immediate", AmEngine::BjerksundStensland, 300.0, 0.30, 0.05, 0.15,
                 365, Option::Call, 100.0);
    emitAmerican("bs_put_deep_itm_immediate", AmEngine::BjerksundStensland, 20.0, 0.02, 0.10, 0.15,
                 365, Option::Put, 100.0);

    // --- q = ln(I/fwd)/sqrt(variance) > 12.5 -> run-away boundary, falls
    //     back to the European greeks. Tiny vol + short maturity + deep OTM.
    emitAmerican("bs_call_runaway_boundary", AmEngine::BjerksundStensland, 100.0, 0.10, 0.10, 0.01,
                 7, Option::Call, 130.0);
    emitAmerican("bs_put_runaway_boundary", AmEngine::BjerksundStensland, 130.0, 0.10, 0.10, 0.01, 7,
                 Option::Put, 100.0);

    // --- the double-boundary refusal: q < 0 and r < q ----------------------
    emitAmerican("bs_rejects_double_boundary", AmEngine::BjerksundStensland, 100.0, -0.02, -0.05,
                 0.25, 365, Option::Call, 100.0);

    // --- guards ------------------------------------------------------------
    emitAmerican("bs_rejects_european_exercise", AmEngine::BjerksundStensland, 100.0, 0.06, 0.05,
                 0.25, 365, Option::Call, 100.0, false, true);
    emitAmerican("bs_rejects_payoff_at_expiry", AmEngine::BjerksundStensland, 100.0, 0.06, 0.05,
                 0.25, 365, Option::Call, 100.0, true, false);
    // Non-plain payoff -> "non-plain payoff given".
    {
        Obj in;
        describeMarket(in, 100.0, 0.06, 0.05, 0.25, 365);
        in.s("payoff", "CashOrNothing")
            .s("type", "Call")
            .n("strike", 100.0)
            .n("cash", 10.0)
            .s("engine", "BjerksundStenslandApproximationEngine")
            .b("payoff_at_expiry", false)
            .b("european_exercise", false);
        Obj ex;
        bool threw = false;
        try {
            const Market m = makeMarket(100.0, 0.06, 0.05, 0.25);
            auto payoff = ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 10.0);
            auto exercise = ext::make_shared<AmericanExercise>(TODAY, TODAY + 365);
            VanillaOption option(payoff, exercise);
            option.setPricingEngine(
                ext::make_shared<BjerksundStenslandApproximationEngine>(m.process));
            option.NPV();
        } catch (const std::exception&) {
            threw = true;
        }
        ex.b("throws", threw);
        addCase("bs_rejects_non_plain_payoff", in, ex);
    }
}

void emitIntegralCases() {
    emitIntegral("integral_call_atm", 100.0, 0.02, 0.05, 0.25, 365, Option::Call, 100.0);
    emitIntegral("integral_call_itm", 120.0, 0.02, 0.05, 0.25, 365, Option::Call, 100.0);
    emitIntegral("integral_call_otm", 80.0, 0.02, 0.05, 0.25, 365, Option::Call, 100.0);
    emitIntegral("integral_put_atm", 100.0, 0.02, 0.05, 0.25, 365, Option::Put, 100.0);
    emitIntegral("integral_put_itm", 80.0, 0.02, 0.05, 0.25, 365, Option::Put, 100.0);
    emitIntegral("integral_put_otm", 120.0, 0.02, 0.05, 0.25, 365, Option::Put, 100.0);
    emitIntegral("integral_call_lowvol", 100.0, 0.02, 0.05, 0.05, 365, Option::Call, 100.0);
    emitIntegral("integral_call_highvol", 100.0, 0.02, 0.05, 0.80, 365, Option::Call, 100.0);
    emitIntegral("integral_call_short", 100.0, 0.02, 0.05, 0.25, 7, Option::Call, 100.0);
    emitIntegral("integral_call_long", 100.0, 0.02, 0.05, 0.25, 3650, Option::Call, 100.0);
    emitIntegral("integral_call_zero_carry", 100.0, 0.05, 0.05, 0.25, 365, Option::Call, 100.0);
    emitIntegral("integral_call_negative_rate", 100.0, 0.02, -0.01, 0.25, 365, Option::Call, 100.0);
    // The integrand applies arguments_.payoff, so binary payoffs work too.
    emitIntegral("integral_cash_or_nothing_call", 100.0, 0.02, 0.05, 0.25, 365, Option::Call, 100.0,
                 "CashOrNothing", 10.0);
    emitIntegral("integral_asset_or_nothing_put", 100.0, 0.02, 0.05, 0.25, 365, Option::Put, 100.0,
                 "AssetOrNothing", 0.0);
    // American exercise -> "not an European Option".
    emitIntegral("integral_rejects_american_exercise", 100.0, 0.02, 0.05, 0.25, 365, Option::Call,
                 100.0, "PlainVanilla", 0.0, true);
}

void emitJumpDiffusionCases() {
    // Haug p.9 Merton-76 grid, as used by the C++ test-suite. The test-suite
    // parameterises by (jumpIntensity, gamma) with
    //     jumpVol      = sqrt((1 - gamma) * variance / jumpIntensity)
    //     meanLogJump  = -0.5 * jumpVol^2
    // so that the total variance is preserved; those two derived numbers are
    // carried in `inputs` explicitly rather than re-derived on the Python side.
    struct Setup {
        const char* name;
        Real spot;
        Real strike;
        Rate q;
        Rate r;
        Volatility vol;
        int days;
        Option::Type type;
        Real intensity;
        Real gamma;
    };
    const Setup setups[] = {
        {"jd_call_itm_low_intensity", 100.0, 80.0, 0.0, 0.08, 0.25, 37, Option::Call, 1.0, 0.25},
        {"jd_call_atm_low_intensity", 100.0, 100.0, 0.0, 0.08, 0.25, 91, Option::Call, 1.0, 0.25},
        {"jd_call_otm_low_intensity", 100.0, 120.0, 0.0, 0.08, 0.25, 183, Option::Call, 1.0, 0.50},
        {"jd_high_intensity_atm", 100.0, 100.0, 0.0, 0.08, 0.25, 183, Option::Call, 10.0, 0.25},
        {"jd_high_intensity_otm", 100.0, 120.0, 0.0, 0.08, 0.25, 183, Option::Call, 10.0, 0.75},
        {"jd_put_atm", 100.0, 100.0, 0.03, 0.08, 0.25, 183, Option::Put, 5.0, 0.50},
        {"jd_put_itm", 90.0, 100.0, 0.03, 0.08, 0.25, 183, Option::Put, 1.0, 0.25},
        {"jd_call_with_dividend", 100.0, 100.0, 0.04, 0.08, 0.25, 365, Option::Call, 2.0, 0.50},
    };
    for (const auto& s : setups) {
        const Real T = Real(s.days) / 365.0;
        const Real variance = s.vol * s.vol * T;
        const Real jumpVol = std::sqrt((1.0 - s.gamma) * variance / (s.intensity * T));
        const Real meanLogJump = -0.5 * jumpVol * jumpVol;
        emitJumpDiffusion(s.name, s.spot, s.q, s.r, s.vol, s.days, s.type, s.strike, s.intensity,
                          meanLogJump, jumpVol);
    }

    // Tuning knobs: relativeAccuracy and maxIterations must both be exposed.
    emitJumpDiffusion("jd_accuracy_1e4", 100.0, 0.0, 0.08, 0.25, 183, Option::Call, 100.0, 5.0,
                      -0.03125, 0.25, 1e-4, 100);
    emitJumpDiffusion("jd_accuracy_1e10", 100.0, 0.0, 0.08, 0.25, 183, Option::Call, 100.0, 5.0,
                      -0.03125, 0.25, 1e-10, 100);
    emitJumpDiffusion("jd_accuracy_1e2", 100.0, 0.0, 0.08, 0.25, 183, Option::Call, 100.0, 5.0,
                      -0.03125, 0.25, 1e-2, 100);
    // maxIterations = 1 -> QL_ENSURE fires.
    emitJumpDiffusion("jd_rejects_too_few_iterations", 100.0, 0.0, 0.08, 0.25, 183, Option::Call,
                      100.0, 5.0, -0.03125, 0.25, 1e-8, 1);
    // Zero jump intensity degenerates to plain Black-Scholes.
    emitJumpDiffusion("jd_zero_intensity_is_black_scholes", 100.0, 0.02, 0.08, 0.25, 183,
                      Option::Call, 100.0, 0.0, 0.0, 0.0);
    // American exercise: the inner AnalyticEuropeanEngine refuses.
    emitJumpDiffusion("jd_rejects_american_exercise", 100.0, 0.0, 0.08, 0.25, 183, Option::Call,
                      100.0, 1.0, -0.03125, 0.25, 1e-4, 100, true);

    emitMertonUnsupported();
}

void emitDiscretizedVanillaCases() {
    // European: exercise condition applied only at the maturity slice.
    emitDiscretizedVanilla("dvo_european_call", 100.0, 0.02, 0.05, 0.25, 365, Option::Call, 100.0,
                           50, "European");
    emitDiscretizedVanilla("dvo_european_put", 100.0, 0.02, 0.05, 0.25, 365, Option::Put, 100.0, 50,
                           "European");
    // American: the `now <= stoppingTimes_[1] && now >= stoppingTimes_[0]`
    // range test makes every slice exercisable.
    emitDiscretizedVanilla("dvo_american_put", 100.0, 0.02, 0.05, 0.25, 365, Option::Put, 100.0, 50,
                           "American");
    emitDiscretizedVanilla("dvo_american_call_with_dividend", 100.0, 0.08, 0.05, 0.25, 365,
                           Option::Call, 100.0, 50, "American");
    // Bermudan: dates deliberately off the grid, so the ctor's
    // grid.closestTime() snapping is observable.
    emitDiscretizedVanilla("dvo_bermudan_put", 100.0, 0.02, 0.05, 0.25, 365, Option::Put, 100.0, 50,
                           "Bermudan", {97, 201, 365});
    // Odd step count and a different moneyness, to catch off-by-one indexing.
    emitDiscretizedVanilla("dvo_american_put_odd_steps", 105.0, 0.02, 0.05, 0.30, 200, Option::Put,
                           100.0, 37, "American");
    emitDiscretizedVanilla("dvo_european_call_few_steps", 100.0, 0.02, 0.05, 0.25, 365,
                           Option::Call, 100.0, 4, "European");
}

}  // namespace

int main() {
    try {
        Settings::instance().evaluationDate() = TODAY;

        emitAtHitCases();
        emitAtExpiryCases();
        emitDigitalCases();
        emitBawCases();
        emitJuCases();
        emitBjerksundCases();
        emitIntegralCases();
        emitJumpDiffusionCases();
        emitDiscretizedVanillaCases();

        emitDocument();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << "\n";
        return 1;
    }
}
