// migration-harness/cpp/probes/v143_experimental_cdsntd/probe.cpp
//
// Reference values for the three ql/experimental/credit classes @ v1.43:
//
//   CdsOption               experimental/credit/cdsoption.hpp:43
//   BlackCdsOptionEngine    experimental/credit/blackcdsoptionengine.hpp:36
//   IntegralNtdEngine       experimental/credit/integralntdengine.hpp:31
//
// SECTION A -- CdsOption + BlackCdsOptionEngine
// ---------------------------------------------
// The engine reads exactly four things off the underlying CDS -- coupons()
// .front()->date(), fairSpread(), runningSpread() and couponLegNPV() -- so the
// underlying needs its own engine (MidPointCdsEngine here) before the option
// can price at all. Pinned:
//
//   * NPV and riskyAnnuity for payer (Protection::Buyer -> Option::Call) and
//     receiver (Seller -> Put) options at several vols and exercise dates;
//   * the non-knock-out payer, which is the ONLY configuration that adds the
//     front-end-protection term. Note the term is multiplied by
//     Integer(callPut), and callPut is Call == +1 there by construction, so
//     the sign is always +1 -- a port that "simplifies" the sign away is
//     right, but a port that uses the SIDE's sign is wrong;
//   * the knocksOut flag on a receiver, which C++ rejects in the CdsOption
//     constructor ("receiver CDS options must knock out"), and the
//     upfront-underlying rejection;
//   * atmRate() == underlying fairSpread();
//   * impliedVolatility(), which re-prices through a private
//     BlackCdsOptionEngine driven off a SimpleQuote and Brent-solves. Pinned
//     at several target values including a round-trip from a known vol.
//
// The engine day-counts T from termStructure_->referenceDate() to the exercise
// date using the DISCOUNT curve's day counter, not the CDS's -- covered by
// giving the two curves different day counters (Actual365Fixed vs Actual360).
//
// SECTION B -- IntegralNtdEngine over a probe-local analytic loss model
// ---------------------------------------------------------------------
// The engine's numerics are entirely in its Riemann loop, and the loop's grid
// is easy to get wrong in a port. The grid is NOT clamped to the accrual end:
// the step shrinks to one day exactly once, the first time d0 + step would
// overshoot, and the walk then lands on accrualEndDate exactly. A port that
// writes `d = min(d0 + step, accrualEnd)` instead agrees whenever the step
// divides the accrual period and disagrees badly when the step EXCEEDS it --
// it collapses the tail into a single lump with one discount factor and one
// accrued amount instead of the daily sequence.
//
// To pin that without drowning the signal in copula quadrature, section B
// installs a probe-local DefaultLossModel whose probAtLeastNEvents is the
// closed form 1 - exp(-n * lambda * t), t = Actual365Fixed year fraction from
// the basket reference date. It is a legitimate loss model as far as Basket is
// concerned (the hooks are protected virtuals and Basket is a friend), it is
// exactly reproducible in any language, and it makes every reported number a
// function of the grid alone. Integration steps are chosen to straddle the
// interesting cases:
//
//   1W  -- divides nothing, several steps then a daily tail;
//   1M  -- close to the 3M accrual period;
//   3M  -- exactly the accrual period, so d0 + step == accrualEnd and the
//          step reduction never fires;
//   6M  -- EXCEEDS the accrual period: C++ walks ~90 daily steps per coupon,
//          the clamping variant takes one. This is the discriminating case.
//
// Both sides (Buyer and Seller), settlePremiumAccrual on and off, and a
// non-zero upfront rate are covered, plus fairPremium and the three
// additionalResults entries under their C++ keys.
//
// SECTION C -- IntegralNtdEngine over the real copula stack
// ----------------------------------------------------------
// Section B proves the loop; section C proves the loop composes with a real
// Basket + ConstantLossModel<GaussianCopulaPolicy>, i.e. that the engine
// prices something an actual user would build. Deliberately small (4 names,
// 1-year schedule) because ConstantLossModel's conditional probability walks
// all 2^n default patterns per quadrature node.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/cdsntd.json.

#include <ql/currencies/europe.hpp>
#include <ql/exercise.hpp>
#include <ql/experimental/credit/basket.hpp>
#include <ql/experimental/credit/blackcdsoptionengine.hpp>
#include <ql/experimental/credit/cdsoption.hpp>
#include <ql/experimental/credit/constantlosslatentmodel.hpp>
#include <ql/experimental/credit/defaultlossmodel.hpp>
#include <ql/experimental/credit/integralntdengine.hpp>
#include <ql/experimental/credit/nthtodefault.hpp>
#include <ql/experimental/credit/pool.hpp>
#include <ql/experimental/math/gaussiancopulapolicy.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/instruments/creditdefaultswap.hpp>
#include <ql/pricingengines/credit/midpointcdsengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

bool g_first = true;

void sep() {
    if (!g_first) std::cout << ",\n";
    g_first = false;
}

void emit(const std::string& name, Real v) {
    sep();
    std::cout << "  \"" << name << "\": " << std::setprecision(17) << v;
}

void emit_int(const std::string& name, long long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
}

// ==========================================================================
// SECTION A -- CdsOption + BlackCdsOptionEngine
// ==========================================================================

const Date A_TODAY(15, January, 2024);

struct CdsFixture {
    Handle<YieldTermStructure> yts;
    Handle<DefaultProbabilityTermStructure> dts;
    Real recovery;
};

CdsFixture makeCdsFixture(Real flatRate, Real hazard, Real recovery) {
    CdsFixture f;
    // Deliberately DIFFERENT day counters: the option engine reads T off the
    // discount curve's day counter (Actual365Fixed), not the CDS schedule's.
    f.yts = Handle<YieldTermStructure>(ext::make_shared<FlatForward>(
        A_TODAY, flatRate, Actual365Fixed(), Continuous, Annual));
    f.dts = Handle<DefaultProbabilityTermStructure>(
        ext::make_shared<FlatHazardRate>(
            A_TODAY, Handle<Quote>(ext::make_shared<SimpleQuote>(hazard)),
            Actual365Fixed()));
    f.recovery = recovery;
    return f;
}

ext::shared_ptr<CreditDefaultSwap> makeUnderlying(const CdsFixture& f,
                                                  Protection::Side side,
                                                  const Date& start,
                                                  const Date& end,
                                                  Rate spread,
                                                  Real notional) {
    Schedule sched = MakeSchedule()
                         .from(start)
                         .to(end)
                         .withTenor(3 * Months)
                         .withCalendar(TARGET())
                         .withTerminationDateConvention(Unadjusted)
                         .withRule(DateGeneration::CDS);
    auto cds = ext::make_shared<CreditDefaultSwap>(
        side, notional, spread, sched, Following, Actual360(),
        /*settlesAccrual*/ true, /*paysAtDefaultTime*/ true, start);
    cds->setPricingEngine(ext::make_shared<MidPointCdsEngine>(
        f.dts, f.recovery, f.yts));
    return cds;
}

void emitCdsOptionCase(const std::string& tag,
                       Protection::Side side,
                       const Date& exerciseDate,
                       const Date& cdsStart,
                       const Date& cdsEnd,
                       Rate spread,
                       Real notional,
                       Volatility vol,
                       bool knocksOut,
                       Real hazard,
                       Real recovery,
                       Real flatRate) {
    CdsFixture f = makeCdsFixture(flatRate, hazard, recovery);
    auto cds = makeUnderlying(f, side, cdsStart, cdsEnd, spread, notional);

    emit(tag + "_underlying_fairSpread", cds->fairSpread());
    emit(tag + "_underlying_couponLegNPV", cds->couponLegNPV());
    emit(tag + "_underlying_NPV", cds->NPV());

    auto volQuote = ext::make_shared<SimpleQuote>(vol);
    CdsOption option(cds, ext::make_shared<EuropeanExercise>(exerciseDate),
                     knocksOut);
    option.setPricingEngine(ext::make_shared<BlackCdsOptionEngine>(
        f.dts, recovery, f.yts, Handle<Quote>(volQuote)));

    emit(tag + "_NPV", option.NPV());
    emit(tag + "_riskyAnnuity", option.riskyAnnuity());
    emit(tag + "_atmRate", option.atmRate());
    emit_int(tag + "_isExpired", option.isExpired() ? 1 : 0);
    emit_int(tag + "_knocksOut", knocksOut ? 1 : 0);
    emit_int(tag + "_side", (long long)side);
    emit(tag + "_vol", vol);
    emit(tag + "_spread", spread);
    emit(tag + "_notional", notional);
    emit(tag + "_hazard", hazard);
    emit(tag + "_recovery", recovery);
    emit(tag + "_flatRate", flatRate);
    emit_int(tag + "_exercise_serial", (long long)exerciseDate.serialNumber());
    emit_int(tag + "_cdsStart_serial", (long long)cdsStart.serialNumber());
    emit_int(tag + "_cdsEnd_serial", (long long)cdsEnd.serialNumber());

    // impliedVolatility. Targets are obtained by RE-PRICING the same option at
    // three vols rather than by scaling the NPV: an arbitrary scaling can land
    // below the option's intrinsic value, where no vol reproduces it and Brent
    // throws "root not bracketed" (it does, for cdso_payer_itm at 0.5 * NPV).
    // Re-priced targets are always attainable, and they test something
    // stronger — that the solve inverts the engine exactly, from a guess of
    // 0.10 that is above the low vol and below the high one.
    Real npv = option.NPV();
    emit(tag + "_impliedVol_roundtrip",
         option.impliedVolatility(npv, f.yts, f.dts, recovery));
    // Tighter accuracy + more evaluations, to prove the knobs are wired.
    emit(tag + "_impliedVol_tight",
         option.impliedVolatility(npv, f.yts, f.dts, recovery, 1e-10, 500));

    const Real volLo = 0.5 * vol;
    const Real volHi = 2.0 * vol;
    volQuote->setValue(volLo);
    Real npvLo = option.NPV();
    volQuote->setValue(volHi);
    Real npvHi = option.NPV();
    volQuote->setValue(vol);  // restore before implying
    emit(tag + "_NPV_at_volLo", npvLo);
    emit(tag + "_NPV_at_volHi", npvHi);
    emit(tag + "_volLo", volLo);
    emit(tag + "_volHi", volHi);
    emit(tag + "_impliedVol_from_lo",
         option.impliedVolatility(npvLo, f.yts, f.dts, recovery, 1e-10, 500));
    emit(tag + "_impliedVol_from_hi",
         option.impliedVolatility(npvHi, f.yts, f.dts, recovery, 1e-10, 500));
}

void sectionCdsOption() {
    Settings::instance().evaluationDate() = A_TODAY;
    emit_int("a_evaluationDate_serial", (long long)A_TODAY.serialNumber());

    // A1: payer (Buyer), knock-out, 6m into a 5y CDS.
    emitCdsOptionCase("cdso_payer_ko", Protection::Buyer,
                      Date(15, July, 2024), Date(20, September, 2024),
                      Date(20, September, 2029), 0.0100, 1.0e7, 0.30, true,
                      0.02, 0.4, 0.03);
    // A2: same, non-knock-out -> front-end protection term fires.
    emitCdsOptionCase("cdso_payer_nko", Protection::Buyer,
                      Date(15, July, 2024), Date(20, September, 2024),
                      Date(20, September, 2029), 0.0100, 1.0e7, 0.30, false,
                      0.02, 0.4, 0.03);
    // A3: receiver (Seller). Must knock out.
    emitCdsOptionCase("cdso_recv_ko", Protection::Seller,
                      Date(15, July, 2024), Date(20, September, 2024),
                      Date(20, September, 2029), 0.0100, 1.0e7, 0.30, true,
                      0.02, 0.4, 0.03);
    // A4: deep OTM payer -- running spread far above the fair spread.
    emitCdsOptionCase("cdso_payer_otm", Protection::Buyer,
                      Date(15, April, 2024), Date(20, June, 2024),
                      Date(20, June, 2027), 0.0500, 5.0e6, 0.45, true,
                      0.01, 0.4, 0.02);
    // A5: deep ITM payer -- running spread far below the fair spread, high
    // hazard, low recovery.
    emitCdsOptionCase("cdso_payer_itm", Protection::Buyer,
                      Date(20, October, 2024), Date(20, December, 2024),
                      Date(20, December, 2031), 0.0025, 2.5e6, 0.60, true,
                      0.05, 0.2, 0.045);
    // A6: low vol, short expiry -- stresses stdDev -> 0 in blackFormula.
    emitCdsOptionCase("cdso_payer_lowvol", Protection::Buyer,
                      Date(15, February, 2024), Date(20, March, 2024),
                      Date(20, March, 2029), 0.0100, 1.0e7, 0.05, true,
                      0.02, 0.4, 0.03);

    // A7: the two constructor rejections.
    {
        CdsFixture f = makeCdsFixture(0.03, 0.02, 0.4);
        auto seller = makeUnderlying(f, Protection::Seller,
                                     Date(20, September, 2024),
                                     Date(20, September, 2029), 0.01, 1.0e7);
        std::string msg = "(no throw)";
        try {
            CdsOption bad(seller,
                          ext::make_shared<EuropeanExercise>(Date(15, July, 2024)),
                          /*knocksOut*/ false);
        } catch (const std::exception& e) {
            msg = e.what();
        }
        emit_str("cdso_reject_receiver_nko", msg);
    }
}

// ==========================================================================
// SECTION B -- IntegralNtdEngine over a probe-local analytic loss model
// ==========================================================================

const Date B_TODAY(15, January, 2024);
const Real B_LAMBDA = 0.03;
const Real B_RECOVERY = 0.35;

//! Closed-form stand-in for a real loss model.
/*! probAtLeastNEvents(n, d) = 1 - exp(-n * lambda * t), with t the
    Actual365Fixed year fraction from B_TODAY. Monotone in both n and d, in
    (0, 1), and reproducible in any language to the last bit given the same
    year fraction -- which is the point: every number section B reports is then
    a function of the engine's integration grid and nothing else.
*/
class AnalyticNtdLossModel : public DefaultLossModel {
  public:
    AnalyticNtdLossModel(Real lambda, Real recovery)
    : lambda_(lambda), recovery_(recovery) {}

  protected:
    // Nothing is cached, so the basket-change hook is a no-op.
    void resetModel() override {}

    Probability probAtLeastNEvents(Size n, const Date& d) const override {
        Time t = Actual365Fixed().yearFraction(B_TODAY, d);
        if (t <= 0.0)
            return 0.0;
        return 1.0 - std::exp(-Real(n) * lambda_ * t);
    }
    Real expectedRecovery(const Date&, Size, const DefaultProbKey&) const override {
        return recovery_;
    }

  private:
    Real lambda_;
    Real recovery_;
};

ext::shared_ptr<Basket> makeAnalyticBasket(Size names, Real notionalPerName,
                                           const Date& basketRefDate) {
    DefaultProbKey key = NorthAmericaCorpDefaultKey(
        EURCurrency(), SeniorSec, Period(), 1.0);
    auto hazardCurve = ext::make_shared<FlatHazardRate>(
        basketRefDate, Handle<Quote>(ext::make_shared<SimpleQuote>(B_LAMBDA)),
        Actual365Fixed());
    std::vector<Issuer::key_curve_pair> curves{
        std::make_pair(key, Handle<DefaultProbabilityTermStructure>(hazardCurve))};
    Issuer issuer(curves);

    auto pool = ext::make_shared<Pool>();
    std::vector<std::string> ids;
    std::vector<Real> notionals;
    for (Size i = 0; i < names; ++i) {
        std::string id = "Name" + std::to_string(i);
        ids.push_back(id);
        notionals.push_back(notionalPerName);
        pool->add(id, issuer, key);
    }
    return ext::make_shared<Basket>(basketRefDate, ids, notionals, pool, 0.0, 1.0);
}

void emitNtdCase(const std::string& tag,
                 const Period& step,
                 Size ntdOrder,
                 Protection::Side side,
                 bool settleAccrual,
                 Rate upfrontRate,
                 Rate premiumRate,
                 const Date& schedStart,
                 const Date& schedEnd,
                 const Period& tenor,
                 Real notional,
                 Size names,
                 const Date& basketRefDate = B_TODAY) {
    auto basket = makeAnalyticBasket(names, notional / Real(names), basketRefDate);
    basket->setLossModel(ext::make_shared<AnalyticNtdLossModel>(B_LAMBDA, B_RECOVERY));

    Schedule sched = MakeSchedule()
                         .from(schedStart)
                         .to(schedEnd)
                         .withTenor(tenor)
                         .withCalendar(TARGET());

    NthToDefault ntd(basket, ntdOrder, side, sched, upfrontRate, premiumRate,
                     Actual360(), notional, settleAccrual);

    auto yts = Handle<YieldTermStructure>(ext::make_shared<FlatForward>(
        B_TODAY, 0.035, Actual365Fixed(), Continuous, Annual));
    ntd.setPricingEngine(ext::make_shared<IntegralNtdEngine>(step, yts));

    // Some configurations make C++ THROW rather than price; those are pinned
    // as the error, because a port that quietly returns a number there has
    // diverged. See ntd_started below.
    std::string err = "";
    try {
        emit(tag + "_NPV", ntd.NPV());
        emit(tag + "_premiumLegNPV", ntd.premiumLegNPV());
        emit(tag + "_protectionLegNPV", ntd.protectionLegNPV());
        emit(tag + "_fairPremium", ntd.fairPremium());
    } catch (const std::exception& e) {
        err = e.what();
    }
    emit_str(tag + "_error", err);
    emit_int(tag + "_step_length", (long long)step.length());
    emit_int(tag + "_step_units", (long long)step.units());
    emit_int(tag + "_ntdOrder", (long long)ntdOrder);
    emit_int(tag + "_side", (long long)side);
    emit_int(tag + "_settleAccrual", settleAccrual ? 1 : 0);
    emit(tag + "_upfrontRate", upfrontRate);
    emit(tag + "_premiumRate", premiumRate);
    emit(tag + "_notional", notional);
    emit_int(tag + "_names", (long long)names);
    emit_int(tag + "_schedStart_serial", (long long)schedStart.serialNumber());
    emit_int(tag + "_schedEnd_serial", (long long)schedEnd.serialNumber());
    emit_int(tag + "_tenor_length", (long long)tenor.length());
    emit_int(tag + "_tenor_units", (long long)tenor.units());
    emit_int(tag + "_basketRefDate_serial", (long long)basketRefDate.serialNumber());

    // Also pin the analytic model's own output at the schedule dates so a
    // failing port can tell "wrong probability" from "wrong grid".
    std::vector<Real> probs;
    for (Size i = 0; i < sched.size(); ++i) {
        Time t = Actual365Fixed().yearFraction(B_TODAY, sched.date(i));
        probs.push_back(t <= 0.0 ? 0.0
                                 : 1.0 - std::exp(-Real(ntdOrder) * B_LAMBDA * t));
    }
    sep();
    std::cout << "  \"" << tag << "_model_probs\": [";
    for (Size i = 0; i < probs.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << probs[i];
    }
    std::cout << "]";
}

void sectionNtdAnalytic() {
    Settings::instance().evaluationDate() = B_TODAY;
    emit_int("b_evaluationDate_serial", (long long)B_TODAY.serialNumber());
    emit("b_lambda", B_LAMBDA);
    emit("b_recovery", B_RECOVERY);

    const Date start(20, March, 2024);
    const Date end(20, March, 2025);

    // Step << accrual period: several full steps then a daily tail.
    emitNtdCase("ntd_w1", 1 * Weeks, 2, Protection::Seller, true, 0.0, 0.02,
                start, end, 3 * Months, 4.0e6, 4);
    // Step comparable to the accrual period.
    emitNtdCase("ntd_m1", 1 * Months, 2, Protection::Seller, true, 0.0, 0.02,
                start, end, 3 * Months, 4.0e6, 4);
    // Step exactly the accrual period: d0 + step lands on accrualEnd, so the
    // one-day reduction never fires.
    emitNtdCase("ntd_m3", 3 * Months, 2, Protection::Seller, true, 0.0, 0.02,
                start, end, 3 * Months, 4.0e6, 4);
    // Step EXCEEDS the accrual period -- the discriminating case. C++ walks
    // daily from the accrual start; a grid clamped to accrualEnd takes one
    // lump step instead.
    emitNtdCase("ntd_m6", 6 * Months, 2, Protection::Seller, true, 0.0, 0.02,
                start, end, 3 * Months, 4.0e6, 4);
    // Same, with an even coarser step and semi-annual coupons.
    emitNtdCase("ntd_y1", 1 * Years, 1, Protection::Seller, true, 0.0, 0.02,
                start, end, 6 * Months, 4.0e6, 4);
    // Buyer side -- every sign flips.
    emitNtdCase("ntd_buyer", 1 * Months, 2, Protection::Buyer, true, 0.0, 0.02,
                start, end, 3 * Months, 4.0e6, 4);
    // Accrual settlement off -- accrualValue stays 0 and fairPremium changes.
    emitNtdCase("ntd_noaccr", 1 * Months, 2, Protection::Seller, false, 0.0,
                0.02, start, end, 3 * Months, 4.0e6, 4);
    // Non-zero upfront.
    emitNtdCase("ntd_upfront", 1 * Months, 2, Protection::Seller, true, 0.01,
                0.02, start, end, 3 * Months, 4.0e6, 4);
    // First order, and an order equal to the basket size.
    emitNtdCase("ntd_first", 1 * Months, 1, Protection::Seller, true, 0.0, 0.02,
                start, end, 3 * Months, 4.0e6, 4);
    emitNtdCase("ntd_last", 1 * Months, 4, Protection::Seller, true, 0.0, 0.02,
                start, end, 3 * Months, 4.0e6, 4);
    // A schedule that STARTS BEFORE the evaluation date, so the integration
    // window is clipped to the curve reference date for the first coupon.
    emitNtdCase("ntd_started", 1 * Months, 2, Protection::Seller, true, 0.0,
                0.02, Date(20, December, 2023), Date(20, December, 2024),
                3 * Months, 4.0e6, 4, Date(20, December, 2023));
}

// ==========================================================================
// SECTION C -- IntegralNtdEngine over the real copula stack
// ==========================================================================

void sectionNtdCopula() {
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;
    emit_int("c_evaluationDate_serial", (long long)today.serialNumber());

    const Size names = 4;
    const Real hazard = 0.015;
    const Real recovery = 0.4;
    const Real correlation = 0.3;
    const Real notional = 4.0e6;

    emit("c_hazard", hazard);
    emit("c_recovery", recovery);
    emit("c_correlation", correlation);
    emit("c_notional", notional);
    emit_int("c_names", (long long)names);

    DefaultProbKey key = NorthAmericaCorpDefaultKey(
        EURCurrency(), SeniorSec, Period(), 1.0);
    auto hazardCurve = ext::make_shared<FlatHazardRate>(
        today, Handle<Quote>(ext::make_shared<SimpleQuote>(hazard)),
        Actual365Fixed());
    std::vector<Issuer::key_curve_pair> curves{
        std::make_pair(key, Handle<DefaultProbabilityTermStructure>(hazardCurve))};
    Issuer issuer(curves);

    auto pool = ext::make_shared<Pool>();
    std::vector<std::string> ids;
    std::vector<Real> notionals;
    for (Size i = 0; i < names; ++i) {
        std::string id = "Name" + std::to_string(i);
        ids.push_back(id);
        notionals.push_back(notional / Real(names));
        pool->add(id, issuer, key);
    }
    auto basket = ext::make_shared<Basket>(today, ids, notionals, pool, 0.0, 1.0);

    // One systemic factor, uniform loading sqrt(rho) -> pairwise correlation
    // exactly `correlation`.
    Real loading = std::sqrt(correlation);
    emit("c_factor_loading", loading);
    std::vector<std::vector<Real> > fctrs(names, std::vector<Real>(1, loading));
    auto lossModel = ext::make_shared<ConstantLossModel<GaussianCopulaPolicy> >(
        fctrs, std::vector<Real>(names, recovery),
        LatentModelIntegrationType::GaussianQuadrature,
        GaussianCopulaPolicy::initTraits());
    basket->setLossModel(lossModel);

    // Pin the model's own probabilities first: if these disagree the NPVs
    // below are meaningless.
    const std::vector<Date> probeDates = {
        Date(20, March, 2024), Date(20, June, 2024),
        Date(20, September, 2024), Date(20, December, 2024)};
    for (Size k = 0; k < probeDates.size(); ++k) {
        for (Size n = 1; n <= names; ++n) {
            emit("c_probAtLeast_" + std::to_string(n) + "_" + std::to_string(k),
                 basket->probAtLeastNEvents(n, probeDates[k]));
        }
        emit_int("c_probeDate_" + std::to_string(k) + "_serial",
                 (long long)probeDates[k].serialNumber());
    }
    emit("c_recoveryRate_0", basket->recoveryRate(probeDates[0], 0));

    Schedule sched = MakeSchedule()
                         .from(Date(20, March, 2024))
                         .to(Date(20, March, 2025))
                         .withTenor(3 * Months)
                         .withCalendar(TARGET());
    auto yts = Handle<YieldTermStructure>(ext::make_shared<FlatForward>(
        today, 0.035, Actual365Fixed(), Continuous, Annual));

    for (Size order = 1; order <= names; ++order) {
        NthToDefault ntd(basket, order, Protection::Seller, sched, 0.0, 0.02,
                         Actual360(), notional, true);
        ntd.setPricingEngine(
            ext::make_shared<IntegralNtdEngine>(1 * Months, yts));
        std::string tag = "c_ntd" + std::to_string(order);
        emit(tag + "_NPV", ntd.NPV());
        emit(tag + "_premiumLegNPV", ntd.premiumLegNPV());
        emit(tag + "_protectionLegNPV", ntd.protectionLegNPV());
        emit(tag + "_fairPremium", ntd.fairPremium());
    }
}

}  // namespace

int main() {
    std::cout << "{\n";
    sectionCdsOption();
    sectionNtdAnalytic();
    sectionNtdCopula();
    std::cout << "\n}\n";
    return 0;
}
