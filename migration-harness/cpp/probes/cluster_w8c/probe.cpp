// Phase 11 W8-C cluster probe.
//
// Reference values for the FX delta-vol + VannaVolga batch (a), the
// Heston Asian/forward analytic batch (b), the mcbasket batch (c) and
// the ExtendedBinomialTree batch (d).
//
// C++ parity:
//   ql/quotes/deltavolquote.hpp
//   ql/pricingengines/blackdeltacalculator.hpp
//   ql/experimental/barrieroption/vannavolgabarrierengine.hpp
//   ql/experimental/barrieroption/vannavolgadoublebarrierengine.hpp
//   ql/experimental/asian/analytic_cont_geom_av_price_heston.hpp
//   ql/experimental/asian/analytic_discr_geom_av_price_heston.hpp
//   ql/experimental/forward/analytichestonforwardeuropeanengine.hpp
//   ql/experimental/lattices/extendedbinomialtree.hpp
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/doublebarrieroption.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/instruments/forwardvanillaoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/pricingengines/blackdeltacalculator.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/barrier/analyticdoublebarrierengine.hpp>
#include <ql/experimental/asian/analytic_cont_geom_av_price_heston.hpp>
#include <ql/experimental/asian/analytic_discr_geom_av_price_heston.hpp>
#include <ql/experimental/barrieroption/vannavolgabarrierengine.hpp>
#include <ql/experimental/barrieroption/vannavolgadoublebarrierengine.hpp>
#include <ql/experimental/forward/analytichestonforwardeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/deltavolquote.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/date.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

// ---------------------------------------------------------------------
// (a) BlackDeltaCalculator: strikeFromDelta round-trip + atmStrike.
// ---------------------------------------------------------------------
void block_black_delta() {
    const Real spot = 1.30265;
    const Real dDiscount = 0.99663;   // exp(-r T) ~ flat
    const Real fDiscount = 0.99965;
    const Real stdDev = 0.08925 * std::sqrt(1.0);  // atm vol * sqrt(T)

    // Forward delta call/put strikeFromDelta
    BlackDeltaCalculator callFwd(Option::Call, DeltaVolQuote::Fwd, spot,
                                 dDiscount, fDiscount, stdDev);
    Real kc = callFwd.strikeFromDelta(0.25);
    emit("bdc_fwd_call_strike", kc);
    // round trip: delta from that strike
    emit("bdc_fwd_call_delta_roundtrip", callFwd.deltaFromStrike(kc));

    BlackDeltaCalculator putFwd(Option::Put, DeltaVolQuote::Fwd, spot,
                                dDiscount, fDiscount, stdDev);
    Real kp = putFwd.strikeFromDelta(-0.25);
    emit("bdc_fwd_put_strike", kp);
    emit("bdc_fwd_put_delta_roundtrip", putFwd.deltaFromStrike(kp));

    // Spot delta variant
    BlackDeltaCalculator callSpot(Option::Call, DeltaVolQuote::Spot, spot,
                                  dDiscount, fDiscount, stdDev);
    Real kcs = callSpot.strikeFromDelta(0.25);
    emit("bdc_spot_call_strike", kcs);
    emit("bdc_spot_call_delta_roundtrip", callSpot.deltaFromStrike(kcs));

    // atmStrike for the various conventions (fwd-delta calculator)
    emit("bdc_atm_fwd", callFwd.atmStrike(DeltaVolQuote::AtmFwd));
    emit("bdc_atm_spot", callFwd.atmStrike(DeltaVolQuote::AtmSpot));
    emit("bdc_atm_dn", callFwd.atmStrike(DeltaVolQuote::AtmDeltaNeutral));
}

// ---------------------------------------------------------------------
// (a) VannaVolgaBarrierEngine: canonical FX barrier values.
//     barrier=1.5, s=1.30265, q=0.0003541, r=0.0033871, t=1,
//     vol25Put=0.10087, volAtm=0.08925, vol25Call=0.08463
// ---------------------------------------------------------------------
struct VVRow {
    Barrier::Type bt;
    Option::Type ot;
    Real strike;
    Real smileVol;   // value.v in the test (interpolated smile vol)
    const char* key;
};

void block_vanna_volga_single() {
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;
    const DayCounter dc = Actual365Fixed();

    const Real s = 1.30265;
    const Real q = 0.0003541;
    const Real r = 0.0033871;
    const Real t = 1.0;
    const Real vol25Put = 0.10087;
    const Real volAtm = 0.08925;
    const Real vol25Call = 0.08463;

    auto spot = ext::make_shared<SimpleQuote>(s);
    auto qSq = ext::make_shared<SimpleQuote>(q);
    auto rSq = ext::make_shared<SimpleQuote>(r);
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, Handle<Quote>(qSq), dc));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, Handle<Quote>(rSq), dc));

    auto vol25PutSq = ext::make_shared<SimpleQuote>(vol25Put);
    auto volAtmSq = ext::make_shared<SimpleQuote>(volAtm);
    auto vol25CallSq = ext::make_shared<SimpleQuote>(vol25Call);

    Handle<DeltaVolQuote> volAtmQuote(ext::make_shared<DeltaVolQuote>(
        Handle<Quote>(volAtmSq), DeltaVolQuote::Fwd, t, DeltaVolQuote::AtmDeltaNeutral));
    Handle<DeltaVolQuote> vol25PutQuote(ext::make_shared<DeltaVolQuote>(
        -0.25, Handle<Quote>(vol25PutSq), t, DeltaVolQuote::Fwd));
    Handle<DeltaVolQuote> vol25CallQuote(ext::make_shared<DeltaVolQuote>(
        0.25, Handle<Quote>(vol25CallSq), t, DeltaVolQuote::Fwd));

    Date exDate = today + Period(365, Days);
    auto exercise = ext::make_shared<EuropeanExercise>(exDate);

    VVRow rows[] = {
        {Barrier::UpOut, Option::Call, 1.13321, 0.11638, "vv_upout_call_k1"},
        {Barrier::UpOut, Option::Call, 1.31179, 0.08925, "vv_upout_call_k3"},
        {Barrier::UpOut, Option::Put,  1.38843, 0.08463, "vv_upout_put_k4"},
        {Barrier::UpIn,  Option::Call, 1.13321, 0.11638, "vv_upin_call_k1"},
        {Barrier::DownOut, Option::Put, 1.31179, 0.08925, "vv_downout_put_k3"},
    };

    for (auto& row : rows) {
        auto payoff = ext::make_shared<PlainVanillaPayoff>(row.ot, row.strike);
        Real barrier = (row.bt == Barrier::DownOut || row.bt == Barrier::DownIn) ? 1.0 : 1.5;
        BarrierOption bo(row.bt, barrier, 0.0, payoff, exercise);
        Real fwd = s * qTS->discount(t) / rTS->discount(t);
        Real bsVanilla = blackFormula(row.ot, row.strike, fwd,
                                      row.smileVol * std::sqrt(t), rTS->discount(t));
        auto eng = ext::make_shared<VannaVolgaBarrierEngine>(
            volAtmQuote, vol25PutQuote, vol25CallQuote, Handle<Quote>(spot),
            rTS, qTS, true, bsVanilla);
        bo.setPricingEngine(eng);
        emit(row.key, bo.NPV());
    }
}

void block_vanna_volga_double() {
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;
    const DayCounter dc = Actual365Fixed();

    const Real s = 1.30265;
    const Real q = 0.0003541;
    const Real r = 0.0033871;
    const Real t = 1.0;

    auto spot = ext::make_shared<SimpleQuote>(s);
    auto qSq = ext::make_shared<SimpleQuote>(q);
    auto rSq = ext::make_shared<SimpleQuote>(r);
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, Handle<Quote>(qSq), dc));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, Handle<Quote>(rSq), dc));

    auto vol25PutSq = ext::make_shared<SimpleQuote>(0.10087);
    auto volAtmSq = ext::make_shared<SimpleQuote>(0.08925);
    auto vol25CallSq = ext::make_shared<SimpleQuote>(0.08463);

    Handle<DeltaVolQuote> volAtmQuote(ext::make_shared<DeltaVolQuote>(
        Handle<Quote>(volAtmSq), DeltaVolQuote::Fwd, t, DeltaVolQuote::AtmDeltaNeutral));
    Handle<DeltaVolQuote> vol25PutQuote(ext::make_shared<DeltaVolQuote>(
        -0.25, Handle<Quote>(vol25PutSq), t, DeltaVolQuote::Fwd));
    Handle<DeltaVolQuote> vol25CallQuote(ext::make_shared<DeltaVolQuote>(
        0.25, Handle<Quote>(vol25CallSq), t, DeltaVolQuote::Fwd));

    Date exDate = today + Period(365, Days);
    auto exercise = ext::make_shared<EuropeanExercise>(exDate);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 1.30);
    DoubleBarrierOption dbo(DoubleBarrier::KnockOut, 1.1, 1.5, 0.0, payoff, exercise);
    auto eng = ext::make_shared<VannaVolgaDoubleBarrierEngine<AnalyticDoubleBarrierEngine>>(
        volAtmQuote, vol25PutQuote, vol25CallQuote, Handle<Quote>(spot), rTS, qTS, false, 0.0, 5);
    dbo.setPricingEngine(eng);
    emit("vv_double_ko_call", dbo.NPV());

    DoubleBarrierOption dboKi(DoubleBarrier::KnockIn, 1.1, 1.5, 0.0, payoff, exercise);
    auto engKi = ext::make_shared<VannaVolgaDoubleBarrierEngine<AnalyticDoubleBarrierEngine>>(
        volAtmQuote, vol25PutQuote, vol25CallQuote, Handle<Quote>(spot), rTS, qTS, false, 0.0, 5);
    dboKi.setPricingEngine(engKi);
    emit("vv_double_ki_call", dboKi.NPV());
}

// ---------------------------------------------------------------------
// (b) Heston geometric-average Asian + Heston forward european.
//     Canonical params from the QuantLib test suite (Kim-Wee table).
// ---------------------------------------------------------------------
void block_heston_asian_forward() {
    const Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;
    const DayCounter dc = Actual365Fixed();

    const Real s0 = 100.0;
    const Real v0 = 0.09;
    const Real kappa = 1.15;
    const Real theta = 0.0348;
    const Real sigma = 0.39;
    const Real rho = -0.64;
    const Real rRate = 0.05;
    const Real qRate = 0.0;

    auto spot = ext::make_shared<SimpleQuote>(s0);
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, rRate, dc));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, qRate, dc));
    auto process = ext::make_shared<HestonProcess>(rTS, qTS, Handle<Quote>(spot),
                                                   v0, kappa, theta, sigma, rho);

    // Continuous geometric-average price Asian, 1y, strike 100, call.
    Date exDate = today + Period(365, Days);
    auto exercise = ext::make_shared<EuropeanExercise>(exDate);
    auto payoffC = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);

    ContinuousAveragingAsianOption contAsian(Average::Geometric, payoffC, exercise);
    contAsian.setPricingEngine(
        ext::make_shared<AnalyticContinuousGeometricAveragePriceAsianHestonEngine>(process));
    emit("heston_cont_geom_asian_call", contAsian.NPV());

    // Discrete geometric-average price Asian: monthly fixings over 1y.
    std::vector<Date> fixingDates;
    for (Size i = 1; i <= 12; ++i)
        fixingDates.push_back(today + Period(i, Months));
    DiscreteAveragingAsianOption discAsian(Average::Geometric, 1.0, 0.0,
                                           fixingDates, payoffC, exercise);
    discAsian.setPricingEngine(
        ext::make_shared<AnalyticDiscreteGeometricAveragePriceAsianHestonEngine>(process));
    emit("heston_discr_geom_asian_call", discAsian.NPV());

    // Heston forward-starting european: reset at 0.5y, expiry 1y, strike-ratio 1.1.
    Date resetDate = today + Period(182, Days);
    auto payoffFwd = ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0);  // strike set by reset
    ForwardVanillaOption fwdOpt(1.1, resetDate, payoffFwd, exercise);
    fwdOpt.setPricingEngine(
        ext::make_shared<AnalyticHestonForwardEuropeanEngine>(process));
    emit("heston_forward_call", fwdOpt.NPV());
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_black_delta();
    block_vanna_volga_single();
    block_vanna_volga_double();
    block_heston_asian_forward();

    // Closing entry (no trailing comma)
    emit("_schema", 1.0, false);
    std::cout << "}\n";
    return 0;
}
