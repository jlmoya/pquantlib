// Phase 11 W11-A cluster probe: MarketModels (BGM) MultiProduct framework +
// core concrete products. The LAST marketmodels wave.
//
//   * THE CANONICAL BGM END-TO-END TEST. Reproduces the marketmodel.cpp
//     `testMultiStepForwardsAndOptionlets` setup (Semiannual schedule to 5y,
//     forwards 0.03 + 0.001*i, abcd-shaped market caplet vols, displacement 0).
//     We emit the full deterministic setup (rateTimes, accruals, todaysForwards,
//     todaysDiscounts, the per-rate flat `volatilities`) PLUS the analytic
//     reference values: expected forward FRA = (f-K)*accrual*P(t_{i+1}), and
//     expected caplet = BlackCalculator(payoff, f, vol*sqrt(rateTimes[i]),
//     P(t_{i+1})*accrual). The Python test reproduces the EXACT setup, builds
//     a MultiProductComposite(MultiStepForwards + MultiStepOptionlets), runs it
//     through AccountingEngine(LogNormalFwdRatePc(FlatVol(...))) with seed 42,
//     and checks the MC means fall within standard-error bounds of these
//     references. This validates the WHOLE W9+W10+W11 marketmodels stack. The
//     reference values are tier EXACT (closed-form Black); the MC comparison is
//     the C++ 2.5-standard-error band (LOOSE / MC).
//
//   * Deterministic next_time_step cash-flow generation for each concrete
//     product, driven by an LMMCurveState set on a flat forward (0.05 for the
//     4-rate world, then deterministic coterminal swap rate / annuity). TIGHT.
//
//   * MarketModelCashRebate.nextTimeStep — fixed-amount one-shot. TIGHT.
//
// C++ parity:
//   ql/models/marketmodels/products/multiproductmultistep.hpp
//   ql/models/marketmodels/products/multiproductonestep.hpp
//   ql/models/marketmodels/products/multiproductcomposite.hpp
//   ql/models/marketmodels/products/singleproductcomposite.hpp
//   ql/models/marketmodels/products/multistep/{multistepnothing,multistepforwards,
//     multistepoptionlets,multistepswap,multistepcoterminalswaps,
//     multistepcoterminalswaptions,cashrebate,exerciseadapter,
//     callspecifiedmultiproduct}.hpp
//   ql/models/marketmodels/products/onestep/{onestepforwards,onestepoptionlets}.hpp
//   test-suite/marketmodel.cpp testMultiStepForwardsAndOptionlets
//   @ v1.42.1 (099987f0).

#include <ql/instruments/payoffs.hpp>
#include <ql/models/marketmodels/curvestate.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/multiproduct.hpp>
#include <ql/models/marketmodels/products/multistep/cashrebate.hpp>
#include <ql/models/marketmodels/products/multistep/multistepcoterminalswaps.hpp>
#include <ql/models/marketmodels/products/multistep/multistepcoterminalswaptions.hpp>
#include <ql/models/marketmodels/products/multistep/multistepforwards.hpp>
#include <ql/models/marketmodels/products/multistep/multistepoptionlets.hpp>
#include <ql/models/marketmodels/products/multistep/multistepswap.hpp>
#include <ql/models/marketmodels/products/onestep/onestepforwards.hpp>
#include <ql/models/marketmodels/products/onestep/onestepoptionlets.hpp>
#include <ql/pricingengines/blackcalculator.hpp>
#include <ql/math/matrix.hpp>
#include <ql/settings.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/simpledaycounter.hpp>
#include <ql/time/schedule.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace QuantLib;

namespace {

bool g_first = true;

void emit(const char* name, Real v) {
    if (!g_first) std::cout << ",\n";
    g_first = false;
    std::cout << "  \"" << name << "\": " << v;
}

void emit_arr(const char* name, const std::vector<Real>& v) {
    if (!g_first) std::cout << ",\n";
    g_first = false;
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

// ---------------------------------------------------------------------------
// Canonical marketmodel.cpp setup (testMultiStepForwardsAndOptionlets).
// ---------------------------------------------------------------------------
struct Canonical {
    std::vector<Time> rateTimes, paymentTimes;
    std::vector<Real> accruals;
    std::vector<Rate> todaysForwards;
    std::vector<DiscountFactor> todaysDiscounts;
    std::vector<Volatility> volatilities;
    Real displacement = 0.0;
};

Canonical makeCanonical() {
    Canonical s;
    Calendar calendar = NullCalendar();
    Date todaysDate = Settings::instance().evaluationDate();
    Date endDate = todaysDate + 5 * Years;
    Schedule dates(todaysDate, endDate, Period(Semiannual), calendar, Following,
                   Following, DateGeneration::Backward, false);
    s.rateTimes = std::vector<Time>(dates.size() - 1);
    s.paymentTimes = std::vector<Time>(s.rateTimes.size() - 1);
    s.accruals = std::vector<Real>(s.rateTimes.size() - 1);
    SimpleDayCounter dayCounter;
    for (Size i = 1; i < dates.size(); ++i)
        s.rateTimes[i - 1] = dayCounter.yearFraction(todaysDate, dates[i]);
    std::copy(s.rateTimes.begin() + 1, s.rateTimes.end(), s.paymentTimes.begin());
    for (Size i = 1; i < s.rateTimes.size(); ++i)
        s.accruals[i - 1] = s.rateTimes[i] - s.rateTimes[i - 1];

    s.todaysForwards = std::vector<Rate>(s.paymentTimes.size());
    for (Size i = 0; i < s.todaysForwards.size(); ++i)
        s.todaysForwards[i] = 0.03 + 0.0010 * i;

    s.todaysDiscounts = std::vector<DiscountFactor>(s.rateTimes.size());
    s.todaysDiscounts[0] = 0.95;
    for (Size i = 1; i < s.rateTimes.size(); ++i)
        s.todaysDiscounts[i] =
            s.todaysDiscounts[i - 1] / (1.0 + s.todaysForwards[i - 1] * s.accruals[i - 1]);

    Volatility mktVols[] = {0.15541283, 0.18719678, 0.20890740, 0.22318179,
                            0.23212717, 0.23731450, 0.23988649, 0.24066384,
                            0.24023111, 0.23900189, 0.23726699, 0.23522952,
                            0.23303022, 0.23076564, 0.22850101, 0.22627951,
                            0.22412881, 0.22206569, 0.22009939};
    s.volatilities = std::vector<Volatility>(s.todaysForwards.size());
    for (Size i = 0; i < std::min(std::size(mktVols), s.todaysForwards.size()); ++i)
        s.volatilities[i] = s.todaysForwards[i] * mktVols[i] /
                            (s.todaysForwards[i] + s.displacement);
    return s;
}

void block_canonical() {
    Canonical s = makeCanonical();
    Size N = s.todaysForwards.size();

    emit_arr("can_rate_times", s.rateTimes);
    emit_arr("can_payment_times", s.paymentTimes);
    emit_arr("can_accruals", s.accruals);
    emit_arr("can_forwards", s.todaysForwards);
    emit_arr("can_discounts", s.todaysDiscounts);
    emit_arr("can_volatilities", s.volatilities);
    emit("can_displacement", s.displacement);
    emit("can_n", (Real)N);

    // Forward strikes = f + 0.01; optionlet payoffs = Call struck at f.
    std::vector<Real> expectedForwards(N), expectedCaplets(N), forwardStrikes(N);
    for (Size i = 0; i < N; ++i) {
        forwardStrikes[i] = s.todaysForwards[i] + 0.01;
        expectedForwards[i] = (s.todaysForwards[i] - forwardStrikes[i]) *
                              s.accruals[i] * s.todaysDiscounts[i + 1];
        Time expiry = s.rateTimes[i];
        ext::shared_ptr<StrikedTypePayoff> payoff(
            new PlainVanillaPayoff(Option::Call, s.todaysForwards[i]));
        expectedCaplets[i] =
            BlackCalculator(payoff, s.todaysForwards[i] + s.displacement,
                            s.volatilities[i] * std::sqrt(expiry),
                            s.todaysDiscounts[i + 1] * s.accruals[i])
                .value();
    }
    emit_arr("can_forward_strikes", forwardStrikes);
    emit_arr("can_expected_forwards", expectedForwards);
    emit_arr("can_expected_caplets", expectedCaplets);
}

// ---------------------------------------------------------------------------
// Deterministic per-product next_time_step probes on a flat-forward state.
//   rateTimes = {0.5, 1.0, 1.5, 2.0}, flat forward 0.05.
// ---------------------------------------------------------------------------
std::vector<Time> rateTimes4() { return {0.5, 1.0, 1.5, 2.0}; }
constexpr Real kF = 0.05;

LMMCurveState flatState() {
    LMMCurveState cs(rateTimes4());
    cs.setOnForwardRates(std::vector<Rate>(3, kF));
    return cs;
}

void block_products_next_step() {
    auto rt = rateTimes4();
    std::vector<Real> accruals = {0.5, 0.5, 0.5};   // taus 1..3
    std::vector<Time> payTimes = {1.0, 1.5, 2.0};   // rateTimes[1:]
    std::vector<Rate> strikes = {0.04, 0.045, 0.05};
    LMMCurveState cs = flatState();

    // --- MultiStepForwards: step 0 generates product 0's flow ---------------
    {
        MultiStepForwards fwd(rt, accruals, payTimes, strikes);
        Size np = fwd.numberOfProducts();
        std::vector<Size> n(np, 0);
        std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
            np, std::vector<MarketModelMultiProduct::CashFlow>(1));
        fwd.reset();
        bool done0 = fwd.nextTimeStep(cs, n, gen);
        emit("fwd_done0", done0 ? 1.0 : 0.0);
        emit("fwd_n0", (Real)n[0]);
        emit("fwd_ti00", (Real)gen[0][0].timeIndex);
        emit("fwd_amt00", gen[0][0].amount);
        bool done1 = fwd.nextTimeStep(cs, n, gen);
        emit("fwd_amt11", gen[1][0].amount);
        bool done2 = fwd.nextTimeStep(cs, n, gen);
        emit("fwd_done2", done2 ? 1.0 : 0.0);
        emit("fwd_amt22", gen[2][0].amount);
        (void)done1;
    }

    // --- MultiStepOptionlets: caplet payoff * accrual -----------------------
    {
        std::vector<ext::shared_ptr<Payoff> > payoffs(3);
        for (Size i = 0; i < 3; ++i)
            payoffs[i].reset(new PlainVanillaPayoff(Option::Call, strikes[i]));
        MultiStepOptionlets opt(rt, accruals, payTimes, payoffs);
        Size np = opt.numberOfProducts();
        std::vector<Size> n(np, 0);
        std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
            np, std::vector<MarketModelMultiProduct::CashFlow>(1));
        opt.reset();
        opt.nextTimeStep(cs, n, gen);
        emit("opt_n0", (Real)n[0]);
        emit("opt_amt00", gen[0][0].amount);
        opt.nextTimeStep(cs, n, gen);
        emit("opt_amt11", gen[1][0].amount);
        bool done2 = opt.nextTimeStep(cs, n, gen);
        emit("opt_done2", done2 ? 1.0 : 0.0);
        emit("opt_amt22", gen[2][0].amount);
    }

    // --- MultiStepSwap: two flows per step (fixed + floating) ---------------
    {
        std::vector<Real> fixedAcc = {0.5, 0.5, 0.5};
        std::vector<Real> floatAcc = {0.5, 0.5, 0.5};
        Real fixedRate = 0.045;
        MultiStepSwap swap(rt, fixedAcc, floatAcc, payTimes, fixedRate, true);
        std::vector<Size> n(1, 0);
        std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
            1, std::vector<MarketModelMultiProduct::CashFlow>(2));
        swap.reset();
        swap.nextTimeStep(cs, n, gen);
        emit("swap_n0", (Real)n[0]);
        emit("swap_fixed0", gen[0][0].amount);
        emit("swap_float0", gen[0][1].amount);
        // payer=false flips the sign
        MultiStepSwap rec(rt, fixedAcc, floatAcc, payTimes, fixedRate, false);
        rec.reset();
        rec.nextTimeStep(cs, n, gen);
        emit("swap_rec_fixed0", gen[0][0].amount);
        emit("swap_rec_float0", gen[0][1].amount);
    }

    // --- MultiStepCoterminalSwaptions: payoff(swapRate)*annuity -------------
    {
        std::vector<ext::shared_ptr<StrikedTypePayoff> > payoffs(3);
        for (Size i = 0; i < 3; ++i)
            payoffs[i].reset(new PlainVanillaPayoff(Option::Call, 0.04));
        MultiStepCoterminalSwaptions swns(rt, payTimes, payoffs);
        Size np = swns.numberOfProducts();
        std::vector<Size> n(np, 0);
        std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
            np, std::vector<MarketModelMultiProduct::CashFlow>(1));
        swns.reset();
        swns.nextTimeStep(cs, n, gen);
        emit("swns_np", (Real)np);
        emit("swns_amt00", gen[0][0].amount);
        emit("swns_swaprate0", cs.coterminalSwapRate(0));
        emit("swns_annuity0", cs.coterminalSwapAnnuity(0, 0));
    }

    // --- MultiStepCoterminalSwaps: cumulative fixed/float per leg ------------
    {
        std::vector<Real> fixedAcc = {0.5, 0.5, 0.5};
        std::vector<Real> floatAcc = {0.5, 0.5, 0.5};
        MultiStepCoterminalSwaps cs_swaps(rt, fixedAcc, floatAcc, payTimes, 0.045);
        Size np = cs_swaps.numberOfProducts();
        std::vector<Size> n(np, 0);
        std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
            np, std::vector<MarketModelMultiProduct::CashFlow>(2));
        cs_swaps.reset();
        cs_swaps.nextTimeStep(cs, n, gen);  // step 0: only product 0 active
        emit("cotsw_np", (Real)np);
        emit("cotsw_n0_step0", (Real)n[0]);
        emit("cotsw_n1_step0", (Real)n[1]);
        emit("cotsw_fixed00", gen[0][0].amount);
        emit("cotsw_float00", gen[0][1].amount);
    }

    // --- OneStepForwards: all flows in a single step ------------------------
    {
        OneStepForwards osf(rt, accruals, payTimes, strikes);
        Size np = osf.numberOfProducts();
        std::vector<Size> n(np, 0);
        std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
            np, std::vector<MarketModelMultiProduct::CashFlow>(1));
        osf.reset();
        bool done = osf.nextTimeStep(cs, n, gen);
        emit("osf_done", done ? 1.0 : 0.0);
        emit("osf_n0", (Real)n[0]);
        emit("osf_amt00", gen[0][0].amount);
        emit("osf_amt11", gen[1][0].amount);
        emit("osf_amt22", gen[2][0].amount);
    }

    // --- OneStepOptionlets: only positive-payoff flows -----------------------
    {
        // strike below forward -> in the money (payoff>0); strike above -> 0.
        std::vector<ext::shared_ptr<Payoff> > payoffs(3);
        payoffs[0].reset(new PlainVanillaPayoff(Option::Call, 0.04));  // ITM
        payoffs[1].reset(new PlainVanillaPayoff(Option::Call, 0.06));  // OTM
        payoffs[2].reset(new PlainVanillaPayoff(Option::Call, 0.045)); // ITM
        OneStepOptionlets oso(rt, accruals, payTimes, payoffs);
        Size np = oso.numberOfProducts();
        std::vector<Size> n(np, 0);
        std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
            np, std::vector<MarketModelMultiProduct::CashFlow>(1));
        oso.reset();
        bool done = oso.nextTimeStep(cs, n, gen);
        emit("oso_done", done ? 1.0 : 0.0);
        emit("oso_n0", (Real)n[0]);
        emit("oso_n1", (Real)n[1]);   // OTM -> 0 flows
        emit("oso_n2", (Real)n[2]);
        emit("oso_amt00", gen[0][0].amount);
        emit("oso_amt22", gen[2][0].amount);
    }
}

void block_cash_rebate() {
    // 2 products, 3 payment/evolution times, amounts[p][t]. Evolution times
    // must be <= the next-to-last rate time (1.5 here), so we use the first
    // three rate times as payment/evolution times.
    auto rt = rateTimes4();
    std::vector<Time> payTimes = {0.5, 1.0, 1.5};
    EvolutionDescription evolution(rt, payTimes);
    Matrix amounts(2, 3);
    amounts[0][0] = 1.0; amounts[0][1] = 2.0; amounts[0][2] = 3.0;
    amounts[1][0] = 10.0; amounts[1][1] = 20.0; amounts[1][2] = 30.0;
    MarketModelCashRebate rebate(evolution, payTimes, amounts, 2);
    LMMCurveState cs = flatState();
    std::vector<Size> n(2, 0);
    std::vector<std::vector<MarketModelMultiProduct::CashFlow> > gen(
        2, std::vector<MarketModelMultiProduct::CashFlow>(1));
    rebate.reset();
    bool done = rebate.nextTimeStep(cs, n, gen);
    emit("rebate_done", done ? 1.0 : 0.0);
    emit("rebate_n0", (Real)n[0]);
    emit("rebate_ti00", (Real)gen[0][0].timeIndex);
    emit("rebate_amt00", gen[0][0].amount);
    emit("rebate_amt10", gen[1][0].amount);
    emit("rebate_maxcf", (Real)rebate.maxNumberOfCashFlowsPerProductPerStep());
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_canonical();
    block_products_next_step();
    block_cash_rebate();

    std::cout << "\n}\n";
    return 0;
}
