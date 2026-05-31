// Phase 11 W11-B cluster probe: MarketModels (BGM) remaining products +
// pathwise (Greeks-aware) products. Builds on W9+W10+W11-A.
//
//   * Deterministic next_time_step cash-flow generation for the remaining
//     multistep/onestep products (CoinitialSwaps, Ratchet, InverseFloater,
//     Tarn, Swaption, PeriodCapletSwaptions, OneStepCoinitialSwaps,
//     OneStepCoterminalSwaps), driven by an LMMCurveState set on a flat
//     forward (0.05, rate times {0.5,1,1.5,2}). TIGHT.
//   * MultiStepTarn / MultiStepRatchet path-dependent accumulation logic over
//     a full deterministic step sequence (the flat state is reused each step,
//     so the accumulation is deterministic). TIGHT.
//   * Pathwise products' next_time_step populating amount[] = value + per-rate
//     derivatives (MultiCaplet/DeflatedCaplet, Swap, CoterminalSwaptionsDeflated
//     + its numerical-FD twin, InverseFloater, CashRebate). TIGHT.
//   * MultiProductPathwiseWrapper wrapping a MarketModelPathwiseMultiCaplet ->
//     ordinary CashFlow.amount equals the pathwise amount[0]. TIGHT.
//
// C++ parity:
//   ql/models/marketmodels/products/multistep/{multistepcoinitialswaps,
//     multistepratchet,multistepinversefloater,multisteptarn,multistepswaption,
//     multistepperiodcapletswaptions,multisteppathwisewrapper}.hpp
//   ql/models/marketmodels/products/onestep/{onestepcoinitialswaps,
//     onestepcoterminalswaps}.hpp
//   ql/models/marketmodels/products/pathwise/{pathwiseproductcaplet,
//     pathwiseproductswap,pathwiseproductswaption,pathwiseproductinversefloater,
//     pathwiseproductcashrebate,pathwiseproductcallspecified}.hpp
//   @ v1.42.1 (099987f0).

#include <ql/instruments/payoffs.hpp>
#include <ql/math/matrix.hpp>
#include <ql/models/marketmodels/curvestate.hpp>
#include <ql/models/marketmodels/curvestates/lmmcurvestate.hpp>
#include <ql/models/marketmodels/evolutiondescription.hpp>
#include <ql/models/marketmodels/multiproduct.hpp>
#include <ql/models/marketmodels/pathwisemultiproduct.hpp>
#include <ql/models/marketmodels/products/multistep/multistepcoinitialswaps.hpp>
#include <ql/models/marketmodels/products/multistep/multistepinversefloater.hpp>
#include <ql/models/marketmodels/products/multistep/multisteppathwisewrapper.hpp>
#include <ql/models/marketmodels/products/multistep/multistepperiodcapletswaptions.hpp>
#include <ql/models/marketmodels/products/multistep/multistepratchet.hpp>
#include <ql/models/marketmodels/products/multistep/multistepswaption.hpp>
#include <ql/models/marketmodels/products/multistep/multisteptarn.hpp>
#include <ql/models/marketmodels/products/onestep/onestepcoinitialswaps.hpp>
#include <ql/models/marketmodels/products/onestep/onestepcoterminalswaps.hpp>
#include <ql/models/marketmodels/products/pathwise/pathwiseproductcaplet.hpp>
#include <ql/models/marketmodels/products/pathwise/pathwiseproductcashrebate.hpp>
#include <ql/models/marketmodels/products/pathwise/pathwiseproductinversefloater.hpp>
#include <ql/models/marketmodels/products/pathwise/pathwiseproductswap.hpp>
#include <ql/models/marketmodels/products/pathwise/pathwiseproductswaption.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
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

std::vector<Time> rateTimes4() { return {0.5, 1.0, 1.5, 2.0}; }
constexpr Real kF = 0.05;

LMMCurveState flatState() {
    LMMCurveState cs(rateTimes4());
    cs.setOnForwardRates(std::vector<Rate>(3, kF));
    return cs;
}

using MP = MarketModelMultiProduct;
using PW = MarketModelPathwiseMultiProduct;

std::vector<std::vector<MP::CashFlow> > mpBuf(Size np, Size mx) {
    return std::vector<std::vector<MP::CashFlow> >(
        np, std::vector<MP::CashFlow>(mx));
}

std::vector<std::vector<PW::CashFlow> > pwBuf(Size np, Size mx, Size rates) {
    std::vector<std::vector<PW::CashFlow> > g(np, std::vector<PW::CashFlow>(mx));
    for (auto& i : g)
        for (auto& j : i)
            j.amount.resize(rates + 1);
    return g;
}

// ---------------------------------------------------------------------------
// (a)+(b): remaining multistep / onestep products on the flat-forward state.
// ---------------------------------------------------------------------------
void block_remaining_products() {
    auto rt = rateTimes4();
    std::vector<Real> accruals = {0.5, 0.5, 0.5};
    std::vector<Time> payTimes = {1.0, 1.5, 2.0};
    LMMCurveState cs = flatState();

    // --- MultiStepCoinitialSwaps: step 0 active for all products -----------
    {
        MultiStepCoinitialSwaps p(rt, accruals, accruals, payTimes, 0.045);
        Size np = p.numberOfProducts();
        auto n = std::vector<Size>(np, 0);
        auto gen = mpBuf(np, 2);
        p.reset();
        bool done0 = p.nextTimeStep(cs, n, gen);
        emit("coin_np", (Real)np);
        emit("coin_done0", done0 ? 1.0 : 0.0);
        emit("coin_n0_step0", (Real)n[0]);
        emit("coin_n2_step0", (Real)n[2]);
        emit("coin_fixed00", gen[0][0].amount);
        emit("coin_float00", gen[0][1].amount);
        emit("coin_ti00", (Real)gen[0][0].timeIndex);
        // step 1: products 1,2 active
        bool done1 = p.nextTimeStep(cs, n, gen);
        emit("coin_n0_step1", (Real)n[0]);
        emit("coin_n1_step1", (Real)n[1]);
        emit("coin_fixed11", gen[1][0].amount);
        (void)done1;
    }

    // --- MultiStepInverseFloater -------------------------------------------
    {
        std::vector<Real> strikes = {0.06, 0.06, 0.06};
        std::vector<Real> mults = {1.0, 1.0, 1.0};
        std::vector<Real> spreads = {0.001, 0.001, 0.001};
        MultiStepInverseFloater p(rt, accruals, accruals, strikes, mults, spreads,
                                  payTimes, true);
        auto n = std::vector<Size>(1, 0);
        auto gen = mpBuf(1, 1);
        p.reset();
        p.nextTimeStep(cs, n, gen);
        emit("invf_n0", (Real)n[0]);
        emit("invf_ti0", (Real)gen[0][0].timeIndex);
        emit("invf_amt0", gen[0][0].amount);
        p.nextTimeStep(cs, n, gen);
        emit("invf_amt1", gen[0][0].amount);
    }

    // --- MultiStepRatchet: path-dependent floor accumulation ---------------
    {
        // gearingFloor=1, gearingFixing=1, spreadFloor=0, spreadFixing=0,
        // initialFloor=0.04, payer. Flat libor 0.05 => coupon=max(floor,0.05).
        MultiStepRatchet p(rt, accruals, payTimes, 1.0, 1.0, 0.0, 0.0, 0.04, true);
        auto n = std::vector<Size>(1, 0);
        auto gen = mpBuf(1, 1);
        p.reset();
        std::vector<Real> ratchetAmts;
        bool done = false;
        for (int s = 0; s < 3 && !done; ++s) {
            done = p.nextTimeStep(cs, n, gen);
            ratchetAmts.push_back(gen[0][0].amount);
        }
        emit_arr("ratchet_amts", ratchetAmts);
        emit("ratchet_done", done ? 1.0 : 0.0);
    }

    // --- MultiStepTarn: path-dependent target redemption -------------------
    {
        // 3 rates; totalCoupon small so it terminates early.
        std::vector<Real> strikes = {0.06, 0.06, 0.06};
        std::vector<Real> mults = {1.0, 1.0, 1.0};
        std::vector<Real> spreads = {0.0, 0.0, 0.0};
        Real totalCoupon = 0.02;  // obvious coupon per step = (0.06-0.05)*0.5=0.005
        MultiStepTarn p(rt, accruals, accruals, payTimes, payTimes, totalCoupon,
                        strikes, mults, spreads);
        auto n = std::vector<Size>(1, 0);
        auto gen = mpBuf(1, 2);
        p.reset();
        emit("tarn_np", (Real)p.numberOfProducts());
        emit("tarn_ncf", (Real)p.maxNumberOfCashFlowsPerProductPerStep());
        std::vector<Real> tarnFloat, tarnFixed;
        std::vector<Real> tarnFloatTi, tarnFixedTi;
        std::vector<Real> tarnDone;
        bool done = false;
        for (int s = 0; s < 3 && !done; ++s) {
            done = p.nextTimeStep(cs, n, gen);
            tarnFloat.push_back(gen[0][0].amount);
            tarnFixed.push_back(gen[0][1].amount);
            tarnFloatTi.push_back((Real)gen[0][0].timeIndex);
            tarnFixedTi.push_back((Real)gen[0][1].timeIndex);
            tarnDone.push_back(done ? 1.0 : 0.0);
        }
        emit_arr("tarn_float", tarnFloat);
        emit_arr("tarn_fixed", tarnFixed);
        emit_arr("tarn_float_ti", tarnFloatTi);
        emit_arr("tarn_fixed_ti", tarnFixedTi);
        emit_arr("tarn_done", tarnDone);
        emit("tarn_steps", (Real)tarnFloat.size());
    }

    // --- MultiStepSwaption: payoff(cmSwapRate)*annuity at startIndex --------
    {
        ext::shared_ptr<StrikedTypePayoff> payoff(
            new PlainVanillaPayoff(Option::Call, 0.04));
        // startIndex=1, endIndex=3 -> swap spanning rates 1..2 (span=2).
        MultiStepSwaption p(rt, 1, 3, payoff);
        emit("swpt_np", (Real)p.numberOfProducts());
        emit("swpt_cmrate", cs.cmSwapRate(1, 2));
        emit("swpt_cmann", cs.cmSwapAnnuity(1, 1, 2));
        auto n = std::vector<Size>(1, 0);
        auto gen = mpBuf(1, 1);
        p.reset();
        // currentIndex_ starts at 0 != startIndex(1): first step is a no-op.
        bool d0 = p.nextTimeStep(cs, n, gen);
        emit("swpt_done0", d0 ? 1.0 : 0.0);
        emit("swpt_n0_step0", (Real)n[0]);
        bool d1 = p.nextTimeStep(cs, n, gen);
        emit("swpt_done1", d1 ? 1.0 : 0.0);
        emit("swpt_n0_step1", (Real)n[0]);
        emit("swpt_amt", gen[0][0].amount);
        emit("swpt_ti", (Real)gen[0][0].timeIndex);
    }

    // --- MultiStepPeriodCapletSwaptions ------------------------------------
    {
        // 3 FRAs, period 1, offset 0 -> numberBigFRAs=3. One fwd/swap payoff,
        // payment time per big FRA.
        std::vector<Time> fwdOptPay = {1.0, 1.5, 2.0};
        std::vector<Time> swpPay = {1.0, 1.5, 2.0};
        std::vector<ext::shared_ptr<StrikedTypePayoff> > fwdPO(3), swpPO(3);
        for (Size i = 0; i < 3; ++i) {
            fwdPO[i].reset(new PlainVanillaPayoff(Option::Call, 0.04));
            swpPO[i].reset(new PlainVanillaPayoff(Option::Call, 0.04));
        }
        MultiStepPeriodCapletSwaptions p(rt, fwdOptPay, swpPay, fwdPO, swpPO, 1, 0);
        Size np = p.numberOfProducts();
        auto n = std::vector<Size>(np, 0);
        auto gen = mpBuf(np, 1);
        p.reset();
        emit("pcs_np", (Real)np);
        bool d0 = p.nextTimeStep(cs, n, gen);
        emit("pcs_done0", d0 ? 1.0 : 0.0);
        emit("pcs_caplet0_n", (Real)n[0]);
        emit("pcs_caplet0_amt", gen[0][0].amount);
        emit("pcs_swaption0_n", (Real)n[3]);
        emit("pcs_swaption0_amt", gen[3][0].amount);
    }

    // --- OneStepCoinitialSwaps ---------------------------------------------
    {
        OneStepCoinitialSwaps p(rt, accruals, accruals, payTimes, 0.045);
        Size np = p.numberOfProducts();
        Size mx = p.maxNumberOfCashFlowsPerProductPerStep();
        auto n = std::vector<Size>(np, 0);
        auto gen = mpBuf(np, mx);
        p.reset();
        bool done = p.nextTimeStep(cs, n, gen);
        emit("oscoin_np", (Real)np);
        emit("oscoin_mx", (Real)mx);
        emit("oscoin_done", done ? 1.0 : 0.0);
        emit("oscoin_n0", (Real)n[0]);
        emit("oscoin_n2", (Real)n[2]);
        // product 2 (longest) gets pairs for indexOfTime 0,1,2.
        emit("oscoin_p2_fixed0", gen[2][0].amount);
        emit("oscoin_p2_float0", gen[2][1].amount);
        emit("oscoin_p2_ti0", (Real)gen[2][0].timeIndex);
        emit("oscoin_p2_ti2", (Real)gen[2][4].timeIndex);
        // product 0 (shortest) gets only indexOfTime 0.
        emit("oscoin_p0_fixed0", gen[0][0].amount);
    }

    // --- OneStepCoterminalSwaps --------------------------------------------
    {
        OneStepCoterminalSwaps p(rt, accruals, accruals, payTimes, 0.045);
        Size np = p.numberOfProducts();
        Size mx = p.maxNumberOfCashFlowsPerProductPerStep();
        auto n = std::vector<Size>(np, 0);
        auto gen = mpBuf(np, mx);
        p.reset();
        bool done = p.nextTimeStep(cs, n, gen);
        emit("oscot_np", (Real)np);
        emit("oscot_mx", (Real)mx);
        emit("oscot_done", done ? 1.0 : 0.0);
        emit("oscot_n0", (Real)n[0]);
        emit("oscot_n2", (Real)n[2]);
        // product 0 (longest) gets pairs for indexOfTime 0,1,2.
        emit("oscot_p0_fixed0", gen[0][0].amount);
        emit("oscot_p0_ti0", (Real)gen[0][0].timeIndex);
        emit("oscot_p0_ti2", (Real)gen[0][4].timeIndex);
        // product 2 (shortest): only indexOfTime 2, at slot (2-2)*2=0.
        emit("oscot_p2_fixed0", gen[2][0].amount);
        emit("oscot_p2_ti0", (Real)gen[2][0].timeIndex);
    }
}

// ---------------------------------------------------------------------------
// (c): pathwise products. amount[0] = value; amount[i+1] = d value / d f_i.
// ---------------------------------------------------------------------------
void block_pathwise_products() {
    auto rt = rateTimes4();
    std::vector<Real> accruals = {0.5, 0.5, 0.5};
    std::vector<Time> payTimes = {1.0, 1.5, 2.0};
    LMMCurveState cs = flatState();
    Size rates = 3;

    // --- MarketModelPathwiseMultiCaplet (undeflated) -----------------------
    {
        std::vector<Rate> strikes = {0.04, 0.045, 0.06};  // ITM, ITM, OTM
        MarketModelPathwiseMultiCaplet p(rt, accruals, payTimes, strikes);
        emit("pwcap_deflated", p.alreadyDeflated() ? 1.0 : 0.0);
        emit("pwcap_np", (Real)p.numberOfProducts());
        auto n = std::vector<Size>(p.numberOfProducts(), 0);
        auto gen = pwBuf(p.numberOfProducts(), 1, rates);
        p.reset();
        p.nextTimeStep(cs, n, gen);  // step 0: ITM
        emit("pwcap_n0", (Real)n[0]);
        emit("pwcap_amt0_val", gen[0][0].amount[0]);
        emit("pwcap_amt0_d1", gen[0][0].amount[1]);
        emit("pwcap_amt0_d2", gen[0][0].amount[2]);
        emit("pwcap_amt0_d3", gen[0][0].amount[3]);
        p.nextTimeStep(cs, n, gen);  // step 1: ITM
        emit("pwcap_amt1_val", gen[1][0].amount[0]);
        emit("pwcap_amt1_d2", gen[1][0].amount[2]);
        bool done2 = p.nextTimeStep(cs, n, gen);  // step 2: OTM => no cash flow
        emit("pwcap_done2", done2 ? 1.0 : 0.0);
        emit("pwcap_n2", (Real)n[2]);
    }

    // --- MarketModelPathwiseMultiDeflatedCaplet ----------------------------
    {
        std::vector<Rate> strikes = {0.04, 0.045, 0.06};
        MarketModelPathwiseMultiDeflatedCaplet p(rt, accruals, payTimes, strikes);
        emit("pwdcap_deflated", p.alreadyDeflated() ? 1.0 : 0.0);
        auto n = std::vector<Size>(p.numberOfProducts(), 0);
        auto gen = pwBuf(p.numberOfProducts(), 1, rates);
        p.reset();
        p.nextTimeStep(cs, n, gen);
        emit("pwdcap_n0", (Real)n[0]);
        emit("pwdcap_amt0_val", gen[0][0].amount[0]);
        emit("pwdcap_amt0_d1", gen[0][0].amount[1]);
        emit("pwdcap_amt0_d2", gen[0][0].amount[2]);
    }

    // --- MarketModelPathwiseSwap -------------------------------------------
    {
        std::vector<Rate> strikes = {0.045, 0.045, 0.045};
        MarketModelPathwiseSwap p(rt, accruals, strikes, 1.0);
        emit("pwswap_deflated", p.alreadyDeflated() ? 1.0 : 0.0);
        emit("pwswap_np", (Real)p.numberOfProducts());
        auto n = std::vector<Size>(1, 0);
        auto gen = pwBuf(1, 1, rates);
        p.reset();
        p.nextTimeStep(cs, n, gen);
        emit("pwswap_n0", (Real)n[0]);
        emit("pwswap_ti0", (Real)gen[0][0].timeIndex);
        emit("pwswap_amt0_val", gen[0][0].amount[0]);
        emit("pwswap_amt0_d1", gen[0][0].amount[1]);
        emit("pwswap_amt0_d2", gen[0][0].amount[2]);
        p.nextTimeStep(cs, n, gen);
        emit("pwswap_amt1_val", gen[0][0].amount[0]);
        emit("pwswap_amt1_d2", gen[0][0].amount[2]);
    }

    // --- MarketModelPathwiseCoterminalSwaptionsDeflated --------------------
    {
        std::vector<Rate> strikes = {0.04, 0.04, 0.04};
        MarketModelPathwiseCoterminalSwaptionsDeflated p(rt, strikes);
        emit("pwswpt_np", (Real)p.numberOfProducts());
        auto n = std::vector<Size>(p.numberOfProducts(), 0);
        auto gen = pwBuf(p.numberOfProducts(), 1, rates);
        p.reset();
        p.nextTimeStep(cs, n, gen);
        emit("pwswpt_n0", (Real)n[0]);
        emit("pwswpt_amt0_val", gen[0][0].amount[0]);
        emit("pwswpt_amt0_d1", gen[0][0].amount[1]);
        emit("pwswpt_amt0_d2", gen[0][0].amount[2]);
        emit("pwswpt_amt0_d3", gen[0][0].amount[3]);
    }

    // --- MarketModelPathwiseCoterminalSwaptionsNumericalDeflated (FD twin) --
    {
        std::vector<Rate> strikes = {0.04, 0.04, 0.04};
        MarketModelPathwiseCoterminalSwaptionsNumericalDeflated p(rt, strikes,
                                                                  1e-6);
        auto n = std::vector<Size>(p.numberOfProducts(), 0);
        auto gen = pwBuf(p.numberOfProducts(), 1, rates);
        p.reset();
        p.nextTimeStep(cs, n, gen);
        emit("pwswptnum_amt0_val", gen[0][0].amount[0]);
        emit("pwswptnum_amt0_d1", gen[0][0].amount[1]);
        emit("pwswptnum_amt0_d2", gen[0][0].amount[2]);
        emit("pwswptnum_amt0_d3", gen[0][0].amount[3]);
    }

    // --- MarketModelPathwiseInverseFloater ---------------------------------
    {
        std::vector<Real> strikes = {0.06, 0.06, 0.06};
        std::vector<Real> mults = {1.0, 1.0, 1.0};
        std::vector<Real> spreads = {0.001, 0.001, 0.001};
        MarketModelPathwiseInverseFloater p(rt, accruals, accruals, strikes,
                                            mults, spreads, payTimes, true);
        emit("pwinvf_np", (Real)p.numberOfProducts());
        auto n = std::vector<Size>(1, 0);
        auto gen = pwBuf(1, 1, rates);
        p.reset();
        p.nextTimeStep(cs, n, gen);
        emit("pwinvf_n0", (Real)n[0]);
        emit("pwinvf_ti0", (Real)gen[0][0].timeIndex);
        emit("pwinvf_amt0_val", gen[0][0].amount[0]);
        emit("pwinvf_amt0_d1", gen[0][0].amount[1]);
        emit("pwinvf_amt0_d2", gen[0][0].amount[2]);
    }

    // --- MarketModelPathwiseCashRebate -------------------------------------
    {
        std::vector<Time> rebPay = {0.5, 1.0, 1.5};
        EvolutionDescription evolution(rt, rebPay);
        Matrix amounts(2, 3);
        amounts[0][0] = 1.0; amounts[0][1] = 2.0; amounts[0][2] = 3.0;
        amounts[1][0] = 10.0; amounts[1][1] = 20.0; amounts[1][2] = 30.0;
        MarketModelPathwiseCashRebate p(evolution, rebPay, amounts, 2);
        emit("pwreb_deflated", p.alreadyDeflated() ? 1.0 : 0.0);
        auto n = std::vector<Size>(2, 0);
        auto gen = pwBuf(2, 1, rates);
        p.reset();
        bool done = p.nextTimeStep(cs, n, gen);
        emit("pwreb_done", done ? 1.0 : 0.0);
        emit("pwreb_n0", (Real)n[0]);
        emit("pwreb_ti00", (Real)gen[0][0].timeIndex);
        emit("pwreb_amt00_val", gen[0][0].amount[0]);
        emit("pwreb_amt00_d1", gen[0][0].amount[1]);
        emit("pwreb_amt10_val", gen[1][0].amount[0]);
    }

    // --- MultiProductPathwiseWrapper(caplet) -------------------------------
    {
        std::vector<Rate> strikes = {0.04, 0.045, 0.06};
        MarketModelPathwiseMultiCaplet inner(rt, accruals, payTimes, strikes);
        MultiProductPathwiseWrapper p(inner);
        emit("wrap_np", (Real)p.numberOfProducts());
        emit("wrap_mx", (Real)p.maxNumberOfCashFlowsPerProductPerStep());
        Size np = p.numberOfProducts();
        auto n = std::vector<Size>(np, 0);
        auto gen = mpBuf(np, p.maxNumberOfCashFlowsPerProductPerStep());
        p.reset();
        p.nextTimeStep(cs, n, gen);  // step 0: ITM caplet
        emit("wrap_n0", (Real)n[0]);
        emit("wrap_amt0", gen[0][0].amount);
        emit("wrap_ti0", (Real)gen[0][0].timeIndex);
        p.nextTimeStep(cs, n, gen);  // step 1
        emit("wrap_amt1", gen[1][0].amount);
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    block_remaining_products();
    block_pathwise_products();

    std::cout << "\n}\n";
    return 0;
}
