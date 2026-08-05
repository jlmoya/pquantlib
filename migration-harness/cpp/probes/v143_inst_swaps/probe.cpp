// migration-harness/cpp/probes/v143_inst_swaps/probe.cpp
//
// Reference values for four instruments ported in coverage wave 5:
//
//   * AssetSwap             (ql/instruments/assetswap.{hpp,cpp})
//   * BMASwap               (ql/instruments/bmaswap.{hpp,cpp})
//   * OvernightIndexFuture  (ql/instruments/overnightindexfuture.{hpp,cpp})
//   * PerpetualFutures      (ql/instruments/perpetualfutures.{hpp,cpp})
//     + DiscountingPerpetualFuturesEngine
//       (ql/pricingengines/futures/discountingperpetualfuturesengine.{hpp,cpp})
//
// WHAT IS PINNED AND WHY
//
// An instrument NPV can match while two errors cancel, so for the two swaps we
// pin the *complete* cashflow listing of both legs alongside every headline
// result: for every flow the payment-date serial and the amount, and for every
// coupon additionally the nominal, accrual start/end serials, the accrual
// period and the rate.  That is the level at which a leg built with the wrong
// convention, the wrong day counter or a dropped constructor argument actually
// shows up.
//
// Every optional constructor argument is swept at a NON-DEFAULT value in its
// own case, because the failure mode this probe exists to catch is an argument
// that is accepted, stored and never forwarded to a leg builder:
//
//   AssetSwap            — floatSchedule, floatingDayCount, parAssetSwap,
//                          gearing, nonParRepayment, dealMaturity,
//                          payBondCoupon (both values), overnight-index branch
//   BMASwap              — Payer / Receiver, liborFraction != 1,
//                          liborSpread != 0, per-leg day counters
//   OvernightIndexFuture — convexityAdjustment (zero AND non-zero),
//                          averagingMethod (Simple AND Compound),
//                          past fixings present in the IndexManager
//   PerpetualFutures     — payoffType, fundingType, fundingFrequency
//                          (Years / Months / Weeks / Days / Hours / continuous),
//                          calendar, day counter; engine: interpolation type
//                          and maxT
//
// The QL_REQUIRE failure branches are recorded as {"raises": true} — the probe
// asserts the throw actually happens, so a C++ branch that stops throwing
// turns the reference itself red rather than silently weakening the test.
//
// All curves, indexes and schedules are built inline from literal tables; the
// evaluation date is pinned explicitly so nothing depends on the machine
// clock.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/swaps.json.

#include <ql/cashflows/coupon.hpp>
#include <ql/cashflows/rateaveraging.hpp>
#include <ql/indexes/bmaindex.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/indexes/indexmanager.hpp>
#include <ql/instruments/assetswap.hpp>
#include <ql/instruments/bmaswap.hpp>
#include <ql/instruments/bonds/fixedratebond.hpp>
#include <ql/instruments/overnightindexfuture.hpp>
#include <ql/instruments/perpetualfutures.hpp>
#include <ql/pricingengines/futures/discountingperpetualfuturesengine.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Pinned market data — deliberately three *different* flat levels so that a
// port that mixes up the forwarding and the discounting curve cannot pass.
// ---------------------------------------------------------------------------
const Date kEval(17, January, 2024);      // Wednesday, no TARGET holiday
const Rate kForwardRate = 0.035;          // ibor / overnight forwarding curve
const Rate kDiscountRate = 0.030;         // swap discounting curve
const Rate kBmaForwardRate = 0.025;       // BMA forwarding curve

// ---------------------------------------------------------------------------
// JSON emission helpers.  Everything is written by hand at setprecision(17).
// ---------------------------------------------------------------------------

void emitFlow(const ext::shared_ptr<CashFlow>& cf, bool comma) {
    std::cout << "        {\"date_serial\": " << cf->date().serialNumber()
              << ", \"amount\": " << cf->amount();
    auto c = ext::dynamic_pointer_cast<Coupon>(cf);
    if (c != nullptr) {
        std::cout << ", \"is_coupon\": true"
                  << ", \"nominal\": " << c->nominal()
                  << ", \"accrual_start_serial\": " << c->accrualStartDate().serialNumber()
                  << ", \"accrual_end_serial\": " << c->accrualEndDate().serialNumber()
                  << ", \"accrual_period\": " << c->accrualPeriod()
                  << ", \"rate\": " << c->rate();
    } else {
        std::cout << ", \"is_coupon\": false";
    }
    std::cout << "}" << (comma ? "," : "") << "\n";
}

void emitLeg(const std::string& key, const Leg& leg, bool comma) {
    std::cout << "      \"" << key << "\": [\n";
    for (Size i = 0; i < leg.size(); ++i)
        emitFlow(leg[i], i + 1 < leg.size());
    std::cout << "      ]" << (comma ? "," : "") << "\n";
}

// Runs `f`, expecting it to throw.  Emits {"raises": <bool>} so a branch that
// silently stops throwing fails the Python test instead of being skipped.
template <class F>
void emitRaises(const std::string& key, F f, bool comma) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception&) {
        threw = true;
    }
    std::cout << "    \"" << key << "\": {\"raises\": " << (threw ? "true" : "false") << "}"
              << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------------------------
// Shared builders
// ---------------------------------------------------------------------------

Handle<YieldTermStructure> flatCurve(Rate r, const DayCounter& dc) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kEval, r, dc, Continuous, Annual));
}

// The asset-swap underlying: a 10y 5% annual TARGET bond issued 15-Jan-2021,
// settling T+3 so that every ibor fixing date of the swap is strictly in the
// future and the reference needs no historical fixings.
ext::shared_ptr<Bond> makeBond() {
    Schedule bondSchedule(Date(15, January, 2021), Date(15, January, 2031), Period(1, Years),
                          TARGET(), Following, Following, DateGeneration::Backward, false);
    return ext::make_shared<FixedRateBond>(3, 100.0, bondSchedule, std::vector<Rate>{0.05},
                                           Thirty360(Thirty360::BondBasis), Following, 100.0,
                                           Date(15, January, 2021));
}

Schedule floatSchedule6M() {
    return Schedule(Date(22, January, 2024), Date(15, January, 2031), Period(6, Months), TARGET(),
                    ModifiedFollowing, ModifiedFollowing, DateGeneration::Backward, false);
}

Schedule floatSchedule3M() {
    return Schedule(Date(22, January, 2024), Date(15, January, 2031), Period(3, Months), TARGET(),
                    ModifiedFollowing, ModifiedFollowing, DateGeneration::Backward, false);
}

// Short (2y) quarterly SOFR schedule — the overnight branch compounds daily
// fixings, so it is kept short to keep the Python test cheap.
Schedule overnightSchedule() {
    return Schedule(Date(22, January, 2024), Date(22, January, 2026), Period(3, Months),
                    UnitedStates(UnitedStates::SOFR), ModifiedFollowing, ModifiedFollowing,
                    DateGeneration::Backward, false);
}

// ---------------------------------------------------------------------------
// AssetSwap
// ---------------------------------------------------------------------------

void emitAssetSwapCase(const std::string& key,
                       const ext::shared_ptr<AssetSwap>& swap,
                       bool comma) {
    std::cout << "    \"" << key << "\": {\n"
              << "      \"npv\": " << swap->NPV() << ",\n"
              << "      \"fair_clean_price\": " << swap->fairCleanPrice() << ",\n"
              << "      \"fair_non_par_repayment\": " << swap->fairNonParRepayment() << ",\n"
              << "      \"fair_spread\": " << swap->fairSpread() << ",\n"
              << "      \"floating_leg_bps\": " << swap->floatingLegBPS() << ",\n"
              << "      \"floating_leg_npv\": " << swap->floatingLegNPV() << ",\n"
              << "      \"par_swap\": " << (swap->parSwap() ? "true" : "false") << ",\n"
              << "      \"spread\": " << swap->spread() << ",\n"
              << "      \"clean_price\": " << swap->cleanPrice() << ",\n"
              << "      \"non_par_repayment\": " << swap->nonParRepayment() << ",\n"
              << "      \"pay_bond_coupon\": " << (swap->payBondCoupon() ? "true" : "false")
              << ",\n"
              << "      \"start_date_serial\": " << swap->startDate().serialNumber() << ",\n"
              << "      \"maturity_date_serial\": " << swap->maturityDate().serialNumber()
              << ",\n";
    emitLeg("bond_leg", swap->bondLeg(), true);
    emitLeg("floating_leg", swap->floatingLeg(), false);
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

void emitAssetSwaps() {
    auto bond = makeBond();
    auto fwd = flatCurve(kForwardRate, Actual365Fixed());
    auto disc = flatCurve(kDiscountRate, Actual365Fixed());
    auto euribor6m = ext::make_shared<Euribor6M>(fwd);
    auto sofr = ext::make_shared<Sofr>(fwd);
    auto engine = ext::make_shared<DiscountingSwapEngine>(disc, false);

    const Real cleanPrice = 102.5;
    const Spread spread = 0.0025;

    struct Case {
        std::string key;
        bool payBondCoupon;
        Schedule floatSchedule;
        DayCounter floatingDayCount;
        bool parSwap;
        Real gearing;
        Real nonParRepayment;
        Date dealMaturity;
        bool overnight;
    };

    std::vector<Case> cases = {
        // 1-2: every optional argument at its default; both sides of payBondCoupon.
        {"base_pay_bond_coupon", true, Schedule(), DayCounter(), true, 1.0, Null<Real>(), Date(),
         false},
        {"base_receive_bond_coupon", false, Schedule(), DayCounter(), true, 1.0, Null<Real>(),
         Date(), false},
        // 3: floatSchedule at a NON-default value (quarterly instead of the
        //    index-tenor schedule derived from the bond).
        {"float_schedule_3m", true, floatSchedule3M(), DayCounter(), true, 1.0, Null<Real>(),
         Date(), false},
        // 4: floatingDayCount at a NON-default value (Thirty360 instead of the
        //    index's Actual/360).
        {"float_daycount_30360", true, Schedule(), Thirty360(Thirty360::BondBasis), true, 1.0,
         Null<Real>(), Date(), false},
        // 5: parAssetSwap = false — market asset swap, notional scaled by the
        //    dirty price and no upfront/backpayment pair.
        {"market_asset_swap", true, Schedule(), DayCounter(), false, 1.0, Null<Real>(), Date(),
         false},
        // 6: gearing at a NON-default value.
        {"gearing_0_9", true, Schedule(), DayCounter(), true, 0.9, Null<Real>(), Date(), false},
        // 7: nonParRepayment at a NON-default value.
        {"non_par_repayment_102", true, Schedule(), DayCounter(), true, 1.0, 102.0, Date(), false},
        // 8: dealMaturity at a NON-default value, deliberately mid-coupon so the
        //    "skipped coupon => accrued SimpleCashFlow" branch is exercised.
        {"deal_maturity_2027", true, Schedule(), DayCounter(), true, 1.0, Null<Real>(),
         Date(15, July, 2027), false},
        // 9: overnight index branch (requires an explicit float schedule).
        {"overnight_sofr", true, overnightSchedule(), DayCounter(), true, 1.0, Null<Real>(),
         Date(), true},
        // 10: overnight + market asset swap + gearing, to prove the overnight
        //     branch forwards the same optional arguments as the ibor branch.
        {"overnight_market_gearing", false, overnightSchedule(), Actual365Fixed(), false, 1.1,
         Null<Real>(), Date(), true},
    };

    std::cout << "  \"asset_swap\": {\n";
    for (Size i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        ext::shared_ptr<IborIndex> index =
            c.overnight ? ext::shared_ptr<IborIndex>(sofr) : ext::shared_ptr<IborIndex>(euribor6m);
        auto swap = ext::make_shared<AssetSwap>(c.payBondCoupon, bond, cleanPrice, index, spread,
                                                c.floatSchedule, c.floatingDayCount, c.parSwap,
                                                c.gearing, c.nonParRepayment, c.dealMaturity);
        swap->setPricingEngine(engine);
        emitAssetSwapCase(c.key, swap, i + 1 < cases.size());
    }
    std::cout << "  },\n";

    // ---- QL_REQUIRE branches ------------------------------------------
    std::cout << "  \"asset_swap_raises\": {\n";
    emitRaises(
        "overnight_without_schedule",
        [&] {
            AssetSwap(true, bond, cleanPrice, sofr, spread);
        },
        true);
    emitRaises(
        "deal_maturity_after_schedule_back",
        [&] {
            AssetSwap(true, bond, cleanPrice, euribor6m, spread, floatSchedule6M(), DayCounter(),
                      true, 1.0, Null<Real>(), Date(15, January, 2035));
        },
        true);
    emitRaises(
        "deal_maturity_before_schedule_front",
        [&] {
            AssetSwap(true, bond, cleanPrice, euribor6m, spread, floatSchedule6M(), DayCounter(),
                      true, 1.0, Null<Real>(), Date(22, January, 2024));
        },
        false);
    std::cout << "  },\n";
}

// ---------------------------------------------------------------------------
// BMASwap
// ---------------------------------------------------------------------------

void emitBmaSwaps() {
    auto liborCurve = flatCurve(kForwardRate, Actual365Fixed());
    auto bmaCurve = flatCurve(kBmaForwardRate, Actual365Fixed());
    auto disc = flatCurve(kDiscountRate, Actual365Fixed());
    auto liborIndex = ext::make_shared<Euribor3M>(liborCurve);
    auto bmaIndex = ext::make_shared<BMAIndex>(bmaCurve);
    auto engine = ext::make_shared<DiscountingSwapEngine>(disc, false);

    // Start 1-Feb-2024: the BMA fixing schedule of the first coupon then rolls
    // back only to Wednesday 31-Jan-2024, i.e. strictly after the evaluation
    // date, so no historical BMA fixing is required.
    const Date start(1, February, 2024);
    const Date end(1, February, 2027);
    const Real nominal = 1.0e6;
    const Real liborFraction = 0.67;
    const Spread liborSpread = 0.0010;

    Schedule liborSchedule(start, end, Period(3, Months), TARGET(), ModifiedFollowing,
                           ModifiedFollowing, DateGeneration::Backward, false);
    Schedule bmaSchedule(start, end, Period(3, Months), UnitedStates(UnitedStates::GovernmentBond),
                         Following, Following, DateGeneration::Backward, false);

    struct Case {
        std::string key;
        Swap::Type type;
    };
    std::vector<Case> cases = {{"payer", Swap::Payer}, {"receiver", Swap::Receiver}};

    std::cout << "  \"bma_swap\": {\n";
    for (Size i = 0; i < cases.size(); ++i) {
        BMASwap swap(cases[i].type, nominal, liborSchedule, liborFraction, liborSpread, liborIndex,
                     Actual360(), bmaSchedule, bmaIndex, Actual365Fixed());
        swap.setPricingEngine(engine);
        std::cout << "    \"" << cases[i].key << "\": {\n"
                  << "      \"npv\": " << swap.NPV() << ",\n"
                  << "      \"fair_libor_fraction\": " << swap.fairLiborFraction() << ",\n"
                  << "      \"fair_libor_spread\": " << swap.fairLiborSpread() << ",\n"
                  << "      \"libor_leg_bps\": " << swap.liborLegBPS() << ",\n"
                  << "      \"bma_leg_bps\": " << swap.bmaLegBPS() << ",\n"
                  << "      \"libor_leg_npv\": " << swap.liborLegNPV() << ",\n"
                  << "      \"bma_leg_npv\": " << swap.bmaLegNPV() << ",\n"
                  << "      \"libor_fraction\": " << swap.liborFraction() << ",\n"
                  << "      \"libor_spread\": " << swap.liborSpread() << ",\n"
                  << "      \"nominal\": " << swap.nominal() << ",\n"
                  << "      \"type\": " << static_cast<int>(swap.type()) << ",\n";
        emitLeg("libor_leg", swap.liborLeg(), true);
        emitLeg("bma_leg", swap.bmaLeg(), false);
        std::cout << "    }" << (i + 1 < cases.size() ? "," : "") << "\n";
    }
    std::cout << "  },\n";
}

// ---------------------------------------------------------------------------
// OvernightIndexFuture
// ---------------------------------------------------------------------------

const Date kFutureValue(20, March, 2024);
const Date kFutureMaturity(19, June, 2024);
const Date kInsideEval(10, April, 2024);  // inside the reference period
const Real kConvexity = 0.0012;

// Deterministic synthetic SOFR history over [valueDate, insideEval]; the exact
// values are emitted so the Python test replays the same IndexManager state.
std::vector<std::pair<Date, Rate>> sofrHistory() {
    Calendar cal = UnitedStates(UnitedStates::SOFR);
    std::vector<std::pair<Date, Rate>> out;
    Size k = 0;
    for (Date d = kFutureValue; d <= kInsideEval; ++d) {
        if (cal.isBusinessDay(d)) {
            out.emplace_back(d, 0.0500 + 0.0001 * static_cast<Real>(k % 11));
            ++k;
        }
    }
    return out;
}

void emitOvernightFutures() {
    auto fwd = flatCurve(kForwardRate, Actual360());
    auto sofr = ext::make_shared<Sofr>(fwd);
    Handle<Quote> convexity(ext::make_shared<SimpleQuote>(kConvexity));

    std::cout << "  \"overnight_index_future\": {\n"
              << "    \"value_date_serial\": " << kFutureValue.serialNumber() << ",\n"
              << "    \"maturity_date_serial\": " << kFutureMaturity.serialNumber() << ",\n"
              << "    \"convexity\": " << kConvexity << ",\n";

    // ---- valued before the reference period: no history needed ---------
    {
        OvernightIndexFuture simpleZero(sofr, kFutureValue, kFutureMaturity, Handle<Quote>(),
                                        RateAveraging::Simple);
        OvernightIndexFuture compoundZero(sofr, kFutureValue, kFutureMaturity, Handle<Quote>(),
                                          RateAveraging::Compound);
        OvernightIndexFuture simpleCa(sofr, kFutureValue, kFutureMaturity, convexity,
                                      RateAveraging::Simple);
        OvernightIndexFuture compoundCa(sofr, kFutureValue, kFutureMaturity, convexity,
                                        RateAveraging::Compound);
        // Default averagingMethod is Compound and the default convexity handle
        // is empty — pinned so a port that flips the default is caught.
        OvernightIndexFuture defaults(sofr, kFutureValue, kFutureMaturity);

        std::cout << "    \"simple_zero_ca\": " << simpleZero.NPV() << ",\n"
                  << "    \"compound_zero_ca\": " << compoundZero.NPV() << ",\n"
                  << "    \"simple_nonzero_ca\": " << simpleCa.NPV() << ",\n"
                  << "    \"compound_nonzero_ca\": " << compoundCa.NPV() << ",\n"
                  << "    \"defaults\": " << defaults.NPV() << ",\n"
                  << "    \"convexity_adjustment_empty\": " << simpleZero.convexityAdjustment()
                  << ",\n"
                  << "    \"convexity_adjustment_quote\": " << simpleCa.convexityAdjustment()
                  << ",\n"
                  << "    \"is_expired_before\": " << (simpleZero.isExpired() ? "true" : "false")
                  << ",\n";
    }

    // ---- valued INSIDE the reference period, with past fixings ---------
    auto history = sofrHistory();
    std::cout << "    \"history\": [\n";
    for (Size i = 0; i < history.size(); ++i)
        std::cout << "      {\"date_serial\": " << history[i].first.serialNumber()
                  << ", \"fixing\": " << history[i].second << "}"
                  << (i + 1 < history.size() ? "," : "") << "\n";
    std::cout << "    ],\n"
              << "    \"inside_eval_serial\": " << kInsideEval.serialNumber() << ",\n";

    {
        for (const auto& p : history)
            sofr->addFixing(p.first, p.second);
        Settings::instance().evaluationDate() = kInsideEval;

        OvernightIndexFuture simplePast(sofr, kFutureValue, kFutureMaturity, Handle<Quote>(),
                                        RateAveraging::Simple);
        OvernightIndexFuture compoundPast(sofr, kFutureValue, kFutureMaturity, Handle<Quote>(),
                                          RateAveraging::Compound);
        OvernightIndexFuture compoundPastCa(sofr, kFutureValue, kFutureMaturity, convexity,
                                            RateAveraging::Compound);
        std::cout << "    \"simple_past_fixings\": " << simplePast.NPV() << ",\n"
                  << "    \"compound_past_fixings\": " << compoundPast.NPV() << ",\n"
                  << "    \"compound_past_fixings_ca\": " << compoundPastCa.NPV() << ",\n";

        // Expired: maturity strictly before the evaluation date.
        Settings::instance().evaluationDate() = Date(1, August, 2024);
        OvernightIndexFuture expired(sofr, kFutureValue, kFutureMaturity, Handle<Quote>(),
                                     RateAveraging::Compound);
        std::cout << "    \"is_expired_after\": " << (expired.isExpired() ? "true" : "false")
                  << "\n";

        Settings::instance().evaluationDate() = kEval;
    }
    std::cout << "  },\n";

    std::cout << "  \"overnight_index_future_raises\": {\n";
    emitRaises(
        "null_index",
        [&] {
            OvernightIndexFuture(ext::shared_ptr<OvernightIndex>(), kFutureValue, kFutureMaturity);
        },
        true);
    emitRaises(
        "missing_past_fixing",
        [&] {
            IndexManager::instance().clearHistories();
            Settings::instance().evaluationDate() = kInsideEval;
            OvernightIndexFuture f(sofr, kFutureValue, kFutureMaturity, Handle<Quote>(),
                                   RateAveraging::Compound);
            f.NPV();
        },
        false);
    IndexManager::instance().clearHistories();
    Settings::instance().evaluationDate() = kEval;
    std::cout << "  },\n";
}

// ---------------------------------------------------------------------------
// PerpetualFutures + DiscountingPerpetualFuturesEngine
// ---------------------------------------------------------------------------

void emitPerpetualFutures() {
    const Real spotValue = 10000.0;
    const Rate domRate = 0.04;
    const Rate forRate = 0.02;

    DayCounter isda = ActualActual(ActualActual::ISDA);
    Handle<YieldTermStructure> domCurve(
        ext::make_shared<FlatForward>(kEval, domRate, isda, Continuous, Annual));
    Handle<YieldTermStructure> forCurve(
        ext::make_shared<FlatForward>(kEval, forRate, isda, Continuous, Annual));
    Handle<Quote> spot(ext::make_shared<SimpleQuote>(spotValue));

    // Single-pillar funding term structure (matches the C++ test-suite case).
    const std::vector<Time> t1{0.0};
    const std::vector<Rate> k1{0.01};
    const std::vector<Spread> i1{0.005};
    // Three-pillar structure, used to make the interpolation type observable.
    const std::vector<Time> t3{0.0, 1.0, 5.0};
    const std::vector<Rate> k3{0.010, 0.020, 0.035};
    const std::vector<Spread> i3{0.005, 0.008, 0.012};

    using Engine = DiscountingPerpetualFuturesEngine;

    struct Case {
        std::string key;
        PerpetualFutures::PayoffType payoff;
        PerpetualFutures::FundingType funding;
        Period freq;
        Calendar cal;
        DayCounter dc;
        bool multiPillar;
        Engine::InterpolationType interp;
        Real maxT;
    };

    std::vector<Case> cases = {
        // Discrete, quarterly — the four payoff x funding combinations.
        {"lin_prev_3m", PerpetualFutures::Linear, PerpetualFutures::FundingWithPreviousSpot,
         Period(3, Months), NullCalendar(), isda, false, Engine::PiecewiseConstant, 60.0},
        {"lin_curr_3m", PerpetualFutures::Linear, PerpetualFutures::FundingWithCurrentSpot,
         Period(3, Months), NullCalendar(), isda, false, Engine::PiecewiseConstant, 60.0},
        {"inv_prev_3m", PerpetualFutures::Inverse, PerpetualFutures::FundingWithPreviousSpot,
         Period(3, Months), NullCalendar(), isda, false, Engine::PiecewiseConstant, 60.0},
        {"inv_curr_3m", PerpetualFutures::Inverse, PerpetualFutures::FundingWithCurrentSpot,
         Period(3, Months), NullCalendar(), isda, false, Engine::PiecewiseConstant, 60.0},
        // Continuous (zero-length funding frequency) — the integration branch.
        {"lin_prev_continuous", PerpetualFutures::Linear,
         PerpetualFutures::FundingWithPreviousSpot, Period(0, Months), NullCalendar(), isda, false,
         Engine::PiecewiseConstant, 60.0},
        {"inv_prev_continuous", PerpetualFutures::Inverse,
         PerpetualFutures::FundingWithPreviousSpot, Period(0, Months), NullCalendar(), isda, false,
         Engine::PiecewiseConstant, 60.0},
        // fundingFrequency units: Years / Weeks / Days / Hours.
        {"lin_prev_1y", PerpetualFutures::Linear, PerpetualFutures::FundingWithPreviousSpot,
         Period(1, Years), NullCalendar(), isda, false, Engine::PiecewiseConstant, 60.0},
        {"lin_prev_1w", PerpetualFutures::Linear, PerpetualFutures::FundingWithPreviousSpot,
         Period(1, Weeks), NullCalendar(), isda, false, Engine::PiecewiseConstant, 20.0},
        {"lin_prev_1d_maxt5", PerpetualFutures::Linear, PerpetualFutures::FundingWithPreviousSpot,
         Period(1, Days), NullCalendar(), isda, false, Engine::PiecewiseConstant, 5.0},
        {"lin_curr_8h_maxt1", PerpetualFutures::Linear, PerpetualFutures::FundingWithCurrentSpot,
         Period(8, Hours), NullCalendar(), isda, false, Engine::PiecewiseConstant, 1.0},
        // Calendar + day counter at NON-default values (both only bite in the
        // Weeks/Days branch, which advances through the calendar).
        {"lin_prev_1w_target_a365", PerpetualFutures::Linear,
         PerpetualFutures::FundingWithPreviousSpot, Period(1, Weeks), TARGET(), Actual365Fixed(),
         false, Engine::PiecewiseConstant, 20.0},
        // Engine interpolation type over a three-pillar funding structure.
        {"lin_prev_3m_pwc3", PerpetualFutures::Linear, PerpetualFutures::FundingWithPreviousSpot,
         Period(3, Months), NullCalendar(), isda, true, Engine::PiecewiseConstant, 60.0},
        {"lin_prev_3m_linear3", PerpetualFutures::Linear,
         PerpetualFutures::FundingWithPreviousSpot, Period(3, Months), NullCalendar(), isda, true,
         Engine::Linear, 60.0},
        {"lin_prev_3m_cubic3", PerpetualFutures::Linear, PerpetualFutures::FundingWithPreviousSpot,
         Period(3, Months), NullCalendar(), isda, true, Engine::CubicSpline, 60.0},
        // maxT at a NON-default value on an otherwise identical case.
        {"lin_prev_3m_maxt30", PerpetualFutures::Linear,
         PerpetualFutures::FundingWithPreviousSpot, Period(3, Months), NullCalendar(), isda, false,
         Engine::PiecewiseConstant, 30.0},
        {"inv_curr_continuous_maxt30", PerpetualFutures::Inverse,
         PerpetualFutures::FundingWithCurrentSpot, Period(0, Months), NullCalendar(), isda, true,
         Engine::Linear, 30.0},
    };

    std::cout << "  \"perpetual_futures\": {\n"
              << "    \"spot\": " << spotValue << ",\n"
              << "    \"dom_rate\": " << domRate << ",\n"
              << "    \"for_rate\": " << forRate << ",\n"
              << "    \"values\": {\n";
    for (Size i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        PerpetualFutures trade(c.payoff, c.funding, c.freq, c.cal, c.dc);
        auto engine = ext::make_shared<Engine>(domCurve, forCurve, spot,
                                               c.multiPillar ? t3 : t1, c.multiPillar ? k3 : k1,
                                               c.multiPillar ? i3 : i1, c.interp, c.maxT);
        trade.setPricingEngine(engine);
        std::cout << "      \"" << c.key << "\": " << trade.NPV()
                  << (i + 1 < cases.size() ? "," : "") << "\n";
    }
    std::cout << "    },\n"
              << "    \"is_expired\": false\n"
              << "  },\n";

    std::cout << "  \"perpetual_futures_raises\": {\n";
    emitRaises(
        "quanto_not_supported",
        [&] {
            PerpetualFutures trade(PerpetualFutures::Quanto);
            trade.setPricingEngine(
                ext::make_shared<Engine>(domCurve, forCurve, spot, t1, k1, i1));
            trade.NPV();
        },
        true);
    emitRaises(
        "empty_funding_times",
        [&] {
            Engine(domCurve, forCurve, spot, std::vector<Time>{}, k1, i1);
        },
        true);
    emitRaises(
        "size_mismatch_rates",
        [&] {
            Engine(domCurve, forCurve, spot, t1, k3, i1);
        },
        true);
    emitRaises(
        "size_mismatch_diffs",
        [&] {
            Engine(domCurve, forCurve, spot, t1, k1, i3);
        },
        true);
    emitRaises(
        "negative_terminal_funding_rate",
        [&] {
            PerpetualFutures trade(PerpetualFutures::Linear);
            trade.setPricingEngine(ext::make_shared<Engine>(domCurve, forCurve, spot, t1,
                                                            std::vector<Rate>{-0.01}, i1));
            trade.NPV();
        },
        true);
    emitRaises(
        "unknown_payoff_type",
        [&] {
            PerpetualFutures trade(PerpetualFutures::PayoffType(-1));
            trade.setPricingEngine(
                ext::make_shared<Engine>(domCurve, forCurve, spot, t1, k1, i1));
            trade.NPV();
        },
        true);
    emitRaises(
        "unknown_funding_type",
        [&] {
            PerpetualFutures trade(PerpetualFutures::Linear,
                                   PerpetualFutures::FundingType(-1));
            trade.setPricingEngine(
                ext::make_shared<Engine>(domCurve, forCurve, spot, t1, k1, i1));
            trade.NPV();
        },
        false);
    std::cout << "  }\n";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = kEval;

    std::cout << "{\n";
    std::cout << "  \"eval_date_serial\": " << kEval.serialNumber() << ",\n";
    emitAssetSwaps();
    emitBmaSwaps();
    emitOvernightFutures();
    emitPerpetualFutures();
    std::cout << "}\n";
    return 0;
}
