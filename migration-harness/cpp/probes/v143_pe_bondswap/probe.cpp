// migration-harness/cpp/probes/v143_pe_bondswap/probe.cpp
//
// Reference values for the "bondswap" cluster of C++ QuantLib v1.43 pricing
// engines / free-function surfaces:
//
//   * BondFunctions                (ql/pricingengines/bond/bondfunctions.{hpp,cpp})
//   * RiskyBondEngine              (ql/pricingengines/bond/riskybondengine.{hpp,cpp})
//   * BachelierCalculator          (ql/pricingengines/bacheliercalculator.{hpp,cpp})
//   * DiscountingFxForwardEngine   (ql/pricingengines/forward/discountingfxforwardengine.{hpp,cpp})
//   * TreeVanillaSwapEngine        (ql/pricingengines/swap/treeswapengine.hpp)
//       -> which exists only to exercise the two class templates
//          LatticeShortRateModelEngine (ql/pricingengines/latticeshortratemodelengine.hpp)
//          GenericModelEngine          (ql/pricingengines/genericmodelengine.hpp)
//   * CounterpartyAdjSwapEngine    (ql/pricingengines/swap/cvaswapengine.{hpp,cpp})
//
// Emits JSON on stdout and nothing else.
//
// =============================================================================
// PART 1 -- BondFunctions
// =============================================================================
//
// BondFunctions is a `struct` of ~40 public statics that adapt CashFlows to a
// Bond: they default the settlement date to `bond.settlementDate()`, they
// re-scale everything by `100.0 / bond.notional(settlement)` so prices are per
// 100 of *current* (amortised) notional, and they guard almost everything with
//     QL_REQUIRE(BondFunctions::isTradable(bond, settlement), "non tradable ...")
// where `isTradable` is exactly `bond.notional(settlement) != 0.0`.
//
// Everything that a port plausibly gets wrong, and is pinned here:
//
//  1. THE `notional()` STEP FUNCTION. `Bond::notional(d)` returns 0 when
//     `d == notionalSchedule_.back()` (the redemption date), i.e. a bond is NOT
//     tradable *on* its own maturity date, only strictly before it. Every
//     `*_maturity` case below pins `isTradable == false` on the maturity date
//     and the resulting throw of accrualStartDate/dirtyPrice/yield/...
//     `accruedAmount` is the one exception: it returns 0.0 instead of throwing
//     (bondfunctions.cpp:229-230).
//
//  2. AMORTISATION. The amortising bond's notional schedule is
//     [null, 2022-01-15, 2023-01-15, 2024-01-15, 2025-01-15, 2026-01-15] with
//     notionals [100, 80, 60, 40, 20, 0]. `notional(d)` returns notionals[i-1]
//     when d < schedule[i] but notionals[i] when d == schedule[i]. So on
//     2024-01-15 the notional is 40, not 60 -- an off-by-one that changes every
//     price (which is divided by it) by 50%.
//
//  3. `previousCouponRate` AGGREGATES *BACKWARDS*. C++ takes a
//     `Leg::const_reverse_iterator` from `previousCashFlow` and calls
//     `aggregateRate(cf, leg.rend())`, so it walks from the previous cashflow
//     towards the START of the leg while `date() == paymentDate`. On the final
//     payment date the leg holds [.., lastCoupon, redemption] (stable-sorted,
//     coupon first), so the reverse walk starting at `redemption` still reaches
//     `lastCoupon` and returns its rate. A port that walks FORWARD from the
//     same index sees only the redemption, finds no Coupon, and returns 0 (or
//     throws). Case `fixed_inspect_after_maturity` pins the C++ answer.
//     `aggregateRate` also SUMS `cp->rate()` over same-date coupons (it does not
//     average them) and QL_ENSUREs that at least one Coupon was found -- so
//     `nextCouponRate`/`previousCouponRate` THROW on a zero-coupon bond, whose
//     only cashflow is a Redemption. Pinned by `zero_*` cases.
//
//  4. THE COUPON INSPECTORS RETURN A NULL Date, NOT 0, when the next cashflow
//     is not a Coupon (cashflows.cpp:246-308). Serial 0 below means "null Date".
//
//  5. `accruedAmount` IS RESCALED, `accruedPeriod`/`accruedDays` ARE NOT.
//     bondfunctions.cpp:224-235 multiplies by 100/notional; the period/day
//     inspectors forward to CashFlows untouched.
//
//  6. `atmRate` WITH AND WITHOUT A PRICE. With `Bond::Price{}` (invalid) the
//     target NPV is the leg's own NPV minus the non-sensitive NPV, so the answer
//     is the par coupon. With a price it is
//     `(dirtyPrice/100 * notional) * discount(npvDate) - nonSensNPV) / bps`.
//     Both are pinned; a port that ignores `price.isValid()` produces the first
//     answer for both.
//
//  7. `yield` IS TEMPLATED ON THE SOLVER (bondfunctions.hpp:167-195) and the
//     non-template overload is only `NewtonSafe` + `setMaxEvaluations`.
//     Different solvers converge to *different* doubles for the same accuracy
//     (1e-10 on the price residual, not on the yield), so all five pinned
//     solvers are pinned separately. Note the template overload has NO
//     maxIterations parameter -- the caller pre-configures the solver.
//
//  8. THE DEPRECATED z-spread OVERLOADS TAKE A DayCounter AND IGNORE IT. In
//     bondfunctions.cpp:500-508 / 530-538 / 570-582 the DayCounter parameter is
//     unnamed and the body forwards to the 5-arg overload. Cases
//     `*_zspread_deprecated_daycounter_ignored` call them with Actual360 and
//     with Thirty360 and pin that BOTH equal the non-deprecated answer. A port
//     must reproduce the discard, not "fix" it.
//
//  9. THE FLOATING-RATE BOND HITS THE UN-FIXED-COUPON BRANCH. Evaluation date is
//     2025-05-15 and the index fixings are only seeded up to 2025-01-13, so the
//     coupon accruing from 2025-07-15 has no fixing and its rate is FORECAST
//     from the index's term structure. Settlement 2025-08-20 therefore makes
//     `nextCouponRate` take the forecast path while settlement 2025-05-20 takes
//     the already-fixed path.
//
// 10. SETTLEMENT BEFORE THE ISSUE DATE. `Coupon::accruedPeriod(d)` returns 0 for
//     `d <= accrualStartDate`, so accrued* are all zero at 2020-01-10 even though
//     the bond is "tradable" there (notional != 0). Pinned by
//     `fixed_inspect_before_issue`.
//
// Market for part 1 (all four bonds share TARGET() and settlementDays = 3):
//   evaluationDate  2025-05-15 (Thursday) => bond.settlementDate() = 2025-05-20
//   fixed       FixedRateBond           2020-01-15 .. 2030-01-15, semiannual, 4%,
//                                       Thirty360(BondBasis), Unadjusted schedule
//   floating    FloatingRateBond        2023-01-15 .. 2028-01-15, semiannual,
//                                       Euribor6M + 50bp, Actual360
//   amortising  AmortizingFixedRateBond 2021-01-15 .. 2026-01-15, annual, 5%,
//                                       notionals {100, 80, 60, 40, 20}
//   zero        ZeroCouponBond          issued 2020-01-15, matures 2030-01-15
//
// =============================================================================
// PART 2 -- BachelierCalculator
// =============================================================================
//
// The normal-model analogue of BlackCalculator. The reason it exists is that it
// must work for NEGATIVE forwards and NEGATIVE strikes, where every log() in
// BlackCalculator is undefined, so those are pinned first. Traps:
//
//  * `value()` CLAMPS AT ZERO: `return discount_ * std::max(result, 0.0);`
//    (bacheliercalculator.cpp:188). No other greek is clamped.
//  * OPTION TYPE IS RECOVERED FROM `alpha_ >= 0`, not stored. For a Call
//    alpha_ = N(d) >= 0; for a Put alpha_ = N(d) - 1 < 0. But an
//    AssetOrNothingPayoff PUT sets alpha_ = 1 - N(d) >= 0, so `value()`,
//    `deltaForward()`, `itmCashProbability()`, `strikeSensitivity()` and
//    `dividendRho()` all take the CALL branch for an asset-or-nothing put.
//    That is C++ behaviour; cases `bachelier_aon_put_*` pin it.
//  * `value()` DOES NOT USE alpha_/beta_/x_ AT ALL -- it recomputes
//    `intrinsic * cum_d_ + stdDev_ * n_d_`. So a CashOrNothing / AssetOrNothing /
//    Gap payoff produces the *plain vanilla* value; only the greeks that read
//    alpha_/beta_ differ. Pinned by `bachelier_con_*` / `bachelier_aon_*`.
//  * `theta` uses `std::log(forward_/spot)`, which is NaN for a negative
//    forward. `bachelier_negfwd_*` pins the greeks that stay finite and does not
//    ask for theta there.
//  * The zero-stdDev branch (stdDev_ < QL_EPSILON) sets cum_d_ to 0.5/1/0 and
//    n_d_ to M_SQRT_2*M_1_SQRTPI (== 1/sqrt(pi) ~ 0.5641895835477563) when
//    F == K. Note `value()` then still adds NO time value because it tests
//    `stdDev_ > QL_EPSILON`, but `vega`/`gamma`/`gammaForward`/`strikeGamma`
//    test `<= QL_EPSILON` and return 0.
//  * `initialize` QL_REQUIREs stdDev >= 0 and discount > 0.
//
// =============================================================================
// PART 3 -- DiscountingFxForwardEngine
// =============================================================================
//
// New in v1.43. Traps: the fair forward rate is `S * dfSource / dfTarget`
// (NOT the other way round), the discount factors are settlement-normalised
// (`discount(T)/discount(settlement)`), and the reported NPV is the
// settlement-date NPV pushed back to the curve reference date -- once with
// `dfSourceSettlement` for the source-currency NPV and once with
// `spot * dfTargetSettlement` for the target-currency NPV, so the two NPVs are
// NOT simply related by the spot rate. Seven additionalResults are pinned by
// name. The engine also QL_REQUIREs both curve reference dates <= settlement and
// spot > 0.
//
// =============================================================================
// PART 4 -- TreeVanillaSwapEngine / LatticeShortRateModelEngine / GenericModelEngine
// =============================================================================
//
// TreeVanillaSwapEngine is the only concrete engine in the library that derives
// straight from LatticeShortRateModelEngine, which derives from
// GenericModelEngine. Both bases carry real behaviour:
//   * GenericModelEngine holds `Handle<ModelType> model_` and registers with it;
//   * LatticeShortRateModelEngine holds `timeSteps_` AND `timeGrid_`, and its
//     `update()` REBUILDS `lattice_` from `timeGrid_` when the grid is non-empty
//     (that is the whole difference between the two constructor flavours: the
//     timeSteps ctor leaves lattice_ null and TreeVanillaSwapEngine builds a
//     fresh lattice inside calculate(); the TimeGrid ctor builds it eagerly and
//     re-builds it on every notification).
// Both flavours are pinned, with the SAME model and a grid chosen to be
// numerically different from the timeSteps grid, so an implementation that
// silently ignores one of the two constructors cannot pass.
//
// =============================================================================
// PART 5 -- CounterpartyAdjSwapEngine
// =============================================================================
//
// Sorensen-Bollier CVA/DVA adjustment: value = riskless NPV
//   - (1-Rc) * sum_i Swaption_i * Q_ctpty(t_{i-1}, t_i)
//   + (1-Ri) * sum_i ReverseSwaption_i * Q_invst(t_{i-1}, t_i)
// Traps:
//  * when the investor default curve handle is EMPTY the ctor substitutes a
//    FlatHazardRate(0, NullCalendar(), 1e-12, ctptyDTS->dayCounter()) -- not a
//    zero curve, a 1e-12 one -- so the DVA term is tiny but NOT exactly zero;
//  * the swaptionlet strip starts at the FIRST fixed pay date >= the default
//    curve's reference date and each swaplet is struck at the *base swap fair
//    rate* `-fixedRate * legNPV[1] / legNPV[0]`;
//  * `results_.fairRate` is NOT the fair rate of the adjusted swap, it is the
//    approximation quoted in cvaswapengine.cpp:211-214;
//  * the constructor taking a `Volatility` and the one taking a `Handle<Quote>`
//    must give identical numbers for the same vol -- pinned.

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/duration.hpp>
#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/bonds/amortizingfixedratebond.hpp>
#include <ql/instruments/bonds/fixedratebond.hpp>
#include <ql/instruments/bonds/floatingratebond.hpp>
#include <ql/instruments/bonds/zerocouponbond.hpp>
#include <ql/instruments/fxforward.hpp>
#include <ql/instruments/makevanillaswap.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/math/solvers1d/bisection.hpp>
#include <ql/math/solvers1d/brent.hpp>
#include <ql/math/solvers1d/finitedifferencenewtonsafe.hpp>
#include <ql/math/solvers1d/newtonsafe.hpp>
#include <ql/math/solvers1d/ridder.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/pricingengines/bacheliercalculator.hpp>
#include <ql/pricingengines/blackcalculator.hpp>
#include <ql/pricingengines/bond/bondfunctions.hpp>
#include <ql/pricingengines/bond/discountingbondengine.hpp>
#include <ql/pricingengines/bond/riskybondengine.hpp>
#include <ql/pricingengines/forward/discountingfxforwardengine.hpp>
#include <ql/pricingengines/swap/cvaswapengine.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/pricingengines/swap/discretizedswap.hpp>
#include <ql/pricingengines/swap/treeswapengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>
#include <ql/timegrid.hpp>
#include <ql/currencies/america.hpp>
#include <ql/currencies/europe.hpp>

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
    //! null Date is emitted as serial 0, which is exactly Date().serialNumber().
    Obj& d(const std::string& k, const Date& v) {
        return i(k, static_cast<long long>(v.serialNumber()));
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

//! Record `key` = f() or `key_throws` = true. Every guard in BondFunctions is a
//! QL_REQUIRE, so "throws" is a first-class expected value here.
template <class F>
void guarded(Obj& o, const std::string& key, F f) {
    try {
        o.n(key, Real(f()));
    } catch (const std::exception&) {
        o.b(key + "_throws", true);
    }
}

template <class F>
void guardedDate(Obj& o, const std::string& key, F f) {
    try {
        o.d(key, f());
    } catch (const std::exception&) {
        o.b(key + "_throws", true);
    }
}

// ===========================================================================
// PART 1 market
// ===========================================================================

const Date kToday(15, May, 2025);

const DayCounter& dc365() {
    static const DayCounter x = Actual365Fixed();
    return x;
}
const DayCounter& dc360() {
    static const DayCounter x = Actual360();
    return x;
}
const DayCounter& dc30360() {
    static const DayCounter x = Thirty360(Thirty360::BondBasis);
    return x;
}

const Calendar& target() {
    static const Calendar c = TARGET();
    return c;
}

ext::shared_ptr<FixedRateBond> makeFixedBond() {
    Schedule sch(Date(15, January, 2020), Date(15, January, 2030), Period(Semiannual), target(),
                 Unadjusted, Unadjusted, DateGeneration::Backward, false);
    return ext::make_shared<FixedRateBond>(3, 100.0, sch, std::vector<Rate>{0.04}, dc30360(),
                                           Following, 100.0, Date(15, January, 2020));
}

ext::shared_ptr<AmortizingFixedRateBond> makeAmortisingBond() {
    Schedule sch(Date(15, January, 2021), Date(15, January, 2026), Period(Annual), target(),
                 Unadjusted, Unadjusted, DateGeneration::Backward, false);
    return ext::make_shared<AmortizingFixedRateBond>(
        3, std::vector<Real>{100.0, 80.0, 60.0, 40.0, 20.0}, sch, std::vector<Rate>{0.05},
        dc30360(), Following, Date(15, January, 2021));
}

ext::shared_ptr<ZeroCouponBond> makeZeroBond() {
    return ext::make_shared<ZeroCouponBond>(3, target(), 100.0, Date(15, January, 2030), Following,
                                            100.0, Date(15, January, 2020));
}

//! Forecast curve for the floater's Euribor6M. Flat 3% continuous / Actual365Fixed.
const Handle<YieldTermStructure>& forecastCurve() {
    static const Handle<YieldTermStructure> h(
        ext::make_shared<FlatForward>(kToday, 0.03, dc365()));
    return h;
}

//! Historical Euribor6M fixings. Only dates strictly before the evaluation date
//! may be seeded, so the 2025-07-15 coupon is deliberately left UNFIXED.
const std::vector<std::pair<Date, Rate>>& floaterFixings() {
    static const std::vector<std::pair<Date, Rate>> f = {
        {Date(11, January, 2023), 0.0250}, {Date(13, July, 2023), 0.0275},
        {Date(11, January, 2024), 0.0300}, {Date(11, July, 2024), 0.0325},
        {Date(13, January, 2025), 0.0280},
    };
    return f;
}

ext::shared_ptr<FloatingRateBond> makeFloatingBond() {
    static const ext::shared_ptr<IborIndex> idx = ext::make_shared<Euribor6M>(forecastCurve());
    Schedule sch(Date(15, January, 2023), Date(15, January, 2028), Period(Semiannual), target(),
                 ModifiedFollowing, ModifiedFollowing, DateGeneration::Backward, false);
    return ext::make_shared<FloatingRateBond>(3, 100.0, sch, idx, dc360(), Following, 2,
                                              std::vector<Real>{1.0}, std::vector<Spread>{0.005});
}

//! Discount curve used by every YieldTermStructure-flavoured BondFunctions call.
//! Reference date == evaluation date, 3.5% annually-compounded, Actual365Fixed.
ext::shared_ptr<YieldTermStructure> discountCurve() {
    static const ext::shared_ptr<YieldTermStructure> c = ext::make_shared<FlatForward>(
        kToday, Handle<Quote>(ext::make_shared<SimpleQuote>(0.035)), dc365(), Compounded, Annual);
    return c;
}

//! Describe a settlement date + everything the Python test needs to rebuild it.
Obj settleInputs(const std::string& bond, const Date& settle) {
    Obj in;
    in.s("bond", bond);
    in.d("settlement_date", settle);
    return in;
}

// ---------------------------------------------------------------------------
// The whole date/coupon inspector surface for one (bond, settlement) pair.
// ---------------------------------------------------------------------------
void inspectorCase(const std::string& name, const std::string& bondId, const Bond& bond,
                   const Date& settle) {
    Obj ex;
    ex.d("start_date", BondFunctions::startDate(bond));
    ex.d("maturity_date", BondFunctions::maturityDate(bond));
    ex.b("is_tradable", BondFunctions::isTradable(bond, settle));
    ex.n("notional", bond.notional(settle));

    ex.d("previous_cash_flow_date", BondFunctions::previousCashFlowDate(bond, settle));
    ex.d("next_cash_flow_date", BondFunctions::nextCashFlowDate(bond, settle));
    ex.n("previous_cash_flow_amount", BondFunctions::previousCashFlowAmount(bond, settle));
    ex.n("next_cash_flow_amount", BondFunctions::nextCashFlowAmount(bond, settle));

    guarded(ex, "previous_coupon_rate",
            [&] { return BondFunctions::previousCouponRate(bond, settle); });
    guarded(ex, "next_coupon_rate", [&] { return BondFunctions::nextCouponRate(bond, settle); });

    guardedDate(ex, "accrual_start_date",
                [&] { return BondFunctions::accrualStartDate(bond, settle); });
    guardedDate(ex, "accrual_end_date", [&] { return BondFunctions::accrualEndDate(bond, settle); });
    guardedDate(ex, "reference_period_start",
                [&] { return BondFunctions::referencePeriodStart(bond, settle); });
    guardedDate(ex, "reference_period_end",
                [&] { return BondFunctions::referencePeriodEnd(bond, settle); });
    guarded(ex, "accrual_period", [&] { return BondFunctions::accrualPeriod(bond, settle); });
    guarded(ex, "accrual_days",
            [&] { return Real(BondFunctions::accrualDays(bond, settle)); });
    guarded(ex, "accrued_period", [&] { return BondFunctions::accruedPeriod(bond, settle); });
    guarded(ex, "accrued_days", [&] { return Real(BondFunctions::accruedDays(bond, settle)); });
    // accruedAmount does NOT throw when non tradable: it returns 0.0.
    ex.n("accrued_amount", BondFunctions::accruedAmount(bond, settle));

    addCase(name, settleInputs(bondId, settle), ex);
}

// ---------------------------------------------------------------------------
// The YieldTermStructure + yield + z-spread surface for one (bond, settlement).
// ---------------------------------------------------------------------------
void pricingCase(const std::string& name, const std::string& bondId, const Bond& bond,
                 const Date& settle, Rate flatYield, Compounding comp, Frequency freq,
                 Real cleanQuote, Spread zSpread) {
    const ext::shared_ptr<YieldTermStructure> curve = discountCurve();
    const InterestRate y(flatYield, dc365(), comp, freq);

    Obj in = settleInputs(bondId, settle);
    in.n("curve_flat_rate", 0.035);
    in.s("curve_day_counter", "Actual365Fixed");
    in.s("curve_compounding", "Compounded");
    in.s("curve_frequency", "Annual");
    in.d("curve_reference_date", kToday);
    in.n("yield", flatYield);
    in.s("yield_day_counter", "Actual365Fixed");
    in.i("yield_compounding", static_cast<long long>(comp));
    in.i("yield_frequency", static_cast<long long>(freq));
    in.n("clean_quote", cleanQuote);
    in.n("z_spread", zSpread);

    Obj ex;
    // --- YieldTermStructure group ---
    guarded(ex, "clean_price_curve", [&] { return BondFunctions::cleanPrice(bond, *curve, settle); });
    guarded(ex, "dirty_price_curve", [&] { return BondFunctions::dirtyPrice(bond, *curve, settle); });
    guarded(ex, "bps_curve", [&] { return BondFunctions::bps(bond, *curve, settle); });
    guarded(ex, "atm_rate_no_price",
            [&] { return BondFunctions::atmRate(bond, *curve, settle); });
    guarded(ex, "atm_rate_clean_price", [&] {
        return BondFunctions::atmRate(bond, *curve, settle, Bond::Price(cleanQuote, Bond::Price::Clean));
    });
    guarded(ex, "atm_rate_dirty_price", [&] {
        return BondFunctions::atmRate(bond, *curve, settle, Bond::Price(cleanQuote, Bond::Price::Dirty));
    });

    // --- InterestRate group ---
    guarded(ex, "clean_price_ir", [&] { return BondFunctions::cleanPrice(bond, y, settle); });
    guarded(ex, "dirty_price_ir", [&] { return BondFunctions::dirtyPrice(bond, y, settle); });
    guarded(ex, "bps_ir", [&] { return BondFunctions::bps(bond, y, settle); });
    guarded(ex, "duration_simple_ir",
            [&] { return BondFunctions::duration(bond, y, Duration::Simple, settle); });
    guarded(ex, "duration_modified_ir",
            [&] { return BondFunctions::duration(bond, y, Duration::Modified, settle); });
    guarded(ex, "duration_macaulay_ir",
            [&] { return BondFunctions::duration(bond, y, Duration::Macaulay, settle); });
    guarded(ex, "convexity_ir", [&] { return BondFunctions::convexity(bond, y, settle); });
    guarded(ex, "basis_point_value_ir",
            [&] { return BondFunctions::basisPointValue(bond, y, settle); });
    guarded(ex, "yield_value_basis_point_ir",
            [&] { return BondFunctions::yieldValueBasisPoint(bond, y, settle); });

    // --- Rate + DayCounter + Compounding + Frequency group ---
    // (Must agree with the InterestRate group exactly -- these overloads only
    //  build an InterestRate and forward.)
    guarded(ex, "clean_price_rate",
            [&] { return BondFunctions::cleanPrice(bond, flatYield, dc365(), comp, freq, settle); });
    guarded(ex, "dirty_price_rate",
            [&] { return BondFunctions::dirtyPrice(bond, flatYield, dc365(), comp, freq, settle); });
    guarded(ex, "bps_rate",
            [&] { return BondFunctions::bps(bond, flatYield, dc365(), comp, freq, settle); });
    guarded(ex, "duration_modified_rate", [&] {
        return BondFunctions::duration(bond, flatYield, dc365(), comp, freq, Duration::Modified,
                                       settle);
    });
    guarded(ex, "convexity_rate",
            [&] { return BondFunctions::convexity(bond, flatYield, dc365(), comp, freq, settle); });
    guarded(ex, "basis_point_value_rate", [&] {
        return BondFunctions::basisPointValue(bond, flatYield, dc365(), comp, freq, settle);
    });
    guarded(ex, "yield_value_basis_point_rate", [&] {
        return BondFunctions::yieldValueBasisPoint(bond, flatYield, dc365(), comp, freq, settle);
    });

    // --- yield (default NewtonSafe overload), clean and dirty quotes ---
    guarded(ex, "yield_clean", [&] {
        return BondFunctions::yield(bond, Bond::Price(cleanQuote, Bond::Price::Clean), dc365(),
                                    comp, freq, settle);
    });
    guarded(ex, "yield_dirty", [&] {
        return BondFunctions::yield(bond, Bond::Price(cleanQuote, Bond::Price::Dirty), dc365(),
                                    comp, freq, settle);
    });

    // --- z-spread group ---
    guarded(ex, "clean_price_zspread", [&] {
        return BondFunctions::cleanPrice(bond, curve, zSpread, comp, freq, settle);
    });
    guarded(ex, "dirty_price_zspread", [&] {
        return BondFunctions::dirtyPrice(bond, curve, zSpread, comp, freq, settle);
    });
    guarded(ex, "z_spread_from_clean", [&] {
        return BondFunctions::zSpread(bond, Bond::Price(cleanQuote, Bond::Price::Clean), curve,
                                      comp, freq, settle);
    });
    guarded(ex, "z_spread_from_dirty", [&] {
        return BondFunctions::zSpread(bond, Bond::Price(cleanQuote, Bond::Price::Dirty), curve,
                                      comp, freq, settle);
    });

    addCase(name, in, ex);
}

// ===========================================================================
// PART 1 driver
// ===========================================================================
void runBondFunctions() {
    const auto fixed = makeFixedBond();
    const auto amort = makeAmortisingBond();
    const auto zero = makeZeroBond();
    const auto floater = makeFloatingBond();

    // --- the market, emitted once so the reference is self-describing ---
    {
        Obj in;
        in.d("evaluation_date", kToday);
        in.s("calendar", "TARGET");
        in.i("settlement_days", 3);
        Obj ex;
        ex.d("fixed_settlement_date", fixed->settlementDate());
        ex.d("fixed_start_date", BondFunctions::startDate(*fixed));
        ex.d("fixed_maturity_date", BondFunctions::maturityDate(*fixed));
        ex.i("fixed_n_cashflows", static_cast<long long>(fixed->cashflows().size()));
        ex.d("amortising_settlement_date", amort->settlementDate());
        ex.d("amortising_start_date", BondFunctions::startDate(*amort));
        ex.d("amortising_maturity_date", BondFunctions::maturityDate(*amort));
        ex.i("amortising_n_cashflows", static_cast<long long>(amort->cashflows().size()));
        ex.d("zero_settlement_date", zero->settlementDate());
        ex.d("zero_start_date", BondFunctions::startDate(*zero));
        ex.d("zero_maturity_date", BondFunctions::maturityDate(*zero));
        ex.i("zero_n_cashflows", static_cast<long long>(zero->cashflows().size()));
        ex.d("floating_settlement_date", floater->settlementDate());
        ex.d("floating_start_date", BondFunctions::startDate(*floater));
        ex.d("floating_maturity_date", BondFunctions::maturityDate(*floater));
        ex.i("floating_n_cashflows", static_cast<long long>(floater->cashflows().size()));
        addCase("market", in, ex);
    }

    // --- default-settlement (Date()) path: must equal bond.settlementDate() ---
    {
        Obj in;
        in.s("bond", "fixed");
        in.s("settlement", "default (Date())");
        Obj ex;
        ex.d("default_settlement_date", fixed->settlementDate());
        ex.b("is_tradable_default", BondFunctions::isTradable(*fixed));
        ex.n("accrued_amount_default", BondFunctions::accruedAmount(*fixed));
        ex.n("accrued_amount_explicit",
             BondFunctions::accruedAmount(*fixed, fixed->settlementDate()));
        ex.d("next_cash_flow_date_default", BondFunctions::nextCashFlowDate(*fixed));
        ex.n("next_coupon_rate_default", BondFunctions::nextCouponRate(*fixed));
        addCase("fixed_default_settlement_date", in, ex);
    }

    // --- dense settlement grid on the fixed-rate bond ---
    inspectorCase("fixed_inspect_before_issue", "fixed", *fixed, Date(10, January, 2020));
    inspectorCase("fixed_inspect_mid_period", "fixed", *fixed, Date(20, May, 2025));
    inspectorCase("fixed_inspect_day_before_coupon", "fixed", *fixed, Date(14, July, 2025));
    inspectorCase("fixed_inspect_on_coupon", "fixed", *fixed, Date(15, July, 2025));
    inspectorCase("fixed_inspect_day_after_coupon", "fixed", *fixed, Date(16, July, 2025));
    inspectorCase("fixed_inspect_day_before_maturity", "fixed", *fixed, Date(14, January, 2030));
    inspectorCase("fixed_inspect_on_maturity", "fixed", *fixed, Date(15, January, 2030));
    inspectorCase("fixed_inspect_after_maturity", "fixed", *fixed, Date(16, January, 2030));

    // --- amortising bond: notional step function ---
    inspectorCase("amortising_inspect_mid_period", "amortising", *amort, Date(20, May, 2025));
    inspectorCase("amortising_inspect_on_amortisation", "amortising", *amort,
                  Date(15, January, 2024));
    inspectorCase("amortising_inspect_day_before_amortisation", "amortising", *amort,
                  Date(14, January, 2024));
    inspectorCase("amortising_inspect_on_maturity", "amortising", *amort, Date(15, January, 2026));

    // --- zero-coupon bond: no Coupon anywhere in the leg ---
    inspectorCase("zero_inspect_mid_period", "zero", *zero, Date(20, May, 2025));
    inspectorCase("zero_inspect_on_maturity", "zero", *zero, Date(15, January, 2030));
    inspectorCase("zero_inspect_after_maturity", "zero", *zero, Date(16, January, 2030));

    // --- floating bond: fixed vs forecast coupon branch ---
    inspectorCase("floating_inspect_fixed_coupon", "floating", *floater, Date(20, May, 2025));
    inspectorCase("floating_inspect_forecast_coupon", "floating", *floater, Date(20, August, 2025));
    inspectorCase("floating_inspect_on_coupon", "floating", *floater, Date(15, July, 2025));
    inspectorCase("floating_inspect_on_maturity", "floating", *floater, Date(17, January, 2028));

    // --- pricing surface ---
    pricingCase("fixed_pricing_mid_period", "fixed", *fixed, Date(20, May, 2025), 0.045,
                Compounded, Semiannual, 101.5, 0.0050);
    pricingCase("fixed_pricing_on_coupon", "fixed", *fixed, Date(15, July, 2025), 0.045, Compounded,
                Semiannual, 99.25, 0.0100);
    pricingCase("fixed_pricing_continuous", "fixed", *fixed, Date(20, May, 2025), 0.045,
                Continuous, NoFrequency, 101.5, 0.0050);
    pricingCase("fixed_pricing_simple", "fixed", *fixed, Date(20, May, 2025), 0.045, Simple,
                Semiannual, 101.5, 0.0050);
    pricingCase("fixed_pricing_on_maturity", "fixed", *fixed, Date(15, January, 2030), 0.045,
                Compounded, Semiannual, 101.5, 0.0050);
    pricingCase("amortising_pricing_mid_period", "amortising", *amort, Date(20, May, 2025), 0.045,
                Compounded, Annual, 100.75, 0.0050);
    pricingCase("zero_pricing_mid_period", "zero", *zero, Date(20, May, 2025), 0.04, Compounded,
                Annual, 82.0, 0.0050);
    pricingCase("floating_pricing_mid_period", "floating", *floater, Date(20, May, 2025), 0.035,
                Compounded, Semiannual, 100.25, 0.0050);

    // --- yield() templated on the solver ------------------------------------
    {
        const Date settle(20, May, 2025);
        const Bond::Price price(101.5, Bond::Price::Clean);
        Obj in = settleInputs("fixed", settle);
        in.n("clean_quote", 101.5);
        in.s("day_counter", "Actual365Fixed");
        in.s("compounding", "Compounded");
        in.s("frequency", "Semiannual");
        in.n("accuracy", 1.0e-10);
        in.n("guess", 0.05);

        Obj ex;
        {
            NewtonSafe s;
            s.setMaxEvaluations(100);
            ex.n("yield_newton_safe", BondFunctions::yield<NewtonSafe>(
                                          s, *fixed, price, dc365(), Compounded, Semiannual, settle));
        }
        {
            Brent s;
            s.setMaxEvaluations(100);
            ex.n("yield_brent", BondFunctions::yield<Brent>(s, *fixed, price, dc365(), Compounded,
                                                            Semiannual, settle));
        }
        {
            Bisection s;
            s.setMaxEvaluations(200);
            ex.n("yield_bisection", BondFunctions::yield<Bisection>(
                                        s, *fixed, price, dc365(), Compounded, Semiannual, settle));
        }
        {
            Ridder s;
            s.setMaxEvaluations(100);
            ex.n("yield_ridder", BondFunctions::yield<Ridder>(s, *fixed, price, dc365(),
                                                              Compounded, Semiannual, settle));
        }
        {
            FiniteDifferenceNewtonSafe s;
            s.setMaxEvaluations(100);
            ex.n("yield_fd_newton_safe",
                 BondFunctions::yield<FiniteDifferenceNewtonSafe>(s, *fixed, price, dc365(),
                                                                  Compounded, Semiannual, settle));
        }
        // The non-template overload IS NewtonSafe with maxIterations evaluations.
        ex.n("yield_default_overload",
             BondFunctions::yield(*fixed, price, dc365(), Compounded, Semiannual, settle));
        // A tighter accuracy and a hostile guess, to prove the parameters are wired.
        {
            NewtonSafe s;
            s.setMaxEvaluations(100);
            ex.n("yield_newton_safe_guess_50pc",
                 BondFunctions::yield<NewtonSafe>(s, *fixed, price, dc365(), Compounded, Semiannual,
                                                  settle, 1.0e-12, 0.50));
        }
        addCase("fixed_yield_solvers", in, ex);
    }

    // --- deprecated z-spread overloads: the DayCounter argument is discarded --
    {
        const Date settle(20, May, 2025);
        const auto curve = discountCurve();
        Obj in = settleInputs("fixed", settle);
        in.n("z_spread", 0.0050);
        in.s("note", "deprecated overloads take a DayCounter and ignore it");

        Obj ex;
        ex.n("clean_price_no_day_counter",
             BondFunctions::cleanPrice(*fixed, curve, 0.0050, Compounded, Semiannual, settle));
        QL_DEPRECATED_DISABLE_WARNING
        ex.n("clean_price_day_counter_a360",
             BondFunctions::cleanPrice(*fixed, curve, 0.0050, dc360(), Compounded, Semiannual,
                                       settle));
        ex.n("clean_price_day_counter_30360",
             BondFunctions::cleanPrice(*fixed, curve, 0.0050, dc30360(), Compounded, Semiannual,
                                       settle));
        ex.n("dirty_price_day_counter_a360",
             BondFunctions::dirtyPrice(*fixed, curve, 0.0050, dc360(), Compounded, Semiannual,
                                       settle));
        ex.n("z_spread_day_counter_a360",
             BondFunctions::zSpread(*fixed, Bond::Price(101.5, Bond::Price::Clean), curve, dc360(),
                                    Compounded, Semiannual, settle));
        QL_DEPRECATED_ENABLE_WARNING
        ex.n("dirty_price_no_day_counter",
             BondFunctions::dirtyPrice(*fixed, curve, 0.0050, Compounded, Semiannual, settle));
        ex.n("z_spread_no_day_counter",
             BondFunctions::zSpread(*fixed, Bond::Price(101.5, Bond::Price::Clean), curve,
                                    Compounded, Semiannual, settle));
        addCase("fixed_zspread_deprecated_daycounter_ignored", in, ex);
    }

    // --- z-spread round trip: zSpread(cleanPrice(s)) == s --------------------
    {
        const Date settle(20, May, 2025);
        const auto curve = discountCurve();
        const Spread s = 0.0125;
        const Real cp = BondFunctions::cleanPrice(*fixed, curve, s, Compounded, Semiannual, settle);
        Obj in = settleInputs("fixed", settle);
        in.n("z_spread_in", s);
        Obj ex;
        ex.n("clean_price", cp);
        ex.n("z_spread_out", BondFunctions::zSpread(*fixed, Bond::Price(cp, Bond::Price::Clean),
                                                    curve, Compounded, Semiannual, settle));
        addCase("fixed_zspread_round_trip", in, ex);
    }
}

// ===========================================================================
// PART 2 -- BachelierCalculator
// ===========================================================================

void bachelierCase(const std::string& name, Option::Type type, Real strike, Real forward,
                   Real stdDev, Real discount, Real spot, Time maturity) {
    Obj in;
    in.s("payoff", "PlainVanillaPayoff");
    in.s("option_type", type == Option::Call ? "Call" : "Put");
    in.n("strike", strike);
    in.n("forward", forward);
    in.n("std_dev", stdDev);
    in.n("discount", discount);
    in.n("spot", spot);
    in.n("maturity", maturity);

    Obj ex;
    try {
        BachelierCalculator c(ext::make_shared<PlainVanillaPayoff>(type, strike), forward, stdDev,
                              discount);
        ex.n("value", c.value());
        ex.n("delta_forward", c.deltaForward());
        ex.n("delta", c.delta(spot));
        ex.n("elasticity_forward", c.elasticityForward());
        ex.n("elasticity", c.elasticity(spot));
        ex.n("gamma_forward", c.gammaForward());
        ex.n("gamma", c.gamma(spot));
        ex.n("vega", c.vega(maturity));
        ex.n("rho", c.rho(maturity));
        ex.n("dividend_rho", c.dividendRho(maturity));
        ex.n("itm_cash_probability", c.itmCashProbability());
        ex.n("itm_asset_probability", c.itmAssetProbability());
        ex.n("strike_sensitivity", c.strikeSensitivity());
        ex.n("strike_gamma", c.strikeGamma());
        ex.n("vanna", c.vanna(maturity));
        ex.n("volga", c.volga(maturity));
        ex.n("alpha", c.alpha());
        ex.n("beta", c.beta());
        // theta is log(forward/spot)-based: only finite when forward/spot > 0.
        if (forward / spot > 0.0) {
            ex.n("theta", c.theta(spot, maturity));
            ex.n("theta_per_day", c.thetaPerDay(spot, maturity));
        }
        // The Option::Type + strike constructor must agree exactly.
        BachelierCalculator c2(type, strike, forward, stdDev, discount);
        ex.n("value_from_type_ctor", c2.value());
        ex.n("delta_forward_from_type_ctor", c2.deltaForward());
    } catch (const std::exception&) {
        ex.b("throws", true);
    }
    addCase(name, in, ex);
}

void runBachelier() {
    // --- classic positive-forward grid, both option types, ITM/ATM/OTM ---
    bachelierCase("bachelier_call_itm", Option::Call, 90.0, 100.0, 20.0, 0.95, 98.0, 1.0);
    bachelierCase("bachelier_call_atm", Option::Call, 100.0, 100.0, 20.0, 0.95, 98.0, 1.0);
    bachelierCase("bachelier_call_otm", Option::Call, 130.0, 100.0, 20.0, 0.95, 98.0, 1.0);
    bachelierCase("bachelier_put_itm", Option::Put, 130.0, 100.0, 20.0, 0.95, 98.0, 1.0);
    bachelierCase("bachelier_put_atm", Option::Put, 100.0, 100.0, 20.0, 0.95, 98.0, 1.0);
    bachelierCase("bachelier_put_otm", Option::Put, 70.0, 100.0, 20.0, 0.95, 98.0, 1.0);

    // --- the point of the Bachelier model: negative forward / negative strike ---
    bachelierCase("bachelier_call_neg_strike", Option::Call, -0.50, 0.25, 0.40, 0.99, 0.25, 2.0);
    bachelierCase("bachelier_put_neg_strike", Option::Put, -0.50, 0.25, 0.40, 0.99, 0.25, 2.0);
    bachelierCase("bachelier_call_neg_fwd_neg_strike", Option::Call, -1.00, -0.75, 0.40, 0.99,
                  -0.75, 2.0);
    bachelierCase("bachelier_put_neg_fwd_neg_strike", Option::Put, -0.50, -0.75, 0.40, 0.99, -0.75,
                  2.0);
    // forward < 0 < spot: theta's log(forward/spot) is NaN, so it is not pinned.
    bachelierCase("bachelier_call_neg_fwd_pos_spot", Option::Call, 0.00, -0.02, 0.30, 0.98, 0.05,
                  1.0);
    bachelierCase("bachelier_call_zero_fwd_zero_strike", Option::Call, 0.0, 0.0, 0.35, 1.0, 1.0,
                  1.0);

    // --- boundary parameters that drive the guards to exactly zero ---
    bachelierCase("bachelier_zero_stddev_atm", Option::Call, 100.0, 100.0, 0.0, 0.95, 98.0, 1.0);
    bachelierCase("bachelier_zero_stddev_itm_call", Option::Call, 90.0, 100.0, 0.0, 0.95, 98.0,
                  1.0);
    bachelierCase("bachelier_zero_stddev_otm_call", Option::Call, 110.0, 100.0, 0.0, 0.95, 98.0,
                  1.0);
    bachelierCase("bachelier_zero_stddev_itm_put", Option::Put, 110.0, 100.0, 0.0, 0.95, 98.0, 1.0);
    // A zero-vol ATM PUT is the only configuration in this grid that reaches the
    // `else return QL_MIN_REAL;` arm of elasticity()/elasticityForward():
    // value() == 0 (<= QL_EPSILON) while deltaForward() == discount*(0.5-1) < 0.
    // QL_MIN_REAL is -DBL_MAX (qldefines.hpp:176), NOT the smallest positive
    // double -- a port that maps it to sys.float_info.min gets the SIGN wrong.
    bachelierCase("bachelier_zero_stddev_atm_put", Option::Put, 100.0, 100.0, 0.0, 0.95, 98.0, 1.0);
    bachelierCase("bachelier_zero_maturity", Option::Call, 100.0, 100.0, 20.0, 0.95, 98.0, 0.0);
    bachelierCase("bachelier_discount_one", Option::Call, 100.0, 100.0, 20.0, 1.0, 100.0, 1.0);

    // --- QL_REQUIRE guards ---
    bachelierCase("bachelier_negative_stddev_throws", Option::Call, 100.0, 100.0, -1.0, 0.95, 98.0,
                  1.0);
    bachelierCase("bachelier_zero_discount_throws", Option::Call, 100.0, 100.0, 20.0, 0.0, 98.0,
                  1.0);
    bachelierCase("bachelier_negative_discount_throws", Option::Call, 100.0, 100.0, 20.0, -0.5,
                  98.0, 1.0);

    // --- non-plain payoffs: value() ignores them, greeks do not ---------------
    auto exotic = [](const std::string& name, const ext::shared_ptr<StrikedTypePayoff>& p,
                     Real forward, Real stdDev, Real discount, Real spot, Time maturity) {
        Obj in;
        in.s("payoff", p->name());
        in.s("option_type", p->optionType() == Option::Call ? "Call" : "Put");
        in.n("strike", p->strike());
        in.n("forward", forward);
        in.n("std_dev", stdDev);
        in.n("discount", discount);
        in.n("spot", spot);
        in.n("maturity", maturity);
        Obj ex;
        try {
            BachelierCalculator c(p, forward, stdDev, discount);
            ex.n("value", c.value());
            ex.n("alpha", c.alpha());
            ex.n("beta", c.beta());
            ex.n("delta_forward", c.deltaForward());
            ex.n("gamma_forward", c.gammaForward());
            ex.n("itm_cash_probability", c.itmCashProbability());
            ex.n("itm_asset_probability", c.itmAssetProbability());
            ex.n("strike_sensitivity", c.strikeSensitivity());
            ex.n("vega", c.vega(maturity));
            ex.n("rho", c.rho(maturity));
            ex.n("dividend_rho", c.dividendRho(maturity));
        } catch (const std::exception&) {
            ex.b("throws", true);
        }
        addCase(name, in, ex);
    };

    exotic("bachelier_con_call",
           ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 7.0), 100.0, 20.0, 0.95, 98.0,
           1.0);
    exotic("bachelier_con_put", ext::make_shared<CashOrNothingPayoff>(Option::Put, 100.0, 7.0),
           100.0, 20.0, 0.95, 98.0, 1.0);
    exotic("bachelier_aon_call",
           ext::make_shared<AssetOrNothingPayoff>(Option::Call, 100.0), 100.0, 20.0, 0.95, 98.0,
           1.0);
    // alpha_ = 1 - N(d) >= 0 for an AON PUT, so every `alpha_ >= 0` test below
    // takes the CALL branch. That is the C++ behaviour and it is pinned here.
    exotic("bachelier_aon_put", ext::make_shared<AssetOrNothingPayoff>(Option::Put, 100.0), 100.0,
           20.0, 0.95, 98.0, 1.0);
    exotic("bachelier_gap_call", ext::make_shared<GapPayoff>(Option::Call, 100.0, 105.0), 100.0,
           20.0, 0.95, 98.0, 1.0);
    // A payoff with no visitor overload must be rejected by Calculator::visit(Payoff&).
    exotic("bachelier_unsupported_payoff_throws",
           ext::make_shared<PercentageStrikePayoff>(Option::Call, 0.9), 100.0, 20.0, 0.95, 98.0,
           1.0);

    {
        // Not part of the bondswap cluster: a single BlackCalculator case whose
        // only purpose is to pin the QL_MIN_REAL arm of
        // BlackCalculator::elasticity / elasticityForward, because the already-
        // ported Python BlackCalculator maps QL_MIN_REAL to the smallest
        // POSITIVE double instead of -DBL_MAX. QL_MIN_REAL is
        //     #define QL_MIN_REAL -((std::numeric_limits<QL_REAL>::max)())
        // (qldefines.hpp:176). A ZERO-vol ATM put has value() == 0 (<=
        // QL_EPSILON) and deltaForward() == discount*(0.5 - 1) = -0.475, which
        // is exactly that arm.
        const Real forward = 100.0, strike = 100.0, stdDev = 0.0, discount = 0.95, spot = 98.0;
        BlackCalculator bc(ext::make_shared<PlainVanillaPayoff>(Option::Put, strike), forward,
                           stdDev, discount);
        Obj in;
        in.s("calculator", "BlackCalculator");
        in.s("option_type", "Put");
        in.n("strike", strike);
        in.n("forward", forward);
        in.n("std_dev", stdDev);
        in.n("discount", discount);
        in.n("spot", spot);
        Obj ex;
        ex.n("value", bc.value());
        ex.n("delta_forward", bc.deltaForward());
        ex.n("elasticity_forward", bc.elasticityForward());
        ex.n("elasticity", bc.elasticity(spot));
        ex.n("ql_min_real", QL_MIN_REAL);
        ex.n("ql_max_real", QL_MAX_REAL);
        addCase("black_calculator_elasticity_min_real_arm", in, ex);
    }
}

// ===========================================================================
// PART 3 -- RiskyBondEngine
// ===========================================================================

void runRiskyBondEngine() {
    const auto bond = makeFixedBond();
    const Handle<YieldTermStructure> yts(
        ext::make_shared<FlatForward>(kToday, 0.035, dc365(), Compounded, Annual));

    auto oneCase = [&](const std::string& name, Real hazardRate, Real recovery) {
        const Handle<DefaultProbabilityTermStructure> dts(
            ext::make_shared<FlatHazardRate>(kToday, hazardRate, dc365()));
        Obj in;
        in.s("bond", "fixed");
        in.n("hazard_rate", hazardRate);
        in.n("recovery_rate", recovery);
        in.n("yield_flat_rate", 0.035);
        in.s("yield_day_counter", "Actual365Fixed");
        in.s("yield_compounding", "Compounded");
        in.s("yield_frequency", "Annual");
        in.d("curve_reference_date", kToday);

        Obj ex;
        auto b = makeFixedBond();
        b->setPricingEngine(ext::make_shared<RiskyBondEngine>(dts, recovery, yts));
        ex.n("npv", b->NPV());
        ex.n("settlement_value", b->settlementValue());
        ex.n("clean_price", b->cleanPrice());
        ex.n("dirty_price", b->dirtyPrice());
        ex.d("settlement_date", b->settlementDate());
        addCase(name, in, ex);
    };

    // Zero hazard rate: every survival probability is 1 and every default
    // probability is 0, so the engine must reproduce the risk-free discounted
    // value exactly. Pinned alongside DiscountingBondEngine for that reason.
    oneCase("risky_bond_zero_hazard_rec40", 0.0, 0.4);
    oneCase("risky_bond_zero_hazard_rec0", 0.0, 0.0);
    oneCase("risky_bond_low_hazard", 0.01, 0.4);
    oneCase("risky_bond_high_hazard", 0.25, 0.4);
    oneCase("risky_bond_high_hazard_zero_recovery", 0.25, 0.0);
    oneCase("risky_bond_high_hazard_full_recovery", 0.25, 1.0);

    {
        // Risk-free reference: DiscountingBondEngine on the same curve. The
        // zero-hazard RiskyBondEngine value must equal this NPV. Note the two
        // engines report NPV at DIFFERENT dates -- RiskyBondEngine uses
        // yieldTS->referenceDate() while DiscountingBondEngine uses the same,
        // so they agree here by construction.
        auto b = makeFixedBond();
        b->setPricingEngine(ext::make_shared<DiscountingBondEngine>(yts));
        Obj in;
        in.s("bond", "fixed");
        in.s("engine", "DiscountingBondEngine");
        Obj ex;
        ex.n("npv", b->NPV());
        ex.n("settlement_value", b->settlementValue());
        ex.n("clean_price", b->cleanPrice());
        addCase("risky_bond_riskfree_reference", in, ex);
    }
}

// ===========================================================================
// PART 4 -- DiscountingFxForwardEngine
// ===========================================================================

void runFxForwardEngine() {
    const Handle<YieldTermStructure> eur(
        ext::make_shared<FlatForward>(kToday, 0.02, dc365(), Compounded, Annual));
    const Handle<YieldTermStructure> usd(
        ext::make_shared<FlatForward>(kToday, 0.045, dc365(), Compounded, Annual));

    auto oneCase = [&](const std::string& name, Real sourceNominal, Real targetNominal,
                       const Date& maturity, bool paySource, Real spot, Natural settlementDays) {
        Obj in;
        in.n("source_nominal", sourceNominal);
        in.s("source_currency", "EUR");
        in.n("target_nominal", targetNominal);
        in.s("target_currency", "USD");
        in.d("maturity_date", maturity);
        in.b("pay_source_currency", paySource);
        in.n("spot_fx", spot);
        in.i("settlement_days", static_cast<long long>(settlementDays));
        in.n("eur_flat_rate", 0.02);
        in.n("usd_flat_rate", 0.045);
        in.s("curve_day_counter", "Actual365Fixed");
        in.s("curve_compounding", "Compounded");
        in.s("curve_frequency", "Annual");
        in.d("curve_reference_date", kToday);
        in.s("payment_calendar", "TARGET");

        Obj ex;
        try {
            FxForward fx(sourceNominal, EURCurrency(), targetNominal, USDCurrency(), maturity,
                         paySource, settlementDays, target());
            fx.setPricingEngine(ext::make_shared<DiscountingFxForwardEngine>(
                eur, usd, Handle<Quote>(ext::make_shared<SimpleQuote>(spot))));
            ex.n("npv", fx.NPV());
            ex.n("fair_forward_rate", fx.fairForwardRate());
            ex.n("npv_source_currency", fx.npvSourceCurrency());
            ex.n("npv_target_currency", fx.npvTargetCurrency());
            ex.d("settlement_date", fx.settlementDate());
            const auto& ar = fx.additionalResults();
            ex.n("ar_spot_fx", ext::any_cast<Real>(ar.at("spotFx")));
            ex.n("ar_source_df", ext::any_cast<Real>(ar.at("sourceCurrencyDiscountFactor")));
            ex.n("ar_target_df", ext::any_cast<Real>(ar.at("targetCurrencyDiscountFactor")));
            ex.n("ar_source_settlement_df",
                 ext::any_cast<Real>(ar.at("sourceCurrencySettlementDiscountFactor")));
            ex.n("ar_target_settlement_df",
                 ext::any_cast<Real>(ar.at("targetCurrencySettlementDiscountFactor")));
            ex.n("ar_source_pv", ext::any_cast<Real>(ar.at("sourceCurrencyPV")));
            ex.n("ar_target_pv", ext::any_cast<Real>(ar.at("targetCurrencyPV")));
            ex.i("n_additional_results", static_cast<long long>(ar.size()));
        } catch (const std::exception&) {
            ex.b("throws", true);
        }
        addCase(name, in, ex);
    };

    const Date m1(15, May, 2026);
    oneCase("fxfwd_pay_source", 1'000'000.0, 1'100'000.0, m1, true, 1.08, 2);
    oneCase("fxfwd_receive_source", 1'000'000.0, 1'100'000.0, m1, false, 1.08, 2);
    // settlementDays = 0 -> settlement date == evaluation date == curve reference
    // date, which is the boundary of the two QL_REQUIREs (ref <= settlement).
    oneCase("fxfwd_zero_settlement_days", 1'000'000.0, 1'100'000.0, m1, true, 1.08, 0);
    oneCase("fxfwd_long_dated", 1'000'000.0, 1'250'000.0, Date(15, May, 2030), true, 1.08, 2);
    // At the fair forward the NPV must be (numerically) zero.
    oneCase("fxfwd_spot_far_from_strike", 1'000'000.0, 1'100'000.0, m1, true, 1.30, 2);
    oneCase("fxfwd_negative_spot_throws", 1'000'000.0, 1'100'000.0, m1, true, -1.08, 2);
    // Maturity BEFORE settlement: C++ has no guard for this. The
    // settlement-normalised discount factors invert (df > 1) and the engine
    // returns a perfectly finite number. Pinned so a port does not add a guard
    // C++ does not have.
    oneCase("fxfwd_maturity_before_settlement", 1'000'000.0, 1'100'000.0, Date(16, May, 2025), true,
            1.08, 5);
}

// ===========================================================================
// PART 5 -- TreeVanillaSwapEngine (and its two template base classes)
// ===========================================================================

void runTreeSwapEngine() {
    const Handle<YieldTermStructure> rts(
        ext::make_shared<FlatForward>(kToday, 0.03, dc365(), Compounded, Annual));
    const auto index = ext::make_shared<Euribor6M>(rts);
    const auto model = ext::make_shared<HullWhite>(rts, 0.05, 0.0075);

    Schedule fixedSch(Date(19, May, 2025), Date(19, May, 2030), Period(Annual), target(),
                      ModifiedFollowing, ModifiedFollowing, DateGeneration::Forward, false);
    Schedule floatSch(Date(19, May, 2025), Date(19, May, 2030), Period(Semiannual), target(),
                      ModifiedFollowing, ModifiedFollowing, DateGeneration::Forward, false);

    auto makeSwap = [&](Swap::Type type) {
        return ext::make_shared<VanillaSwap>(type, 1'000'000.0, fixedSch, 0.03, dc30360(), floatSch,
                                             index, 0.0, dc360());
    };

    // The mandatory times DiscretizedSwap asks for. The TimeGrid flavour of the
    // engine only works if the grid CONTAINS them (otherwise
    // TimeGrid::index throws "using inadequate tree"), so the Python test has
    // to build the grid the same way -- from DiscretizedSwap::mandatoryTimes().
    auto mandatoryTimes = [&](Swap::Type type) {
        auto swap = makeSwap(type);
        VanillaSwap::arguments args;
        swap->setupArguments(&args);
        DiscretizedSwap dswap(args, rts->referenceDate(), rts->dayCounter());
        return dswap.mandatoryTimes();
    };

    auto oneCase = [&](const std::string& name, Swap::Type type, Size timeSteps,
                       bool useTimeGrid) {
        Obj in;
        in.s("swap_type", type == Swap::Payer ? "Payer" : "Receiver");
        in.n("nominal", 1'000'000.0);
        in.n("fixed_rate", 0.03);
        in.s("fixed_day_counter", "Thirty360(BondBasis)");
        in.s("float_day_counter", "Actual360");
        in.s("index", "Euribor6M");
        in.n("hull_white_a", 0.05);
        in.n("hull_white_sigma", 0.0075);
        in.n("curve_flat_rate", 0.03);
        in.d("curve_reference_date", kToday);
        in.d("effective_date", Date(19, May, 2025));
        in.d("termination_date", Date(19, May, 2030));
        in.i("time_steps", static_cast<long long>(timeSteps));
        in.b("use_time_grid", useTimeGrid);
        in.s("time_grid_source",
             useTimeGrid ? "TimeGrid(DiscretizedSwap::mandatoryTimes(), timeSteps)" : "n/a");

        Obj ex;
        try {
            auto swap = makeSwap(type);
            if (useTimeGrid) {
                // The TimeGrid ctor builds the lattice EAGERLY inside
                // LatticeShortRateModelEngine and rebuilds it in update();
                // the timeSteps ctor leaves lattice_ null and
                // TreeVanillaSwapEngine::calculate() builds one per call.
                const std::vector<Time> times = mandatoryTimes(type);
                TimeGrid grid(times.begin(), times.end(), timeSteps);
                ex.i("grid_size", static_cast<long long>(grid.size()));
                ex.n("grid_back", grid.back());
                swap->setPricingEngine(ext::make_shared<TreeVanillaSwapEngine>(model, grid, rts));
            } else {
                swap->setPricingEngine(
                    ext::make_shared<TreeVanillaSwapEngine>(model, timeSteps, rts));
            }
            ex.n("npv", swap->NPV());
        } catch (const std::exception&) {
            ex.b("throws", true);
            addCase(name, in, ex);
            return;
        }
        // TreeVanillaSwapEngine::calculate() assigns ONLY results_.value. Every
        // other VanillaSwap result stays Null, so these accessors throw. Pinned
        // explicitly so a port does not invent them.
        auto accessorThrows = [&](Real (VanillaSwap::*f)() const) {
            auto s2 = makeSwap(type);
            s2->setPricingEngine(ext::make_shared<TreeVanillaSwapEngine>(model, timeSteps, rts));
            try {
                ((*s2).*f)();
                return false;
            } catch (const std::exception&) {
                return true;
            }
        };
        ex.b("fixed_leg_npv_throws", accessorThrows(&VanillaSwap::fixedLegNPV));
        ex.b("floating_leg_npv_throws", accessorThrows(&VanillaSwap::floatingLegNPV));
        ex.b("fair_rate_throws", accessorThrows(&VanillaSwap::fairRate));
        ex.b("fair_spread_throws", accessorThrows(&VanillaSwap::fairSpread));
        addCase(name, in, ex);
    };

    oneCase("tree_swap_payer_steps40", Swap::Payer, 40, false);
    oneCase("tree_swap_receiver_steps40", Swap::Receiver, 40, false);
    oneCase("tree_swap_payer_steps80", Swap::Payer, 80, false);
    // Same steps as tree_swap_payer_steps40 built from the same mandatory times
    // => the eager-lattice path must give the IDENTICAL number.
    oneCase("tree_swap_payer_timegrid40", Swap::Payer, 40, true);
    oneCase("tree_swap_receiver_timegrid40", Swap::Receiver, 40, true);
    // A denser grid => a different number, so a port that ignores the TimeGrid
    // constructor cannot pass both TimeGrid cases.
    oneCase("tree_swap_payer_timegrid200", Swap::Payer, 200, true);
    // timeSteps == 0 is rejected by LatticeShortRateModelEngine's QL_REQUIRE.
    oneCase("tree_swap_zero_steps_throws", Swap::Payer, 0, false);

    {
        // DiscountingSwapEngine on the same curve, for scale: the tree price
        // must be close but NOT equal (the lattice discretises the float leg).
        auto swap = makeSwap(Swap::Payer);
        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(rts));
        Obj in;
        in.s("engine", "DiscountingSwapEngine");
        Obj ex;
        ex.n("npv", swap->NPV());
        ex.n("fair_rate", swap->fairRate());
        addCase("tree_swap_discounting_reference", in, ex);
    }
}

// ===========================================================================
// PART 6 -- CounterpartyAdjSwapEngine
// ===========================================================================

void runCvaSwapEngine() {
    const Handle<YieldTermStructure> rts(
        ext::make_shared<FlatForward>(kToday, 0.03, dc365(), Compounded, Annual));
    const auto index = ext::make_shared<Euribor6M>(rts);

    auto makeSwap = [&](Swap::Type type) {
        Schedule fixedSch(Date(19, May, 2025), Date(19, May, 2030), Period(Annual), target(),
                          ModifiedFollowing, ModifiedFollowing, DateGeneration::Forward, false);
        Schedule floatSch(Date(19, May, 2025), Date(19, May, 2030), Period(Semiannual), target(),
                          ModifiedFollowing, ModifiedFollowing, DateGeneration::Forward, false);
        return ext::make_shared<VanillaSwap>(type, 1'000'000.0, fixedSch, 0.03, dc30360(), floatSch,
                                             index, 0.0, dc360());
    };

    auto oneCase = [&](const std::string& name, Swap::Type type, Real blackVol, Real ctptyHazard,
                       Real ctptyRecovery, bool withInvestor, Real invstHazard,
                       Real invstRecovery, bool useQuoteCtor) {
        Obj in;
        in.s("swap_type", type == Swap::Payer ? "Payer" : "Receiver");
        in.n("nominal", 1'000'000.0);
        in.n("fixed_rate", 0.03);
        in.n("black_vol", blackVol);
        in.n("ctpty_hazard_rate", ctptyHazard);
        in.n("ctpty_recovery_rate", ctptyRecovery);
        in.b("with_investor_curve", withInvestor);
        in.n("invst_hazard_rate", invstHazard);
        in.n("invst_recovery_rate", invstRecovery);
        in.b("use_quote_ctor", useQuoteCtor);
        in.n("curve_flat_rate", 0.03);
        in.d("curve_reference_date", kToday);
        in.d("effective_date", Date(19, May, 2025));
        in.d("termination_date", Date(19, May, 2030));

        Obj ex;
        try {
            const Handle<DefaultProbabilityTermStructure> ctpty(
                ext::make_shared<FlatHazardRate>(kToday, ctptyHazard, dc365()));
            Handle<DefaultProbabilityTermStructure> invst;
            if (withInvestor)
                invst = Handle<DefaultProbabilityTermStructure>(
                    ext::make_shared<FlatHazardRate>(kToday, invstHazard, dc365()));

            auto swap = makeSwap(type);
            ext::shared_ptr<PricingEngine> engine;
            if (useQuoteCtor) {
                engine = ext::make_shared<CounterpartyAdjSwapEngine>(
                    rts, Handle<Quote>(ext::make_shared<SimpleQuote>(blackVol)), ctpty,
                    ctptyRecovery, invst, invstRecovery);
            } else {
                engine = ext::make_shared<CounterpartyAdjSwapEngine>(rts, blackVol, ctpty,
                                                                     ctptyRecovery, invst,
                                                                     invstRecovery);
            }
            swap->setPricingEngine(engine);
            ex.n("npv", swap->NPV());
            ex.n("fair_rate", swap->fairRate());
        } catch (const std::exception&) {
            ex.b("throws", true);
        }
        addCase(name, in, ex);
    };

    // Both constructors, same vol -> identical numbers.
    oneCase("cva_payer_vol_ctor", Swap::Payer, 0.20, 0.02, 0.4, false, 0.0, 0.999, false);
    oneCase("cva_payer_quote_ctor", Swap::Payer, 0.20, 0.02, 0.4, false, 0.0, 0.999, true);
    oneCase("cva_receiver_vol_ctor", Swap::Receiver, 0.20, 0.02, 0.4, false, 0.0, 0.999, false);
    // Counterparty only vs both curves supplied.
    oneCase("cva_both_default_curves", Swap::Payer, 0.20, 0.02, 0.4, true, 0.01, 0.35, false);
    oneCase("cva_both_default_curves_receiver", Swap::Receiver, 0.20, 0.02, 0.4, true, 0.01, 0.35,
            false);
    // Vol and recovery knobs.
    oneCase("cva_high_vol", Swap::Payer, 0.60, 0.02, 0.4, false, 0.0, 0.999, false);
    oneCase("cva_zero_ctpty_hazard", Swap::Payer, 0.20, 0.0, 0.4, false, 0.0, 0.999, false);
    oneCase("cva_full_recovery", Swap::Payer, 0.20, 0.02, 1.0, false, 0.0, 0.999, false);

    {
        // Riskless reference on the same curve.
        auto swap = makeSwap(Swap::Payer);
        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(rts));
        Obj in;
        in.s("engine", "DiscountingSwapEngine");
        Obj ex;
        ex.n("npv", swap->NPV());
        ex.n("fair_rate", swap->fairRate());
        ex.n("fixed_leg_npv", swap->fixedLegNPV());
        ex.n("floating_leg_npv", swap->floatingLegNPV());
        addCase("cva_riskless_reference", in, ex);
    }
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    // Seed the floater's historical fixings before any bond is built.
    {
        const auto idx = ext::make_shared<Euribor6M>(forecastCurve());
        for (const auto& f : floaterFixings())
            idx->addFixing(f.first, f.second, true);
    }

    runBondFunctions();
    runBachelier();
    runRiskyBondEngine();
    runFxForwardEngine();
    runTreeSwapEngine();
    runCvaSwapEngine();

    emitDocument();
    return 0;
}
