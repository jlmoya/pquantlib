// migration-harness/cpp/probes/v143_inst_bondsbtp/probe.cpp
//
// instruments/bonds, part 2 — the Italian government-bond family from
// ql/instruments/bonds/btp.{hpp,cpp} plus ConvertibleFloatingRateBond from
// ql/instruments/bonds/convertiblebonds.{hpp,cpp}, at C++ QuantLib v1.43:
//
//   * CCTEU                                — Euribor6M-indexed floater
//   * BTP                                  — fixed-rate, both ctors
//   * RendistatoBasket                     — weights + the QL_REQUIRE branches
//   * RendistatoCalculator                 — LazyObject, equivalent swap
//   * RendistatoEquivalentSwapLengthQuote  — Quote adapter
//   * RendistatoEquivalentSwapSpreadQuote  — Quote adapter
//   * ConvertibleFloatingRateBond          — under the binomial TF engine
//
// WHAT IS PINNED AND WHY
// ----------------------
// These classes are mostly *wiring*: each one hands a hard-coded set of
// conventions to a general-purpose base (FloatingRateBond / FixedRateBond /
// MakeVanillaSwap / IborLeg). A single wrong convention — the NullCalendar in
// the schedule, the Unadjusted/Unadjusted pair, endOfMonth=true, the
// ModifiedFollowing+TARGET payment calendar of a BTP, the Actual/360 of a
// CCTEU, the ClosestRounding(5) on accruedAmount — moves one number and is
// invisible in an NPV alone, because two such errors can cancel. So for every
// bond we pin the COMPLETE cashflow listing (payment-date serial and amount
// for every flow; nominal, accrual start/end serials, accrual period, rate and
// ex-coupon serial for every coupon) as well as NPV / clean / dirty / accrued
// / yield.
//
// For RendistatoCalculator we pin not only yield()/duration()/
// equivalentSwapLength()/equivalentSwapSpread() but every per-bond and
// per-swap vector it exposes. The equivalent-swap index is chosen by walking
// swapBondDurations_ until it exceeds the basket duration, so a wrong basket
// weighting or a wrong swap-bond yield shifts the index and nothing else —
// only the vectors make that visible. Note that the walk `break`s, so the
// tail of swapRates_ / swapBondDurations_ is left at Null<Real>() and the tail
// of swapBondYields_ at its 0.05 seed; that observable tail is emitted too
// (Null<Real>() as JSON null).
//
// Every optional constructor argument is exercised at a NON-default value in
// its own variant, so an argument that is accepted and silently dropped fails
// here rather than years later.
//
// Deterministic inputs: evaluation date pinned; two DIFFERENT zero curves
// (forecast vs discount) built inline from literal tables so that a
// forecast/discount mix-up shows up; Euribor6M historic fixings are collected
// from the constructed coupons (every fixing date strictly before the
// evaluation date) and emitted, so the Python side replays exactly the same
// history.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/bondsbtp.json.

#include <ql/cashflows/coupon.hpp>
#include <ql/cashflows/floatingratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/exercise.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/bonds/btp.hpp>
#include <ql/instruments/bonds/convertiblebonds.hpp>
#include <ql/instruments/callabilityschedule.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/pricingengines/bond/binomialconvertibleengine.hpp>
#include <ql/pricingengines/bond/discountingbondengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/schedule.hpp>
#include <ql/utilities/null.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------- JSON ----

using KV = std::vector<std::string>;

std::string qs(const std::string& s) { return "\"" + s + "\""; }

std::string num(Real x) {
    if (x == Null<Real>())
        return "null";
    std::ostringstream os;
    os << std::setprecision(17) << x;
    return os.str();
}

std::string inum(long long v) { return std::to_string(v); }

std::string boolean(bool v) { return v ? "true" : "false"; }

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0)
            out += sep;
        out += v[i];
    }
    return out;
}

void add(KV& kv, const std::string& key, const std::string& value) {
    kv.push_back(qs(key) + ":" + value);
}

std::string obj(const KV& kv) { return "{" + join(kv, ",") + "}"; }

std::string arr(const std::vector<std::string>& items) {
    return "[" + join(items, ",") + "]";
}

std::string realArray(const std::vector<Real>& xs) {
    std::vector<std::string> items;
    items.reserve(xs.size());
    for (Real x : xs)
        items.push_back(num(x));
    return arr(items);
}

// ------------------------------------------------------- market inputs ----

// Deliberately NOT the 15th: a 15-Jun evaluation date makes the CCTEU accrual
// exactly 63/360, whose accrued amount already has 5 decimals — which would let
// a missing ClosestRounding(5) pass unnoticed. 16-Jun gives 64/360.
const Date kToday(16, June, 2026);

// Forecast curve: zero rates (Actual/365 Fixed, continuously compounded,
// linearly interpolated) on literal node dates out to 30 years.
const Date kCurveDates[] = {
    Date(16, June, 2026),     Date(15, December, 2026), Date(15, June, 2027),
    Date(15, June, 2028),     Date(15, June, 2029),     Date(15, June, 2031),
    Date(15, June, 2033),     Date(15, June, 2036),     Date(15, June, 2041),
    Date(15, June, 2046),     Date(15, June, 2056),
};
const Rate kForecastRates[] = {0.0200, 0.0215, 0.0230, 0.0250, 0.0265, 0.0285,
                               0.0300, 0.0315, 0.0330, 0.0340, 0.0350};
// Deliberately a DIFFERENT shape, so a forecast/discount swap is visible.
const Rate kDiscountRates[] = {0.0180, 0.0190, 0.0205, 0.0222, 0.0236, 0.0255,
                               0.0268, 0.0282, 0.0296, 0.0305, 0.0315};
const Size kCurveSize = sizeof(kCurveDates) / sizeof(kCurveDates[0]);

std::vector<Date> curveDates() {
    return {kCurveDates, kCurveDates + kCurveSize};
}

ext::shared_ptr<YieldTermStructure> zeroCurve(const Rate* rates) {
    std::vector<Rate> r(rates, rates + kCurveSize);
    return ext::make_shared<ZeroCurve>(curveDates(), r, Actual365Fixed());
}

// --------------------------------------------------------- leg emitter ----

std::string emitLeg(const Leg& leg) {
    std::vector<std::string> flows;
    flows.reserve(leg.size());
    for (const auto& cf : leg) {
        KV kv;
        add(kv, "date", inum(cf->date().serialNumber()));
        add(kv, "amount", num(cf->amount()));
        add(kv, "ex_coupon", inum(cf->exCouponDate().serialNumber()));
        auto cp = ext::dynamic_pointer_cast<Coupon>(cf);
        if (cp != nullptr) {
            add(kv, "is_coupon", boolean(true));
            add(kv, "nominal", num(cp->nominal()));
            add(kv, "accrual_start", inum(cp->accrualStartDate().serialNumber()));
            add(kv, "accrual_end", inum(cp->accrualEndDate().serialNumber()));
            add(kv, "accrual_period", num(cp->accrualPeriod()));
            add(kv, "rate", num(cp->rate()));
        } else {
            add(kv, "is_coupon", boolean(false));
        }
        flows.push_back(obj(kv));
    }
    return arr(flows);
}

// Common bond observables: everything a wrong convention would move.
KV bondCore(const Bond& bond) {
    KV kv;
    add(kv, "settlement_date", inum(bond.settlementDate().serialNumber()));
    add(kv, "issue_date", inum(bond.issueDate().serialNumber()));
    add(kv, "start_date", inum(bond.startDate().serialNumber()));
    add(kv, "maturity_date", inum(bond.maturityDate().serialNumber()));
    add(kv, "n_cashflows", inum(static_cast<long long>(bond.cashflows().size())));
    add(kv, "notional", num(bond.notional(bond.settlementDate())));
    add(kv, "accrued", num(bond.accruedAmount()));
    add(kv, "npv", num(bond.NPV()));
    add(kv, "clean_price", num(bond.cleanPrice()));
    add(kv, "dirty_price", num(bond.dirtyPrice()));
    add(kv, "settlement_value", num(bond.settlementValue()));
    add(kv, "cashflows", emitLeg(bond.cashflows()));
    return kv;
}

// -------------------------------------------------- historical fixings ----

// Collect every fixing date strictly before the evaluation date used by the
// coupons of `leg`; those are the fixings the Python side must replay.
void collectPastFixings(const Leg& leg, std::set<Date>& out) {
    for (const auto& cf : leg) {
        auto frc = ext::dynamic_pointer_cast<FloatingRateCoupon>(cf);
        if (frc != nullptr && frc->fixingDate() < kToday)
            out.insert(frc->fixingDate());
    }
}

// Deterministic synthetic history — arbitrary but reproducible, and varying
// enough that a coupon reading the wrong fixing date reads a wrong rate.
Rate syntheticFixing(Size i) { return 0.0175 + 0.0015 * static_cast<Real>(i % 7); }

} // namespace

int main() {
    std::cout << std::setprecision(17);

    Settings::instance().evaluationDate() = kToday;

    Handle<YieldTermStructure> fwdCurve(zeroCurve(kForecastRates));
    Handle<YieldTermStructure> discCurve(zeroCurve(kDiscountRates));

    auto euribor6m = ext::make_shared<Euribor6M>(fwdCurve);

    // =================================================================
    // Instruments — constructed BEFORE any fixing is added (construction
    // never needs a fixing; only amount()/rate() does).
    // =================================================================

    // --- CCTEU ---------------------------------------------------------
    // Seasoned floater: startDate/issueDate in the past, so accruedAmount()
    // and the ClosestRounding(5) it applies are actually exercised.
    CCTEU ccteu(Date(15, October, 2031), 0.0075, fwdCurve, Date(15, April, 2024),
                Date(15, April, 2024));
    // Non-default `spread` only.
    CCTEU ccteuSpread(Date(15, October, 2031), 0.0250, fwdCurve, Date(15, April, 2024),
                      Date(15, April, 2024));
    // Non-default `startDate` only (a shorter bond off the same maturity).
    CCTEU ccteuStart(Date(15, October, 2031), 0.0075, fwdCurve, Date(15, October, 2025),
                     Date(15, October, 2025));
    // Non-default `issueDate`: an issue date AFTER today+2 moves
    // settlementDate() (Bond::settlementDate = max(advance(today), issueDate)).
    CCTEU ccteuFutureIssue(Date(15, October, 2032), 0.0075, fwdCurve, Date(15, April, 2027),
                           Date(15, April, 2027));

    // --- BTP -----------------------------------------------------------
    BTP btp(Date(1, August, 2033), 0.0450, Date(1, August, 2023), Date(1, August, 2023));
    // Legacy non-par redemption ctor (the 5-arg overload).
    BTP btpRedemption(Date(15, May, 2037), 0.0300, 99.999, Date(15, May, 2017),
                      Date(15, May, 2017));

    // --- Rendistato basket ---------------------------------------------
    auto btpA = ext::make_shared<BTP>(Date(1, August, 2029), 0.0350, Date(1, August, 2019),
                                      Date(1, August, 2019));
    auto btpB = ext::make_shared<BTP>(Date(1, February, 2032), 0.0400, Date(1, February, 2022),
                                      Date(1, February, 2022));
    auto btpC = ext::make_shared<BTP>(Date(1, August, 2033), 0.0450, Date(1, August, 2023),
                                      Date(1, August, 2023));
    auto btpD = ext::make_shared<BTP>(Date(15, May, 2037), 0.0300, 99.999, Date(15, May, 2017),
                                      Date(15, May, 2017));
    std::vector<ext::shared_ptr<BTP> > btps = {btpA, btpB, btpC, btpD};
    std::vector<Real> outstandings = {25.0e9, 30.0e9, 20.0e9, 15.0e9};
    std::vector<ext::shared_ptr<SimpleQuote> > priceQuotes = {
        ext::make_shared<SimpleQuote>(101.25), ext::make_shared<SimpleQuote>(103.10),
        ext::make_shared<SimpleQuote>(105.40), ext::make_shared<SimpleQuote>(92.75)};
    std::vector<Handle<Quote> > cleanPrices;
    for (const auto& q : priceQuotes)
        cleanPrices.emplace_back(q);

    auto basket = ext::make_shared<RendistatoBasket>(btps, outstandings, cleanPrices);
    auto calculator = ext::make_shared<RendistatoCalculator>(basket, euribor6m, discCurve);

    // --- ConvertibleFloatingRateBond -----------------------------------
    // Month-end schedule so that the ex-coupon endOfMonth flag has an
    // observable effect (30-Jun payment dates: -1M with EOM lands on 31-May,
    // without EOM on 30-May).
    Schedule convSchedule(Date(30, June, 2024), Date(30, June, 2034), Period(6, Months),
                          TARGET(), Following, Following, DateGeneration::Backward, true);
    const Date convIssue(30, June, 2024);
    const Date convMaturity = convSchedule.endDate();

    // --- fixings -------------------------------------------------------
    // Collect from every floating leg we will price, then replay.
    std::set<Date> pastFixings;
    collectPastFixings(ccteu.cashflows(), pastFixings);
    collectPastFixings(ccteuSpread.cashflows(), pastFixings);
    collectPastFixings(ccteuStart.cashflows(), pastFixings);
    collectPastFixings(ccteuFutureIssue.cashflows(), pastFixings);
    {
        // The convertible-floater variants shift fixing dates via fixingDays,
        // so build throwaway legs for each fixingDays value we use.
        for (Natural fd : {Natural(2), Natural(5)}) {
            Leg probeLeg = IborLeg(convSchedule, euribor6m)
                               .withPaymentDayCounter(Actual360())
                               .withNotionals(100.0)
                               .withPaymentAdjustment(convSchedule.businessDayConvention())
                               .withFixingDays(fd)
                               .withSpreads(std::vector<Spread>(1, 0.005));
            collectPastFixings(probeLeg, pastFixings);
        }
    }

    std::vector<std::string> fixingItems;
    {
        Size i = 0;
        for (const Date& d : pastFixings) {
            Rate r = syntheticFixing(i++);
            euribor6m->addFixing(d, r);
            KV kv;
            add(kv, "date", inum(d.serialNumber()));
            add(kv, "value", num(r));
            fixingItems.push_back(obj(kv));
        }
    }

    // --- engines -------------------------------------------------------
    auto bondEngine = ext::make_shared<DiscountingBondEngine>(discCurve);
    ccteu.setPricingEngine(bondEngine);
    ccteuSpread.setPricingEngine(bondEngine);
    ccteuStart.setPricingEngine(bondEngine);
    ccteuFutureIssue.setPricingEngine(bondEngine);
    btp.setPricingEngine(bondEngine);
    btpRedemption.setPricingEngine(bondEngine);

    const ActualActual isma(ActualActual::ISMA);

    // =================================================================
    // Emission
    // =================================================================
    std::vector<std::string> top;

    // ---- market inputs, so Python replays exactly this scenario -------
    {
        KV kv;
        add(kv, "evaluation_date", inum(kToday.serialNumber()));
        std::vector<std::string> ds;
        for (const Date& d : curveDates())
            ds.push_back(inum(d.serialNumber()));
        add(kv, "curve_dates", arr(ds));
        add(kv, "forecast_rates",
            realArray(std::vector<Real>(kForecastRates, kForecastRates + kCurveSize)));
        add(kv, "discount_rates",
            realArray(std::vector<Real>(kDiscountRates, kDiscountRates + kCurveSize)));
        add(kv, "euribor6m_fixings", arr(fixingItems));
        top.push_back(qs("market") + ":" + obj(kv));
    }

    // ---- CCTEU --------------------------------------------------------
    {
        KV kv = bondCore(ccteu);
        // ClosestRounding(5) is applied on top of FloatingRateBond::accruedAmount;
        // both are pinned so a missing rounding cannot hide.
        add(kv, "accrued_unrounded",
            num(ccteu.FloatingRateBond::accruedAmount(ccteu.settlementDate())));
        add(kv, "yield", num(ccteu.yield(isma, Compounded, Annual)));
        top.push_back(qs("ccteu") + ":" + obj(kv));
    }
    {
        KV kv = bondCore(ccteuSpread);
        add(kv, "accrued_unrounded",
            num(ccteuSpread.FloatingRateBond::accruedAmount(ccteuSpread.settlementDate())));
        top.push_back(qs("ccteu_spread") + ":" + obj(kv));
    }
    {
        KV kv = bondCore(ccteuStart);
        top.push_back(qs("ccteu_start") + ":" + obj(kv));
    }
    {
        KV kv = bondCore(ccteuFutureIssue);
        top.push_back(qs("ccteu_future_issue") + ":" + obj(kv));
    }

    // ---- BTP ----------------------------------------------------------
    {
        KV kv = bondCore(btp);
        add(kv, "accrued_unrounded",
            num(btp.FixedRateBond::accruedAmount(btp.settlementDate())));
        add(kv, "frequency", inum(static_cast<long long>(btp.frequency())));
        add(kv, "day_counter", qs(btp.dayCounter().name()));
        // NOTE: BTP::yield(Real, ...) HIDES every Bond::yield overload in C++, so the
// engine-driven yield is only reachable through explicit qualification.
        add(kv, "yield_engine", num(btp.Bond::yield(isma, Compounded, Annual)));
        // BTP::yield(cleanPrice, ...) — the BTP-convention yield helper.
        add(kv, "btp_yield_default", num(btp.yield(101.75)));
        add(kv, "btp_yield_at_date",
            num(btp.yield(101.75, Date(3, August, 2026), 1.0e-12, 200)));
        add(kv, "btp_yield_other_price", num(btp.yield(97.25)));
        top.push_back(qs("btp") + ":" + obj(kv));
    }
    {
        KV kv = bondCore(btpRedemption);
        add(kv, "btp_yield_default", num(btpRedemption.yield(93.50)));
        top.push_back(qs("btp_redemption") + ":" + obj(kv));
    }

    // ---- RendistatoBasket ---------------------------------------------
    {
        KV kv;
        add(kv, "size", inum(static_cast<long long>(basket->size())));
        add(kv, "outstanding", num(basket->outstanding()));
        add(kv, "weights", realArray(basket->weights()));
        add(kv, "outstandings", realArray(basket->outstandings()));
        add(kv, "n_quotes", inum(static_cast<long long>(basket->cleanPriceQuotes().size())));
        add(kv, "n_btps", inum(static_cast<long long>(basket->btps().size())));
        top.push_back(qs("basket") + ":" + obj(kv));
    }
    {
        // Same bonds, DIFFERENT outstandings: size() is unchanged, weights are not.
        std::vector<Real> other = {10.0e9, 10.0e9, 40.0e9, 40.0e9};
        auto reweighted = ext::make_shared<RendistatoBasket>(btps, other, cleanPrices);
        KV kv;
        add(kv, "size", inum(static_cast<long long>(reweighted->size())));
        add(kv, "outstanding", num(reweighted->outstanding()));
        add(kv, "weights", realArray(reweighted->weights()));
        top.push_back(qs("basket_reweighted") + ":" + obj(kv));
    }
    {
        // The four QL_REQUIRE branches of the RendistatoBasket ctor.
        KV kv;
        auto raises = [&](const std::vector<ext::shared_ptr<BTP> >& bs,
                          const std::vector<Real>& os,
                          const std::vector<Handle<Quote> >& qsv) {
            try {
                RendistatoBasket b(bs, os, qsv);
                return false;
            } catch (...) {
                return true;
            }
        };
        add(kv, "empty_basket",
            obj({qs("raises") + ":" +
                 boolean(raises({}, {}, {}))}));
        add(kv, "outstandings_size_mismatch",
            obj({qs("raises") + ":" +
                 boolean(raises(btps, {25.0e9, 30.0e9, 20.0e9}, cleanPrices))}));
        add(kv, "quotes_size_mismatch",
            obj({qs("raises") + ":" +
                 boolean(raises(btps, outstandings,
                                {cleanPrices[0], cleanPrices[1], cleanPrices[2]}))}));
        add(kv, "negative_outstanding",
            obj({qs("raises") + ":" +
                 boolean(raises(btps, {25.0e9, -1.0, 20.0e9, 15.0e9}, cleanPrices))}));
        top.push_back(qs("basket_raises") + ":" + obj(kv));
    }

    // ---- RendistatoCalculator -----------------------------------------
    auto emitCalculator = [&](const RendistatoCalculator& r) {
        KV kv;
        add(kv, "yield", num(r.yield()));
        add(kv, "duration", num(r.duration()));
        add(kv, "yields", realArray(r.yields()));
        add(kv, "durations", realArray(r.durations()));
        add(kv, "swap_lengths", realArray(r.swapLengths()));
        add(kv, "swap_rates", realArray(r.swapRates()));
        add(kv, "swap_yields", realArray(r.swapYields()));
        add(kv, "swap_durations", realArray(r.swapDurations()));
        add(kv, "equivalent_swap_length", num(r.equivalentSwapLength()));
        add(kv, "equivalent_swap_rate", num(r.equivalentSwapRate()));
        add(kv, "equivalent_swap_yield", num(r.equivalentSwapYield()));
        add(kv, "equivalent_swap_duration", num(r.equivalentSwapDuration()));
        add(kv, "equivalent_swap_spread", num(r.equivalentSwapSpread()));
        add(kv, "equivalent_swap_fixed_rate", num(r.equivalentSwap()->fixedRate()));
        add(kv, "equivalent_swap_fair_rate", num(r.equivalentSwap()->fairRate()));
        add(kv, "equivalent_swap_fixed_dates",
            inum(static_cast<long long>(r.equivalentSwap()->fixedSchedule().size())));
        return obj(kv);
    };

    auto lengthQuote = ext::make_shared<RendistatoEquivalentSwapLengthQuote>(calculator);
    auto spreadQuote = ext::make_shared<RendistatoEquivalentSwapSpreadQuote>(calculator);

    top.push_back(qs("rendistato") + ":" + emitCalculator(*calculator));
    {
        KV kv;
        add(kv, "length_value", num(lengthQuote->value()));
        add(kv, "length_is_valid", boolean(lengthQuote->isValid()));
        add(kv, "spread_value", num(spreadQuote->value()));
        add(kv, "spread_is_valid", boolean(spreadQuote->isValid()));
        top.push_back(qs("quotes_before") + ":" + obj(kv));
    }

    // Bump a basket clean-price quote: RendistatoBasket observes the quotes,
    // RendistatoCalculator observes the basket, so the LazyObject must
    // invalidate and both adapter quotes must move.
    priceQuotes[0]->setValue(95.00);
    priceQuotes[3]->setValue(88.00);

    top.push_back(qs("rendistato_after_bump") + ":" + emitCalculator(*calculator));
    {
        KV kv;
        add(kv, "length_value", num(lengthQuote->value()));
        add(kv, "length_is_valid", boolean(lengthQuote->isValid()));
        add(kv, "spread_value", num(spreadQuote->value()));
        add(kv, "spread_is_valid", boolean(spreadQuote->isValid()));
        top.push_back(qs("quotes_after") + ":" + obj(kv));
    }
    // restore
    priceQuotes[0]->setValue(101.25);
    priceQuotes[3]->setValue(92.75);

    {
        // isValid() must be false when the underlying calculation throws:
        // an invalid (Null-valued) clean-price quote makes
        // performCalculations() raise inside RendistatoCalculator.
        std::vector<Handle<Quote> > brokenQuotes = {
            Handle<Quote>(ext::make_shared<SimpleQuote>()), cleanPrices[1], cleanPrices[2],
            cleanPrices[3]};
        auto brokenBasket = ext::make_shared<RendistatoBasket>(btps, outstandings, brokenQuotes);
        auto brokenCalc =
            ext::make_shared<RendistatoCalculator>(brokenBasket, euribor6m, discCurve);
        RendistatoEquivalentSwapLengthQuote lq(brokenCalc);
        RendistatoEquivalentSwapSpreadQuote sq(brokenCalc);
        KV kv;
        add(kv, "length_is_valid", boolean(lq.isValid()));
        add(kv, "spread_is_valid", boolean(sq.isValid()));
        top.push_back(qs("quotes_invalid") + ":" + obj(kv));
    }

    // ---- ConvertibleFloatingRateBond ----------------------------------
    {
        DayCounter convDc = Actual360();
        Handle<Quote> underlying(ext::make_shared<SimpleQuote>(50.0));
        Handle<YieldTermStructure> divYield(
            ext::make_shared<FlatForward>(kToday, 0.02, convDc));
        Handle<YieldTermStructure> riskFree(
            ext::make_shared<FlatForward>(kToday, 0.05, convDc));
        Handle<BlackVolTermStructure> vol(
            ext::make_shared<BlackConstantVol>(kToday, TARGET(), 0.15, convDc));
        auto process = ext::make_shared<BlackScholesMertonProcess>(underlying, divYield,
                                                                   riskFree, vol);
        const Size timeSteps = 201;

        ext::shared_ptr<Exercise> europeanExercise =
            ext::make_shared<EuropeanExercise>(convMaturity);
        ext::shared_ptr<Exercise> americanExercise =
            ext::make_shared<AmericanExercise>(convIssue, convMaturity);

        CallabilitySchedule noCall;
        CallabilitySchedule callPut;
        callPut.push_back(ext::make_shared<SoftCallability>(
            Bond::Price(108.0, Bond::Price::Clean), Date(30, June, 2029), 1.10));
        callPut.push_back(ext::make_shared<Callability>(
            Bond::Price(101.0, Bond::Price::Clean), Callability::Put, Date(30, June, 2031)));

        struct Variant {
            const char* key;
            bool american;
            Real conversionRatio;
            Real creditSpread;
            Spread legSpread;
            bool withCallPut;
            Real redemption;
            Natural fixingDays;
            Natural settlementDays;
            Period exCouponPeriod;
            bool exCouponUseNullCalendar;
            BusinessDayConvention exCouponConvention;
            bool exCouponEndOfMonth;
        };

        const Period noPeriod;
        const Period oneMonth(1, Months);
        // key, american, convRatio, creditSpread, legSpread, callPut, redemption,
        // fixingDays, settlementDays, exPeriod, exNullCalendar, exConvention, exEOM
        const std::vector<Variant> variants = {
            // base — every optional argument at its DEFAULT
            {"base", false, 2.0, 0.005, 0.005, false, 100.0, 2, 3, noPeriod, false, Unadjusted,
             false},
            {"american", true, 2.0, 0.005, 0.005, false, 100.0, 2, 3, noPeriod, false, Unadjusted,
             false},
            // one non-default at a time
            {"conv_ratio_3", false, 3.0, 0.005, 0.005, false, 100.0, 2, 3, noPeriod, false,
             Unadjusted, false},
            {"credit_spread_2pc", false, 2.0, 0.020, 0.005, false, 100.0, 2, 3, noPeriod, false,
             Unadjusted, false},
            {"leg_spread_2pc", false, 2.0, 0.005, 0.020, false, 100.0, 2, 3, noPeriod, false,
             Unadjusted, false},
            {"call_put", true, 2.0, 0.005, 0.005, true, 100.0, 2, 3, noPeriod, false, Unadjusted,
             false},
            {"redemption_102", false, 2.0, 0.005, 0.005, false, 102.0, 2, 3, noPeriod, false,
             Unadjusted, false},
            {"fixing_days_5", false, 2.0, 0.005, 0.005, false, 100.0, 5, 3, noPeriod, false,
             Unadjusted, false},
            {"settlement_days_10", false, 2.0, 0.005, 0.005, false, 100.0, 2, 10, noPeriod, false,
             Unadjusted, false},
            // the four ex-coupon arguments, each moved off its default in turn
            {"ex_coupon_1m", false, 2.0, 0.005, 0.005, false, 100.0, 2, 3, oneMonth, false,
             Unadjusted, false},
            {"ex_coupon_1m_preceding", false, 2.0, 0.005, 0.005, false, 100.0, 2, 3, oneMonth,
             false, Preceding, false},
            {"ex_coupon_1m_nullcal_preceding", false, 2.0, 0.005, 0.005, false, 100.0, 2, 3,
             oneMonth, true, Preceding, false},
            {"ex_coupon_1m_eom", false, 2.0, 0.005, 0.005, false, 100.0, 2, 3, oneMonth, false,
             Unadjusted, true},
        };

        std::vector<std::string> items;
        for (const Variant& v : variants) {
            // A default-constructed Calendar() is what the C++ default argument
            // supplies; the leg builder then falls back to the schedule calendar.
            Calendar exCal = Calendar();
            if (v.exCouponPeriod != Period())
                exCal = v.exCouponUseNullCalendar ? Calendar(NullCalendar()) : Calendar(TARGET());
            ConvertibleFloatingRateBond bond(
                v.american ? americanExercise : europeanExercise, v.conversionRatio,
                v.withCallPut ? callPut : noCall, convIssue, v.settlementDays, euribor6m,
                v.fixingDays, std::vector<Spread>(1, v.legSpread), convDc, convSchedule,
                v.redemption, v.exCouponPeriod, exCal, v.exCouponConvention,
                v.exCouponEndOfMonth);
            bond.setPricingEngine(ext::make_shared<BinomialConvertibleEngine<CoxRossRubinstein> >(
                process, timeSteps, Handle<Quote>(ext::make_shared<SimpleQuote>(v.creditSpread))));

            KV kv;
            add(kv, "key", qs(v.key));
            add(kv, "npv", num(bond.NPV()));
            add(kv, "conversion_ratio", num(bond.conversionRatio()));
            add(kv, "n_callability",
                inum(static_cast<long long>(bond.callability().size())));
            add(kv, "settlement_date", inum(bond.settlementDate().serialNumber()));
            add(kv, "settlement_days", inum(static_cast<long long>(bond.settlementDays())));
            add(kv, "maturity_date", inum(bond.maturityDate().serialNumber()));
            add(kv, "accrued", num(bond.accruedAmount()));
            add(kv, "n_cashflows", inum(static_cast<long long>(bond.cashflows().size())));
            add(kv, "cashflows", emitLeg(bond.cashflows()));
            items.push_back(obj(kv));
        }

        KV kv;
        add(kv, "time_steps", inum(static_cast<long long>(timeSteps)));
        add(kv, "spot", num(50.0));
        add(kv, "dividend_yield", num(0.02));
        add(kv, "risk_free_rate", num(0.05));
        add(kv, "volatility", num(0.15));
        add(kv, "schedule_start", inum(convSchedule.startDate().serialNumber()));
        add(kv, "schedule_end", inum(convMaturity.serialNumber()));
        add(kv, "call_date", inum(Date(30, June, 2029).serialNumber()));
        add(kv, "put_date", inum(Date(30, June, 2031).serialNumber()));
        add(kv, "variants", arr(items));
        top.push_back(qs("convertible_frn") + ":" + obj(kv));
    }

    std::cout << "{\n  " << join(top, ",\n  ") << "\n}\n";
    return 0;
}
