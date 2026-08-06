// migration-harness/cpp/probes/v143_quotes_tail/probe.cpp
//
// Reference values for the seven ql/quotes/* classes PQuantLib was missing
// against C++ QuantLib v1.43:
//
//   ImpliedStdDevQuote                   ql/quotes/impliedstddevquote.{hpp,cpp}
//   EurodollarFuturesImpliedStdDevQuote  ql/quotes/eurodollarfuturesquote.{hpp,cpp}
//   ForwardValueQuote                    ql/quotes/forwardvaluequote.{hpp,cpp}
//   ForwardSwapQuote                     ql/quotes/forwardswapquote.{hpp,cpp}
//   FuturesConvAdjustmentQuote           ql/quotes/futuresconvadjustmentquote.{hpp,cpp}
//   LastFixingQuote                      ql/quotes/lastfixingquote.{hpp,cpp}
//   MultiCompositeQuote                  ql/quotes/multicompositequote.hpp
//
// The evaluation date is PINNED to 2026-06-15 (a Monday) for the whole run.
// Four of the seven read Settings::instance().evaluationDate() — directly
// (FuturesConvAdjustmentQuote::value, LastFixingQuote::referenceDate,
// ForwardSwapQuote::initializeDates) or through Index::fixing — so anything
// emitted here is wall-clock dependent unless the date is nailed down. The
// pytest module pins the same date; see the `kToday` constant below.
//
// What is pinned and why:
//
//   * ImpliedStdDevQuote — value() for Call and Put, ATM and away from the
//     money. `accuracy` and `maxIter` are ALSO probed at values that change
//     the answer, because both are trailing defaulted scalars and an
//     accepted-then-discarded scalar is the defect class this port is prone
//     to: `coarse_accuracy` stops the Newton-safe solve early (so the answer
//     differs from the converged root AND depends on `guess`), and
//     `max_iter_1` exhausts the evaluation budget, which raises inside
//     performCalculations and is swallowed to 0.0 by the try/catch.
//     `restart_uses_previous_result_as_guess` pins the mutable-member
//     semantics: impliedStdev_ seeds the NEXT solve, so a coarse-accuracy
//     re-solve after the price moves lands somewhere a fresh solve would not.
//     `empty_forward_is_swallowed` vs `empty_price_throws` pin the exact
//     boundary of the try block — price_->value() is dereferenced OUTSIDE it
//     and forward_->value() INSIDE, so the two empty handles behave
//     differently. A port that wraps the whole body in one try/except gets
//     `empty_price_throws` wrong.
//
//   * EurodollarFuturesImpliedStdDevQuote — BOTH branches of the
//     strike_ > forwardValue test, with deliberately different call and put
//     prices so a port that took the wrong branch (or the wrong price) cannot
//     reproduce either. Note the two inversions this class performs and the
//     probe pins: the constructor stores strike_ = 100 - strike, and the
//     branch that fires when the (rate) strike is ABOVE the (rate) forward
//     prices a CALL off the PUT quote. isValid() is probed with the
//     branch-irrelevant price handle empty (still valid) and with the
//     branch-relevant one empty (invalid).
//
//   * ForwardValueQuote — a forecast fixing (date after the evaluation date)
//     and a historical one (date before it, served from the fixing history),
//     plus isValid() which is unconditionally true even for a date with no
//     fixing at all.
//
//   * ForwardSwapQuote — valueDate/startDate/fixingDate AND value(), at a
//     NON-ZERO forward start and a NON-ZERO spread (both are silently
//     droppable), with the zero-spread and the empty-spread-handle cases
//     alongside so the spread's contribution is visible in the reference
//     itself. The evaluation date is then moved and every date and the value
//     re-read, which is the only thing that exercises update()'s
//     initializeDates() re-snap.
//
//   * FuturesConvAdjustmentQuote — value() and all four inspectors, from BOTH
//     constructors (explicit date and IMM code) which must agree, at two
//     distinct (volatility, meanReversion) pairs so neither can be swapped or
//     ignored. The value is re-read after the evaluation date moves, because
//     rate_ is a cache keyed on nothing but the update() notification.
//
//   * LastFixingQuote — referenceDate() and value() with the evaluation date
//     AFTER the last stored fixing (min() picks the fixing) and BEFORE it
//     (min() picks the evaluation date), which is the whole content of the
//     class; plus isValid()/value() on an index with an empty history.
//
//   * MultiCompositeQuote — value() through a function whose coefficients
//     differ per element, so element ORDER is pinned, not just membership;
//     inputValue(i) for every i; isValid() with an empty handle and with an
//     invalid quote; the recompute after an element moves; and the
//     "invalid MultiCompositeQuote" failure.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/quotes/tail.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/currencies/europe.hpp>
#include <ql/handle.hpp>
#include <ql/index.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/math/array.hpp>
#include <ql/option.hpp>
#include <ql/quotes/eurodollarfuturesquote.hpp>
#include <ql/quotes/forwardswapquote.hpp>
#include <ql/quotes/forwardvaluequote.hpp>
#include <ql/quotes/futuresconvadjustmentquote.hpp>
#include <ql/quotes/impliedstddevquote.hpp>
#include <ql/quotes/lastfixingquote.hpp>
#include <ql/quotes/multicompositequote.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/imm.hpp>

using namespace QuantLib;

namespace {

// A Monday, so nothing below rolls off a weekend by accident. Every scenario
// in this file is evaluated with Settings::instance().evaluationDate() equal
// to this date; the pytest module pins the same one.
const Date kToday(15, June, 2026);
// Used only by the two update()-driven scenarios (ForwardSwapQuote and
// FuturesConvAdjustmentQuote). A Wednesday, one week on.
const Date kLater(24, June, 2026);

bool gFirst = true;

void key(const std::string& k) {
    if (!gFirst)
        std::cout << ",\n";
    gFirst = false;
    std::cout << "  \"" << k << "\": ";
}

void emitReal(const std::string& k, Real v) {
    key(k);
    std::cout << v;
}

void emitInt(const std::string& k, long v) {
    key(k);
    std::cout << v;
}

void emitBool(const std::string& k, bool v) {
    key(k);
    std::cout << (v ? "true" : "false");
}

void emitString(const std::string& k, const std::string& v) {
    key(k);
    std::cout << "\"" << v << "\"";
}

// Evaluates `f`, emitting either <k> = the value, or <k>_raises = true.
template <class F>
void emitRealOrRaises(const std::string& k, F f) {
    try {
        Real v = f();
        emitReal(k, v);
        emitBool(k + "_raises", false);
    } catch (std::exception&) {
        emitBool(k + "_raises", true);
    }
}

Handle<Quote> quote(Real v) {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(v));
}

// ---------------------------------------------------------------------------
// ImpliedStdDevQuote
// ---------------------------------------------------------------------------

void emitImpliedStdDev(const std::string& k,
                       Option::Type type,
                       Real forward,
                       Real price,
                       Real strike,
                       Real guess,
                       Real accuracy,
                       Natural maxIter) {
    ImpliedStdDevQuote q(type, quote(forward), quote(price), strike, guess, accuracy, maxIter);
    emitRealOrRaises(k, [&] { return q.value(); });
    emitBool(k + "_is_valid", q.isValid());
}

void sectionImpliedStdDev() {
    // Converged solves: Call/Put, ATM and away from the money, at an accuracy
    // tight enough that the answer is the root and not an artefact of the path.
    emitImpliedStdDev("isdq_call_atm", Option::Call, 100.0, 5.0, 100.0, 0.15, 1.0e-10, 100);
    emitImpliedStdDev("isdq_put_itm", Option::Put, 100.0, 12.0, 110.0, 0.30, 1.0e-10, 100);
    emitImpliedStdDev("isdq_call_otm_rate", Option::Call, 0.05, 0.002, 0.06, 0.25, 1.0e-10, 100);
    emitImpliedStdDev("isdq_put_otm_rate", Option::Put, 0.05, 0.0015, 0.04, 0.25, 1.0e-10, 100);
    // Same inputs as isdq_call_atm at the DEFAULT accuracy (1e-6) and maxIter
    // (100), reached through the defaulted trailing parameters.
    {
        ImpliedStdDevQuote q(Option::Call, quote(100.0), quote(5.0), 100.0, 0.15);
        emitReal("isdq_defaults", q.value());
    }
    // Coarse accuracy: the solve stops early, so the answer differs from the
    // converged root above AND depends on the guess. Two guesses, one below
    // and one above the root, to make the guess-dependence explicit.
    emitImpliedStdDev("isdq_coarse_low_guess", Option::Call, 100.0, 5.0, 100.0, 0.02, 0.5, 100);
    emitImpliedStdDev("isdq_coarse_high_guess", Option::Call, 100.0, 5.0, 100.0, 2.0, 0.5, 100);
    // maxIter exhausted -> Error inside performCalculations -> swallowed to 0.
    emitImpliedStdDev("isdq_max_iter_1", Option::Call, 100.0, 5.0, 100.0, 2.0, 1.0e-10, 1);
    // Price below intrinsic: blackFormulaImpliedStdDev rejects it (negative
    // put implied by put-call parity) -> swallowed to 0.
    emitImpliedStdDev("isdq_price_below_intrinsic", Option::Call, 100.0, 0.5, 90.0, 0.15, 1.0e-10,
                      100);

    // Invalid inputs: isValid() false, and value() still returns something
    // (0.0) because the failing dereference happens inside the try block.
    {
        ImpliedStdDevQuote q(Option::Call, quote(100.0),
                             Handle<Quote>(ext::make_shared<SimpleQuote>()), 100.0, 0.15, 1.0e-10,
                             100);
        emitBool("isdq_invalid_price_is_valid", q.isValid());
    }
    {
        ImpliedStdDevQuote q(Option::Call, Handle<Quote>(ext::make_shared<SimpleQuote>()),
                             quote(5.0), 100.0, 0.15, 1.0e-10, 100);
        emitBool("isdq_invalid_forward_is_valid", q.isValid());
    }
    // Empty handles. forward_->value() is INSIDE the try, price_->value() is
    // NOT: the two behave differently.
    {
        ImpliedStdDevQuote q(Option::Call, Handle<Quote>(), quote(5.0), 100.0, 0.15, 1.0e-10, 100);
        emitBool("isdq_empty_forward_is_valid", q.isValid());
        emitRealOrRaises("isdq_empty_forward_value", [&] { return q.value(); });
    }
    {
        ImpliedStdDevQuote q(Option::Call, quote(100.0), Handle<Quote>(), 100.0, 0.15, 1.0e-10, 100);
        emitBool("isdq_empty_price_is_valid", q.isValid());
        emitRealOrRaises("isdq_empty_price_value", [&] { return q.value(); });
    }

    // Mutable-guess semantics: the second solve is seeded with the FIRST
    // solve's result, not with the constructor's guess. At coarse accuracy
    // that is observable.
    {
        auto price = ext::make_shared<SimpleQuote>(5.0);
        ImpliedStdDevQuote q(Option::Call, quote(100.0), Handle<Quote>(price), 100.0, 2.0, 0.5,
                             100);
        emitReal("isdq_restart_first", q.value());
        price->setValue(9.0);
        emitReal("isdq_restart_second", q.value());
        // A fresh quote on the same inputs, seeded from the constructor guess,
        // lands elsewhere — which is what makes the case above meaningful.
        ImpliedStdDevQuote fresh(Option::Call, quote(100.0), quote(9.0), 100.0, 2.0, 0.5, 100);
        emitReal("isdq_restart_fresh", fresh.value());
    }
}

// ---------------------------------------------------------------------------
// EurodollarFuturesImpliedStdDevQuote
// ---------------------------------------------------------------------------

void sectionEurodollar() {
    // forward quote is a FUTURES PRICE: 94.85 -> rate forward of 5.15.
    // strike is likewise a futures price and the ctor stores 100 - strike.
    const Real kForwardPrice = 94.85;
    const Real kCallPrice = 0.25;
    const Real kPutPrice = 0.30;

    // strike 94.0 -> strike_ = 6.0 > forwardValue 5.15 -> Call priced off the
    // PUT quote.
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), quote(kCallPrice),
                                              quote(kPutPrice), 94.0, 0.15, 1.0e-10, 100);
        emitReal("edf_put_branch", q.value());
        emitBool("edf_put_branch_is_valid", q.isValid());
    }
    // strike 95.0 -> strike_ = 5.0 <= forwardValue 5.15 -> Put priced off the
    // CALL quote.
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), quote(kCallPrice),
                                              quote(kPutPrice), 95.0, 0.15, 1.0e-10, 100);
        emitReal("edf_call_branch", q.value());
        emitBool("edf_call_branch_is_valid", q.isValid());
    }
    // Defaulted guess (.15), accuracy (1e-6) and maxIter (100).
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), quote(kCallPrice),
                                              quote(kPutPrice), 95.0);
        emitReal("edf_defaults", q.value());
    }
    // Coarse accuracy makes the guess observable.
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), quote(kCallPrice),
                                              quote(kPutPrice), 95.0, 2.0, 0.5, 100);
        emitReal("edf_coarse_high_guess", q.value());
    }
    // maxIter exhausted. Unlike ImpliedStdDevQuote this class does NOT catch,
    // so the Error propagates out of value().
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), quote(kCallPrice),
                                              quote(kPutPrice), 95.0, 2.0, 1.0e-10, 1);
        emitRealOrRaises("edf_max_iter_1", [&] { return q.value(); });
    }

    // isValid(): on the put branch only the put handle matters, and vice versa.
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), Handle<Quote>(),
                                              quote(kPutPrice), 94.0, 0.15, 1.0e-10, 100);
        emitBool("edf_put_branch_empty_call_is_valid", q.isValid());
    }
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), quote(kCallPrice),
                                              Handle<Quote>(), 94.0, 0.15, 1.0e-10, 100);
        emitBool("edf_put_branch_empty_put_is_valid", q.isValid());
    }
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), quote(kCallPrice),
                                              Handle<Quote>(), 95.0, 0.15, 1.0e-10, 100);
        emitBool("edf_call_branch_empty_put_is_valid", q.isValid());
    }
    {
        EurodollarFuturesImpliedStdDevQuote q(quote(kForwardPrice), Handle<Quote>(),
                                              quote(kPutPrice), 95.0, 0.15, 1.0e-10, 100);
        emitBool("edf_call_branch_empty_call_is_valid", q.isValid());
    }
    {
        EurodollarFuturesImpliedStdDevQuote q(Handle<Quote>(), quote(kCallPrice), quote(kPutPrice),
                                              95.0, 0.15, 1.0e-10, 100);
        emitBool("edf_empty_forward_is_valid", q.isValid());
    }
    {
        EurodollarFuturesImpliedStdDevQuote q(Handle<Quote>(ext::make_shared<SimpleQuote>()),
                                              quote(kCallPrice), quote(kPutPrice), 95.0, 0.15,
                                              1.0e-10, 100);
        emitBool("edf_invalid_forward_is_valid", q.isValid());
    }
}

// ---------------------------------------------------------------------------
// Shared index / curve plumbing for the index-backed quotes
// ---------------------------------------------------------------------------

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kToday, r, Actual365Fixed()));
}

// A private family name per scenario: IndexManager keys the fixing history on
// the index NAME, so two scenarios sharing a name would share fixings.
ext::shared_ptr<IborIndex> makeIbor(const std::string& family,
                                    const Period& tenor,
                                    const Handle<YieldTermStructure>& curve) {
    return ext::make_shared<IborIndex>(family, tenor, 2, EURCurrency(), TARGET(),
                                       ModifiedFollowing, false, Actual360(), curve);
}

// ---------------------------------------------------------------------------
// ForwardValueQuote
// ---------------------------------------------------------------------------

void sectionForwardValue() {
    auto index = makeIbor("FVQtest", 6 * Months, flatCurve(0.0325));
    // A past fixing, so the historical branch of Index::fixing has something
    // to return.
    const Date past(5, June, 2026);
    index->addFixing(past, 0.0287);

    {
        const Date fixingDate(17, September, 2026);
        ForwardValueQuote q(index, fixingDate);
        emitInt("fvq_forecast_fixing_serial", fixingDate.serialNumber());
        emitReal("fvq_forecast_value", q.value());
        emitBool("fvq_forecast_is_valid", q.isValid());
    }
    {
        ForwardValueQuote q(index, past);
        emitInt("fvq_past_fixing_serial", past.serialNumber());
        emitReal("fvq_past_value", q.value());
        emitBool("fvq_past_is_valid", q.isValid());
    }
    {
        // isValid() is unconditionally true, even for a date whose fixing
        // cannot be produced at all (a Sunday is not a valid fixing date).
        ForwardValueQuote q(index, Date(14, June, 2026));
        emitBool("fvq_unfixable_is_valid", q.isValid());
        emitRealOrRaises("fvq_unfixable_value", [&] { return q.value(); });
    }
    emitString("fvq_index_name", index->name());
}

// ---------------------------------------------------------------------------
// ForwardSwapQuote
// ---------------------------------------------------------------------------

void emitForwardSwap(const std::string& k, ForwardSwapQuote& q) {
    emitInt(k + "_value_date_serial", q.valueDate().serialNumber());
    emitInt(k + "_start_date_serial", q.startDate().serialNumber());
    emitInt(k + "_fixing_date_serial", q.fixingDate().serialNumber());
    emitReal(k + "_value", q.value());
    emitBool(k + "_is_valid", q.isValid());
}

void sectionForwardSwap() {
    auto curve = flatCurve(0.0325);
    auto ibor = makeIbor("FSQtest", 6 * Months, curve);
    auto swapIndex = ext::make_shared<SwapIndex>(
        "FSQtestSwap", 5 * Years, 2, EURCurrency(), TARGET(), 1 * Years, Unadjusted,
        Thirty360(Thirty360::BondBasis), ibor);

    const Period fwdStart(3, Months);
    {
        ForwardSwapQuote q(swapIndex, quote(0.0025), fwdStart);
        emitForwardSwap("fsq_spread25bp", q);
    }
    {
        ForwardSwapQuote q(swapIndex, quote(0.0), fwdStart);
        emitForwardSwap("fsq_spread0", q);
    }
    {
        // Empty spread handle: treated as zero spread, and isValid() skips the
        // spread check entirely.
        ForwardSwapQuote q(swapIndex, Handle<Quote>(), fwdStart);
        emitForwardSwap("fsq_spread_empty", q);
    }
    {
        // Zero forward start, to pin that fwdStart is threaded at all.
        ForwardSwapQuote q(swapIndex, quote(0.0025), 0 * Days);
        emitForwardSwap("fsq_fwdstart0", q);
    }
    {
        // Invalid spread quote -> isValid() false, but the dates and the
        // swap-side calculation are unaffected.
        ForwardSwapQuote q(swapIndex, Handle<Quote>(ext::make_shared<SimpleQuote>()), fwdStart);
        emitBool("fsq_invalid_spread_is_valid", q.isValid());
    }
    {
        // update() re-snaps every date off the new evaluation date.
        ForwardSwapQuote q(swapIndex, quote(0.0025), fwdStart);
        emitForwardSwap("fsq_before_roll", q);
        Settings::instance().evaluationDate() = kLater;
        emitForwardSwap("fsq_after_roll", q);
        Settings::instance().evaluationDate() = kToday;
        emitForwardSwap("fsq_after_roll_back", q);
    }
}

// ---------------------------------------------------------------------------
// FuturesConvAdjustmentQuote
// ---------------------------------------------------------------------------

void emitFuturesConvAdj(const std::string& k, FuturesConvAdjustmentQuote& q) {
    emitReal(k + "_value", q.value());
    emitReal(k + "_futures_value", q.futuresValue());
    emitReal(k + "_volatility", q.volatility());
    emitReal(k + "_mean_reversion", q.meanReversion());
    emitInt(k + "_imm_date_serial", q.immDate().serialNumber());
    emitBool(k + "_is_valid", q.isValid());
}

void sectionFuturesConvAdj() {
    auto index = makeIbor("FCAtest", 3 * Months, flatCurve(0.0325));
    const std::string immCode = "U6";
    const Date immDate = IMM::date(immCode);
    emitInt("fca_imm_code_resolves_to_serial", immDate.serialNumber());
    emitInt("fca_index_maturity_serial", index->maturityDate(immDate).serialNumber());

    {
        FuturesConvAdjustmentQuote q(index, immDate, quote(97.85), quote(0.011), quote(0.03));
        emitFuturesConvAdj("fca_by_date", q);
    }
    {
        // The IMM-code constructor must land on exactly the same numbers.
        FuturesConvAdjustmentQuote q(index, immCode, quote(97.85), quote(0.011), quote(0.03));
        emitFuturesConvAdj("fca_by_imm_code", q);
    }
    {
        // Distinct volatility / meanReversion, so neither can be swapped for
        // the other nor dropped.
        FuturesConvAdjustmentQuote q(index, immDate, quote(97.85), quote(0.02), quote(0.05));
        emitFuturesConvAdj("fca_alt_params", q);
    }
    {
        // Same numbers, volatility and meanReversion exchanged: a port that
        // mixed the two arguments up reproduces fca_alt_params but not this.
        FuturesConvAdjustmentQuote q(index, immDate, quote(97.85), quote(0.05), quote(0.02));
        emitFuturesConvAdj("fca_params_swapped", q);
    }
    {
        FuturesConvAdjustmentQuote q(index, immDate, quote(97.85), quote(0.011), quote(0.03));
        emitReal("fca_before_roll_value", q.value());
        Settings::instance().evaluationDate() = kLater;
        emitReal("fca_after_roll_value", q.value());
        Settings::instance().evaluationDate() = kToday;
        emitReal("fca_after_roll_back_value", q.value());
    }
    {
        FuturesConvAdjustmentQuote q(index, immDate,
                                     Handle<Quote>(ext::make_shared<SimpleQuote>()), quote(0.011),
                                     quote(0.03));
        emitBool("fca_invalid_futures_is_valid", q.isValid());
    }
    {
        FuturesConvAdjustmentQuote q(index, immDate, quote(97.85),
                                     Handle<Quote>(ext::make_shared<SimpleQuote>()), quote(0.03));
        emitBool("fca_invalid_vol_is_valid", q.isValid());
    }
    {
        FuturesConvAdjustmentQuote q(index, immDate, quote(97.85), quote(0.011),
                                     Handle<Quote>(ext::make_shared<SimpleQuote>()));
        emitBool("fca_invalid_mr_is_valid", q.isValid());
    }
    {
        FuturesConvAdjustmentQuote q(index, immDate, Handle<Quote>(), quote(0.011), quote(0.03));
        emitBool("fca_empty_futures_is_valid", q.isValid());
    }
}

// ---------------------------------------------------------------------------
// LastFixingQuote
// ---------------------------------------------------------------------------

void sectionLastFixing() {
    auto index = makeIbor("LFQtest", 6 * Months, flatCurve(0.0325));
    // Every TARGET business day from 2026-06-01 to 2026-06-12, so any
    // referenceDate() the min() can produce inside the window has a fixing.
    const int kDays[] = {1, 2, 3, 4, 5, 8, 9, 10, 11, 12};
    Real v = 0.0301;
    for (int d : kDays) {
        index->addFixing(Date(d, June, 2026), v);
        v += 0.0001;
    }
    emitInt("lfq_last_fixing_date_serial", index->timeSeries().lastDate().serialNumber());

    {
        // Evaluation date AFTER the last fixing -> min() picks the fixing.
        LastFixingQuote q(index);
        emitInt("lfq_after_reference_serial", q.referenceDate().serialNumber());
        emitReal("lfq_after_value", q.value());
        emitBool("lfq_after_is_valid", q.isValid());
        emitString("lfq_index_name", q.index()->name());
    }
    {
        // Evaluation date BEFORE the last fixing -> min() picks the evaluation
        // date, so the quote reads back through the history, not off the end.
        Settings::instance().evaluationDate() = Date(9, June, 2026);
        LastFixingQuote q(index);
        emitInt("lfq_before_reference_serial", q.referenceDate().serialNumber());
        emitReal("lfq_before_value", q.value());
        emitBool("lfq_before_is_valid", q.isValid());
        Settings::instance().evaluationDate() = kToday;
    }
    {
        // Empty history: invalid, and value() refuses.
        auto empty = makeIbor("LFQempty", 6 * Months, flatCurve(0.0325));
        LastFixingQuote q(empty);
        emitBool("lfq_empty_is_valid", q.isValid());
        emitRealOrRaises("lfq_empty_value", [&] { return q.value(); });
    }
}

// ---------------------------------------------------------------------------
// MultiCompositeQuote
// ---------------------------------------------------------------------------

void sectionMultiComposite() {
    // Per-element coefficients: the reference pins element ORDER, not just the
    // set of inputs.
    auto weighted = [](Array a) { return a[0] + 10.0 * a[1] + 100.0 * a[2]; };

    auto e0 = ext::make_shared<SimpleQuote>(2.0);
    auto e1 = ext::make_shared<SimpleQuote>(3.0);
    auto e2 = ext::make_shared<SimpleQuote>(5.0);
    std::vector<Handle<Quote>> elems{Handle<Quote>(e0), Handle<Quote>(e1), Handle<Quote>(e2)};

    {
        MultiCompositeQuote<decltype(weighted)> q(elems, weighted);
        emitReal("mcq_weighted_value", q.value());
        emitBool("mcq_weighted_is_valid", q.isValid());
        emitReal("mcq_input_value_0", q.inputValue(0));
        emitReal("mcq_input_value_1", q.inputValue(1));
        emitReal("mcq_input_value_2", q.inputValue(2));
        emitRealOrRaises("mcq_input_value_3", [&] { return q.inputValue(3); });
        // An element moves -> update() drops the cache -> next value() differs.
        e1->setValue(4.0);
        emitReal("mcq_after_element_moves", q.value());
        e1->setValue(3.0);
    }
    {
        // A non-linear function over the whole Array, to pin that the callback
        // really receives every element.
        auto norm = [](Array a) {
            Real s = 0.0;
            for (Real x : a)
                s += x * x;
            return std::sqrt(s);
        };
        MultiCompositeQuote<decltype(norm)> q(elems, norm);
        emitReal("mcq_norm_value", q.value());
        emitInt("mcq_norm_size", static_cast<long>(elems.size()));
    }
    {
        // One invalid element -> whole quote invalid, and value() refuses.
        std::vector<Handle<Quote>> withInvalid{Handle<Quote>(e0),
                                               Handle<Quote>(ext::make_shared<SimpleQuote>()),
                                               Handle<Quote>(e2)};
        MultiCompositeQuote<decltype(weighted)> q(withInvalid, weighted);
        emitBool("mcq_invalid_element_is_valid", q.isValid());
        emitRealOrRaises("mcq_invalid_element_value", [&] { return q.value(); });
    }
    {
        // One empty handle -> same.
        std::vector<Handle<Quote>> withEmpty{Handle<Quote>(e0), Handle<Quote>(),
                                             Handle<Quote>(e2)};
        MultiCompositeQuote<decltype(weighted)> q(withEmpty, weighted);
        emitBool("mcq_empty_element_is_valid", q.isValid());
        emitRealOrRaises("mcq_empty_element_value", [&] { return q.value(); });
    }
    {
        // Empty element list: vacuously valid, and the function is handed a
        // zero-length Array.
        auto count = [](Array a) { return static_cast<Real>(a.size()); };
        MultiCompositeQuote<decltype(count)> q(std::vector<Handle<Quote>>{}, count);
        emitBool("mcq_no_elements_is_valid", q.isValid());
        emitReal("mcq_no_elements_value", q.value());
    }
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = kToday;

    std::cout << "{\n";
    emitInt("evaluation_date_serial", kToday.serialNumber());
    emitInt("roll_date_serial", kLater.serialNumber());

    sectionImpliedStdDev();
    sectionEurodollar();
    sectionForwardValue();
    sectionForwardSwap();
    sectionFuturesConvAdj();
    sectionLastFixing();
    sectionMultiComposite();

    std::cout << "\n}" << std::endl;
    return 0;
}
