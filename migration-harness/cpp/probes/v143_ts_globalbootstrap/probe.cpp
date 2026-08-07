// migration-harness/cpp/probes/v143_ts_globalbootstrap/probe.cpp
//
// Pins ql/termstructures/globalbootstrap.{hpp,cpp},
//      ql/termstructures/globalbootstrapvars.{hpp,cpp} and
//      ql/termstructures/multicurve.{hpp,cpp} (v1.43).
//
// GlobalBootstrap defaults its optimiser to LevenbergMarquardt, and the
// pquantlib LevenbergMarquardt is a scipy delegation whose C++-parity tests
// are xfailed. So the CONVERGED curve is the one thing that cannot be pinned
// bit-for-bit. Every layer below the optimiser therefore gets probed on its
// own, at a FIXED input, so the port's own arithmetic is nailed down and the
// optimiser is the only loose joint:
//
//   1. initialize()            globalbootstrap.hpp:242-317 — dates/times, the
//                              alive-instrument selection (pillarDate() >
//                              firstDate), alive additional helpers, the
//                              surviving additional dates (date <= firstDate
//                              removed, then sort + unique), maxDate_ and the
//                              initial data_. Pure date/selection logic.
//   2. setupCostFunction()     globalbootstrap.hpp:319-376 — the returned
//                              guess array, i.e. Traits::transformInverse of
//                              Traits::guess run through Traits::updateGuess,
//                              followed by the additional variables' guesses.
//   3. evaluateCostFunction()  globalbootstrap.hpp:391-403 at a FIXED x fed
//                              through setCostFunctionArgument first:
//                              quoteError() * weight per alive instrument,
//                              then the additional penalty terms.
//   4. SimpleQuoteVariables    globalbootstrapvars.cpp — transformDirect /
//                              transformInverse (both the Null-lower-bound
//                              and the exp/log branch) and initialize() in
//                              both validData branches, including the
//                              detail::get() fallback past the end of a SHORT
//                              vector, which returns v.back() and NOT the
//                              default (ql/utilities/vectors.hpp:33-42).
//   5. runMultiCurveBootstrap  globalbootstrap.cpp:50-116 — the concatenation
//                              LAYOUT: per-contributor guess sizes, the
//                              offsets at which each contributor's slice of x
//                              is handed back, and the concatenated result at
//                              a fixed x. Captured from the REAL lambda by
//                              handing MultiCurveBootstrap an OptimizationMethod
//                              that evaluates Problem::values() once at a
//                              chosen point and returns a succeeded status.
//   6. the converged curves, single-curve and multi-curve-with-a-cycle.
//
// The curve is deliberately built with a reference date LATER than one of its
// helpers' pillar dates, so the alive/dead split is actually exercised: an
// overnight deposit fixing today matures before the curve's reference date and
// must be dropped, together with its instrument weight.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/globalbootstrap.json.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/currencies/europe.hpp>
#include <ql/math/interpolations/loginterpolation.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/method.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/globalbootstrap.hpp>
#include <ql/termstructures/globalbootstrapvars.hpp>
#include <ql/termstructures/multicurve.hpp>
#include <ql/termstructures/yield/bootstraptraits.hpp>
#include <ql/termstructures/yield/piecewiseyieldcurve.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/termstructures/yield/zerospreadedtermstructure.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

using namespace QuantLib;

namespace {

// -------------------------------------------------------------------------
// fixtures
// -------------------------------------------------------------------------

// 23 October 2025 is a Thursday; TARGET spot (+2 business days) is Monday
// 27 October 2025, which is the curves' reference date throughout.
const Date kToday(23, October, 2025);

Date settlementDate() { return TARGET().advance(kToday, 2, Days); }

typedef PiecewiseYieldCurve<Discount, LogLinear, GlobalBootstrap> Curve;
// PiecewiseYieldCurve holds a GlobalBootstrap<itself> BY VALUE, so the two
// templates are mutually recursive. Naming GlobalBootstrap<Curve> first makes
// the compiler instantiate the bootstrap outermost, and it then needs a Curve
// that is still mid-instantiation. Forcing Curve to be completed first breaks
// the cycle: by the time its bootstrap_ member is instantiated, Curve's
// traits_type / interpolator_type typedefs are already declared.
static_assert(sizeof(Curve) > 0, "force PiecewiseYieldCurve to instantiate first");
typedef Curve::bootstrap_type Bootstrap;
typedef BootstrapHelper<YieldTermStructure> Helper;

// A deposit helper. fixingDays == 0 with a 1-day tenor produces a pillar of
// today + 1 business day, i.e. BEFORE the curve's reference date, which is how
// the dead-instrument branch of initialize() gets exercised.
ext::shared_ptr<RateHelper>
deposit(Rate rate, Integer n, TimeUnit unit, Natural fixingDays) {
    return ext::make_shared<DepositRateHelper>(rate, Period(n, unit), fixingDays, TARGET(),
                                               ModifiedFollowing, true, Actual360());
}

// instruments_ of the probe curve: [0] is dead (matures 24 Oct, before the
// 27 Oct reference date), [1..4] are alive.
std::vector<ext::shared_ptr<RateHelper>> makeInstruments() {
    return {deposit(0.021, 1, Days, 0), deposit(0.022, 1, Months, 2),
            deposit(0.024, 3, Months, 2), deposit(0.026, 6, Months, 2),
            deposit(0.028, 1, Years, 2)};
}

// One weight per instrument, all different, so that dropping the dead one is
// visible in aliveInstrumentWeights_.
std::vector<Real> instrumentWeights() { return {0.5, 1.0, 2.0, 1.0, 1.5}; }

// additionalHelpers_: [0] is dead (matures 27 Oct == reference date, and the
// test is pillarDate() > firstDate, so equality means dead), [1] is alive.
std::vector<ext::shared_ptr<Helper>> makeAdditionalHelpers() {
    return {deposit(0.020, 2, Days, 0), deposit(0.027, 9, Months, 2)};
}

// additionalDates_: deliberately unsorted, with three dates at or before the
// reference date (which initialize() removes) and one duplicate of an existing
// instrument pillar (which std::unique collapses).
std::vector<Date> additionalDates() {
    Calendar cal = TARGET();
    Date settl = settlementDate();
    return {kToday - 1,                    // removed: < firstDate
            settl,                         // removed: == firstDate
            cal.advance(settl, 2, Months), // kept
            cal.advance(settl, 1, Months), // duplicate of the 1M deposit pillar
            cal.advance(settl, 4, Months), // kept
            kToday - 2};                   // removed: < firstDate
}

// Two penalty terms, a pure function of (times, data) so the port can
// reproduce them without any market data.
Array penalties(const std::vector<Time>& t, const std::vector<Real>& d) {
    Array a(2);
    a[0] = 10.0 * (d[1] - d[0]) + t[1];
    a[1] = 5.0 * (d[d.size() - 1] - d[d.size() - 2]) - t[t.size() - 1];
    return a;
}

// The same two terms as a zero-argument functor: GlobalBootstrap's third
// constructor (hpp:186-206) wraps this into the two-argument shape by
// discarding both arguments.
Array constantPenalties() {
    Array a(2);
    a[0] = 0.25;
    a[1] = -0.5;
    return a;
}

// The fixed point at which every cost function below is evaluated. Offsetting
// the guess keeps x in a regime where the curve stays positive but no residual
// is accidentally zero.
Array probePoint(const Array& guess) {
    Array x(guess.size());
    for (Size i = 0; i < guess.size(); ++i)
        x[i] = guess[i] + 0.001 * static_cast<Real>(i + 1);
    return x;
}

ext::shared_ptr<Curve> makeCurve(const std::vector<ext::shared_ptr<RateHelper>>& instruments,
                                 Bootstrap bootstrap) {
    return ext::make_shared<Curve>(settlementDate(), instruments, Actual365Fixed(), LogLinear(),
                                   std::move(bootstrap));
}

// The full-featured bootstrap: additional helpers, additional dates, the
// two-argument penalty shape and per-instrument weights.
Bootstrap fullBootstrap() {
    return Bootstrap(makeAdditionalHelpers(), additionalDates, penalties, Null<Real>(), nullptr,
                     nullptr, nullptr, instrumentWeights());
}

const MultiCurveBootstrapContributor* contributorOf(const ext::shared_ptr<Curve>& c) {
    return c->multiCurveBootstrapContributor();
}

// -------------------------------------------------------------------------
// an OptimizationMethod that does not optimise
// -------------------------------------------------------------------------
//
// It records the Problem's initial value (which for MultiCurveBootstrap is the
// concatenated global guess), evaluates the REAL cost function once at
// probePoint(guess), records the result, and reports a status for which
// EndCriteria::succeeded() is true so the caller's QL_REQUIRE passes. This is
// how the concatenation layout of globalbootstrap.cpp:61-100 is observed
// without running an optimiser.
class CapturingMethod : public OptimizationMethod {
  public:
    EndCriteria::Type minimize(Problem& P, const EndCriteria&) override {
        P.reset();
        guess_ = P.currentValue();
        x_ = probePoint(guess_);
        values_ = P.values(x_);
        return EndCriteria::StationaryPoint;
    }
    Array guess_, x_, values_;
};

// -------------------------------------------------------------------------
// JSON emission
// -------------------------------------------------------------------------

std::string jsonArray(const Array& a) {
    std::ostringstream os;
    os << std::setprecision(17);
    os << "[";
    for (Size i = 0; i < a.size(); ++i)
        os << (i ? ", " : "") << a[i];
    os << "]";
    return os.str();
}

std::string jsonArray(const std::vector<Real>& v) {
    std::ostringstream os;
    os << std::setprecision(17);
    os << "[";
    for (Size i = 0; i < v.size(); ++i)
        os << (i ? ", " : "") << v[i];
    os << "]";
    return os.str();
}

std::string jsonArray(const std::vector<Size>& v) {
    std::ostringstream os;
    os << "[";
    for (Size i = 0; i < v.size(); ++i)
        os << (i ? ", " : "") << v[i];
    os << "]";
    return os.str();
}

std::string jsonDates(const std::vector<Date>& v) {
    std::ostringstream os;
    os << "[";
    for (Size i = 0; i < v.size(); ++i)
        os << (i ? ", " : "") << v[i].serialNumber();
    os << "]";
    return os.str();
}

std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        if (c == '\n')
            continue;
        out += c;
    }
    return out + "\"";
}

// -------------------------------------------------------------------------
// sections
// -------------------------------------------------------------------------

// 4. SimpleQuoteVariables — globalbootstrapvars.cpp:10-48.
void emitSimpleQuoteVariables() {
    std::cout << "  \"simple_quote_variables\": {\n";

    // (a) no bounds, no guesses: detail::get on an EMPTY vector returns the
    // default, so lb == Null<Real>() and both transforms are the identity;
    // the initial guess is 0.0 and it is written back into the quotes.
    {
        std::vector<ext::shared_ptr<SimpleQuote>> quotes = {
            ext::make_shared<SimpleQuote>(0.11), ext::make_shared<SimpleQuote>(0.22),
            ext::make_shared<SimpleQuote>(0.33)};
        SimpleQuoteVariables v(quotes);
        Array invalid = v.initialize(false);
        std::vector<Real> afterInvalid;
        for (const auto& q : quotes)
            afterInvalid.push_back(q->value());
        Array x(3);
        x[0] = -0.7;
        x[1] = 0.4;
        x[2] = 1.3;
        v.update(x);
        std::vector<Real> afterUpdate;
        for (const auto& q : quotes)
            afterUpdate.push_back(q->value());
        Array valid = v.initialize(true);
        std::cout << "    \"no_bounds\": {\n"
                  << "      \"initialize_invalid\": " << jsonArray(invalid) << ",\n"
                  << "      \"quotes_after_initialize_invalid\": " << jsonArray(afterInvalid)
                  << ",\n"
                  << "      \"update_x\": " << jsonArray(x) << ",\n"
                  << "      \"quotes_after_update\": " << jsonArray(afterUpdate) << ",\n"
                  << "      \"initialize_valid\": " << jsonArray(valid) << "\n"
                  << "    },\n";
    }

    // (b) SHORT initialGuesses / lowerBounds: three quotes but only two of
    // each. detail::get returns v.back() past the end of a NON-EMPTY vector
    // (vectors.hpp:38-40), so quote 2 inherits quote 1's guess AND bound.
    {
        std::vector<ext::shared_ptr<SimpleQuote>> quotes = {
            ext::make_shared<SimpleQuote>(), ext::make_shared<SimpleQuote>(),
            ext::make_shared<SimpleQuote>()};
        SimpleQuoteVariables v(quotes, {2.0, 0.5}, {0.0, -1.0});
        Array invalid = v.initialize(false);
        std::vector<Real> afterInvalid;
        for (const auto& q : quotes)
            afterInvalid.push_back(q->value());
        Array x(3);
        x[0] = 0.3;
        x[1] = -0.2;
        x[2] = 0.9;
        v.update(x);
        std::vector<Real> afterUpdate;
        for (const auto& q : quotes)
            afterUpdate.push_back(q->value());
        Array valid = v.initialize(true);
        std::cout << "    \"short_vectors\": {\n"
                  << "      \"initial_guesses\": [2.0, 0.5],\n"
                  << "      \"lower_bounds\": [0.0, -1.0],\n"
                  << "      \"initialize_invalid\": " << jsonArray(invalid) << ",\n"
                  << "      \"quotes_after_initialize_invalid\": " << jsonArray(afterInvalid)
                  << ",\n"
                  << "      \"update_x\": " << jsonArray(x) << ",\n"
                  << "      \"quotes_after_update\": " << jsonArray(afterUpdate) << ",\n"
                  << "      \"initialize_valid\": " << jsonArray(valid) << "\n"
                  << "    },\n";
    }

    // (c) guesses but no bounds: identity transforms with non-zero guesses.
    {
        std::vector<ext::shared_ptr<SimpleQuote>> quotes = {
            ext::make_shared<SimpleQuote>(), ext::make_shared<SimpleQuote>()};
        SimpleQuoteVariables v(quotes, {0.3});
        Array invalid = v.initialize(false);
        std::vector<Real> afterInvalid;
        for (const auto& q : quotes)
            afterInvalid.push_back(q->value());
        std::cout << "    \"guesses_no_bounds\": {\n"
                  << "      \"initial_guesses\": [0.3],\n"
                  << "      \"initialize_invalid\": " << jsonArray(invalid) << ",\n"
                  << "      \"quotes_after_initialize_invalid\": " << jsonArray(afterInvalid)
                  << "\n"
                  << "    },\n";
    }

    // (d) the two constructor QL_REQUIREs — globalbootstrapvars.cpp:15-16.
    {
        std::vector<ext::shared_ptr<SimpleQuote>> quotes = {ext::make_shared<SimpleQuote>(0.1)};
        std::string guessMsg, boundMsg;
        try {
            SimpleQuoteVariables v(quotes, {1.0, 2.0});
        } catch (const std::exception& e) {
            guessMsg = e.what();
        }
        try {
            SimpleQuoteVariables v(quotes, {}, {1.0, 2.0});
        } catch (const std::exception& e) {
            boundMsg = e.what();
        }
        std::cout << "    \"too_many_initial_guesses_message\": " << jsonString(guessMsg) << ",\n"
                  << "    \"too_many_lower_bounds_message\": " << jsonString(boundMsg) << "\n";
    }

    std::cout << "  },\n";
}

// The EndCriteria GlobalBootstrap::setup builds by default, hpp:228. Emitting
// its accessors pins the constructor's own Null-substitution behaviour for
// these exact arguments (maxStationaryStateIterations stays 10 rather than
// being derived from maxIterations).
void emitSetupDefaults() {
    const Real accuracy = 1.0e-12;
    EndCriteria ec(1000, 10, accuracy, accuracy, accuracy);
    // MultiCurveBootstrap(optimizer, endCriteria) substitutes this accuracy
    // for whichever argument is null — globalbootstrap.cpp:34.
    const Real mcAccuracy = 1.0e-10;
    EndCriteria mcEc(1000, 10, mcAccuracy, mcAccuracy, mcAccuracy);
    std::cout << "  \"setup_defaults\": {\n"
              << "    \"curve_accuracy\": " << 1.0e-12 << ",\n"
              << "    \"multi_curve_null_accuracy\": " << mcAccuracy << ",\n"
              << "    \"end_criteria_max_iterations\": " << ec.maxIterations() << ",\n"
              << "    \"end_criteria_max_stationary_state_iterations\": "
              << ec.maxStationaryStateIterations() << ",\n"
              << "    \"end_criteria_root_epsilon\": " << ec.rootEpsilon() << ",\n"
              << "    \"end_criteria_function_epsilon\": " << ec.functionEpsilon() << ",\n"
              << "    \"end_criteria_gradient_norm_epsilon\": " << ec.gradientNormEpsilon()
              << ",\n"
              << "    \"multi_curve_end_criteria_root_epsilon\": " << mcEc.rootEpsilon() << "\n"
              << "  },\n";
}

// The QL_REQUIRE on instrumentWeights_ — globalbootstrap.hpp:232-235.
void emitWeightsRequire() {
    std::string msg;
    try {
        auto instruments = makeInstruments();
        auto curve = makeCurve(instruments, Bootstrap(Null<Real>(), nullptr, nullptr, {1.0, 2.0}));
    } catch (const std::exception& e) {
        msg = e.what();
    }
    std::cout << "  \"instrument_weights_require_message\": " << jsonString(msg) << ",\n";
}

// 1-3. initialize / setupCostFunction / evaluateCostFunction, all at a fixed
// point, on the full-featured bootstrap.
void emitSingleCurve() {
    auto instruments = makeInstruments();
    auto curve = makeCurve(instruments, fullBootstrap());
    const auto* c = contributorOf(curve);

    // setupCostFunction() triggers initialize() (hpp:331-332); nothing reads
    // times_/data_ before this point.
    Array guess = c->setupCostFunction();

    // The alive split is not directly readable, so it is reconstructed from
    // the two observable consequences: which pillar dates ended up on the
    // grid, and which instruments the cost function scores.
    std::vector<Date> pillars;
    for (const auto& h : instruments)
        pillars.push_back(h->pillarDate());
    std::vector<Date> additionalPillars;
    auto addHelpers = makeAdditionalHelpers();
    for (const auto& h : addHelpers)
        additionalPillars.push_back(h->pillarDate());

    Array x = probePoint(guess);
    c->setCostFunctionArgument(x);
    Array residuals = c->evaluateCostFunction();

    // data_ after setCostFunctionArgument: Traits::transformDirect(x[i], ...)
    // written through Traits::updateGuess.
    std::vector<Real> dataAtX = curve->data();

    std::cout << "  \"single_curve\": {\n"
              << "    \"reference_date\": " << curve->referenceDate().serialNumber() << ",\n"
              << "    \"instrument_pillar_dates\": " << jsonDates(pillars) << ",\n"
              << "    \"additional_helper_pillar_dates\": " << jsonDates(additionalPillars)
              << ",\n"
              << "    \"raw_additional_dates\": " << jsonDates(additionalDates()) << ",\n"
              << "    \"instrument_weights\": " << jsonArray(instrumentWeights()) << ",\n"
              << "    \"dates\": " << jsonDates(curve->dates()) << ",\n"
              << "    \"times\": " << jsonArray(curve->times()) << ",\n"
              << "    \"max_date\": " << curve->maxDate().serialNumber() << ",\n"
              << "    \"n_alive_instruments\": " << residuals.size() - 2 << ",\n"
              << "    \"guess\": " << jsonArray(guess) << ",\n"
              << "    \"x\": " << jsonArray(x) << ",\n"
              << "    \"data_at_x\": " << jsonArray(dataAtX) << ",\n"
              << "    \"residuals_at_x\": " << jsonArray(residuals) << "\n"
              << "  },\n";
}

// The initial data_ that initialize() installs when the curve is not valid:
// dates.size() copies of Traits::initialValue (hpp:313). Probed on a bare
// bootstrap so no penalty/additional machinery is involved.
void emitInitialData() {
    auto instruments = makeInstruments();
    auto curve = makeCurve(instruments, Bootstrap(Null<Real>(), nullptr, nullptr,
                                                  instrumentWeights()));
    const auto* c = contributorOf(curve);
    // Reading data_ BEFORE setupCostFunction would read a vector that
    // initialize() has not sized yet, so the guess call comes first and the
    // initial value is reported as the scalar Traits::initialValue instead.
    Array guess = c->setupCostFunction();
    std::cout << "  \"bare_curve\": {\n"
              << "    \"dates\": " << jsonDates(curve->dates()) << ",\n"
              << "    \"times\": " << jsonArray(curve->times()) << ",\n"
              << "    \"max_date\": " << curve->maxDate().serialNumber() << ",\n"
              << "    \"initial_value\": "
              << Discount::initialValue(static_cast<const YieldTermStructure*>(curve.get()))
              << ",\n"
              << "    \"guess\": " << jsonArray(guess) << "\n"
              << "  },\n";
}

// The zero-argument penalty shape, hpp:186-206: the wrapper discards times and
// data and returns f(). Same fixed point as the bare curve above, so the two
// residual vectors differ only by the appended constants.
void emitConstantPenalties() {
    auto instruments = makeInstruments();
    auto curve = makeCurve(instruments,
                           Bootstrap({}, nullptr, constantPenalties, Null<Real>(), nullptr, nullptr,
                                     nullptr, instrumentWeights()));
    const auto* c = contributorOf(curve);
    Array guess = c->setupCostFunction();
    Array x = probePoint(guess);
    c->setCostFunctionArgument(x);
    Array residuals = c->evaluateCostFunction();
    std::cout << "  \"constant_penalties\": {\n"
              << "    \"dates\": " << jsonDates(curve->dates()) << ",\n"
              << "    \"guess\": " << jsonArray(guess) << ",\n"
              << "    \"residuals_at_x\": " << jsonArray(residuals) << "\n"
              << "  },\n";
}

// 5. The MultiCurveBootstrap concatenation layout, globalbootstrap.cpp:50-116.
// Two curves of DIFFERENT sizes, the second one carrying penalties so that its
// residual count differs from its guess count — a port that reuses the guess
// sizes to slice the results gets this wrong.
void emitMultiCurveLayout() {
    auto instrumentsA = makeInstruments();
    // A shorter second curve: the 1M and 6M deposits only.
    std::vector<ext::shared_ptr<RateHelper>> instrumentsB = {deposit(0.019, 1, Months, 2),
                                                             deposit(0.023, 6, Months, 2)};

    auto curveA = makeCurve(instrumentsA, Bootstrap(Null<Real>(), nullptr, nullptr,
                                                    instrumentWeights()));
    auto curveB = makeCurve(instrumentsB, Bootstrap({}, nullptr, constantPenalties, Null<Real>(),
                                                    nullptr, nullptr, nullptr, {}));

    auto method = ext::make_shared<CapturingMethod>();
    auto endCriteria = ext::make_shared<EndCriteria>(1000, 10, 1e-10, 1e-10, 1e-10);
    auto mcb = ext::make_shared<MultiCurveBootstrap>(method, endCriteria);
    mcb->add(contributorOf(curveA));
    mcb->add(contributorOf(curveB));
    mcb->runMultiCurveBootstrap();

    std::vector<Size> guessSizes = {curveA->times().size() - 1, curveB->times().size() - 1};

    std::cout << "  \"multi_curve_layout\": {\n"
              << "    \"guess_sizes\": " << jsonArray(guessSizes) << ",\n"
              << "    \"offsets\": [0, " << guessSizes[0] << "],\n"
              << "    \"global_guess\": " << jsonArray(method->guess_) << ",\n"
              << "    \"x\": " << jsonArray(method->x_) << ",\n"
              << "    \"global_values_at_x\": " << jsonArray(method->values_) << ",\n"
              << "    \"curve_a_dates\": " << jsonDates(curveA->dates()) << ",\n"
              << "    \"curve_b_dates\": " << jsonDates(curveB->dates()) << ",\n"
              << "    \"curve_a_data_at_x\": " << jsonArray(curveA->data()) << ",\n"
              << "    \"curve_b_data_at_x\": " << jsonArray(curveB->data()) << "\n"
              << "  },\n";
}

// 6a. The converged single curve. Optimiser-dependent, so reported for
// information and pinned only at the tolerance the port can actually reach.
void emitConverged() {
    auto instruments = makeInstruments();
    auto curve = makeCurve(instruments, Bootstrap(Null<Real>(), nullptr, nullptr,
                                                  instrumentWeights()));
    curve->enableExtrapolation();
    std::vector<Real> discounts, quoteErrors;
    for (const auto& d : curve->dates())
        discounts.push_back(curve->discount(d));
    // instruments[0] is the dead overnight deposit: GlobalBootstrap never
    // calls setTermStructure on it, so quoteError() would throw "term
    // structure not set". Only the alive helpers are scored.
    for (Size i = 1; i < instruments.size(); ++i)
        quoteErrors.push_back(instruments[i]->quoteError());
    std::cout << "  \"converged_single_curve\": {\n"
              << "    \"dates\": " << jsonDates(curve->dates()) << ",\n"
              << "    \"times\": " << jsonArray(curve->times()) << ",\n"
              << "    \"data\": " << jsonArray(curve->data()) << ",\n"
              << "    \"discounts_at_pillars\": " << jsonArray(discounts) << ",\n"
              << "    \"quote_errors\": " << jsonArray(quoteErrors) << "\n"
              << "  },\n";
}

// 6b. MultiCurve on a genuine dependency cycle: a piecewise curve whose swap
// helpers discount off a spreaded curve that is itself built off the piecewise
// curve. multicurve.cpp:32-71.
void emitMultiCurveCycle() {
    RelinkableHandle<YieldTermStructure> internal3m, internalOis;

    auto index3m = ext::make_shared<IborIndex>("EUR3M", 3 * Months, 2, EURCurrency(), TARGET(),
                                               ModifiedFollowing, false, Actual360(), internal3m);
    Handle<Quote> q(ext::make_shared<SimpleQuote>(0.03));
    Handle<Quote> spread(ext::make_shared<SimpleQuote>(-0.01));

    std::vector<ext::shared_ptr<RateHelper>> helpers3m;
    for (Size i = 1; i <= 5; ++i)
        helpers3m.push_back(ext::make_shared<SwapRateHelper>(
            q, static_cast<Integer>(i) * Years, TARGET(), Annual, Following,
            Thirty360(Thirty360::BondBasis), index3m, Handle<Quote>(), 0 * Days, internalOis));

    const Real accuracy = 1.0e-10;
    auto multiCurve = ext::make_shared<MultiCurve>(accuracy);
    ext::shared_ptr<YieldTermStructure> ptr3m = ext::make_shared<Curve>(
        kToday, helpers3m, Actual360(), LogLinear(), Bootstrap(accuracy));
    Handle<YieldTermStructure> curve3m =
        multiCurve->addBootstrappedCurve(internal3m, std::move(ptr3m));
    ext::shared_ptr<YieldTermStructure> ptrOis =
        ext::make_shared<ZeroSpreadedTermStructure>(internal3m, spread);
    Handle<YieldTermStructure> curveOis =
        multiCurve->addNonBootstrappedCurve(internalOis, std::move(ptrOis));

    std::vector<Date> pillars;
    for (const auto& h : helpers3m)
        pillars.push_back(h->pillarDate());

    std::vector<Real> discounts3m, discountsOis, quoteErrors;
    for (const auto& d : pillars) {
        discounts3m.push_back(curve3m->discount(d));
        discountsOis.push_back(curveOis->discount(d));
    }
    for (const auto& h : helpers3m)
        quoteErrors.push_back(h->quoteError());

    const Real zeroOis = curveOis->zeroRate(1.0, Continuous).rate();
    const Real zero3m = curve3m->zeroRate(1.0, Continuous).rate();

    std::cout << "  \"multi_curve_cycle\": {\n"
              << "    \"reference_date\": " << curve3m->referenceDate().serialNumber() << ",\n"
              << "    \"pillar_dates\": " << jsonDates(pillars) << ",\n"
              << "    \"discounts_3m\": " << jsonArray(discounts3m) << ",\n"
              << "    \"discounts_ois\": " << jsonArray(discountsOis) << ",\n"
              << "    \"quote_errors\": " << jsonArray(quoteErrors) << ",\n"
              << "    \"zero_3m_1y\": " << zero3m << ",\n"
              << "    \"zero_ois_1y\": " << zeroOis << ",\n"
              << "    \"zero_spread_1y\": " << zeroOis - zero3m << ",\n"
              << "    \"spread\": " << spread->value() << ",\n"
              << "    \"quote\": " << q->value() << "\n"
              << "  }\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = kToday;

    std::cout << "{\n";
    std::cout << "  \"evaluation_date\": " << kToday.serialNumber() << ",\n";
    std::cout << "  \"settlement_date\": " << settlementDate().serialNumber() << ",\n";
    emitSimpleQuoteVariables();
    emitSetupDefaults();
    emitWeightsRequire();
    emitInitialData();
    emitSingleCurve();
    emitConstantPenalties();
    emitMultiCurveLayout();
    emitConverged();
    emitMultiCurveCycle();
    std::cout << "}" << std::endl;
    return 0;
}
