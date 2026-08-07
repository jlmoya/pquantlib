// migration-harness/cpp/probes/v143_ts_fittedbond/probe.cpp
//
// Pins FittedBondDiscountCurve + FittingMethod + the seven concrete fitting
// methods in ql/termstructures/yield/nonlinearfittingmethods.{hpp,cpp} (v1.43).
//
// The fit itself is TWO nested iterative processes: a Simplex over a cost
// function that internally runs a NewtonSafe root-find per bond (in
// FittingMethod::init, fittedbonddiscountcurve.cpp:214).  A probe that only
// pinned the converged answer would be untestable: any single-ULP difference
// anywhere flips a simplex comparison and the whole path diverges.  So the
// output is layered, and only the last layer touches an optimizer:
//
//   layer1_*  FittingMethod::discount(x, t) / size() / basisFunction / the
//             Bernstein basis.  Pure arithmetic, no bonds, no solver.
//   layer2_*  FittingMethod::init()'s weight vector: raw ytm, raw modified
//             duration, and the 1/duration weights normalised by
//             1/sqrt(sum of squares) (cpp:202-226).  One root-find per bond,
//             no simplex.
//   layer3_*  FittingCost::values(x) / value(x) at a FIXED x, reached by
//             handing the fitting method a stub OptimizationMethod that
//             evaluates the problem once at the guess and returns.  Exercises
//             the real (private, otherwise unreachable) FittingCost including
//             the L2 penalty terms (cpp:314-336).  Still no simplex.
//   layer4_*  the real Simplex fit, plus the refit / resetGuess behaviour
//             (cpp:295) and the Clone<FittingMethod> ownership semantics.
//
// Emits ONE JSON object on stdout and nothing else.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ql/instruments/bonds/fixedratebond.hpp>
#include <ql/instruments/bonds/zerocouponbond.hpp>
#include <ql/math/bernsteinpolynomial.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/method.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/pricingengines/bond/bondfunctions.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/bondhelpers.hpp>
#include <ql/termstructures/yield/fittedbonddiscountcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/nonlinearfittingmethods.hpp>
#include <ql/time/calendars/canada.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

// Evaluation date; the same asof as test-suite/fittedbonddiscountcurve.cpp:82.
const Date kEval(15, July, 2019);

// ---------------------------------------------------------------------------
// tiny JSON emitter
// ---------------------------------------------------------------------------

std::string num(Real x) {
    std::ostringstream os;
    os << std::setprecision(17) << x;
    return os.str();
}

std::string arr(const Array& a) {
    std::string s = "[";
    for (Size i = 0; i < a.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += num(a[i]);
    }
    return s + "]";
}

std::string arr(const std::vector<Real>& a) {
    std::string s = "[";
    for (Size i = 0; i < a.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += num(a[i]);
    }
    return s + "]";
}

// ---------------------------------------------------------------------------
// layer 1 — discount(x, t) for a bare fitting method
// ---------------------------------------------------------------------------

// The times every layer-1 case is sampled at. 0.0 and 1e-6 exercise the
// near-zero end (Nelson-Siegel/Svensson divide by (kappa+eps)*(t+eps));
// 30.0 sits past every maxCutoffTime used below.
const std::vector<Time> kTimes = {0.0, 1.0e-6, 0.25, 0.9999, 1.0,
                                  3.0, 5.0,    7.5,  20.0,   30.0};

// size() is public on the base but re-declared private/protected in most of the
// derived classes, so it is only reachable through a base reference.
Size sz(const FittedBondDiscountCurve::FittingMethod& m) { return m.size(); }

std::string discountRow(const FittedBondDiscountCurve::FittingMethod& m, const Array& x) {
    std::vector<Real> d;
    d.reserve(kTimes.size());
    for (Time t : kTimes)
        d.push_back(m.discount(x, t));
    return "{\"size\": " + std::to_string(m.size()) + ", \"x\": " + arr(x) +
           ", \"d\": " + arr(d) + "}";
}

// ---------------------------------------------------------------------------
// layer 3 — a stub optimizer that evaluates the problem once, at the guess
// ---------------------------------------------------------------------------

class ProbeOptimizer : public OptimizationMethod {
  public:
    mutable Array x0, vals;
    mutable Real val = 0.0;
    mutable bool called = false;

    EndCriteria::Type minimize(Problem& P, const EndCriteria&) override {
        // MANDATORY: Problem's constructor leaves functionEvaluation_ and
        // functionValue_ INDETERMINATE (problem.hpp:45-47, 138-142); only
        // reset() initialises them. Every shipped OptimizationMethod calls it
        // first, and without it FittingMethod::calculate() copies uninitialised
        // memory into numberOfIterations_ (observed: the value changed between
        // two runs of this probe).
        P.reset();
        x0 = P.currentValue();
        vals = P.values(x0);   // FittingCost::values(x)  (cpp:314-336)
        val = P.value(x0);     // FittingCost::value(x)   (cpp:304-312)
        P.setFunctionValue(val);
        called = true;
        return EndCriteria::StationaryPoint;
    }
};

// ---------------------------------------------------------------------------
// bond sets
// ---------------------------------------------------------------------------

// Four Canadian government bonds, verbatim from
// test-suite/fittedbonddiscountcurve.cpp:86-124.
struct BondSet {
    std::vector<ext::shared_ptr<Bond> > bonds;
    std::vector<ext::shared_ptr<BondHelper> > helpers;
    std::vector<Real> quotes;
};

BondSet canadaSet() {
    BondSet s;
    s.quotes = {101.2100, 100.6270, 99.9210, 101.6700};

    s.bonds.push_back(ext::make_shared<FixedRateBond>(
        2, 100.0,
        Schedule(Date(1, February, 2013), Date(3, February, 2020), 6 * Months, Canada(),
                 Following, Following, DateGeneration::Forward, false, Date(3, August, 2013)),
        std::vector<Rate>(1, 0.046), ActualActual(ActualActual::ISDA)));

    s.bonds.push_back(ext::make_shared<FixedRateBond>(
        2, 100.0,
        Schedule(Date(12, June, 2015), Date(12, June, 2020), 6 * Months, Canada(), Following,
                 Following, DateGeneration::Forward, false, Date(12, December, 2015)),
        std::vector<Rate>(1, 0.0295), ActualActual(ActualActual::ISDA)));

    s.bonds.push_back(ext::make_shared<FixedRateBond>(
        2, 100.0,
        Schedule(Date(24, November, 2017), Date(24, November, 2020), 6 * Months, Canada(),
                 Following, Following, DateGeneration::Forward, false, Date(24, May, 2018)),
        std::vector<Rate>(1, 0.02689), ActualActual(ActualActual::ISDA)));

    s.bonds.push_back(ext::make_shared<FixedRateBond>(
        2, 100.0,
        Schedule(Date(21, February, 2017), Date(21, February, 2022), 6 * Months, Canada(),
                 Following, Following, DateGeneration::Forward, false, Date(21, August, 2017)),
        std::vector<Rate>(1, 0.0338), ActualActual(ActualActual::ISDA)));

    for (Size i = 0; i < s.bonds.size(); ++i)
        s.helpers.push_back(ext::make_shared<BondHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(s.quotes[i])), s.bonds[i]));
    return s;
}

// Four zero-coupon bonds, from test-suite/fittedbonddiscountcurve.cpp:229-238.
// Cheap to price, so this is the set the heavier layer-4 fits use.
BondSet zeroSet() {
    BondSet s;
    s.quotes = {99.0, 98.0, 95.0, 90.0};
    const Period tenors[] = {Period(1, Years), Period(2, Years), Period(5, Years),
                             Period(10, Years)};
    for (Size i = 0; i < 4; ++i) {
        s.bonds.push_back(
            ext::make_shared<ZeroCouponBond>(3, TARGET(), 100.0, kEval + tenors[i]));
        s.helpers.push_back(ext::make_shared<BondHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(s.quotes[i])), s.bonds[i]));
    }
    return s;
}

// Reproduce FittingMethod::init()'s per-bond intermediates, unweighted
// (fittedbonddiscountcurve.cpp:190-226). The conventions are hard-coded there:
// the curve's day counter, Compounded, Annual, Duration::Modified, and the
// BOND's own settlement date (not the curve's reference date).
void emitWeightInputs(const BondSet& s, const DayCounter& curveDC) {
    std::vector<Real> ytms, durs, settle;
    for (const auto& h : s.helpers) {
        const auto& bond = h->bond();
        Bond::Price price(h->quote()->value(), h->priceType());
        Date bondSettlement = bond->settlementDate();
        Rate ytm = BondFunctions::yield(*bond, price, curveDC, Compounded, Annual,
                                        bondSettlement);
        Time dur = BondFunctions::duration(*bond, ytm, curveDC, Compounded, Annual,
                                           Duration::Modified, bondSettlement);
        ytms.push_back(ytm);
        durs.push_back(dur);
        settle.push_back(static_cast<Real>(bondSettlement.serialNumber()));
    }
    std::cout << "    \"ytm\": " << arr(ytms) << ",\n"
              << "    \"duration\": " << arr(durs) << ",\n"
              << "    \"settlement\": " << arr(settle) << "\n";
}

// The times the fitted curves are sampled at in layers 3 and 4.
const std::vector<Time> kCurveTimes = {0.1, 0.5, 1.0, 2.0, 2.6, 5.0};

void emitCurveSamples(FittedBondDiscountCurve& curve) {
    std::vector<Real> dfs, zeros;
    for (Time t : kCurveTimes) {
        dfs.push_back(curve.discount(t, true));
        zeros.push_back(curve.zeroRate(t, Continuous).rate());
    }
    std::cout << "    \"t\": " << arr(kCurveTimes) << ",\n"
              << "    \"discount\": " << arr(dfs) << ",\n"
              << "    \"zero_continuous\": " << arr(zeros) << ",\n";
}

const DayCounter kAct365 = Actual365Fixed();

void emitFitResults(FittedBondDiscountCurve& curve);

// One real Simplex fit, emitted as a complete JSON object.
void emitFit(const std::string& key,
             const FittedBondDiscountCurve::FittingMethod& method,
             const BondSet& s,
             const Array& guess,
             Real accuracy,
             Size maxEval) {
    FittedBondDiscountCurve curve(kEval, s.helpers, kAct365, method, accuracy, maxEval, guess);
    curve.enableExtrapolation();
    std::cout << "  \"" << key << "\": {\n";
    emitFitResults(curve);
    std::vector<Real> dfs, zeros;
    for (Time t : kCurveTimes) {
        dfs.push_back(curve.discount(t, true));
        zeros.push_back(curve.zeroRate(t, Continuous).rate());
    }
    std::cout << "    \"t\": " << arr(kCurveTimes) << ",\n"
              << "    \"discount\": " << arr(dfs) << ",\n"
              << "    \"zero_continuous\": " << arr(zeros) << ",\n"
              << "    \"guess\": " << arr(guess) << ",\n"
              << "    \"n_bonds\": " << curve.numberOfBonds() << "\n"
              << "  },\n";
}

// The same fit truncated at a series of iteration caps. Used to localise where
// a port's simplex path first parts company with this one: the caps below the
// divergence point must reproduce exactly.
void emitCapSweep(const std::string& key,
                  const FittedBondDiscountCurve::FittingMethod& method,
                  const BondSet& s,
                  const Array& guess,
                  Real accuracy,
                  const std::vector<Size>& caps,
                  Size maxStationary = 100) {
    // maxStationaryStateIterations does NOT steer the simplex: it is only the
    // seed of the unconditional checkStationaryPoint call at the exit
    // (simplex.cpp:141-152), so lowering it to allow caps below 100 leaves the
    // iterate path untouched.
    std::cout << "  \"" << key << "\": {\n";
    for (Size k = 0; k < caps.size(); ++k) {
        FittedBondDiscountCurve curve(kEval, s.helpers, kAct365, method, accuracy, caps[k],
                                      guess, 1.0, maxStationary);
        curve.enableExtrapolation();
        const auto& r = curve.fitResults();
        std::cout << "    \"" << caps[k] << "\": {\"solution\": " << arr(r.solution())
                  << ", \"iterations\": " << r.numberOfIterations()
                  << ", \"cost\": " << num(r.minimumCostValue())
                  << ", \"error_code\": " << static_cast<int>(r.errorCode()) << "}"
                  << (k + 1 == caps.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
}

// FittingCost at the guess, via the stub optimizer, for an arbitrary method.
void emitCostAtGuess(const std::string& key,
                     const FittedBondDiscountCurve::FittingMethod& method,
                     const ext::shared_ptr<ProbeOptimizer>& opt,
                     const BondSet& s,
                     const Array& guess) {
    FittedBondDiscountCurve curve(kEval, s.helpers, kAct365, method, 1e-10, 10000, guess);
    curve.enableExtrapolation();
    const auto& r = curve.fitResults();
    std::cout << "  \"" << key << "\": {\n"
              << "    \"weights\": " << arr(r.weights()) << ",\n"
              << "    \"cost_x\": " << arr(opt->x0) << ",\n"
              << "    \"cost_values\": " << arr(opt->vals) << ",\n"
              << "    \"cost_value\": " << num(opt->val) << "\n"
              << "  },\n";
}

void emitFitResults(FittedBondDiscountCurve& curve) {
    const auto& r = curve.fitResults();
    std::cout << "    \"solution\": " << arr(r.solution()) << ",\n"
              << "    \"number_of_iterations\": " << r.numberOfIterations() << ",\n"
              << "    \"minimum_cost_value\": " << num(r.minimumCostValue()) << ",\n"
              << "    \"error_code\": " << static_cast<int>(r.errorCode()) << ",\n"
              << "    \"weights\": " << arr(r.weights()) << ",\n"
              << "    \"max_date\": " << curve.maxDate().serialNumber() << ",\n";
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kEval;
    std::cout << std::setprecision(17);

    const DayCounter act365 = Actual365Fixed();
    const Date maxDate10y = kEval + Period(10, Years);

    std::cout << "{\n";

    // -----------------------------------------------------------------------
    std::cout << "  \"meta\": {\n"
              << "    \"eval_date\": " << kEval.serialNumber() << ",\n"
              << "    \"max_date_10y\": " << maxDate10y.serialNumber() << ",\n"
              << "    \"null_real\": " << num(Null<Real>()) << ",\n"
              << "    \"times\": " << arr(kTimes) << "\n"
              << "  },\n";

    // =======================================================================
    // LAYER 1 — sizes under every constructor overload
    // =======================================================================
    std::cout << "  \"layer1_size\": {\n";
    {
        // ExponentialSplinesFitting::size() (cpp:69-73) depends on BOTH
        // constrainAtZero and whether fixedKappa is Null<Real>(). All four.
        std::cout << "    \"exp_c1_kfree_n9\": " << sz(ExponentialSplinesFitting(true)) << ",\n"
                  << "    \"exp_c0_kfree_n9\": " << sz(ExponentialSplinesFitting(false)) << ",\n"
                  << "    \"exp_c1_kfix_n9\": " << sz(ExponentialSplinesFitting(true, 9, 0.05))
                  << ",\n"
                  << "    \"exp_c0_kfix_n9\": " << sz(ExponentialSplinesFitting(false, 9, 0.05))
                  << ",\n"
                  << "    \"exp_c1_kfree_n5\": "
                  << sz(ExponentialSplinesFitting(true, 5, Null<Real>())) << ",\n"
                  << "    \"exp_c1_kfix_n2\": " << sz(ExponentialSplinesFitting(true, 2, 0.05))
                  << ",\n"
                  << "    \"exp_ctor_l2_c1_n7_kfix\": "
                  << sz(ExponentialSplinesFitting(true, Array(), Array(), 0.0, QL_MAX_REAL, 7,
                                                  0.05))
                  << ",\n";

        std::cout << "    \"nelson_siegel\": " << sz(NelsonSiegelFitting()) << ",\n"
                  << "    \"svensson\": " << sz(SvenssonFitting()) << ",\n";

        // CubicBSplinesFitting: basisFunctions = knots-4, minus one when
        // constrained at zero (cpp:199-213).
        const std::vector<Time> knots12 = {-30.0, -20.0, -10.0, 0.0, 5.0,  10.0,
                                           15.0,  20.0,  25.0,  30.0, 40.0, 50.0};
        const std::vector<Time> knots9 = {-10.0, -5.0, 0.0, 4.0, 8.0, 12.0, 20.0, 30.0, 40.0};
        std::cout << "    \"bspline_k12_c1\": " << sz(CubicBSplinesFitting(knots12, true))
                  << ",\n"
                  << "    \"bspline_k12_c0\": " << sz(CubicBSplinesFitting(knots12, false))
                  << ",\n"
                  << "    \"bspline_k9_c1\": " << sz(CubicBSplinesFitting(knots9, true)) << ",\n";

        // NaturalCubicFitting pushes 0.0, sorts, and de-dups at 1e-14
        // (cpp:279-284) — so a knot vector that already contains 0 keeps its
        // length while one that does not grows by one.
        std::cout << "    \"natcubic_with_zero\": "
                  << sz(NaturalCubicFitting(std::vector<Time>{0.0, 1.0, 3.0, 7.0, 15.0})) << ",\n"
                  << "    \"natcubic_without_zero\": "
                  << sz(NaturalCubicFitting(std::vector<Time>{1.0, 3.0, 7.0, 15.0})) << ",\n"
                  << "    \"natcubic_unsorted_dup\": "
                  << sz(NaturalCubicFitting(std::vector<Time>{7.0, 1.0, 3.0, 1.0, 15.0, 0.0}))
                  << ",\n"
                  << "    \"natcubic_near_dup_1e15\": "
                  << sz(NaturalCubicFitting(std::vector<Time>{1.0, 1.0 + 1e-15, 3.0})) << ",\n";

        std::cout << "    \"poly_d3_c1\": " << sz(SimplePolynomialFitting(3, true)) << ",\n"
                  << "    \"poly_d3_c0\": " << sz(SimplePolynomialFitting(3, false)) << ",\n"
                  << "    \"poly_d1_c1\": " << sz(SimplePolynomialFitting(1, true)) << ",\n";

        auto inner = ext::make_shared<SvenssonFitting>();
        auto flat = ext::make_shared<FlatForward>(kEval, 0.02, act365, Continuous, Annual);
        flat->enableExtrapolation();
        std::cout << "    \"spread_over_svensson\": "
                  << sz(SpreadFittingMethod(inner, Handle<YieldTermStructure>(flat))) << "\n";
    }
    std::cout << "  },\n";

    // =======================================================================
    // LAYER 1 — discount(x, t)
    // =======================================================================
    std::cout << "  \"layer1_discount\": {\n";
    {
        // --- exponential splines ------------------------------------------
        // constrainAtZero: d(t) = coeff*e^{-k t} + sum_i x[i] e^{-k (i+2) t},
        // coeff = 1 - sum x[i]  (cpp:87-97); x[N-1] is kappa when free.
        Array esFree(9);
        esFree[0] = 0.30;  esFree[1] = -0.15; esFree[2] = 0.08;
        esFree[3] = -0.04; esFree[4] = 0.02;  esFree[5] = -0.01;
        esFree[6] = 0.005; esFree[7] = -0.002;
        esFree[8] = 0.055; // kappa
        std::cout << "    \"exp_c1_kfree\": "
                  << discountRow(ExponentialSplinesFitting(true), esFree) << ",\n";

        Array esFree10(10);
        for (Size i = 0; i < 9; ++i)
            esFree10[i] = esFree[i];
        esFree10[8] = -0.0007;
        esFree10[9] = 0.055; // kappa
        std::cout << "    \"exp_c0_kfree\": "
                  << discountRow(ExponentialSplinesFitting(false), esFree10) << ",\n";

        Array esFix(8);
        for (Size i = 0; i < 8; ++i)
            esFix[i] = esFree[i];
        std::cout << "    \"exp_c1_kfix\": "
                  << discountRow(ExponentialSplinesFitting(true, 9, 0.05), esFix) << ",\n";

        Array esFix9(9);
        for (Size i = 0; i < 9; ++i)
            esFix9[i] = esFree10[i];
        std::cout << "    \"exp_c0_kfix\": "
                  << discountRow(ExponentialSplinesFitting(false, 9, 0.05), esFix9) << ",\n";

        // --- Nelson-Siegel -------------------------------------------------
        Array ns = {0.03, -0.012, 0.021, 1.7};
        std::cout << "    \"nelson_siegel\": " << discountRow(NelsonSiegelFitting(), ns)
                  << ",\n";

        // Cutoffs: minCutoffTime 1.0 / maxCutoffTime 5.0 makes BOTH branches of
        // FittingMethod::discount (fittedbonddiscountcurve.hpp:358-371) live at
        // the sampled times.
        std::cout << "    \"nelson_siegel_cutoff\": "
                  << discountRow(NelsonSiegelFitting(Array(),
                                                     ext::shared_ptr<OptimizationMethod>(),
                                                     Array(), 1.0, 5.0),
                                 ns)
                  << ",\n";

        // --- Svensson ------------------------------------------------------
        Array sv = {0.032, -0.011, 0.019, 0.006, 1.9, 0.42};
        std::cout << "    \"svensson\": " << discountRow(SvenssonFitting(), sv) << ",\n";
        std::cout << "    \"svensson_cutoff\": "
                  << discountRow(SvenssonFitting(Array(), Array(), 0.5, 8.0), sv) << ",\n";

        // --- cubic B-splines -----------------------------------------------
        const std::vector<Time> knots12 = {-30.0, -20.0, -10.0, 0.0, 5.0,  10.0,
                                           15.0,  20.0,  25.0,  30.0, 40.0, 50.0};
        Array bs7 = {0.99, 0.95, 0.90, 0.84, 0.77, 0.70, 0.62};
        std::cout << "    \"bspline_c1\": "
                  << discountRow(CubicBSplinesFitting(knots12, true), bs7) << ",\n";
        Array bs8 = {1.00, 0.99, 0.95, 0.90, 0.84, 0.77, 0.70, 0.62};
        std::cout << "    \"bspline_c0\": "
                  << discountRow(CubicBSplinesFitting(knots12, false), bs8) << ",\n";

        // --- natural cubic --------------------------------------------------
        // knotTimes gains 0.0 at the front, so x is the nodal discount at
        // 1/3/7/15/30 and d(0) is pinned to 1 (cpp:320-338). t is CLAMPED to
        // [0, 30], which is why t = 0.0 and t = 30.0 are not extrapolations.
        Array nc = {0.972, 0.918, 0.815, 0.640, 0.410};
        std::cout << "    \"natural_cubic\": "
                  << discountRow(NaturalCubicFitting(
                                     std::vector<Time>{1.0, 3.0, 7.0, 15.0, 30.0}),
                                 nc)
                  << ",\n";
        // Same effective knot set reached the other way: unsorted, duplicated,
        // and already containing 0 (cpp:279-284).
        Array nc4 = {0.972, 0.918, 0.815, 0.640};
        std::cout << "    \"natural_cubic_dedup\": "
                  << discountRow(NaturalCubicFitting(
                                     std::vector<Time>{7.0, 1.0, 3.0, 1.0, 15.0, 0.0}),
                                 nc4)
                  << ",\n";

        // --- simple polynomial ----------------------------------------------
        // constrainAtZero: d = 1 + sum x[i] B_{i+1}^{i+1}(t) (cpp:381-385), and
        // B_n^n(t) = t^n, so this is 1 + x0 t + x1 t^2 + x2 t^3.
        Array p3 = {-0.028, 0.0006, -1.0e-5};
        std::cout << "    \"poly_c1\": "
                  << discountRow(SimplePolynomialFitting(3, true), p3) << ",\n";
        // x[0] deliberately != 1.0 so the unconstrained branch cannot coincide
        // with the constrained one — otherwise a port that confused the two
        // would agree here.
        Array p4 = {0.98, -0.028, 0.0006, -1.0e-5};
        std::cout << "    \"poly_c0\": "
                  << discountRow(SimplePolynomialFitting(3, false), p4) << "\n";
    }
    std::cout << "  },\n";

    // =======================================================================
    // LAYER 1 — CubicBSplines basis functions + the Bernstein basis
    // =======================================================================
    std::cout << "  \"layer1_bspline_basis\": {\n";
    {
        const std::vector<Time> knots12 = {-30.0, -20.0, -10.0, 0.0, 5.0,  10.0,
                                           15.0,  20.0,  25.0,  30.0, 40.0, 50.0};
        CubicBSplinesFitting f(knots12, true);
        for (Integer i = 0; i <= 7; ++i) {
            std::vector<Real> row;
            for (Time t : kTimes)
                row.push_back(f.basisFunction(i, t));
            std::cout << "    \"N" << i << "\": " << arr(row)
                      << (i == 7 ? "\n" : ",\n");
        }
    }
    std::cout << "  },\n";

    std::cout << "  \"layer1_bernstein\": {\n";
    {
        // The exact calls SimplePolynomialFitting::discountFunction makes
        // (cpp:378-385): get(i,i,t) unconstrained, get(i+1,i+1,t) constrained.
        for (Natural i = 0; i <= 4; ++i) {
            std::vector<Real> row;
            for (Time t : kTimes)
                row.push_back(BernsteinPolynomial::get(i, i, t));
            std::cout << "    \"B" << i << "_" << i << "\": " << arr(row)
                      << (i == 4 ? "\n" : ",\n");
        }
    }
    std::cout << "  },\n";

    // =======================================================================
    // "don't fit" mode — maxEvaluations == 0, both constructor overloads
    // =======================================================================
    std::cout << "  \"no_fit\": {\n";
    {
        // Parameters verbatim from test-suite/fittedbonddiscountcurve.cpp:49-59.
        Array parameters = {-51293.44,  -212240.36, 168668.51, 88792.74, 120712.13,
                            -34332.83,  -66479.66,  13605.17,  0.0};
        ExponentialSplinesFitting method;

        FittedBondDiscountCurve c1(kEval, method, parameters, maxDate10y, act365);
        FittedBondDiscountCurve c2(0, TARGET(), method, parameters, maxDate10y, act365);

        std::vector<Real> d1, d2;
        for (Time t : {0.5, 1.0, 3.0, 7.0, 9.9})
            d1.push_back(c1.discount(t));
        for (Time t : {0.5, 1.0, 3.0, 7.0, 9.9})
            d2.push_back(c2.discount(t));

        // The upstream parameter set has kappa == x[8] == 0.0, which collapses
        // the exponential-splines discount to identically 1.0 (every
        // exp(-kappa*n*t) is 1 and the constrained coefficient completes the
        // sum). Great for the "does it plumb" question, useless as an
        // arithmetic pin — so a second, non-degenerate set is fitted too.
        Array parameters2 = {0.30, -0.15, 0.08, -0.04, 0.02, -0.01, 0.005, -0.002, 0.055};
        FittedBondDiscountCurve c3(kEval, method, parameters2, maxDate10y, act365);
        std::vector<Real> d3;
        for (Time t : {0.5, 1.0, 3.0, 7.0, 9.9})
            d3.push_back(c3.discount(t));

        std::cout << "    \"reference_date_c1\": " << c1.referenceDate().serialNumber() << ",\n"
                  << "    \"reference_date_c2\": " << c2.referenceDate().serialNumber() << ",\n"
                  << "    \"number_of_bonds\": " << c1.numberOfBonds() << ",\n"
                  << "    \"max_date\": " << c1.maxDate().serialNumber() << ",\n"
                  << "    \"solution\": " << arr(c1.fitResults().solution()) << ",\n"
                  << "    \"number_of_iterations\": " << c1.fitResults().numberOfIterations()
                  << ",\n"
                  << "    \"minimum_cost_value\": " << num(c1.fitResults().minimumCostValue())
                  << ",\n"
                  << "    \"error_code\": " << static_cast<int>(c1.fitResults().errorCode())
                  << ",\n"
                  << "    \"weights\": " << arr(c1.fitResults().weights()) << ",\n"
                  << "    \"t\": [0.5, 1, 3, 7, 9.9],\n"
                  << "    \"discount_c1\": " << arr(d1) << ",\n"
                  << "    \"discount_c2\": " << arr(d2) << ",\n"
                  << "    \"parameters_nondegenerate\": " << arr(parameters2) << ",\n"
                  << "    \"discount_c3\": " << arr(d3) << "\n";
    }
    std::cout << "  },\n";

    // =======================================================================
    // SpreadFittingMethod — rebase + discountingCurve->discount(t, true)
    // =======================================================================
    std::cout << "  \"spread\": {\n";
    {
        Array x = {0.004, -0.002, 0.003, 1.4};

        // (a) discounting curve anchored 30 days BEFORE the fitted curve, so
        //     rebase_ = discountingCurve->discount(curve->referenceDate()) != 1
        //     (cpp:418-429).
        auto base = ext::make_shared<FlatForward>(kEval - 30, 0.021, act365, Continuous, Annual);
        base->enableExtrapolation();
        auto inner = ext::make_shared<NelsonSiegelFitting>();
        SpreadFittingMethod spread(inner, Handle<YieldTermStructure>(base), 0.5, 8.0);

        FittedBondDiscountCurve curve(kEval, spread, x, maxDate10y, act365);
        curve.enableExtrapolation();
        std::vector<Real> d;
        for (Time t : kTimes)
            d.push_back(curve.discount(t, true));

        // (b) same reference date -> rebase_ == 1.0.
        auto base2 = ext::make_shared<FlatForward>(kEval, 0.021, act365, Continuous, Annual);
        base2->enableExtrapolation();
        auto inner2 = ext::make_shared<NelsonSiegelFitting>();
        SpreadFittingMethod spread2(inner2, Handle<YieldTermStructure>(base2));
        FittedBondDiscountCurve curve2(kEval, spread2, x, maxDate10y, act365);
        curve2.enableExtrapolation();
        std::vector<Real> d2;
        for (Time t : kTimes)
            d2.push_back(curve2.discount(t, true));

        std::cout << "    \"x\": " << arr(x) << ",\n"
                  << "    \"rebase\": " << num(base->discount(kEval)) << ",\n"
                  << "    \"base_reference_date\": " << base->referenceDate().serialNumber()
                  << ",\n"
                  << "    \"t\": " << arr(kTimes) << ",\n"
                  << "    \"d_rebased\": " << arr(d) << ",\n"
                  << "    \"d_same_reference\": " << arr(d2) << "\n";
    }
    std::cout << "  },\n";

    // =======================================================================
    // LAYER 2 — init()'s weight inputs, for both bond sets
    // =======================================================================
    std::cout << "  \"layer2_canada\": {\n";
    emitWeightInputs(canadaSet(), act365);
    std::cout << "  },\n";

    std::cout << "  \"layer2_zeros\": {\n";
    emitWeightInputs(zeroSet(), act365);
    std::cout << "  },\n";

    // =======================================================================
    // LAYER 3 — FittingCost at a fixed x, via the stub optimizer
    // =======================================================================
    std::cout << "  \"layer3_ns_canada\": {\n";
    {
        BondSet s = canadaSet();
        auto opt = ext::make_shared<ProbeOptimizer>();
        Array guess = {0.0317, 5.0, -3.6796, 24.1703};
        NelsonSiegelFitting method(Array(), opt, Array());
        FittedBondDiscountCurve curve(kEval, s.helpers, act365, method, 1e-10, 10000, guess);
        curve.enableExtrapolation();
        emitFitResults(curve);
        std::cout << "    \"cost_x\": " << arr(opt->x0) << ",\n"
                  << "    \"cost_values\": " << arr(opt->vals) << ",\n"
                  << "    \"cost_value\": " << num(opt->val) << ",\n";
        emitCurveSamples(curve);
        std::cout << "    \"n_bonds\": " << curve.numberOfBonds() << "\n";
    }
    std::cout << "  },\n";

    std::cout << "  \"layer3_ns_zeros_l2\": {\n";
    {
        // L2 penalty: values() grows to n + N entries, the tail being
        // l2[i]*(x[i]-guess[i])^2 (cpp:329-334). Because the stub optimizer
        // evaluates AT the guess, every penalty term is exactly zero here —
        // which is itself the discriminating check that the penalty is
        // measured from the guess and not from the origin.
        BondSet s = zeroSet();
        auto opt = ext::make_shared<ProbeOptimizer>();
        Array l2 = {0.25, 0.5, 0.75, 1.0};
        Array guess = {0.021, -0.004, 0.011, 1.3};
        NelsonSiegelFitting method(Array(), opt, l2);
        FittedBondDiscountCurve curve(kEval, s.helpers, act365, method, 1e-10, 10000, guess);
        curve.enableExtrapolation();
        emitFitResults(curve);
        std::cout << "    \"cost_x\": " << arr(opt->x0) << ",\n"
                  << "    \"cost_values\": " << arr(opt->vals) << ",\n"
                  << "    \"cost_value\": " << num(opt->val) << ",\n";
        emitCurveSamples(curve);
        std::cout << "    \"n_bonds\": " << curve.numberOfBonds() << "\n";
    }
    std::cout << "  },\n";

    std::cout << "  \"layer3_l2_offset\": {\n";
    {
        // Same as above but the guess handed to the CURVE differs from the
        // point the optimizer is evaluated at... it cannot: the optimizer sees
        // problem.currentValue() == guessSolution_. So instead, give explicit
        // (non-duration) weights, which skips the whole yield/duration path and
        // isolates values()[i] = (w_i * quoteError_i)^2 (cpp:323-327).
        BondSet s = zeroSet();
        auto opt = ext::make_shared<ProbeOptimizer>();
        Array weights = {0.1, 0.2, 0.3, 0.4};
        Array guess = {0.021, -0.004, 0.011, 1.3};
        NelsonSiegelFitting method(weights, opt, Array());
        FittedBondDiscountCurve curve(kEval, s.helpers, act365, method, 1e-10, 10000, guess);
        curve.enableExtrapolation();
        emitFitResults(curve);
        std::cout << "    \"cost_x\": " << arr(opt->x0) << ",\n"
                  << "    \"cost_values\": " << arr(opt->vals) << ",\n"
                  << "    \"cost_value\": " << num(opt->val) << ",\n";
        emitCurveSamples(curve);
        std::cout << "    \"n_bonds\": " << curve.numberOfBonds() << "\n";
    }
    std::cout << "  },\n";

    // =======================================================================
    // LAYER 4 — the real Simplex fit
    // =======================================================================
    {
        // Two free parameters over four zero-coupon bonds: the simplex
        // converges on simplex SIZE well before the iteration cap, so the exit
        // is StationaryPoint and the path is short.
        Array polyGuess = {-0.02, 0.0};
        emitFit("layer4_poly_zeros", SimplePolynomialFitting(2, true), zeroSet(), polyGuess,
                1e-10, 5000);

        // Four parameters over four bonds: fits to machine precision but keeps
        // wandering, so the exit is MaxIterations at the (deliberately low)
        // cap. The stiffest reproducibility test in the file.
        Array nsGuess0 = {0.021, -0.004, 0.011, 1.3};
        emitFit("layer4_ns_zeros", NelsonSiegelFitting(), zeroSet(), nsGuess0, 1e-9, 400);
    }

    // =======================================================================
    // LAYER 4 — one real fit per concrete fitting method, so that every
    // discountFunction is exercised through the optimizer and not only at
    // hand-picked parameters.
    // =======================================================================
    {
        BondSet zeros = zeroSet();
        BondSet canada = canadaSet();

        // The upstream test's scenario (test-suite:139-147), capped at 1000
        // iterations instead of 10000 to keep the path short; the exit is
        // MaxIterations either way.
        Array nsGuess = {0.0317, 5.0, -3.6796, 24.1703};
        emitFit("layer4_ns_canada", NelsonSiegelFitting(), canada, nsGuess, 1e-10, 1000);

        // Same, with the cutoffs the upstream test's "method2" uses: flat
        // forward before the first and after the last bond maturity
        // (test-suite:130-133).
        Real minT = kAct365.yearFraction(kEval, canada.helpers.front()->bond()->maturityDate());
        Real maxT = kAct365.yearFraction(kEval, canada.helpers.back()->bond()->maturityDate());
        emitFit("layer4_ns_canada_cutoff",
                NelsonSiegelFitting(Array(), ext::shared_ptr<OptimizationMethod>(), Array(),
                                    minT, maxT),
                canada, nsGuess, 1e-10, 1000);
        std::cout << "  \"layer4_ns_canada_cutoff_times\": {\"min\": " << num(minT)
                  << ", \"max\": " << num(maxT) << "},\n";

        Array svGuess = {0.011, -0.001, -0.001, 0.0005, 0.3, 1.2};
        emitFit("layer4_svensson_zeros", SvenssonFitting(), zeros, svGuess, 1e-9, 300);

        // numCoeffs 4 with kappa fixed -> size 3.
        Array esGuess = {0.2, 0.1, 0.05};
        emitFit("layer4_exp_zeros", ExponentialSplinesFitting(true, 4, 0.05), zeros, esGuess,
                1e-9, 500);

        const std::vector<Time> knots9 = {-10.0, -5.0, 0.0, 4.0, 8.0, 12.0, 20.0, 30.0, 40.0};
        Array bsGuess = {0.98, 0.94, 0.88, 0.80};
        emitFit("layer4_bspline_zeros", CubicBSplinesFitting(knots9, true), zeros, bsGuess,
                1e-9, 500);

        Array ncGuess = {0.99, 0.98, 0.95, 0.90};
        emitFit("layer4_natcubic_zeros",
                NaturalCubicFitting(std::vector<Time>{1.0, 2.0, 5.0, 10.0}), zeros, ncGuess,
                1e-9, 500);

        // A real fit through SpreadFittingMethod, so init()'s rebase and the
        // extrapolating discountingCurve_ call are on the optimizer's path.
        auto base = ext::make_shared<FlatForward>(kEval - 30, 0.012, act365, Continuous, Annual);
        base->enableExtrapolation();
        auto inner = ext::make_shared<NelsonSiegelFitting>();
        Array spGuess = {0.0, 0.0, 0.0, 1.0};
        emitFit("layer4_spread_zeros",
                SpreadFittingMethod(inner, Handle<YieldTermStructure>(base), 0.0, QL_MAX_REAL),
                zeros, spGuess, 1e-9, 300);

        // Constraint threading: the same one-parameter problem with and
        // without PositiveConstraint. Simplex builds its initial simplex with
        // Constraint::update, which HALVES the step until feasible, so the two
        // take different paths and land on opposite signs (test-suite:326-334).
        Array cGuess = {0.01};
        emitFit("layer4_unconstrained", SimplePolynomialFitting(1, true), zeros, cGuess, 1e-10,
                5000);
        emitFit("layer4_positive_constrained",
                SimplePolynomialFitting(1, true, Array(),
                                        ext::shared_ptr<OptimizationMethod>(), Array(), 0.0,
                                        QL_MAX_REAL, PositiveConstraint()),
                zeros, cGuess, 1e-10, 5000);

        // --- divergence localisation -------------------------------------
        // Both fits below sit in a FLAT valley: exp splines because three free
        // coefficients cannot fit four bonds (so the simplex crawls along a
        // ridge), cubic B-splines because four coefficients fit four bonds
        // EXACTLY (so the cost bottoms out near 1e-25 and the last steps
        // compare noise). Truncating the same fit at a ladder of caps says
        // where any two implementations first take different branches.
        auto esOpt = ext::make_shared<ProbeOptimizer>();
        emitCostAtGuess("layer3_exp_zeros",
                        ExponentialSplinesFitting(true, Array(), esOpt, Array(), 0.0,
                                                  QL_MAX_REAL, 4, 0.05),
                        esOpt, zeros, esGuess);
        auto bsOpt = ext::make_shared<ProbeOptimizer>();
        emitCostAtGuess("layer3_bspline_zeros",
                        CubicBSplinesFitting(knots9, true, Array(), bsOpt, Array()), bsOpt,
                        zeros, bsGuess);
        emitCapSweep("layer4_exp_zeros_caps", ExponentialSplinesFitting(true, 4, 0.05), zeros,
                     esGuess, 1e-9, {110, 150, 200, 220, 230, 240, 250});
        // Same fit truncated much earlier, to bracket the FIRST differing
        // branch rather than only observing that the endpoints differ.
        emitCapSweep("layer4_exp_zeros_earlycaps", ExponentialSplinesFitting(true, 4, 0.05),
                     zeros, esGuess, 1e-9,
                     {6, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 80, 100}, 5);
        emitCapSweep("layer4_bspline_zeros_caps", CubicBSplinesFitting(knots9, true), zeros,
                     bsGuess, 1e-9, {110, 200, 300, 400, 500, 600, 700, 800, 840});
    }

    // =======================================================================
    // resetGuess / refit — calculate() writes the solution back into
    // curve_->guessSolution_ (cpp:295), so a plain refit restarts from the
    // previous answer while resetGuess restarts from the original.
    // =======================================================================
    std::cout << "  \"layer4_refit\": {\n";
    {
        BondSet s = zeroSet();
        SimplePolynomialFitting method(2, true);
        Array guess = {-0.02, 0.0};
        FittedBondDiscountCurve curve(kEval, s.helpers, act365, method, 1e-10, 5000, guess);
        curve.enableExtrapolation();

        Array sol1 = curve.fitResults().solution();
        Integer n1 = curve.fitResults().numberOfIterations();
        Real c1 = curve.fitResults().minimumCostValue();

        curve.update(); // invalidate; the next calculate() starts from sol1
        Array sol2 = curve.fitResults().solution();
        Integer n2 = curve.fitResults().numberOfIterations();
        Real c2 = curve.fitResults().minimumCostValue();

        curve.resetGuess(guess); // back to the original starting point
        Array sol3 = curve.fitResults().solution();
        Integer n3 = curve.fitResults().numberOfIterations();
        Real c3 = curve.fitResults().minimumCostValue();

        std::cout << "    \"solution_1\": " << arr(sol1) << ",\n"
                  << "    \"iterations_1\": " << n1 << ",\n"
                  << "    \"cost_1\": " << num(c1) << ",\n"
                  << "    \"solution_2\": " << arr(sol2) << ",\n"
                  << "    \"iterations_2\": " << n2 << ",\n"
                  << "    \"cost_2\": " << num(c2) << ",\n"
                  << "    \"solution_3\": " << arr(sol3) << ",\n"
                  << "    \"iterations_3\": " << n3 << ",\n"
                  << "    \"cost_3\": " << num(c3) << "\n";
    }
    std::cout << "  },\n";

    // =======================================================================
    // Clone<FittingMethod> ownership: the curve holds a CLONE, so the caller's
    // instance never sees the fit (fittedbonddiscountcurve.hpp:164).
    // =======================================================================
    std::cout << "  \"clone_ownership\": {\n";
    {
        BondSet s = zeroSet();
        SimplePolynomialFitting caller(2, true);
        Array guess = {-0.02, 0.0};
        FittedBondDiscountCurve curve(kEval, s.helpers, act365, caller, 1e-10, 5000, guess);
        curve.enableExtrapolation();
        curve.fitResults(); // forces the fit

        // NOT emitted: caller.numberOfIterations() / caller.minimumCostValue().
        // fittedbonddiscountcurve.hpp:273-275 declares numberOfIterations_ and
        // costValue_ with NO initializer, so on an object that was never fitted
        // they are indeterminate — reading them would put uninitialised memory
        // into the reference file. errorCode_ IS initialised (hpp:277).
        std::cout << "    \"caller_solution_size\": " << caller.solution().size() << ",\n"
                  << "    \"caller_weights_size\": " << caller.weights().size() << ",\n"
                  << "    \"caller_error_code\": " << static_cast<int>(caller.errorCode())
                  << ",\n"
                  << "    \"curve_solution_size\": " << curve.fitResults().solution().size()
                  << ",\n"
                  << "    \"curve_weights_size\": " << curve.fitResults().weights().size()
                  << "\n";
    }
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
