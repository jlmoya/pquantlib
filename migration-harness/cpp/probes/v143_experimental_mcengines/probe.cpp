// migration-harness/cpp/probes/v143_experimental_mcengines/probe.cpp
//
// Reference values for the ql/experimental Monte-Carlo *engine* family @ v1.43:
//
//   ql/experimental/mcbasket/mclongstaffschwartzpathengine.hpp
//       MCLongstaffSchwartzPathEngine
//   ql/experimental/mcbasket/longstaffschwartzmultipathpricer.hpp
//       LongstaffSchwartzMultiPathPricer, LongstaffSchwartzMultiPathPricer::PathInfo
//   ql/experimental/mcbasket/mcamericanpathengine.hpp
//       MCAmericanPathEngine, MakeMCAmericanPathEngine
//   ql/experimental/mcbasket/mcpathbasketengine.hpp
//       MakeMCPathBasketEngine
//   ql/experimental/exoticoptions/mceverestengine.hpp   MakeMCEverestEngine
//   ql/experimental/exoticoptions/mchimalayaengine.hpp  MakeMCHimalayaEngine
//   ql/experimental/exoticoptions/mcpagodaengine.hpp    MakeMCPagodaEngine
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// These are Monte-Carlo engines over a *reproducible* stream: MersenneTwister
// (integer state, bit-exact across ports) or Sobol, mapped through
// InverseCumulativeNormal. So a fixed seed pins an exact NPV, not a
// statistical band, and that is what every block below emits.
//
// The one thing that breaks pathwise reproducibility is NOT the RNG: it is
// StochasticProcessArray, which premultiplies the Gaussian increment by
// pseudoSqrt(correlation, SalvagingAlgorithm::Spectral). That pseudo-root is
// basis-dependent — any M with M*M^T == corr is a valid answer, and different
// eigen-solvers pick different M. C++ uses SymmetricSchurDecomposition
// (Jacobi, eigenvalues DESCENDING) and then normalizePseudoRoot().
//
// Block Z pins that matrix explicitly so a port can check its own choice
// instead of assuming. Note the special case that the rest of the probe leans
// on: for the IDENTITY correlation the pseudo-root is exactly the identity in
// any basis, so identity-correlated baskets are pathwise reproducible
// regardless of the eigen-solver. Blocks A..F therefore use identity
// correlation for the pinned-NPV cases, and block Z carries the correlated
// evidence separately.
//
// Blocks:
//   Z  pseudoSqrt(Spectral) for identity and for a non-trivial correlation,
//      plus the raw SymmetricSchurDecomposition eigen-system behind it.
//   A  MCEverestEngine   / MakeMCEverestEngine    (Pseudo + LowDiscrepancy)
//   B  MCPagodaEngine    / MakeMCPagodaEngine     (Pseudo + LowDiscrepancy)
//   C  MCHimalayaEngine  / MakeMCHimalayaEngine   (Pseudo + LowDiscrepancy)
//   D  MCPathBasketEngine/ MakeMCPathBasketEngine (Pseudo + LowDiscrepancy)
//   E  LongstaffSchwartzMultiPathPricer + PathInfo, driven by HAND-BUILT paths
//      so the regression is pinned independently of any path generator:
//      per-path PathInfo, the per-exercise-date regression coefficients, the
//      lowerBounds_ vector after calibrate(), and operator() per path.
//   F  MCAmericanPathEngine / MakeMCAmericanPathEngine end-to-end.
//   G  the named-parameter guards of every MakeMC* builder (exact messages).
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/mcengines.json.

#include <ql/experimental/exoticoptions/everestoption.hpp>
#include <ql/experimental/exoticoptions/himalayaoption.hpp>
#include <ql/experimental/exoticoptions/mceverestengine.hpp>
#include <ql/experimental/exoticoptions/mchimalayaengine.hpp>
#include <ql/experimental/exoticoptions/mcpagodaengine.hpp>
#include <ql/experimental/exoticoptions/pagodaoption.hpp>
#include <ql/experimental/mcbasket/adaptedpathpayoff.hpp>
#include <ql/experimental/mcbasket/mcamericanpathengine.hpp>
#include <ql/experimental/mcbasket/mcpathbasketengine.hpp>
#include <ql/experimental/mcbasket/pathmultiassetoption.hpp>
#include <ql/exercise.hpp>
#include <ql/math/matrixutilities/pseudosqrt.hpp>
#include <ql/math/matrixutilities/symmetricschurdecomposition.hpp>
#include <ql/methods/montecarlo/lsmbasissystem.hpp>
#include <ql/methods/montecarlo/multipath.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/stochasticprocessarray.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/impliedtermstructure.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --------------------------------------------------------------------------
// JSON emission (flat top-level object; the pquantlib harness convention)
// --------------------------------------------------------------------------

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

void emit_bool(const std::string& name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

void emit_iarr(const std::string& name, const std::vector<long long>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]";
}

void emit_matrix(const std::string& name, const Matrix& m) {
    std::vector<Real> flat;
    for (Size i = 0; i < m.rows(); ++i)
        for (Size j = 0; j < m.columns(); ++j)
            flat.push_back(m[i][j]);
    emit_arr(name, flat);
}

void emit_array(const std::string& name, const Array& a) {
    emit_arr(name, std::vector<Real>(a.begin(), a.end()));
}

// Run `f`, emit the QuantLib error message it raises (or "" if it does not).
template <class F>
void emit_throws(const std::string& name, F f) {
    std::string msg;
    try {
        f();
    } catch (const std::exception& e) {
        msg = e.what();
    }
    // QuantLib decorates messages with the source location when
    // QL_ERROR_LINES is on; keep only the final sentence, which is the
    // QL_REQUIRE text a port can reproduce.
    const std::size_t nl = msg.find_last_of('\n');
    if (nl != std::string::npos) msg = msg.substr(nl + 1);
    emit_str(name, msg);
}

// Instrument::errorEstimate() does not return Null when the engine left the
// result unset -- it raises. Pin that, since it is how a caller observes
// "this RNG policy publishes no error estimate".
void emit_error_estimate_absent(const std::string& tag, const Instrument& option) {
    std::string msg;
    try {
        (void)option.errorEstimate();
    } catch (const std::exception& e) {
        msg = e.what();
    }
    const std::size_t nl = msg.find_last_of('\n');
    if (nl != std::string::npos) msg = msg.substr(nl + 1);
    emit_str(tag + "_error_estimate_raises", msg);
}

// --------------------------------------------------------------------------
// Shared market
// --------------------------------------------------------------------------

const Date TODAY(15, January, 2024);
const Rate RISK_FREE = 0.05;
const Rate DIVIDEND = 0.02;

const Real SPOTS[3] = {100.0, 95.0, 105.0};
const Volatility VOLS[3] = {0.20, 0.25, 0.30};

DayCounter dayCounter() { return Actual365Fixed(); }

Handle<YieldTermStructure> flatRate(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(TODAY, r, dayCounter()));
}

ext::shared_ptr<StochasticProcess1D> bsmProcess(Size i) {
    Handle<Quote> spot(ext::make_shared<SimpleQuote>(SPOTS[i]));
    Handle<BlackVolTermStructure> vol(ext::make_shared<BlackConstantVol>(
        TODAY, NullCalendar(), VOLS[i], dayCounter()));
    return ext::make_shared<BlackScholesMertonProcess>(
        spot, flatRate(DIVIDEND), flatRate(RISK_FREE), vol);
}

// n independent assets: the pseudo-root of the identity is the identity in
// every basis, so these paths are eigen-solver independent.
ext::shared_ptr<StochasticProcessArray> independentBasket(Size n) {
    std::vector<ext::shared_ptr<StochasticProcess1D> > procs;
    for (Size i = 0; i < n; ++i) procs.push_back(bsmProcess(i));
    Matrix corr(n, n, 0.0);
    for (Size i = 0; i < n; ++i) corr[i][i] = 1.0;
    return ext::make_shared<StochasticProcessArray>(procs, corr);
}

Matrix correlated3() {
    Matrix corr(3, 3);
    corr[0][0] = 1.0;  corr[0][1] = 0.5;  corr[0][2] = 0.4;
    corr[1][0] = 0.5;  corr[1][1] = 1.0;  corr[1][2] = 0.3;
    corr[2][0] = 0.4;  corr[2][1] = 0.3;  corr[2][2] = 1.0;
    return corr;
}

std::vector<Date> threeFixings() {
    return {TODAY + Period(4, Months), TODAY + Period(8, Months),
            TODAY + Period(12, Months)};
}

// --------------------------------------------------------------------------
// The one concrete PathPayoff this probe uses everywhere.
//
//   payments[i]  = coupon               at i == 0
//                + max(avg(T) - kCall, 0) at i == last
//   exercises[i] = max(kPut - avg(i), 0)  for i >= 1  (none at i == 0)
//   states[i]    = { avg(i) }             for i >= 1
//
// avg(i) is the arithmetic mean over assets at fixing i. basisSystemDimension
// is 1, so LsmBasisSystem::multiPathBasisSystem(1, 2, Monomial) gives the
// three functions {1, x, x^2}.
//
// Leaving fixing 0 non-exercisable is deliberate: it drives the "states empty
// => cannot exercise" branch of both calibrate() and operator().
// --------------------------------------------------------------------------

class BasketPathPayoff : public AdaptedPathPayoff {
  public:
    BasketPathPayoff(Real coupon, Real kCall, Real kPut)
    : coupon_(coupon), kCall_(kCall), kPut_(kPut) {}

    std::string name() const override { return "basket-path-payoff"; }
    std::string description() const override {
        return "coupon at first fixing, basket call at last, American basket put in between";
    }
    Size basisSystemDimension() const override { return 1; }

  protected:
    void operator()(ValuationData& data) const override {
        const Size nTimes = data.numberOfTimes();
        const Size nAssets = data.numberOfAssets();
        for (Size i = 0; i < nTimes; ++i) {
            Real sum = 0.0;
            for (Size j = 0; j < nAssets; ++j) sum += data.getAssetValue(i, j);
            const Real avg = sum / static_cast<Real>(nAssets);

            Real payment = 0.0;
            if (i == 0) payment += coupon_;
            if (i == nTimes - 1) payment += std::max(avg - kCall_, 0.0);
            data.setPayoffValue(i, payment);

            if (i > 0) {
                Array state(1, avg);
                data.setExerciseData(i, std::max(kPut_ - avg, 0.0), state);
            }
        }
    }

  private:
    Real coupon_, kCall_, kPut_;
};

// --------------------------------------------------------------------------
// PathMultiAssetOption is abstract in C++ (pathPayoff/fixingDates are pure
// virtual and the library ships no concrete). This is the minimal concrete.
// --------------------------------------------------------------------------

class ConcretePathMultiAssetOption : public PathMultiAssetOption {
  public:
    ConcretePathMultiAssetOption(ext::shared_ptr<PathPayoff> payoff,
                                 std::vector<Date> fixings)
    : payoff_(std::move(payoff)), fixings_(std::move(fixings)) {}

    ext::shared_ptr<PathPayoff> pathPayoff() const override { return payoff_; }
    std::vector<Date> fixingDates() const override { return fixings_; }
    bool isExpired() const override { return false; }

  private:
    ext::shared_ptr<PathPayoff> payoff_;
    std::vector<Date> fixings_;
};

// --------------------------------------------------------------------------
// Subclass that publishes LongstaffSchwartzMultiPathPricer's protected guts:
// PathInfo, transformPath(), coeff_, lowerBounds_, calibrationPhase_.
// --------------------------------------------------------------------------

class OpenLSMPricer : public LongstaffSchwartzMultiPathPricer {
  public:
    using LongstaffSchwartzMultiPathPricer::LongstaffSchwartzMultiPathPricer;
    using LongstaffSchwartzMultiPathPricer::PathInfo;
    using LongstaffSchwartzMultiPathPricer::calibrationPhase_;
    using LongstaffSchwartzMultiPathPricer::coeff_;
    using LongstaffSchwartzMultiPathPricer::lowerBounds_;
    using LongstaffSchwartzMultiPathPricer::transformPath;
    using LongstaffSchwartzMultiPathPricer::v_;
};

// --------------------------------------------------------------------------
// Block Z — the pseudo-root that StochasticProcessArray premultiplies by.
// --------------------------------------------------------------------------

void blockZ() {
    Matrix id(3, 3, 0.0);
    for (Size i = 0; i < 3; ++i) id[i][i] = 1.0;

    emit_matrix("Z_pseudoSqrt_spectral_identity3", pseudoSqrt(id, SalvagingAlgorithm::Spectral));

    const Matrix corr = correlated3();
    emit_matrix("Z_corr3", corr);
    emit_matrix("Z_pseudoSqrt_spectral_corr3", pseudoSqrt(corr, SalvagingAlgorithm::Spectral));

    SymmetricSchurDecomposition jd(corr);
    emit_array("Z_schur_eigenvalues_corr3", jd.eigenvalues());
    emit_matrix("Z_schur_eigenvectors_corr3", jd.eigenvectors());
    emit_str("Z_note",
             "pseudoSqrt(Spectral) = jd.eigenvectors() * diag(sqrt(max(lambda,0))) "
             "then normalizePseudoRoot(); eigenvalues come out of "
             "SymmetricSchurDecomposition in DECREASING order");

    // Row norms of the pseudo-root: normalizePseudoRoot() forces these to
    // sqrt(matrix[i][i]) == 1 for a unit-diagonal correlation.
    const Matrix root = pseudoSqrt(corr, SalvagingAlgorithm::Spectral);
    std::vector<Real> norms;
    for (Size i = 0; i < root.rows(); ++i) {
        Real n = 0.0;
        for (Size j = 0; j < root.columns(); ++j) n += root[i][j] * root[i][j];
        norms.push_back(std::sqrt(n));
    }
    emit_arr("Z_pseudoSqrt_row_norms_corr3", norms);
}

// --------------------------------------------------------------------------
// Block A — MCEverestEngine / MakeMCEverestEngine
// --------------------------------------------------------------------------

void everestCase(const std::string& tag,
                 const ext::shared_ptr<PricingEngine>& engine,
                 Real notional,
                 Rate guarantee,
                 const ext::shared_ptr<Exercise>& exercise,
                 bool hasErrorEstimate) {
    EverestOption option(notional, guarantee, exercise);
    option.setPricingEngine(engine);
    emit(tag + "_npv", option.NPV());
    emit(tag + "_yield", option.yield());
    if (hasErrorEstimate)
        emit(tag + "_error_estimate", option.errorEstimate());
    else
        emit_error_estimate_absent(tag, option);
}

void blockA() {
    Settings::instance().evaluationDate() = TODAY;
    auto processes = independentBasket(3);
    auto exercise = ext::make_shared<EuropeanExercise>(TODAY + Period(1, Years));
    const Real notional = 1.0e6;
    const Rate guarantee = 0.03;

    emit_int("A_notional", static_cast<long long>(notional));
    emit("A_guarantee", guarantee);

    // A1 — pseudo-random, fixed step count, fixed sample count.
    everestCase("A1",
                MakeMCEverestEngine<PseudoRandom>(processes)
                    .withSteps(4)
                    .withSamples(255)
                    .withSeed(42),
                notional, guarantee, exercise, true);

    // A2 — steps-per-year + antithetic.
    everestCase("A2",
                MakeMCEverestEngine<PseudoRandom>(processes)
                    .withStepsPerYear(12)
                    .withSamples(255)
                    .withAntitheticVariate()
                    .withSeed(7),
                notional, guarantee, exercise, true);

    // A3 — low-discrepancy: no error estimate is published at all.
    everestCase("A3",
                MakeMCEverestEngine<LowDiscrepancy>(processes)
                    .withSteps(4)
                    .withSamples(256)
                    .withSeed(11),
                notional, guarantee, exercise, false);

    // A4 — tolerance-driven termination (minSamples defaults to 1023).
    {
        EverestOption option(notional, guarantee, exercise);
        auto engine = MakeMCEverestEngine<PseudoRandom>(processes)
                          .withSteps(2)
                          .withAbsoluteTolerance(4000.0)
                          .withMaxSamples(20000)
                          .withSeed(3);
        option.setPricingEngine(engine);
        emit("A4_npv", option.NPV());
        emit("A4_error_estimate", option.errorEstimate());
    }
}

// --------------------------------------------------------------------------
// Block B — MCPagodaEngine / MakeMCPagodaEngine
// --------------------------------------------------------------------------

void blockB() {
    Settings::instance().evaluationDate() = TODAY;
    auto processes = independentBasket(3);
    const std::vector<Date> fixings = threeFixings();
    const Real roof = 15.0;
    const Real fraction = 0.62;

    emit("B_roof", roof);
    emit("B_fraction", fraction);

    {
        PagodaOption option(fixings, roof, fraction);
        option.setPricingEngine(MakeMCPagodaEngine<PseudoRandom>(processes)
                                    .withSamples(255)
                                    .withSeed(42));
        emit("B1_npv", option.NPV());
        emit("B1_error_estimate", option.errorEstimate());
    }
    {
        PagodaOption option(fixings, roof, fraction);
        option.setPricingEngine(MakeMCPagodaEngine<PseudoRandom>(processes)
                                    .withSamples(255)
                                    .withAntitheticVariate()
                                    .withSeed(42));
        emit("B2_npv", option.NPV());
        emit("B2_error_estimate", option.errorEstimate());
    }
    {
        PagodaOption option(fixings, roof, fraction);
        option.setPricingEngine(MakeMCPagodaEngine<LowDiscrepancy>(processes)
                                    .withSamples(256)
                                    .withSeed(11));
        emit("B3_npv", option.NPV());
        emit_error_estimate_absent("B3", option);
    }
}

// --------------------------------------------------------------------------
// Block C — MCHimalayaEngine / MakeMCHimalayaEngine
// --------------------------------------------------------------------------

void blockC() {
    Settings::instance().evaluationDate() = TODAY;
    auto processes = independentBasket(3);
    const std::vector<Date> fixings = threeFixings();
    const Real strike = 100.0;

    emit("C_strike", strike);

    {
        HimalayaOption option(fixings, strike);
        option.setPricingEngine(MakeMCHimalayaEngine<PseudoRandom>(processes)
                                    .withSamples(255)
                                    .withSeed(42));
        emit("C1_npv", option.NPV());
        emit("C1_error_estimate", option.errorEstimate());
    }
    {
        HimalayaOption option(fixings, strike);
        option.setPricingEngine(MakeMCHimalayaEngine<PseudoRandom>(processes)
                                    .withSamples(255)
                                    .withAntitheticVariate()
                                    .withSeed(42));
        emit("C2_npv", option.NPV());
        emit("C2_error_estimate", option.errorEstimate());
    }
    {
        HimalayaOption option(fixings, strike);
        option.setPricingEngine(MakeMCHimalayaEngine<LowDiscrepancy>(processes)
                                    .withSamples(256)
                                    .withSeed(11));
        emit("C3_npv", option.NPV());
        emit_error_estimate_absent("C3", option);
    }
}

// --------------------------------------------------------------------------
// Block D — MCPathBasketEngine / MakeMCPathBasketEngine
// --------------------------------------------------------------------------

const Real D_COUPON = 1.5;
const Real D_KCALL = 100.0;
const Real D_KPUT = 105.0;

void blockD() {
    Settings::instance().evaluationDate() = TODAY;
    auto processes = independentBasket(3);
    const std::vector<Date> fixings = threeFixings();
    auto payoff = ext::make_shared<BasketPathPayoff>(D_COUPON, D_KCALL, D_KPUT);

    emit("D_coupon", D_COUPON);
    emit("D_call_strike", D_KCALL);
    emit("D_put_strike", D_KPUT);
    emit_int("D_basis_system_dimension", static_cast<long long>(payoff->basisSystemDimension()));
    emit_str("D_payoff_name", payoff->name());
    emit_str("D_payoff_description", payoff->description());

    {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCPathBasketEngine<PseudoRandom>(processes)
                                    .withSteps(3)
                                    .withSamples(255)
                                    .withSeed(42));
        emit("D1_npv", option.NPV());
        emit("D1_error_estimate", option.errorEstimate());
    }
    {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCPathBasketEngine<PseudoRandom>(processes)
                                    .withStepsPerYear(6)
                                    .withSamples(255)
                                    .withAntitheticVariate()
                                    .withSeed(42));
        emit("D2_npv", option.NPV());
        emit("D2_error_estimate", option.errorEstimate());
    }
    {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCPathBasketEngine<LowDiscrepancy>(processes)
                                    .withSteps(3)
                                    .withSamples(256)
                                    .withSeed(11));
        emit("D3_npv", option.NPV());
        emit_error_estimate_absent("D3", option);
    }
}

// --------------------------------------------------------------------------
// Block E — LongstaffSchwartzMultiPathPricer + PathInfo on hand-built paths.
//
// One asset, four grid nodes (t = 0, 1/3, 2/3, 1), three fixings at grid
// positions {1, 2, 3}. Eight calibration paths and four evaluation paths,
// scripted so that the ITM set at each exercise date is big enough for the
// three-function basis (v_.size() == 3).
// --------------------------------------------------------------------------

const Real E_SPOT_GRID[4] = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};

// values[p][k] is the asset level at grid node k for path p; node 0 is the
// spot and is never read by timePositions_ = {1,2,3}.
const Real E_CALIB[8][4] = {
    {100.0, 108.0, 112.0, 120.0},
    {100.0,  96.0,  92.0,  88.0},
    {100.0, 102.0,  99.0, 104.0},
    {100.0,  94.0,  97.0,  93.0},
    {100.0, 110.0, 101.0,  95.0},
    {100.0,  90.0,  85.0, 101.0},
    {100.0, 105.0, 115.0, 109.0},
    {100.0,  98.0, 103.0,  91.0},
};

const Real E_EVAL[4][4] = {
    {100.0, 101.0,  96.0,  99.0},
    {100.0,  92.0, 106.0, 113.0},
    {100.0, 115.0, 118.0,  84.0},
    {100.0,  99.0,  99.0,  99.0},
};

MultiPath handBuiltPath(const Real (&values)[4], const TimeGrid& grid) {
    Path p(grid);
    for (Size k = 0; k < 4; ++k) p[k] = values[k];
    return MultiPath(std::vector<Path>{p});
}

void blockE() {
    Settings::instance().evaluationDate() = TODAY;

    const std::vector<Time> times(E_SPOT_GRID, E_SPOT_GRID + 4);
    TimeGrid grid(times.begin(), times.end());

    const std::vector<Size> timePositions = {1, 2, 3};
    const std::vector<Date> fixings = threeFixings();

    Handle<YieldTermStructure> riskFree = flatRate(RISK_FREE);
    Array discounts(timePositions.size());
    std::vector<Handle<YieldTermStructure> > forwardTS(timePositions.size());
    for (Size i = 0; i < timePositions.size(); ++i) {
        discounts[i] = riskFree->discount(times[timePositions[i]]);
        forwardTS[i] = Handle<YieldTermStructure>(
            ext::make_shared<ImpliedTermStructure>(riskFree, fixings[i]));
    }
    emit_array("E_discounts", discounts);
    emit_arr("E_times", std::vector<Real>(times.begin(), times.end()));

    auto payoff = ext::make_shared<BasketPathPayoff>(D_COUPON, D_KCALL, D_KPUT);
    OpenLSMPricer pricer(payoff, timePositions, forwardTS, discounts, 2,
                         LsmBasisSystem::Monomial);

    emit_int("E_basis_size", static_cast<long long>(pricer.v_.size()));
    emit_bool("E_calibration_phase_initial", pricer.calibrationPhase_);

    // -- PathInfo, straight out of transformPath() -------------------------
    for (Size p = 0; p < 8; ++p) {
        MultiPath mp = handBuiltPath(E_CALIB[p], grid);
        OpenLSMPricer::PathInfo info = pricer.transformPath(mp);
        const std::string tag = "E_pathinfo_" + std::to_string(p);
        emit_int(tag + "_path_length", static_cast<long long>(info.pathLength()));
        emit_array(tag + "_payments", info.payments);
        emit_array(tag + "_exercises", info.exercises);
        std::vector<long long> stateSizes;
        std::vector<Real> stateValues;
        for (const Array& s : info.states) {
            stateSizes.push_back(static_cast<long long>(s.size()));
            for (Real v : s) stateValues.push_back(v);
        }
        emit_iarr(tag + "_state_sizes", stateSizes);
        emit_arr(tag + "_state_values", stateValues);
    }

    // -- calibration phase: operator() records and returns 0 ---------------
    std::vector<Real> calibrationReturns;
    for (Size p = 0; p < 8; ++p) {
        MultiPath mp = handBuiltPath(E_CALIB[p], grid);
        calibrationReturns.push_back(pricer(mp));
    }
    emit_arr("E_calibration_phase_returns", calibrationReturns);

    pricer.calibrate();
    emit_bool("E_calibration_phase_after", pricer.calibrationPhase_);

    // -- the regression itself ---------------------------------------------
    for (Size i = 0; i + 1 < timePositions.size(); ++i) {
        emit_array("E_coeff_" + std::to_string(i), pricer.coeff_[i]);
        emit_int("E_coeff_" + std::to_string(i) + "_size",
                 static_cast<long long>(pricer.coeff_[i].size()));
    }
    std::vector<Real> lowerBounds;
    for (Size i = 0; i < timePositions.size(); ++i)
        lowerBounds.push_back(pricer.lowerBounds_[i]);
    emit_arr("E_lower_bounds", lowerBounds);

    // -- pricing phase ------------------------------------------------------
    std::vector<Real> evalPrices;
    for (auto& p : E_EVAL) {
        MultiPath mp = handBuiltPath(p, grid);
        evalPrices.push_back(pricer(mp));
    }
    emit_arr("E_eval_prices", evalPrices);

    // The calibration paths re-priced after calibration — this is exactly
    // what MCLongstaffSchwartzPathEngine does (see block F note).
    std::vector<Real> calibPricesAfter;
    for (auto& p : E_CALIB) {
        MultiPath mp = handBuiltPath(p, grid);
        calibPricesAfter.push_back(pricer(mp));
    }
    emit_arr("E_calib_prices_after", calibPricesAfter);
}

// --------------------------------------------------------------------------
// Block F — MCAmericanPathEngine / MakeMCAmericanPathEngine end to end.
//
// C++ parity note that a port must reproduce: MCLongstaffSchwartzPathEngine::
// calculate() builds an mcModel_, runs nCalibrationSamples_ through it,
// calibrates -- and then calls McSimulation::calculate(), which THROWS THAT
// MODEL AWAY and builds a fresh one off a fresh pathGenerator() with the same
// seed. So the calibration statistics never reach the result, and the pricing
// paths are the same paths the regression was fitted on (in-sample).
// --------------------------------------------------------------------------

void blockF() {
    Settings::instance().evaluationDate() = TODAY;
    auto processes = independentBasket(3);
    const std::vector<Date> fixings = threeFixings();
    auto payoff = ext::make_shared<BasketPathPayoff>(D_COUPON, D_KCALL, D_KPUT);

    {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCAmericanPathEngine<PseudoRandom>(processes)
                                    .withSteps(3)
                                    .withSamples(255)
                                    .withCalibrationSamples(128)
                                    .withSeed(42));
        emit("F1_npv", option.NPV());
        emit("F1_error_estimate", option.errorEstimate());
    }
    {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCAmericanPathEngine<PseudoRandom>(processes)
                                    .withStepsPerYear(6)
                                    .withSamples(255)
                                    .withCalibrationSamples(128)
                                    .withAntitheticVariate()
                                    .withSeed(42));
        emit("F2_npv", option.NPV());
        emit("F2_error_estimate", option.errorEstimate());
    }
    {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCAmericanPathEngine<LowDiscrepancy>(processes)
                                    .withSteps(3)
                                    .withSamples(256)
                                    .withCalibrationSamples(128)
                                    .withSeed(11));
        emit("F3_npv", option.NPV());
        emit_error_estimate_absent("F3", option);
    }
    // F4 — the default calibration-sample count is 2048, not Null.
    {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCAmericanPathEngine<PseudoRandom>(processes)
                                    .withSteps(3)
                                    .withSamples(255)
                                    .withSeed(42));
        emit("F4_npv_default_calibration_samples", option.NPV());
    }
    emit_int("F_default_calibration_samples", 2048);
    emit_str("F_note",
             "MCLongstaffSchwartzPathEngine::calculate calibrates on an mcModel_ that "
             "McSimulation::calculate then discards, rebuilding the generator from the "
             "same seed: pricing is in-sample on the calibration paths");
}

// --------------------------------------------------------------------------
// Block G — named-parameter guards of the MakeMC* builders.
// --------------------------------------------------------------------------

void blockG() {
    Settings::instance().evaluationDate() = TODAY;
    auto processes = independentBasket(3);

    emit_throws("G_everest_samples_after_tolerance", [&] {
        MakeMCEverestEngine<PseudoRandom>(processes).withAbsoluteTolerance(1.0).withSamples(10);
    });
    emit_throws("G_everest_tolerance_after_samples", [&] {
        MakeMCEverestEngine<PseudoRandom>(processes).withSamples(10).withAbsoluteTolerance(1.0);
    });
    emit_throws("G_everest_tolerance_lowdiscrepancy", [&] {
        MakeMCEverestEngine<LowDiscrepancy>(processes).withAbsoluteTolerance(1.0);
    });
    emit_throws("G_everest_no_steps", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCEverestEngine<PseudoRandom>(processes).withSamples(10);
        (void)e;
    });
    emit_throws("G_everest_steps_overspecified", [&] {
        ext::shared_ptr<PricingEngine> e = MakeMCEverestEngine<PseudoRandom>(processes)
                                               .withSteps(2)
                                               .withStepsPerYear(2)
                                               .withSamples(10);
        (void)e;
    });
    emit_throws("G_american_no_steps", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCAmericanPathEngine<PseudoRandom>(processes).withSamples(10);
        (void)e;
    });
    emit_throws("G_american_steps_overspecified", [&] {
        ext::shared_ptr<PricingEngine> e = MakeMCAmericanPathEngine<PseudoRandom>(processes)
                                               .withSteps(2)
                                               .withStepsPerYear(2)
                                               .withSamples(10);
        (void)e;
    });
    emit_throws("G_himalaya_tolerance_lowdiscrepancy", [&] {
        MakeMCHimalayaEngine<LowDiscrepancy>(processes).withAbsoluteTolerance(1.0);
    });
    emit_throws("G_pagoda_samples_after_tolerance", [&] {
        MakeMCPagodaEngine<PseudoRandom>(processes).withAbsoluteTolerance(1.0).withSamples(10);
    });
    emit_throws("G_pathbasket_samples_after_tolerance", [&] {
        MakeMCPathBasketEngine<PseudoRandom>(processes).withAbsoluteTolerance(1.0).withSamples(10);
    });

    // MakeMCPathBasketEngine, unlike Everest/American, does NOT guard the
    // step count in its conversion operator: the engine constructor raises.
    emit_throws("G_pathbasket_no_steps", [&] {
        ext::shared_ptr<PricingEngine> e =
            MakeMCPathBasketEngine<PseudoRandom>(processes).withSamples(10);
        (void)e;
    });
    // Himalaya/Pagoda builders take no steps at all — their grid is the
    // fixing dates. Converting with nothing but samples must succeed.
    {
        bool ok = true;
        try {
            ext::shared_ptr<PricingEngine> e =
                MakeMCHimalayaEngine<PseudoRandom>(processes).withSamples(10);
            ok = static_cast<bool>(e);
        } catch (...) {
            ok = false;
        }
        emit_bool("G_himalaya_converts_without_steps", ok);
    }

    emit_int("G_pseudorandom_allows_error_estimate",
             static_cast<long long>(PseudoRandom::allowsErrorEstimate));
    emit_int("G_lowdiscrepancy_allows_error_estimate",
             static_cast<long long>(LowDiscrepancy::allowsErrorEstimate));
}

// --------------------------------------------------------------------------
// Block H — the single-sample branch of `if constexpr (allowsErrorEstimate)`.
//
// Every one of these engines writes results_.errorEstimate unconditionally
// once RNG::allowsErrorEstimate is true. GeneralStatistics::variance refuses
// on N == 1 ("sample number <=1, unsufficient"), so a PseudoRandom engine
// asked for a single sample raises out of NPV() itself -- it does not quietly
// leave the error estimate unset. LowDiscrepancy never enters that branch, so
// the same single-sample run prices fine and only errorEstimate() complains.
// --------------------------------------------------------------------------

void blockH() {
    Settings::instance().evaluationDate() = TODAY;
    auto processes = independentBasket(3);
    auto exercise = ext::make_shared<EuropeanExercise>(TODAY + Period(1, Years));
    const std::vector<Date> fixings = threeFixings();
    auto payoff = ext::make_shared<BasketPathPayoff>(D_COUPON, D_KCALL, D_KPUT);

    emit_throws("H_everest_single_sample", [&] {
        EverestOption option(1.0e6, 0.03, exercise);
        option.setPricingEngine(MakeMCEverestEngine<PseudoRandom>(processes)
                                    .withSteps(2)
                                    .withSamples(1)
                                    .withSeed(42));
        (void)option.NPV();
    });
    emit_throws("H_himalaya_single_sample", [&] {
        HimalayaOption option(fixings, 100.0);
        option.setPricingEngine(
            MakeMCHimalayaEngine<PseudoRandom>(processes).withSamples(1).withSeed(42));
        (void)option.NPV();
    });
    emit_throws("H_pagoda_single_sample", [&] {
        PagodaOption option(fixings, 15.0, 0.62);
        option.setPricingEngine(
            MakeMCPagodaEngine<PseudoRandom>(processes).withSamples(1).withSeed(42));
        (void)option.NPV();
    });
    emit_throws("H_pathbasket_single_sample", [&] {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCPathBasketEngine<PseudoRandom>(processes)
                                    .withSteps(3)
                                    .withSamples(1)
                                    .withSeed(42));
        (void)option.NPV();
    });
    emit_throws("H_american_single_sample", [&] {
        ConcretePathMultiAssetOption option(payoff, fixings);
        option.setPricingEngine(MakeMCAmericanPathEngine<PseudoRandom>(processes)
                                    .withSteps(3)
                                    .withSamples(1)
                                    .withCalibrationSamples(128)
                                    .withSeed(42));
        (void)option.NPV();
    });

    // LowDiscrepancy skips the branch entirely: one sample prices fine.
    {
        EverestOption option(1.0e6, 0.03, exercise);
        option.setPricingEngine(MakeMCEverestEngine<LowDiscrepancy>(processes)
                                    .withSteps(2)
                                    .withSamples(1)
                                    .withSeed(11));
        emit("H_everest_lowdiscrepancy_single_sample_npv", option.NPV());
        emit_error_estimate_absent("H_everest_lowdiscrepancy_single_sample", option);
    }
}

}  // namespace

int main() {
    std::cout << "{\n";
    blockZ();
    blockA();
    blockB();
    blockC();
    blockD();
    blockE();
    blockF();
    blockG();
    blockH();
    std::cout << "\n}\n";
    return 0;
}
