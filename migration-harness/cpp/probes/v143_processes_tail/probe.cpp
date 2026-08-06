// migration-harness/cpp/probes/v143_processes_tail/probe.cpp
//
// Reference values for the ql/processes coverage tail, v1.43:
//
//   EndEulerDiscretization          ql/processes/endeulerdiscretization.{hpp,cpp}
//   GeometricBrownianMotionProcess  ql/processes/geometricbrownianprocess.{hpp,cpp}
//   GarmanKohlagenProcess           ql/processes/blackscholesprocess.{hpp,cpp}
//   HullWhiteProcess                ql/processes/hullwhiteprocess.{hpp,cpp}
//   Merton76Process                 ql/processes/merton76process.{hpp,cpp}
//   MfStateProcess                  ql/processes/mfstateprocess.{hpp,cpp}
//   JointStochasticProcess          ql/processes/jointstochasticprocess.{hpp,cpp}
//   HybridHestonHullWhiteProcess    ql/processes/hybridhestonhullwhiteprocess.{hpp,cpp}
//   HestonSLVProcess                ql/processes/hestonslvprocess.{hpp,cpp}
//
// What is pinned and WHY:
//
//  * The FULL StochasticProcess surface for each process — size(), factors(),
//    initialValues(), drift(), diffusion(), expectation(), stdDeviation()
//    or covariance(), evolve() with a FIXED dw vector, and apply() — sampled
//    at several (t, x) points. Any single one of those in isolation lets a
//    sign error or a transposed matrix survive; the drift/diffusion pair plus
//    a fixed-dw evolve pins the composition too.
//
//  * EVERY constructor argument at a NON-DEFAULT value. An argument that is
//    accepted and then silently discarded is the defect class this port is
//    prone to, and it is invisible when every argument sits at its default.
//    Concretely:
//      - GeometricBrownianMotionProcess: initialValue/mue/sigma all distinct
//        and none of them 0 or 1.
//      - GarmanKohlagenProcess: foreign and domestic curves are DIFFERENT
//        non-flat curves, so swapping them changes the drift; forceDiscretization
//        is emitted at both false and true.
//      - HullWhiteProcess: a and sigma both non-default; the curve is non-flat
//        so alpha(t)'s f(t) term is observable.
//      - Merton76Process: jumpIntensity/logMeanJump/logJumpVolatility are three
//        adjacent same-typed Handle<Quote> arguments with three distinct values,
//        so a permutation is caught.
//      - MfStateProcess: reversion emitted at a non-zero value AND at zero (the
//        reversionZero_ arm), with a genuinely non-uniform vols vector.
//      - JointStochasticProcess: the `factors` argument is emitted at its
//        Null<Size>() default (-> modelFactors) AND at an explicit 2.
//      - HybridHestonHullWhiteProcess: corrEquityShortRate non-zero, and BOTH
//        Discretization values (Euler and BSMHullWhite) are pinned — they take
//        different branches of evolve().
//      - HestonSLVProcess: mixingFactor at 0.75, NOT its 1.0 default; the
//        leverage function is a genuine (t, strike) surface, so a port that
//        ignored either argument of localVol(t, x, true) diverges.
//
//  * NON-FLAT risk-free and dividend curves throughout. With flat curves the
//    instantaneous forward rate forwardRate(t, t, Continuous) and the
//    one-basis-point forward forwardRate(t, t+1e-4, Continuous) coincide
//    exactly, which hides which of the two a port actually calls. HestonProcess
//    (a dependency of both HybridHestonHullWhiteProcess and HestonSLVProcess)
//    uses the instantaneous form; GeneralizedBlackScholesProcess uses the
//    1e-4 form. Section "heston_dependency" pins HestonProcess::drift directly
//    so that difference is nailed down rather than assumed.
//
//  * EndEulerDiscretization is pinned side-by-side with EulerDiscretization on
//    processes whose drift AND diffusion genuinely depend on t (MfStateProcess,
//    HullWhiteProcess and a StochasticProcessArray of two MfStateProcesses).
//    On a time-homogeneous process the two discretizations are identical and
//    the test would prove nothing.
//
//  * JointStochasticProcess is abstract and has NO concrete subclass anywhere
//    in QuantLib v1.43, so the probe defines one (ProbeJointProcess below) and
//    the pytest module defines the identical subclass. preEvolve() records the
//    internal `dv` vector, which is the correlated Brownian increment the base
//    class feeds to its constituents; emitting it pins the SVD + rankReducedSqrt
//    machinery inside evolve() rather than only its end result. evolve() is
//    called TWICE with the same (t0, dt) on the non-state-dependent instance so
//    the correlationCache_ hit path is exercised.
//
//  * The evaluation date is pinned (Settings::instance().evaluationDate()); the
//    pytest module pins the identical date via an autouse fixture.
//
// Emits JSON on stdout.

#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/math/matrixutilities/pseudosqrt.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/endeulerdiscretization.hpp>
#include <ql/processes/eulerdiscretization.hpp>
#include <ql/processes/geometricbrownianprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/processes/hestonslvprocess.hpp>
#include <ql/processes/hullwhiteprocess.hpp>
#include <ql/processes/hybridhestonhullwhiteprocess.hpp>
#include <ql/processes/jointstochasticprocess.hpp>
#include <ql/processes/merton76process.hpp>
#include <ql/processes/mfstateprocess.hpp>
#include <ql/processes/stochasticprocessarray.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/fixedlocalvolsurface.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kToday(15, June, 2026);
const Actual365Fixed kA365;
const NullCalendar kNullCal;

// --- JSON helpers -----------------------------------------------------------

void key(const char* k) { std::cout << "\"" << k << "\": "; }

void jreal(const char* k, Real v, bool comma = true) {
    key(k);
    std::cout << v << (comma ? ",\n" : "\n");
}

void jsize(const char* k, Size v, bool comma = true) {
    key(k);
    std::cout << v << (comma ? ",\n" : "\n");
}

void jbool(const char* k, bool v, bool comma = true) {
    key(k);
    std::cout << (v ? "true" : "false") << (comma ? ",\n" : "\n");
}

void jarray(const char* k, const Array& a, bool comma = true) {
    key(k);
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i)
        std::cout << (i ? ", " : "") << a[i];
    std::cout << "]" << (comma ? ",\n" : "\n");
}

void jmatrix(const char* k, const Matrix& m, bool comma = true) {
    key(k);
    std::cout << "[";
    for (Size i = 0; i < m.rows(); ++i) {
        std::cout << (i ? ", [" : "[");
        for (Size j = 0; j < m.columns(); ++j)
            std::cout << (j ? ", " : "") << m[i][j];
        std::cout << "]";
    }
    std::cout << "]" << (comma ? ",\n" : "\n");
}

// --- market data ------------------------------------------------------------
//
// Deliberately NON-FLAT. Both curves are Linear-interpolated continuously
// compounded zero curves over the same date grid but with different levels and
// different slopes, so (a) forwardRate(t, t) != forwardRate(t, t + 1e-4), and
// (b) swapping the two curves changes every drift.

std::vector<Date> curveDates() {
    return {kToday, kToday + Period(6, Months), kToday + Period(1, Years),
            kToday + Period(3, Years), kToday + Period(7, Years),
            kToday + Period(15, Years)};
}

ext::shared_ptr<YieldTermStructure> domesticCurve() {
    const std::vector<Rate> zeros = {0.0180, 0.0225, 0.0265, 0.0310, 0.0345, 0.0362};
    return ext::make_shared<ZeroCurve>(curveDates(), zeros, kA365);
}

ext::shared_ptr<YieldTermStructure> foreignCurve() {
    const std::vector<Rate> zeros = {0.0090, 0.0105, 0.0135, 0.0118, 0.0142, 0.0175};
    return ext::make_shared<ZeroCurve>(curveDates(), zeros, kA365);
}

Handle<YieldTermStructure> domesticHandle() {
    return Handle<YieldTermStructure>(domesticCurve());
}
Handle<YieldTermStructure> foreignHandle() {
    return Handle<YieldTermStructure>(foreignCurve());
}

Handle<Quote> quote(Real v) { return Handle<Quote>(ext::make_shared<SimpleQuote>(v)); }

Handle<BlackVolTermStructure> blackVol(Volatility v) {
    return Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(kToday, kNullCal, v, kA365));
}

// The leverage surface for HestonSLVProcess: a genuine (time, strike) grid, so
// that a port which dropped either argument of localVol(t, x, true) — or
// transposed the matrix — diverges immediately.
ext::shared_ptr<LocalVolTermStructure> leverageSurface() {
    const std::vector<Time> times = {0.25, 0.50, 1.00, 2.00};
    const std::vector<Real> strikes = {80.0, 90.0, 100.0, 110.0, 125.0};
    auto m = ext::make_shared<Matrix>(strikes.size(), times.size());
    // Distinct in BOTH directions: no row and no column is constant.
    const Real vals[5][4] = {{1.35, 1.28, 1.19, 1.11}, {1.22, 1.16, 1.09, 1.04},
                             {1.05, 1.02, 1.00, 0.98}, {0.94, 0.93, 0.92, 0.915},
                             {0.86, 0.87, 0.885, 0.90}};
    for (Size i = 0; i < strikes.size(); ++i)
        for (Size j = 0; j < times.size(); ++j)
            (*m)[i][j] = vals[i][j];
    return ext::make_shared<FixedLocalVolSurface>(kToday, times, strikes, m, kA365);
}

// --- MfStateProcess fixtures ------------------------------------------------

Array arr(std::initializer_list<Real> v) { return Array(v.begin(), v.end()); }

// StochasticProcess1D declares the whole vector-form StochasticProcess
// interface (size / initialValues / drift(Array) / diffusion(Array) /
// expectation / stdDeviation / covariance / evolve(Array) / apply(Array))
// PRIVATE, so those overrides are only reachable through a base reference.
// That is itself worth recording: a 1-D process is a StochasticProcess, but
// only when you look at it as one.
const StochasticProcess& asND(const StochasticProcess1D& p) { return p; }

// Records WHICH QL_REQUIRE fires for an invalid construction. Several of these
// classes range-check in the constructor BODY while a member initialiser has
// already run and thrown its own message first; guessing which one wins is
// exactly the kind of assumption this harness exists to remove.
std::string failureOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return std::string(e.what());
    }
    return std::string("<no exception>");
}

void jstring(const char* k, const std::string& v, bool comma = true) {
    key(k);
    std::cout << "\"";
    for (char c : v) {
        if (c == '"' || c == '\\')
            std::cout << '\\';
        if (c == '\n')
            std::cout << "\\n";
        else
            std::cout << c;
    }
    std::cout << "\"" << (comma ? ",\n" : "\n");
}

ext::shared_ptr<MfStateProcess> mfA() {
    // reversion NON-ZERO; times strictly increasing; vols genuinely non-uniform.
    return ext::make_shared<MfStateProcess>(0.031, arr({0.5, 1.0, 2.0, 5.0}),
                                            arr({0.0050, 0.0075, 0.0090, 0.0110, 0.0130}));
}

ext::shared_ptr<MfStateProcess> mfB() {
    return ext::make_shared<MfStateProcess>(0.017, arr({0.75, 1.5, 3.0}),
                                            arr({0.0062, 0.0081, 0.0104, 0.0121}));
}

ext::shared_ptr<MfStateProcess> mfZero() {
    // reversionZero_ arm.
    return ext::make_shared<MfStateProcess>(0.0, arr({0.5, 1.0, 2.0, 5.0}),
                                            arr({0.0050, 0.0075, 0.0090, 0.0110, 0.0130}));
}

ext::shared_ptr<MfStateProcess> mfEmptyTimes(Real reversion) {
    // times_.empty() arm of variance().
    return ext::make_shared<MfStateProcess>(reversion, Array(), arr({0.0083}));
}

// --- concrete JointStochasticProcess ---------------------------------------
//
// JointStochasticProcess is abstract and QuantLib v1.43 ships no concrete
// subclass, so the probe supplies the minimal one. The pytest module defines a
// byte-for-byte equivalent so the base-class arithmetic is what is compared.

class ProbeJointProcess : public JointStochasticProcess {
  public:
    ProbeJointProcess(std::vector<ext::shared_ptr<StochasticProcess> > l,
                      Size factors,
                      bool stateDependent)
    : JointStochasticProcess(std::move(l), factors), stateDependent_(stateDependent) {}

    void preEvolve(Time, const Array&, Time, const Array& dv) const override {
        lastDv_ = dv;  // records the correlated increment the base class built
    }

    Array postEvolve(Time, const Array&, Time, const Array&, const Array& y0) const override {
        Array r = y0;
        r[0] += 0.125;  // deliberately observable, so a dropped postEvolve shows
        return r;
    }

    DiscountFactor numeraire(Time t, const Array& x) const override {
        return std::exp(-x[x.size() - 1] * t);
    }

    bool correlationIsStateDependent() const override { return stateDependent_; }

    Matrix crossModelCorrelation(Time t0, const Array& x0) const override {
        Matrix m(size(), size(), 0.0);
        // Zero inside each constituent's own block and on the diagonal: only the
        // cross-model entries are the joint process's business.
        const Real scale = stateDependent_ ? Real(std::tanh(x0[0] / 200.0) * 4.0) : Real(1.0);
        m[0][2] = m[2][0] = 0.35 * scale;
        m[1][2] = m[2][1] = -0.20 * scale;
        (void)t0;
        return m;
    }

    const Array& lastDv() const { return lastDv_; }

  private:
    const bool stateDependent_;
    mutable Array lastDv_;
};

std::vector<ext::shared_ptr<StochasticProcess> > jointConstituents() {
    // Constituent 0: a 2-D StochasticProcessArray of two GBMs (size 2, factors 2)
    // Constituent 1: HullWhiteProcess (size 1, factors 1)
    // -> size_ = 3, modelFactors_ = 3, and slice() is genuinely exercised.
    std::vector<ext::shared_ptr<StochasticProcess1D> > inner = {
        ext::make_shared<GeometricBrownianMotionProcess>(100.0, 0.055, 0.21),
        ext::make_shared<GeometricBrownianMotionProcess>(85.0, 0.032, 0.34)};
    Matrix corr(2, 2);
    corr[0][0] = corr[1][1] = 1.0;
    corr[0][1] = corr[1][0] = 0.25;
    return {ext::make_shared<StochasticProcessArray>(inner, corr),
            ext::make_shared<HullWhiteProcess>(domesticHandle(), 0.07, 0.013)};
}

void emitJoint(const char* name, Size factors, bool stateDependent) {
    ProbeJointProcess p(jointConstituents(), factors, stateDependent);

    const Array x0 = arr({102.5, 88.0, 0.0215});
    const Time t0 = 0.75;
    const Time dt = 0.5;

    key(name);
    std::cout << "{\n";
    jsize("size", p.size());
    jsize("factors", p.factors());
    jbool("correlation_is_state_dependent", p.correlationIsStateDependent());
    jarray("initial_values", p.initialValues());
    jarray("drift", p.drift(t0, x0));
    jmatrix("diffusion", p.diffusion(t0, x0));
    jarray("expectation", p.expectation(t0, x0, dt));
    jmatrix("covariance", p.covariance(t0, x0, dt));
    jmatrix("std_deviation", p.stdDeviation(t0, x0, dt));
    jmatrix("cross_model_correlation", p.crossModelCorrelation(t0, x0));
    jarray("apply", p.apply(x0, arr({0.031, -0.018, 0.0007})));
    jreal("numeraire", p.numeraire(t0, x0));
    // JointStochasticProcess::time delegates to l_[0]->time(date). Constituent
    // 0 here is a StochasticProcessArray of GBMs, and neither of those carries
    // a daycounter, so the delegation lands on the base StochasticProcess::time
    // which QL_FAILs. That IS the C++ behaviour and is pinned as such; the
    // successful delegation is pinned separately in "joint_time".
    jbool("time_raises", true);

    // dw has p.factors() entries.
    Array dw(p.factors());
    const Real dws[3] = {0.63, -1.21, 0.47};
    for (Size i = 0; i < dw.size(); ++i)
        dw[i] = dws[i];
    const Array e1 = p.evolve(t0, x0, dt, dw);
    const Array dv1 = p.lastDv();

    // Same (t0, dt), different dw: on the non-state-dependent instance this is
    // the correlationCache_ HIT path.
    Array dw2(p.factors());
    const Real dws2[3] = {-0.29, 0.88, -1.04};
    for (Size i = 0; i < dw2.size(); ++i)
        dw2[i] = dws2[i];
    const Array e2 = p.evolve(t0, x0, dt, dw2);
    const Array dv2 = p.lastDv();

    jarray("dw1", dw);
    jarray("pre_evolve_dv1", dv1);
    jarray("evolve1", e1);
    jarray("dw2", dw2);
    jarray("pre_evolve_dv2", dv2);
    jarray("evolve2", e2, false);
    std::cout << "},\n";
}

// --- Euler / EndEuler side-by-side -----------------------------------------

void emit1dDiscretizations(const char* name, const StochasticProcess1D& p, Time t0, Real x0,
                           Time dt) {
    EulerDiscretization euler;
    EndEulerDiscretization endEuler;
    key(name);
    std::cout << "{\n";
    jreal("t0", t0);
    jreal("x0", x0);
    jreal("dt", dt);
    jreal("euler_drift", euler.drift(p, t0, x0, dt));
    jreal("euler_diffusion", euler.diffusion(p, t0, x0, dt));
    jreal("euler_variance", euler.variance(p, t0, x0, dt));
    jreal("end_euler_drift", endEuler.drift(p, t0, x0, dt));
    jreal("end_euler_diffusion", endEuler.diffusion(p, t0, x0, dt));
    jreal("end_euler_variance", endEuler.variance(p, t0, x0, dt), false);
    std::cout << "},\n";
}

void emitNdDiscretizations(const char* name, const StochasticProcess& p, Time t0, const Array& x0,
                           Time dt) {
    EulerDiscretization euler;
    EndEulerDiscretization endEuler;
    key(name);
    std::cout << "{\n";
    jreal("t0", t0);
    jarray("x0", x0);
    jreal("dt", dt);
    jarray("euler_drift", euler.drift(p, t0, x0, dt));
    jmatrix("euler_diffusion", euler.diffusion(p, t0, x0, dt));
    jmatrix("euler_covariance", euler.covariance(p, t0, x0, dt));
    jarray("end_euler_drift", endEuler.drift(p, t0, x0, dt));
    jmatrix("end_euler_diffusion", endEuler.diffusion(p, t0, x0, dt));
    jmatrix("end_euler_covariance", endEuler.covariance(p, t0, x0, dt), false);
    std::cout << "},\n";
}

} // namespace

int main() {
    try {
        Settings::instance().evaluationDate() = kToday;
        std::cout << std::setprecision(17);
        std::cout << "{\n";
        jsize("today_serial", kToday.serialNumber());

        // =====================================================================
        // GeometricBrownianMotionProcess
        // =====================================================================
        {
            // All three ctor arguments non-default and mutually distinct.
            GeometricBrownianMotionProcess p(95.5, 0.073, 0.231);
            key("gbm");
            std::cout << "{\n";
            jreal("init_initial_value", 95.5);
            jreal("init_mue", 0.073);
            jreal("init_sigma", 0.231);
            jsize("size", asND(p).size());
            jsize("factors", p.factors());
            jreal("x0", p.x0());
            jarray("initial_values", asND(p).initialValues());
            jreal("drift_t0_x95", p.drift(0.0, 95.5));
            jreal("drift_t2_x120", p.drift(2.0, 120.0));
            jreal("drift_t2_xneg", p.drift(2.0, -30.0));
            jreal("diffusion_t0_x95", p.diffusion(0.0, 95.5));
            jreal("diffusion_t2_x120", p.diffusion(2.0, 120.0));
            jarray("drift_vec", asND(p).drift(1.5, arr({110.0})));
            jmatrix("diffusion_mat", asND(p).diffusion(1.5, arr({110.0})));
            jreal("expectation", p.expectation(1.5, 110.0, 0.25));
            jreal("std_deviation", p.stdDeviation(1.5, 110.0, 0.25));
            jreal("variance", p.variance(1.5, 110.0, 0.25));
            jreal("apply", p.apply(110.0, 3.75));
            jreal("evolve", p.evolve(1.5, 110.0, 0.25, 0.83));
            jarray("evolve_vec", asND(p).evolve(1.5, arr({110.0}), 0.25, arr({0.83})));
            jmatrix("covariance_mat", asND(p).covariance(1.5, arr({110.0}), 0.25), false);
            std::cout << "},\n";
        }

        // =====================================================================
        // GarmanKohlagenProcess
        // =====================================================================
        {
            // foreign != domestic, both non-flat -> a swap of the two arguments
            // moves every drift.
            const Handle<Quote> spot = quote(1.2345);
            const Handle<BlackVolTermStructure> vol = blackVol(0.1725);

            GarmanKohlagenProcess p(spot, foreignHandle(), domesticHandle(), vol);
            GarmanKohlagenProcess pForced(
                spot, foreignHandle(), domesticHandle(), vol,
                ext::shared_ptr<StochasticProcess1D::discretization>(new EulerDiscretization),
                true);

            key("garman_kohlagen");
            std::cout << "{\n";
            jreal("init_spot", 1.2345);
            jreal("init_vol", 0.1725);
            jsize("size", asND(p).size());
            jsize("factors", p.factors());
            jreal("x0", p.x0());
            // These two prove the argument mapping: C++ forwards
            // foreignRiskFreeTS -> dividendYield and domesticRiskFreeTS -> riskFreeRate.
            jreal("dividend_yield_discount_2y", p.dividendYield()->discount(2.0));
            jreal("risk_free_rate_discount_2y", p.riskFreeRate()->discount(2.0));
            jreal("state_variable_value", p.stateVariable()->value());
            jreal("black_vol_1y_atm", p.blackVolatility()->blackVol(1.0, 1.2345, true));
            jreal("local_vol_1y_atm", p.localVolatility()->localVol(1.0, 1.2345, true));
            jreal("drift_t0", p.drift(0.0, 1.2345));
            jreal("drift_t1_5", p.drift(1.5, 1.31));
            jreal("diffusion_t1_5", p.diffusion(1.5, 1.31));
            jreal("expectation", p.expectation(1.5, 1.31, 0.5));
            jreal("std_deviation", p.stdDeviation(1.5, 1.31, 0.5));
            jreal("variance", p.variance(1.5, 1.31, 0.5));
            jreal("apply", p.apply(1.31, 0.042));
            jreal("evolve", p.evolve(1.5, 1.31, 0.5, -0.62));
            jreal("time_1y", p.time(kToday + 365));
            // forceDiscretization=true takes the Euler branch of evolve/variance.
            jreal("forced_variance", pForced.variance(1.5, 1.31, 0.5));
            jreal("forced_std_deviation", pForced.stdDeviation(1.5, 1.31, 0.5));
            jreal("forced_evolve", pForced.evolve(1.5, 1.31, 0.5, -0.62), false);
            std::cout << "},\n";
        }

        // =====================================================================
        // HullWhiteProcess
        // =====================================================================
        {
            HullWhiteProcess p(domesticHandle(), 0.07, 0.013);
            key("hull_white");
            std::cout << "{\n";
            jreal("init_a", 0.07);
            jreal("init_sigma", 0.013);
            jreal("a", p.a());
            jreal("sigma", p.sigma());
            jsize("size", asND(p).size());
            jsize("factors", p.factors());
            jreal("x0", p.x0());
            jarray("initial_values", asND(p).initialValues());
            jreal("alpha_t0", p.alpha(0.0));
            jreal("alpha_t1", p.alpha(1.0));
            jreal("alpha_t5", p.alpha(5.0));
            jreal("drift_t0", p.drift(0.0, 0.018));
            jreal("drift_t1_5", p.drift(1.5, 0.0245));
            jreal("drift_t4", p.drift(4.0, 0.031));
            jreal("diffusion_t1_5", p.diffusion(1.5, 0.0245));
            jreal("expectation", p.expectation(1.5, 0.0245, 0.5));
            jreal("std_deviation", p.stdDeviation(1.5, 0.0245, 0.5));
            jreal("variance", p.variance(1.5, 0.0245, 0.5));
            jreal("apply", p.apply(0.0245, 0.0031));
            jreal("evolve", p.evolve(1.5, 0.0245, 0.5, 0.71));
            jarray("drift_vec", asND(p).drift(1.5, arr({0.0245})));
            jmatrix("diffusion_mat", asND(p).diffusion(1.5, arr({0.0245})));
            jmatrix("covariance_mat", asND(p).covariance(1.5, arr({0.0245}), 0.5), false);
            std::cout << "},\n";
        }

        // =====================================================================
        // Merton76Process
        // =====================================================================
        {
            // The three Handle<Quote> jump arguments are adjacent and same-typed:
            // three distinct values catch any permutation.
            Merton76Process p(quote(97.25), foreignHandle(), domesticHandle(), blackVol(0.2130),
                              quote(1.37), quote(-0.041), quote(0.0925));
            key("merton76");
            std::cout << "{\n";
            jsize("size", asND(p).size());
            jsize("factors", p.factors());
            jreal("x0", p.x0());
            jarray("initial_values", asND(p).initialValues());
            jreal("state_variable", p.stateVariable()->value());
            jreal("jump_intensity", p.jumpIntensity()->value());
            jreal("log_mean_jump", p.logMeanJump()->value());
            jreal("log_jump_volatility", p.logJumpVolatility()->value());
            // dividendYield <- the 2nd ctor argument (foreign), riskFreeRate <- 3rd.
            jreal("dividend_yield_discount_2y", p.dividendYield()->discount(2.0));
            jreal("risk_free_rate_discount_2y", p.riskFreeRate()->discount(2.0));
            jreal("black_vol_1y", p.blackVolatility()->blackVol(1.0, 97.25, true));
            jreal("time_1y", p.time(kToday + 365));
            jreal("time_3y", p.time(kToday + 1095), false);
            std::cout << "},\n";
        }

        // =====================================================================
        // MfStateProcess
        // =====================================================================
        {
            auto a = mfA();
            auto z = mfZero();
            auto e0 = mfEmptyTimes(0.0);
            auto en = mfEmptyTimes(0.029);

            key("mf_state");
            std::cout << "{\n";
            jsize("size", asND(*a).size());
            jsize("factors", a->factors());
            jreal("x0", a->x0());
            jarray("initial_values", asND(*a).initialValues());
            jreal("drift", a->drift(1.25, 0.03));

            // diffusion is piecewise constant in t; sample inside every bucket
            // AND exactly ON the nodes (upper_bound is strict, so t == node
            // still selects the bucket ABOVE it).
            const Time ts[9] = {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 6.0};
            for (int i = 0; i < 9; ++i) {
                const std::string k = "diffusion_t" + std::to_string(i);
                key(k.c_str());
                std::cout << a->diffusion(ts[i], 0.02) << ",\n";
            }

            jreal("expectation", a->expectation(1.25, 0.037, 0.5));
            jreal("expectation_other_t", a->expectation(3.0, 0.037, 1.5));

            // variance: spanning zero, one and several buckets; and dt below
            // QL_EPSILON (the early-return arm).
            jreal("variance_within_bucket", a->variance(1.25, 0.02, 0.2));
            jreal("variance_one_boundary", a->variance(0.75, 0.02, 0.5));
            jreal("variance_many_buckets", a->variance(0.25, 0.02, 5.0));
            jreal("variance_on_node", a->variance(1.0, 0.02, 1.0));
            jreal("variance_tiny_dt", a->variance(1.25, 0.02, 1e-17));
            jreal("std_deviation", a->stdDeviation(1.25, 0.02, 0.2));
            jreal("evolve", a->evolve(1.25, 0.037, 0.5, -0.43));
            jreal("apply", a->apply(0.037, 0.0021));

            // reversion == 0 arm
            jreal("zero_variance_within_bucket", z->variance(1.25, 0.02, 0.2));
            jreal("zero_variance_many_buckets", z->variance(0.25, 0.02, 5.0));
            jreal("zero_std_deviation", z->stdDeviation(1.25, 0.02, 0.2));
            jreal("zero_diffusion_t1_5", z->diffusion(1.5, 0.02));

            // times_.empty() arm, both reversion branches
            jreal("empty_zero_variance", e0->variance(1.25, 0.02, 0.5));
            jreal("empty_nonzero_variance", en->variance(1.25, 0.02, 0.5));
            jreal("empty_diffusion", e0->diffusion(3.3, 0.02));
            jreal("empty_nonzero_std_deviation", en->stdDeviation(1.25, 0.02, 0.5), false);
            std::cout << "},\n";
        }

        // =====================================================================
        // EndEulerDiscretization (side by side with EulerDiscretization)
        // =====================================================================
        {
            // MfStateProcess: diffusion is genuinely t-dependent, and dt is
            // chosen so t0 and t0+dt land in DIFFERENT vol buckets.
            emit1dDiscretizations("end_euler_mf", *mfA(), 0.75, 0.021, 0.5);
            // HullWhiteProcess: drift is t-dependent, diffusion is not.
            HullWhiteProcess hw(domesticHandle(), 0.07, 0.013);
            emit1dDiscretizations("end_euler_hw", hw, 1.5, 0.0245, 0.75);

            // Multi-D: two correlated MfStateProcesses.
            std::vector<ext::shared_ptr<StochasticProcess1D> > inner = {mfA(), mfB()};
            Matrix corr(2, 2);
            corr[0][0] = corr[1][1] = 1.0;
            corr[0][1] = corr[1][0] = -0.35;
            StochasticProcessArray spa(inner, corr);
            emitNdDiscretizations("end_euler_array", spa, 0.75, arr({0.021, 0.014}), 0.5);
        }

        // =====================================================================
        // HestonProcess dependency check
        //
        // HybridHestonHullWhiteProcess::drift and HestonSLVProcess::drift both
        // read forward rates. HestonProcess uses the INSTANTANEOUS forward
        // forwardRate(t, t, Continuous); GeneralizedBlackScholesProcess uses
        // forwardRate(t, t + 1e-4, Continuous). Both are emitted here on the
        // same non-flat curve so the difference is a fact in the reference file
        // rather than an assumption.
        // =====================================================================
        {
            auto heston = ext::make_shared<HestonProcess>(domesticHandle(), foreignHandle(),
                                                          quote(102.5), 0.0625, 1.35, 0.0475,
                                                          0.62, -0.40);
            key("heston_dependency");
            std::cout << "{\n";
            jsize("size", heston->size());
            jsize("factors", heston->factors());
            jarray("initial_values", heston->initialValues());
            jarray("drift_t0", heston->drift(0.0, arr({102.5, 0.0625})));
            jarray("drift_t1_5", heston->drift(1.5, arr({108.0, 0.0512})));
            jarray("drift_t4", heston->drift(4.0, arr({95.0, 0.0710})));
            jarray("drift_negative_v", heston->drift(1.5, arr({108.0, -0.002})));
            jmatrix("diffusion_t1_5", heston->diffusion(1.5, arr({108.0, 0.0512})));
            jmatrix("diffusion_zero_v", heston->diffusion(1.5, arr({108.0, 0.0})));
            jarray("apply", heston->apply(arr({108.0, 0.0512}), arr({0.031, -0.0044})));
            jreal("time_1y", heston->time(kToday + 365));
            // The two forward-rate conventions, spelled out.
            jreal("rf_instantaneous_fwd_t1_5",
                  domesticCurve()->forwardRate(1.5, 1.5, Continuous).rate());
            jreal("rf_1e4_fwd_t1_5",
                  domesticCurve()->forwardRate(1.5, 1.5001, Continuous, NoFrequency, true).rate());
            jreal("div_instantaneous_fwd_t1_5",
                  foreignCurve()->forwardRate(1.5, 1.5, Continuous).rate());
            jreal("div_1e4_fwd_t1_5",
                  foreignCurve()->forwardRate(1.5, 1.5001, Continuous, NoFrequency, true).rate(),
                  false);
            std::cout << "},\n";
        }

        // =====================================================================
        // HybridHestonHullWhiteProcess — BOTH Discretization values
        // =====================================================================
        {
            auto heston = ext::make_shared<HestonProcess>(domesticHandle(), foreignHandle(),
                                                          quote(102.5), 0.0625, 1.35, 0.0475,
                                                          0.62, -0.40);
            auto hwf = ext::make_shared<HullWhiteForwardProcess>(domesticHandle(), 0.07, 0.013);
            hwf->setForwardMeasureTime(3.0);

            const Real corrEquityShortRate = -0.35;  // non-zero
            const Array x0 = arr({102.5, 0.0625, 0.0215});
            const Array dw = arr({0.63, -1.21, 0.47});
            const Time t0 = 0.75;
            const Time dt = 0.5;

            const char* names[2] = {"hhw_bsm", "hhw_euler"};
            HybridHestonHullWhiteProcess::Discretization discs[2] = {
                HybridHestonHullWhiteProcess::BSMHullWhite,
                HybridHestonHullWhiteProcess::Euler};

            for (int k = 0; k < 2; ++k) {
                HybridHestonHullWhiteProcess p(heston, hwf, corrEquityShortRate, discs[k]);
                key(names[k]);
                std::cout << "{\n";
                jreal("init_corr_equity_short_rate", corrEquityShortRate);
                jreal("forward_measure_time", hwf->getForwardMeasureTime());
                jsize("discretization", static_cast<Size>(p.discretization()));
                jsize("size", p.size());
                jsize("factors", p.factors());
                jreal("eta", p.eta());
                jarray("initial_values", p.initialValues());
                jarray("drift", p.drift(t0, x0));
                jmatrix("diffusion", p.diffusion(t0, x0));
                jarray("apply", p.apply(x0, arr({0.031, -0.0044, 0.0007})));
                jarray("evolve", p.evolve(t0, x0, dt, dw));
                jarray("evolve_other", p.evolve(1.75, arr({97.0, 0.0410, 0.0288}), 0.25,
                                                arr({-0.44, 0.91, -0.13})));
                jreal("numeraire_t0", p.numeraire(t0, x0));
                jreal("numeraire_t2", p.numeraire(2.0, arr({95.0, 0.05, 0.0310})));
                jreal("numeraire_t3", p.numeraire(3.0, arr({95.0, 0.05, 0.0310})));
                jreal("time_1y", p.time(kToday + 365), false);
                std::cout << "},\n";
            }
        }

        // =====================================================================
        // HestonSLVProcess — mixingFactor at a NON-default 0.75
        // =====================================================================
        {
            auto heston = ext::make_shared<HestonProcess>(domesticHandle(), foreignHandle(),
                                                          quote(102.5), 0.0625, 1.35, 0.0475,
                                                          0.62, -0.40);
            auto lev = leverageSurface();
            HestonSLVProcess p(heston, lev, 0.75);
            HestonSLVProcess pDefault(heston, lev);  // mixingFactor == 1.0

            key("heston_slv");
            std::cout << "{\n";
            jreal("init_mixing_factor", 0.75);
            jsize("size", p.size());
            jsize("factors", p.factors());
            jreal("v0", p.v0());
            jreal("rho", p.rho());
            jreal("kappa", p.kappa());
            jreal("theta", p.theta());
            jreal("sigma", p.sigma());
            jreal("mixing_factor", p.mixingFactor());
            jreal("default_mixing_factor", pDefault.mixingFactor());
            jreal("s0", p.s0()->value());
            jreal("dividend_yield_discount_2y", p.dividendYield()->discount(2.0));
            jreal("risk_free_rate_discount_2y", p.riskFreeRate()->discount(2.0));
            jreal("leverage_t0_5_k100", p.leverageFct()->localVol(0.5, 100.0, true));
            jreal("leverage_t1_25_k95", p.leverageFct()->localVol(1.25, 95.0, true));
            jarray("initial_values", p.initialValues());
            jarray("apply", p.apply(arr({108.0, 0.0512}), arr({0.031, -0.0044})));
            jarray("drift_t0_5", p.drift(0.5, arr({100.0, 0.0625})));
            jarray("drift_t1_25", p.drift(1.25, arr({95.0, 0.0410})));
            jmatrix("diffusion_t0_5", p.diffusion(0.5, arr({100.0, 0.0625})));
            jmatrix("diffusion_t1_25", p.diffusion(1.25, arr({95.0, 0.0410})));
            jreal("time_1y", p.time(kToday + 365));
            // mixingFactor must move drift/diffusion/evolve: emit the default too.
            jmatrix("default_diffusion_t0_5", pDefault.diffusion(0.5, arr({100.0, 0.0625})));

            // evolve() has three arithmetic outcomes; all three are pinned, and
            // the psi that selects between them is emitted alongside so the test
            // can assert which branch it is exercising rather than assume.
            //   psi  = s2/(m*m) with
            //   ex   = exp(-kappa*dt)
            //   m    = theta + (v-theta)*ex
            //   s2   = v*mixedSigma^2*ex/kappa*(1-ex)
            //          + theta*mixedSigma^2/(2*kappa)*(1-ex)^2
            const Real mixedSigma = 0.75 * 0.62;
            const Real kappa = 1.35, theta = 0.0475;
            auto psiOf = [&](Real v, Time dt) {
                const Real ex = std::exp(-kappa * dt);
                const Real m = theta + (v - theta) * ex;
                const Real s2 = v * mixedSigma * mixedSigma * ex / kappa * (1 - ex) +
                                theta * mixedSigma * mixedSigma / (2 * kappa) * (1 - ex) * (1 - ex);
                return s2 / (m * m);
            };
            // psi < 1.5 branch (large v, small dt)
            jreal("psi_low", psiOf(0.25, 0.05));
            jarray("evolve_psi_low", p.evolve(0.5, arr({100.0, 0.25}), 0.05, arr({0.63, -0.42})));
            // psi >= 1.5 branch with u > p  (exponential tail)
            jreal("psi_high", psiOf(0.001, 3.0));
            jarray("evolve_psi_high", p.evolve(0.5, arr({100.0, 0.001}), 3.0, arr({0.63, 0.85})));
            // psi >= 1.5 branch with u <= p  (variance pinned to exactly 0.0)
            jarray("evolve_psi_high_zero",
                   p.evolve(0.5, arr({100.0, 0.001}), 3.0, arr({0.63, -1.85})));
            jarray("default_evolve_psi_low",
                   pDefault.evolve(0.5, arr({100.0, 0.25}), 0.05, arr({0.63, -0.42})));
            jarray("default_evolve_psi_high",
                   pDefault.evolve(0.5, arr({100.0, 0.001}), 3.0, arr({0.63, 0.85})), false);
            std::cout << "},\n";
        }

        // =====================================================================
        // JointStochasticProcess (via the probe's concrete subclass)
        // =====================================================================
        emitJoint("joint_default_factors", Null<Size>(), false);
        emitJoint("joint_two_factors", 2, false);
        emitJoint("joint_state_dependent", Null<Size>(), true);

        // JointStochasticProcess::time -> l_[0]->time(date), pinned on a
        // constituent list whose head DOES carry a daycounter.
        {
            std::vector<ext::shared_ptr<StochasticProcess> > l = {
                ext::make_shared<GarmanKohlagenProcess>(quote(1.2345), foreignHandle(),
                                                        domesticHandle(), blackVol(0.1725)),
                ext::make_shared<HullWhiteProcess>(domesticHandle(), 0.07, 0.013)};
            ProbeJointProcess p(l, Null<Size>(), false);
            key("joint_time");
            std::cout << "{\n";
            jsize("size", p.size());
            jsize("factors", p.factors());
            jreal("time_1y", p.time(kToday + 365));
            jreal("time_3y", p.time(kToday + 1095), false);
            std::cout << "},\n";
        }

        // =====================================================================
        // StochasticProcessArray + the pseudoSqrt(Spectral) it is built on
        //
        // StochasticProcessArray::{diffusion,stdDeviation,evolve} expose the
        // pseudo-root ITSELF, not just ``M M^T``, so the column ORDER and the
        // per-column SIGN of pseudoSqrt(corr, Spectral) are observable. The
        // identity case is included on purpose: its eigenvalues are degenerate,
        // and SymmetricSchurDecomposition breaks that tie by sorting the
        // (eigenvalue, eigenvector) PAIRS with std::greater<>, i.e.
        // lexicographically on the eigenvector — which a naive
        // "reverse the ascending eigensolver output" does not reproduce.
        // =====================================================================
        {
            key("spa");
            std::cout << "{\n";
            Matrix identity(2, 2, 0.0);
            identity[0][0] = identity[1][1] = 1.0;
            Matrix pos(2, 2);
            pos[0][0] = pos[1][1] = 1.0;
            pos[0][1] = pos[1][0] = 0.25;
            Matrix neg(2, 2);
            neg[0][0] = neg[1][1] = 1.0;
            neg[0][1] = neg[1][0] = -0.35;
            Matrix identity3(3, 3, 0.0);
            for (Size i = 0; i < 3; ++i)
                identity3[i][i] = 1.0;

            jmatrix("pseudo_sqrt_identity2", pseudoSqrt(identity, SalvagingAlgorithm::Spectral));
            jmatrix("pseudo_sqrt_identity3", pseudoSqrt(identity3, SalvagingAlgorithm::Spectral));
            jmatrix("pseudo_sqrt_pos", pseudoSqrt(pos, SalvagingAlgorithm::Spectral));
            jmatrix("pseudo_sqrt_neg", pseudoSqrt(neg, SalvagingAlgorithm::Spectral));
            jmatrix("rank_reduced_sqrt_identity2",
                    rankReducedSqrt(identity, 2, 1.0, SalvagingAlgorithm::Spectral));

            // The array under an IDENTITY correlation: evolve() must reduce to
            // the component-wise 1-D evolve, which is only true if the
            // pseudo-root is the identity (not a permutation of it).
            {
                std::vector<ext::shared_ptr<StochasticProcess1D> > inner = {
                    ext::make_shared<GeometricBrownianMotionProcess>(100.0, 0.055, 0.21),
                    ext::make_shared<GeometricBrownianMotionProcess>(85.0, 0.032, 0.34)};
                StochasticProcessArray spaI(inner, identity);
                StochasticProcessArray spaP(inner, pos);
                StochasticProcessArray spaN(inner, neg);
                const Array x0 = arr({102.0, 88.0});
                const Array dw = arr({0.5, -0.3});
                jmatrix("identity_correlation", spaI.correlation());
                jmatrix("identity_diffusion", spaI.diffusion(0.75, x0));
                jmatrix("identity_std_deviation", spaI.stdDeviation(0.75, x0, 0.5));
                jarray("identity_evolve", spaI.evolve(0.75, x0, 0.5, dw));
                jreal("identity_evolve_component0",
                      inner[0]->evolve(0.75, x0[0], 0.5, dw[0]));
                jreal("identity_evolve_component1",
                      inner[1]->evolve(0.75, x0[1], 0.5, dw[1]));
                jmatrix("pos_diffusion", spaP.diffusion(0.75, x0));
                jmatrix("pos_std_deviation", spaP.stdDeviation(0.75, x0, 0.5));
                jarray("pos_evolve", spaP.evolve(0.75, x0, 0.5, dw));
                jmatrix("neg_diffusion", spaN.diffusion(0.75, x0));
                jmatrix("neg_covariance", spaN.covariance(0.75, x0, 0.5));
                jarray("neg_evolve", spaN.evolve(0.75, x0, 0.5, dw), false);
            }
            std::cout << "},\n";
        }

        // =====================================================================
        // Which QL_REQUIRE actually fires on invalid construction
        // =====================================================================
        {
            key("errors");
            std::cout << "{\n";
            jstring("hw_negative_a", failureOf([] {
                        HullWhiteProcess p(domesticHandle(), -0.01, 0.013);
                    }));
            // NOTE: the OrnsteinUhlenbeckProcess member initialiser runs BEFORE
            // the constructor body, and it checks the volatility itself, so the
            // "negative sigma given" QL_REQUIRE in HullWhiteProcess's body is
            // unreachable for a negative sigma.
            jstring("hw_negative_sigma", failureOf([] {
                        HullWhiteProcess p(domesticHandle(), 0.07, -0.013);
                    }));
            // HullWhiteForwardProcess has NO QL_REQUIREs of its own, but its
            // OrnsteinUhlenbeckProcess member initialiser still rejects a
            // negative volatility — so "no checks" does not mean "accepts
            // anything".
            jstring("hwf_negative_a", failureOf([] {
                        HullWhiteForwardProcess p(domesticHandle(), -0.01, 0.013);
                    }));
            jstring("hwf_negative_sigma", failureOf([] {
                        HullWhiteForwardProcess p(domesticHandle(), 0.07, -0.013);
                    }));
            jstring("mf_bad_vol_count", failureOf([] {
                        MfStateProcess p(0.03, arr({0.5, 1.0}), arr({0.01, 0.01}));
                    }));
            jstring("mf_times_not_increasing", failureOf([] {
                        MfStateProcess p(0.03, arr({1.0, 0.5}), arr({0.01, 0.01, 0.01}));
                    }));
            jstring("mf_negative_vol", failureOf([] {
                        MfStateProcess p(0.03, arr({0.5, 1.0}), arr({0.01, -0.01, 0.01}));
                    }));
            jstring("mf_empty_vols", failureOf([] {
                        MfStateProcess p(0.03, Array(), Array());
                    }));
            jstring("joint_too_many_factors", failureOf([] {
                        ProbeJointProcess p(jointConstituents(), 4, false);
                    }));
            {
                auto heston = ext::make_shared<HestonProcess>(
                    domesticHandle(), foreignHandle(), quote(102.5), 0.0625, 1.35, 0.0475,
                    0.62, -0.40);
                auto hwf =
                    ext::make_shared<HullWhiteForwardProcess>(domesticHandle(), 0.07, 0.013);
                hwf->setForwardMeasureTime(3.0);
                jstring("hhw_indefinite_correlation", failureOf([&] {
                            HybridHestonHullWhiteProcess p(heston, hwf, -0.95);
                        }));
                auto hwf0 =
                    ext::make_shared<HullWhiteForwardProcess>(domesticHandle(), 0.07, 0.0);
                hwf0->setForwardMeasureTime(3.0);
                // Likewise: the HullWhite MODEL member initialiser rejects a
                // zero sigma before the body's "positive vol" QL_REQUIRE runs.
                jstring("hhw_zero_hw_sigma", failureOf([&] {
                            HybridHestonHullWhiteProcess p(heston, hwf0, -0.35);
                        }),
                        false);
            }
            std::cout << "},\n";
        }

        std::cout << "  \"end\": true\n}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
}
