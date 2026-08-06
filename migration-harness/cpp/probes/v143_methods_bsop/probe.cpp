// v1.43 methods coverage: FdmBlackScholesOp's localVol and quanto branches.
//
// ql/methods/finitedifferences/operators/fdmblackscholesop.{hpp,cpp}
// @ v1.43 (6b57206e0).
//
// setTime() has four arms — {constant vol, local vol} x {no quanto, quanto} —
// and the pquantlib port only had the first. This probe pins all four, on the
// same mesh and the same market, as:
//
//   * the operator itself, read out through apply() on three interleaved
//     indicator vectors, which between them determine every band of a
//     tridiagonal operator without needing access to its private diagonals,
//     and
//   * a full Fdm1DimSolver rollback, so the branches are pinned end to end
//     rather than only at the operator level.
//
// The strike-dependent local-vol cases are driven by an EXTERNAL
// LocalVolTermStructure (the RampLocalVol below, sigma(t,S) = 0.15 + 0.05
// log(S/100) + 0.02 t), handed to GeneralizedBlackScholesProcess through the
// constructor that takes one. That is deliberate: it keeps this probe from
// depending on Dupire, so what it pins is FdmBlackScholesOp's own arithmetic
// — the exp(locations) spot grid, the per-node variance loop, the per-node
// axpyb, and the vector overload of quantoAdjustment — rather than the
// quality of a local-vol surface construction.
//
// The flat_lv case does exercise the derived path: with a BlackConstantVol,
// GeneralizedBlackScholesProcess::localVolatility() returns a LocalConstantVol
// and the local-vol arm must reproduce the constant-vol arm.

#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdm2dblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdm1dimsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmsolverdesc.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/localvoltermstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

void emit(const std::string& name, Real v) {
    std::cout << "  \"" << name << "\": " << v << ",\n";
}

const Date TODAY(15, January, 2024);
const DayCounter DC = Actual365Fixed();

Handle<YieldTermStructure> flat(Rate r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(TODAY, r, DC));
}

// A closed-form, strike- AND time-dependent local vol, so both sides of the
// cross-validation compute exactly the same function and what is compared is
// FdmBlackScholesOp, not a surface construction.
class RampLocalVol : public LocalVolTermStructure {
  public:
    RampLocalVol() : LocalVolTermStructure(TODAY, NullCalendar(), Following, DC) {}
    Date maxDate() const override { return Date::maxDate(); }
    Real minStrike() const override { return 0.0; }
    Real maxStrike() const override { return QL_MAX_REAL; }

  protected:
    Volatility localVolImpl(Time t, Real s) const override {
        return 0.15 + 0.05 * std::log(s / 100.0) + 0.02 * t;
    }
};

ext::shared_ptr<FdmStepConditionComposite> emptyConditions() {
    FdmStepConditionComposite::Conditions stepConds;
    std::list<std::vector<Time> > stoppingTimes;
    return ext::make_shared<FdmStepConditionComposite>(stoppingTimes, stepConds);
}

ext::shared_ptr<FdmQuantoHelper> quantoHelper() {
    // domestic 5%, foreign 3%, fx vol 12%, correlation 0.4, ATM fx 1.25
    return ext::make_shared<FdmQuantoHelper>(
        *flat(0.05), *flat(0.03),
        *Handle<BlackVolTermStructure>(
            ext::make_shared<BlackConstantVol>(TODAY, NullCalendar(), 0.12, DC)),
        0.4, 1.25);
}

// apply() on three interleaved indicator vectors determines every band.
void emitApply(const std::string& p, const FdmBlackScholesOp& op, Size n) {
    for (Size k = 0; k < 3; ++k) {
        Array e(n, 0.0);
        for (Size i = k; i < n; i += 3)
            e[i] = 1.0;
        const Array y = op.apply(e);
        emit(p + "apply" + std::to_string(k) + "_0", y[0]);
        emit(p + "apply" + std::to_string(k) + "_1", y[1]);
        emit(p + "apply" + std::to_string(k) + "_mid", y[n / 2]);
        emit(p + "apply" + std::to_string(k) + "_last", y[n - 1]);
    }
}

struct Case {
    const char* name;
    bool localVol;   // ask the op for the process's localVolatility()
    bool external;   // ... which is the RampLocalVol rather than the derived one
    bool quanto;
};

void block(const Case& c) {
    const Real strike = 100.0;
    const Size n = 25;
    auto m1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), n);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    const Handle<Quote> s0(ext::make_shared<SimpleQuote>(100.0));
    const Handle<BlackVolTermStructure> vol(
        ext::make_shared<BlackConstantVol>(TODAY, NullCalendar(), 0.20, DC));

    ext::shared_ptr<GeneralizedBlackScholesProcess> process;
    if (c.external) {
        process = ext::make_shared<GeneralizedBlackScholesProcess>(
            s0, flat(0.02), flat(0.05), vol,
            Handle<LocalVolTermStructure>(ext::make_shared<RampLocalVol>()));
    } else {
        process = ext::make_shared<BlackScholesMertonProcess>(
            s0, flat(0.02), flat(0.05), vol);
    }

    const std::string p = std::string(c.name) + "_";
    const auto qh = c.quanto ? quantoHelper() : ext::shared_ptr<FdmQuantoHelper>();

    {
        FdmBlackScholesOp op(mesher, process, strike, c.localVol, -Null<Real>(), 0, qh);
        op.setTime(0.0, 0.25);
        emitApply(p, op, n);
        // a second setTime, to pin that the coefficients are recomputed (and,
        // for the ramp, that they move with the mid-point time)
        op.setTime(0.5, 0.75);
        const Array y = op.apply(Array(n, 1.0));
        emit(p + "t2_apply_0", y[0]);
        emit(p + "t2_apply_mid", y[n / 2]);
        emit(p + "t2_apply_last", y[n - 1]);
    }

    {
        FdmBoundaryConditionSet bcSet;
        FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 0};
        auto op = ext::make_shared<FdmBlackScholesOp>(
            mesher, process, strike, c.localVol, -Null<Real>(), 0, qh);
        Fdm1DimSolver solver(desc, FdmSchemeDesc::Douglas(), op);
        emit(p + "value_90", solver.interpolateAt(std::log(90.0)));
        emit(p + "value_100", solver.interpolateAt(std::log(100.0)));
        emit(p + "value_110", solver.interpolateAt(std::log(110.0)));
        emit(p + "dx_100", solver.derivativeX(std::log(100.0)));
    }
}

// A local vol that throws below a spot level, to pin the
// illegalLocalVolOverwrite branch: a NON-negative override substitutes for
// the vol when the lookup throws; a negative one lets it propagate.
class ThrowingLocalVol : public LocalVolTermStructure {
  public:
    ThrowingLocalVol() : LocalVolTermStructure(TODAY, NullCalendar(), Following, DC) {}
    Date maxDate() const override { return Date::maxDate(); }
    Real minStrike() const override { return 0.0; }
    Real maxStrike() const override { return QL_MAX_REAL; }

  protected:
    Volatility localVolImpl(Time t, Real s) const override {
        QL_REQUIRE(s > 80.0, "no local vol below 80");
        return 0.15 + 0.05 * std::log(s / 100.0) + 0.02 * t;
    }
};

void blockOverwrite() {
    const Real strike = 100.0;
    const Size n = 25;
    auto m1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), n);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);
    auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)), flat(0.02), flat(0.05),
        Handle<BlackVolTermStructure>(
            ext::make_shared<BlackConstantVol>(TODAY, NullCalendar(), 0.20, DC)),
        Handle<LocalVolTermStructure>(ext::make_shared<ThrowingLocalVol>()));

    FdmBlackScholesOp op(mesher, process, strike, true, 0.35, 0,
                         ext::shared_ptr<FdmQuantoHelper>());
    op.setTime(0.0, 0.25);
    emitApply("ovr_", op, n);
}

// ---------------------------------------------------------------------------
// Fdm2dBlackScholesOp's own localVol branch: it does NOT reuse the
// sub-operators' variance, it rescales the mixed-derivative template by the
// product of the two *local* vols per node (note: the vols, not the
// variances, and with illegalLocalVolOverwrite applied per asset).
// Same mesh and market as the constant-vol case already in
// v143_methods_operators, so the two are directly comparable.
// ---------------------------------------------------------------------------
void block2d() {
    auto mesher = ext::make_shared<FdmMesherComposite>(
        ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(200.0), 5),
        ext::make_shared<Uniform1dMesher>(std::log(45.0), std::log(180.0), 4));

    const Handle<BlackVolTermStructure> v1(
        ext::make_shared<BlackConstantVol>(TODAY, NullCalendar(), 0.25, DC));
    const Handle<BlackVolTermStructure> v2(
        ext::make_shared<BlackConstantVol>(TODAY, NullCalendar(), 0.30, DC));
    const Handle<LocalVolTermStructure> lv(ext::make_shared<RampLocalVol>());

    auto p1 = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)), flat(0.02), flat(0.05), v1, lv);
    auto p2 = ext::make_shared<GeneralizedBlackScholesProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(90.0)), flat(0.01), flat(0.05), v2, lv);

    const Size n = mesher->layout()->size();
    Array ramp(n);
    for (Size i = 0; i < n; ++i)
        ramp[i] = 1.0 + 0.5 * Real(i);

    for (int lvFlag = 0; lvFlag < 2; ++lvFlag) {
        Fdm2dBlackScholesOp op(mesher, p1, p2, 0.4, 1.0, lvFlag != 0, -Null<Real>());
        op.setTime(0.1, 0.35);
        const std::string p = lvFlag ? "op2d_lv_" : "op2d_plain_";
        const Array a = op.apply(ramp);
        const Array m = op.apply_mixed(ramp);
        const Array d0 = op.apply_direction(0, ramp);
        for (Size i = 0; i < n; ++i) {
            emit(p + "apply_" + std::to_string(i), a[i]);
            emit(p + "mixed_" + std::to_string(i), m[i]);
        }
        emit(p + "dir0_0", d0[0]);
        emit(p + "dir0_last", d0[n - 1]);
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // constant vol, and the derived-LocalConstantVol arm that must match it
    block({"flat_plain", false, false, false});
    block({"flat_lv", true, false, false});
    // an external, strike- and time-dependent local vol: the arms diverge
    block({"ramp_plain", false, true, false});
    block({"ramp_lv", true, true, false});
    // and the quanto adjustment, scalar overload then vector overload
    block({"flat_plain_q", false, false, true});
    block({"ramp_lv_q", true, true, true});

    blockOverwrite();
    block2d();

    std::cout << "  \"_end\": 0\n}\n";
    return 0;
}
