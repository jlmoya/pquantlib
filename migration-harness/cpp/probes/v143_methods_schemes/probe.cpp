// migration-harness/cpp/probes/v143_methods_schemes/probe.cpp
//
// Reference values for the ql/methods/finitedifferences/schemes cluster plus
// ql/methods/finitedifferences/trbdf2.hpp at C++ QuantLib v1.43 (pinned
// submodule @ 6b57206e0).  Emits references/v143/methods/schemes.json.
//
// What is pinned here, and why
// ----------------------------
//
// The eight classes under test are *pure algebra over an operator*: none of
// them owns any numerics of its own beyond the sequence of apply / solve calls
// it makes and the coefficients it multiplies them by.  So the reference
// fixture has to be an operator whose every primitive is bit-reproducible in
// the port, otherwise a scheme bug and an operator bug are indistinguishable.
//
// Block S ("synthetic") therefore uses `ProbeOp`, a hand-written
// FdmLinearOpComposite with TWO directions and a genuine mixed term, whose
// apply / apply_direction / apply_mixed / solve_splitting / preconditioner are
// written as explicit loops with dyadic-rational coefficients.  The Python
// test re-implements exactly the same loops.  That is what makes the
// comparison meaningful:
//
//   * size() == 2 drives the `for (i=0; i < map_->size(); ++i)` splitting loop
//     of Douglas / Craig-Sneyd / Hundsdorfer / modified Craig-Sneyd more than
//     once, which a 1-D operator cannot do;
//   * apply_mixed() is non-zero, which is the ONLY thing distinguishing
//     CraigSneydScheme from HundsdorferScheme (`apply_mixed(y-a)` versus
//     `apply(y-a)`) and ModifiedCraigSneydScheme from CraigSneydScheme (the
//     extra `(0.5-mu)*dt*apply(y-a)` term).  With a 1-D op all three schemes
//     collapse onto each other and the test proves nothing;
//   * map_->size() != 1 sends TrBDF2Scheme down its BiCGstab branch.
//
// Block S is emitted twice per scheme: once with an empty bc_set, once with a
// `ProbeBC` boundary condition, so that every one of the five
// BoundaryConditionSchemeHelper fan-out methods is observable in the output.
// ProbeBC is a Dirichlet-shaped condition: applyBeforeApplying/BeforeSolving
// zero the operator's first and last row (idempotent, exactly what
// DirichletBC does to a TridiagonalOperator), applyAfterApplying/AfterSolving
// pin the two boundary entries of the array.
//
// Block H drives BoundaryConditionSchemeHelper's five methods directly, so the
// class is pinned on its own and not only through its callers.
//
// Block B ("Black-Scholes") repeats every scheme on the real 1-D
// FdmBlackScholesOp over an FdmBlackScholesMesher, with the same fixture the
// Python test-suite already uses for the schemes cluster (21 points, S0 = 100,
// r = 5%, q = 0, sigma = 20%, T = 1, K = 100, Actual365Fixed, NullCalendar,
// reference date 15-Jun-2026).  This is the integration check: it pins that a
// scheme wired to a production operator reproduces C++, and it incidentally
// cross-validates FdmBlackScholesOp / FdmBlackScholesMesher, which until now
// were only self-consistency-tested on the Python side.  In 1-D the splitting
// loop runs once and apply_mixed is zero, so Douglas / Craig-Sneyd /
// Hundsdorfer / modified Craig-Sneyd must agree with each other -- that
// degeneracy is itself pinned.
//
// Block T pins TRBDF2<TridiagonalOperator> from trbdf2.hpp -- a different
// class from TrBDF2Scheme, living on the pre-1.0 operator-algebra framework
// (applyTo / solveFor / identity / operator algebra) rather than on
// FdmLinearOpComposite.  Four variants: time-constant and time-dependent
// operator, each with and without a boundary-condition set (DirichletBC lower
// + NeumannBC upper).  The time-dependent variant is what exercises the three
// `if (L_.isTimeDependent())` re-derivations inside step(); the bc variants
// exercise the fact that the BDF2 parts are mutated in place by
// applyBeforeApplying and reused across steps.
//
// C++ parity:
//   ql/methods/finitedifferences/schemes/boundaryconditionschemehelper.hpp
//   ql/methods/finitedifferences/schemes/craigsneydscheme.{hpp,cpp}
//   ql/methods/finitedifferences/schemes/douglasscheme.{hpp,cpp}
//   ql/methods/finitedifferences/schemes/hundsdorferscheme.{hpp,cpp}
//   ql/methods/finitedifferences/schemes/methodoflinesscheme.{hpp,cpp}
//   ql/methods/finitedifferences/schemes/modifiedcraigsneydscheme.{hpp,cpp}
//   ql/methods/finitedifferences/schemes/trbdf2scheme.hpp
//   ql/methods/finitedifferences/trbdf2.hpp
//   @ v1.43 (6b57206e0).

#include <ql/handle.hpp>
#include <ql/methods/finitedifferences/boundarycondition.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/schemes/boundaryconditionschemehelper.hpp>
#include <ql/methods/finitedifferences/schemes/craigsneydscheme.hpp>
#include <ql/methods/finitedifferences/schemes/cranknicolsonscheme.hpp>
#include <ql/methods/finitedifferences/schemes/douglasscheme.hpp>
#include <ql/methods/finitedifferences/schemes/hundsdorferscheme.hpp>
#include <ql/methods/finitedifferences/schemes/methodoflinesscheme.hpp>
#include <ql/methods/finitedifferences/schemes/modifiedcraigsneydscheme.hpp>
#include <ql/methods/finitedifferences/schemes/trbdf2scheme.hpp>
#include <ql/methods/finitedifferences/trbdf2.hpp>
#include <ql/methods/finitedifferences/tridiagonaloperator.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/date.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

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

void emit_int(const std::string& name, long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i != 0U) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

std::vector<Real> to_vector(const Array& a) {
    return {a.begin(), a.end()};
}

// --------------------------------------------------------------------------
// Block S fixture: a two-direction FdmLinearOpComposite with a mixed term.
//
// Every coefficient is a dyadic rational so that the coefficient arithmetic
// itself is exact in binary64 and any C++/Python discrepancy has to come from
// the scheme, not from the fixture.
// --------------------------------------------------------------------------

const Size PROBE_N = 7;

class ProbeOp : public FdmLinearOpComposite {
  public:
    ProbeOp() : mask_(PROBE_N, 1.0) {}

    Size size() const override { return 2; }

    void setTime(Time t1, Time t2) override {
        t1_ = t1;
        t2_ = t2;
        tm_ = 0.5 * (t1 + t2);
    }

    // A Dirichlet-shaped boundary condition zeroes the first and last row of
    // the operator; ProbeBC does it through this hook.  Idempotent.
    void maskBoundary() {
        mask_[0] = 0.0;
        mask_[PROBE_N - 1] = 0.0;
    }

    // Diffusion-shaped: strictly negative diagonal, positive off-diagonals,
    // weakly diagonally dominant -- so `a + dt*L*a` decays, exactly as a
    // production FD generator does, and none of the six schemes runs into a
    // growth mode that would swamp a genuine port defect.
    Real lower(Size d, Size i) const {
        return d == 0 ? (1.0 - 0.0625 * Real(i)) : (0.375 + 0.015625 * Real(i));
    }
    Real diag(Size d, Size i) const {
        return d == 0 ? (-2.5 - 0.25 * tm_ - 0.0625 * Real(i)) : (-1.25 - 0.5 * tm_);
    }
    Real upper(Size d, Size i) const {
        return d == 0 ? (0.875 + 0.03125 * Real(i)) : (0.4375 - 0.03125 * Real(i));
    }

    Array apply_direction(Size d, const Array& r) const override {
        Array out(PROBE_N, 0.0);
        for (Size i = 0; i < PROBE_N; ++i) {
            Real v = diag(d, i) * r[i];
            if (i > 0) v += lower(d, i) * r[i - 1];
            if (i + 1 < PROBE_N) v += upper(d, i) * r[i + 1];
            out[i] = mask_[i] * v;
        }
        return out;
    }

    Array apply_mixed(const Array& r) const override {
        Array out(PROBE_N, 0.0);
        for (Size i = 0; i < PROBE_N; ++i)
            out[i] = mask_[i] * (0.0625 + 0.125 * tm_)
                     * (r[(i + 2) % PROBE_N] - r[(i + PROBE_N - 2) % PROBE_N]);
        return out;
    }

    Array apply(const Array& r) const override {
        return apply_direction(0, r) + apply_direction(1, r) + apply_mixed(r);
    }

    // Solve (I + s * L_d) x = r, with L_d the masked direction-d tridiagonal.
    // Thomas algorithm, written exactly as TridiagonalOperator::solveFor.
    Array solve_splitting(Size d, const Array& r, Real s) const override {
        Array x(PROBE_N, 0.0);
        std::vector<Real> temp(PROBE_N, 0.0);
        std::vector<Real> a(PROBE_N, 0.0), b(PROBE_N, 0.0), c(PROBE_N, 0.0);
        for (Size i = 0; i < PROBE_N; ++i) {
            a[i] = s * mask_[i] * lower(d, i);
            b[i] = 1.0 + s * mask_[i] * diag(d, i);
            c[i] = s * mask_[i] * upper(d, i);
        }
        Real bet = b[0];
        x[0] = r[0] / bet;
        for (Size j = 1; j < PROBE_N; ++j) {
            temp[j] = c[j - 1] / bet;
            bet = b[j] - a[j] * temp[j];
            x[j] = (r[j] - a[j] * x[j - 1]) / bet;
        }
        for (Size j = PROBE_N - 2; j > 0; --j) x[j] -= temp[j + 1] * x[j + 1];
        x[0] -= temp[1] * x[1];
        return x;
    }

    Array preconditioner(const Array& r, Real s) const override {
        return solve_splitting(0, r, s);
    }

  private:
    std::vector<Real> mask_;
    Time t1_ = 0.0, t2_ = 0.0, tm_ = 0.0;
};

class ProbeBC : public BoundaryCondition<FdmLinearOp> {
  public:
    void applyBeforeApplying(FdmLinearOp& op) const override {
        dynamic_cast<ProbeOp&>(op).maskBoundary();
    }
    void applyAfterApplying(Array& u) const override {
        u[0] = v_;
        u[u.size() - 1] = -v_;
    }
    void applyBeforeSolving(FdmLinearOp& op, Array& rhs) const override {
        dynamic_cast<ProbeOp&>(op).maskBoundary();
        rhs[0] = v_;
        rhs[rhs.size() - 1] = -v_;
    }
    void applyAfterSolving(Array& u) const override {
        u[0] = 2.0 * v_;
        u[u.size() - 1] = -2.0 * v_;
    }
    void setTime(Time t) override { v_ = 1.0 + 0.5 * t; }

  private:
    Real v_ = 1.0;
};

Array probeStart() {
    Array a(PROBE_N);
    a[0] = 1.0;
    a[1] = 1.5;
    a[2] = 2.25;
    a[3] = 1.75;
    a[4] = 0.5;
    a[5] = -0.75;
    a[6] = 2.0;
    return a;
}

OperatorTraits<FdmLinearOp>::bc_set makeBcSet(bool withBc) {
    OperatorTraits<FdmLinearOp>::bc_set s;
    if (withBc) s.push_back(ext::make_shared<ProbeBC>());
    return s;
}

const Time SYN_T = 1.0;
const Time SYN_DT = 0.25;

// Run `steps` successive steps of `scheme` from SYN_T backwards and emit the
// state after the first and after the last one.
template <class Scheme>
void runSynthetic(const std::string& prefix, Scheme& scheme) {
    Array a = probeStart();
    scheme.setStep(SYN_DT);
    Time t = SYN_T;
    scheme.step(a, t);
    emit_arr(prefix + "_step1", to_vector(a));
    for (int k = 1; k < 3; ++k) {
        t -= SYN_DT;
        scheme.step(a, t);
    }
    emit_arr(prefix + "_step3", to_vector(a));
}

// --------------------------------------------------------------------------
// Block H -- BoundaryConditionSchemeHelper on its own.
// --------------------------------------------------------------------------

void block_helper() {
    auto op = ext::make_shared<ProbeOp>();
    op->setTime(0.75, 1.0);

    const BoundaryConditionSchemeHelper helper(makeBcSet(true));
    Array a = probeStart();

    // Before setTime the condition's stored value is its constructed default.
    Array y0 = op->apply(a);
    emit_arr("helper_apply_unmasked", to_vector(y0));

    helper.setTime(0.5);  // -> v = 1.25
    helper.applyBeforeApplying(*op);
    Array y1 = op->apply(a);
    emit_arr("helper_apply_masked", to_vector(y1));

    helper.applyAfterApplying(y1);
    emit_arr("helper_after_applying", to_vector(y1));

    Array rhs = probeStart();
    helper.applyBeforeSolving(*op, rhs);
    emit_arr("helper_before_solving_rhs", to_vector(rhs));

    Array solved = op->solve_splitting(0, rhs, -0.25);
    emit_arr("helper_solved", to_vector(solved));

    helper.applyAfterSolving(solved);
    emit_arr("helper_after_solving", to_vector(solved));

    // Empty bc_set: every method is a no-op.
    const BoundaryConditionSchemeHelper empty{OperatorTraits<FdmLinearOp>::bc_set()};
    Array untouched = probeStart();
    empty.setTime(0.5);
    empty.applyAfterApplying(untouched);
    empty.applyAfterSolving(untouched);
    emit_arr("helper_empty_noop", to_vector(untouched));
}

// --------------------------------------------------------------------------
// Block S -- the six FdmLinearOpComposite schemes on ProbeOp.
// --------------------------------------------------------------------------

void block_synthetic_variant(const std::string& suffix, bool withBc) {
    {
        auto op = ext::make_shared<ProbeOp>();
        DouglasScheme s(0.5, op, makeBcSet(withBc));
        runSynthetic("syn_douglas" + suffix, s);
    }
    {
        auto op = ext::make_shared<ProbeOp>();
        CraigSneydScheme s(0.5, 0.5, op, makeBcSet(withBc));
        runSynthetic("syn_craigsneyd" + suffix, s);
    }
    {
        auto op = ext::make_shared<ProbeOp>();
        HundsdorferScheme s(0.5 + std::sqrt(3.0) / 6.0, 0.5, op, makeBcSet(withBc));
        runSynthetic("syn_hundsdorfer" + suffix, s);
    }
    {
        auto op = ext::make_shared<ProbeOp>();
        ModifiedCraigSneydScheme s(1.0 / 3.0, 1.0 / 3.0, op, makeBcSet(withBc));
        runSynthetic("syn_modcraigsneyd" + suffix, s);
    }
    {
        auto op = ext::make_shared<ProbeOp>();
        MethodOfLinesScheme s(1e-6, 0.1, op, makeBcSet(withBc));
        runSynthetic("syn_mol" + suffix, s);
    }
    {
        // map_->size() == 2 -> the BiCGstab branch of TrBDF2Scheme::step.
        auto op = ext::make_shared<ProbeOp>();
        auto trapezoidal = ext::make_shared<DouglasScheme>(0.5, op, makeBcSet(withBc));
        TrBDF2Scheme<DouglasScheme> s(2.0 - std::sqrt(2.0), op, trapezoidal, makeBcSet(withBc));
        runSynthetic("syn_trbdf2" + suffix, s);
        emit_int("syn_trbdf2" + suffix + "_iterations", long(s.numberOfIterations()));
    }
    {
        // ... and the GMRES branch, same fixture.
        auto op = ext::make_shared<ProbeOp>();
        auto trapezoidal = ext::make_shared<DouglasScheme>(0.5, op, makeBcSet(withBc));
        TrBDF2Scheme<DouglasScheme> s(2.0 - std::sqrt(2.0), op, trapezoidal, makeBcSet(withBc),
                                      1e-8, TrBDF2Scheme<DouglasScheme>::GMRES);
        runSynthetic("syn_trbdf2_gmres" + suffix, s);
        emit_int("syn_trbdf2_gmres" + suffix + "_iterations", long(s.numberOfIterations()));
    }
}

void block_synthetic() {
    block_synthetic_variant("", false);
    block_synthetic_variant("_bc", true);
}

// --------------------------------------------------------------------------
// Block B -- the same schemes on the production 1-D FdmBlackScholesOp.
// --------------------------------------------------------------------------

ext::shared_ptr<GeneralizedBlackScholesProcess> makeBsmProcess(const Date& ref) {
    const DayCounter dc = Actual365Fixed();
    const Calendar cal = NullCalendar();
    Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(ref, 0.05, dc, Continuous, Annual));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(ref, 0.00, dc, Continuous, Annual));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(ref, cal, 0.20, dc));
    return ext::make_shared<GeneralizedBlackScholesProcess>(spot, qTS, rTS, volTS);
}

const Time BSM_T = 1.0;
const Time BSM_DT = 0.05;

template <class Scheme>
void runBsm(const std::string& prefix, Scheme& scheme, const Array& start) {
    Array a = start;
    scheme.setStep(BSM_DT);
    Time t = BSM_T;
    scheme.step(a, t);
    emit_arr(prefix + "_step1", to_vector(a));
    for (int k = 1; k < 4; ++k) {
        t -= BSM_DT;
        scheme.step(a, t);
    }
    emit_arr(prefix + "_step4", to_vector(a));
}

void block_bsm(const Date& ref) {
    auto process = makeBsmProcess(ref);
    auto bsMesher = ext::make_shared<FdmBlackScholesMesher>(21, process, 1.0, 100.0);
    auto mesher = ext::make_shared<FdmMesherComposite>(bsMesher);

    emit_arr("bsm_mesher_locations", bsMesher->locations());

    // Call payoff at the mesh nodes -- a realistic, non-smooth start vector.
    const Size n = mesher->layout()->size();
    Array start(n);
    for (Size i = 0; i < n; ++i)
        start[i] = std::max(std::exp(bsMesher->locations()[i]) - 100.0, 0.0);
    emit_arr("bsm_start", to_vector(start));

    {
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, process, 100.0);
        op->setTime(0.95, 1.0);
        emit_arr("bsm_op_apply_start", to_vector(op->apply(start)));
        emit_arr("bsm_op_solve_splitting", to_vector(op->solve_splitting(0, start, -0.025)));
    }
    {
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, process, 100.0);
        DouglasScheme s(0.5, op);
        runBsm("bsm_douglas", s, start);
    }
    {
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, process, 100.0);
        CraigSneydScheme s(0.5, 0.5, op);
        runBsm("bsm_craigsneyd", s, start);
    }
    {
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, process, 100.0);
        HundsdorferScheme s(0.5 + std::sqrt(3.0) / 6.0, 0.5, op);
        runBsm("bsm_hundsdorfer", s, start);
    }
    {
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, process, 100.0);
        ModifiedCraigSneydScheme s(1.0 / 3.0, 1.0 / 3.0, op);
        runBsm("bsm_modcraigsneyd", s, start);
    }
    {
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, process, 100.0);
        MethodOfLinesScheme s(1e-6, 0.1, op);
        runBsm("bsm_mol", s, start);
    }
    {
        // map_->size() == 1 -> the direct solve_splitting branch of
        // TrBDF2Scheme::step, and CrankNicolsonScheme as the trapezoidal half.
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, process, 100.0);
        auto trapezoidal = ext::make_shared<CrankNicolsonScheme>(0.5, op);
        TrBDF2Scheme<CrankNicolsonScheme> s(2.0 - std::sqrt(2.0), op, trapezoidal);
        runBsm("bsm_trbdf2", s, start);
        emit_int("bsm_trbdf2_iterations", long(s.numberOfIterations()));
    }
}

// --------------------------------------------------------------------------
// Block T -- TRBDF2<TridiagonalOperator> (trbdf2.hpp, old FD framework).
// --------------------------------------------------------------------------

const Size TRI_N = 7;

// The old framework's MixedScheme/TRBDF2 evolve `du/dt = -L u`, so a
// positive-diagonal / negative-off-diagonal L is the decaying (diffusion)
// case; that keeps three successive steps from amplifying the state.
class TriTimeSetter : public TridiagonalOperator::TimeSetter {
  public:
    void setTime(Time t, TridiagonalOperator& L) const override {
        L.setFirstRow(2.0 + 0.5 * t, -1.0);
        for (Size i = 1; i < TRI_N - 1; ++i)
            L.setMidRow(i, -1.0 + 0.0625 * Real(i), 2.0 + 0.5 * t, -1.0 + 0.03125 * Real(i));
        L.setLastRow(-1.0, 2.0 + 0.5 * t);
    }
};

class TimeDepTridiagonalOperator : public TridiagonalOperator {
  public:
    TimeDepTridiagonalOperator() : TridiagonalOperator(TRI_N) {
        timeSetter_ = ext::make_shared<TriTimeSetter>();
        TriTimeSetter().setTime(0.0, *this);
    }
};

TridiagonalOperator makeConstantTridiagonal() {
    TridiagonalOperator L(TRI_N);
    L.setFirstRow(2.0, -1.0);
    for (Size i = 1; i < TRI_N - 1; ++i)
        L.setMidRow(i, -1.0 + 0.0625 * Real(i), 2.0, -1.0 + 0.03125 * Real(i));
    L.setLastRow(-1.0, 2.0);
    return L;
}

QL_DEPRECATED_DISABLE_WARNING

OperatorTraits<TridiagonalOperator>::bc_set makeTriBcSet(bool withBc) {
    OperatorTraits<TridiagonalOperator>::bc_set s;
    if (withBc) {
        s.push_back(ext::make_shared<DirichletBC>(0.5, BoundaryCondition<TridiagonalOperator>::Lower));
        s.push_back(ext::make_shared<NeumannBC>(0.25, BoundaryCondition<TridiagonalOperator>::Upper));
    }
    return s;
}

QL_DEPRECATED_ENABLE_WARNING

void runTrbdf2(const std::string& prefix, const TridiagonalOperator& L, bool withBc) {
    TRBDF2<TridiagonalOperator> evolver(L, makeTriBcSet(withBc));
    evolver.setStep(0.25);

    Array a = probeStart();
    Time t = 1.0;
    evolver.step(a, t);
    emit_arr(prefix + "_step1", to_vector(a));
    for (int k = 1; k < 3; ++k) {
        t -= 0.25;
        evolver.step(a, t);
    }
    emit_arr(prefix + "_step3", to_vector(a));
}

void block_trbdf2_old() {
    const TridiagonalOperator constantL = makeConstantTridiagonal();
    emit_arr("tri_const_lower", to_vector(constantL.lowerDiagonal()));
    emit_arr("tri_const_diag", to_vector(constantL.diagonal()));
    emit_arr("tri_const_upper", to_vector(constantL.upperDiagonal()));
    emit_int("tri_const_is_time_dependent", constantL.isTimeDependent() ? 1 : 0);

    const TridiagonalOperator timeDepL = TimeDepTridiagonalOperator();
    emit_int("tri_timedep_is_time_dependent", timeDepL.isTimeDependent() ? 1 : 0);

    runTrbdf2("trbdf2_const", constantL, false);
    runTrbdf2("trbdf2_const_bc", constantL, true);
    runTrbdf2("trbdf2_timedep", timeDepL, false);
    runTrbdf2("trbdf2_timedep_bc", timeDepL, true);
}

}  // namespace

int main() {
    const Date ref(15, June, 2026);
    Settings::instance().evaluationDate() = ref;

    std::cout << "{\n";
    block_helper();
    block_synthetic();
    block_bsm(ref);
    block_trbdf2_old();
    std::cout << "\n}" << std::endl;
    return 0;
}
