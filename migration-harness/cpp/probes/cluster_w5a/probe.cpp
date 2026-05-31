// Phase 11 W5-A cluster probe: ExtOU + Kluge process FD infrastructure.
//
// Captures reference values for:
//
//   * Glued1dMesher: locations as union of two Uniform1dMesher
//     (overlap & no-overlap variants); spacings dplus/dminus at sample
//     nodes.
//
//   * FdmExtendedOrnsteinUhlenbeckOp on a Uniform1d log-spot mesh
//     after setTime(t1, t2): map.apply on synthetic vectors (constant,
//     linear, x^2) at fixed (speed, sigma, level, rTS) params.
//
//   * FdmExtOUJumpOp on a 2D mesh (Uniform1d x Uniform1d) after
//     setTime: apply on a constant vector; apply_direction on the y
//     factor; integro_part diag sum.
//
//   * FdmKlugeExtOUOp on a 3D mesh (Uniform1d^3) after setTime: apply
//     on constant + linear-in-direction-2 vectors.
//
//   * FdmExtOUJumpSolver value_at(x0, y0) NPV at maturity for a
//     spread payoff (LOOSE).
//
//   * FdmKlugeExtOUSolver value_at(x0, y0, u0) NPV for a spread
//     payoff (LOOSE).
//
//   * FdmSimple2dExtOUSolver value_at(x0, t0) NPV (LOOSE).
//
//   * FdmSimple3dExtOUJumpSolver value_at(x0, y0, t0) NPV (LOOSE).
//
//   * FdmExpExtOUInnerValueCalculator inner_value at known state.
//
//   * FdmSpreadPayoffInnerValue inner_value at known states.
//
// C++ parity:
//   ql/experimental/finitedifferences/glued1dmesher.hpp
//   ql/experimental/finitedifferences/fdmextendedornsteinuhlenbeckop.hpp
//   ql/experimental/finitedifferences/fdmextoujumpop.hpp
//   ql/experimental/finitedifferences/fdmklugeextouop.hpp
//   ql/experimental/finitedifferences/fdmextoujumpsolver.hpp
//   ql/experimental/finitedifferences/fdmklugeextousolver.hpp
//   ql/experimental/finitedifferences/fdmsimple2dextousolver.hpp
//   ql/experimental/finitedifferences/fdmsimple3dextoujumpsolver.hpp
//   ql/experimental/finitedifferences/fdmexpextouinnervaluecalculator.hpp
//   ql/experimental/finitedifferences/fdmspreadpayoffinnervalue.hpp
//   @ v1.42.1 (099987f0).
//
// Note: Settings::evaluationDate() is intentionally NOT set in main()
// because the combined translation-unit size + Apple Clang -O3
// optimisation can leave the `inline operator!=(Date, Date)` as an
// external reference that the QL dylib does not export.

#include <ql/experimental/finitedifferences/fdmexpextouinnervaluecalculator.hpp>
#include <ql/experimental/finitedifferences/fdmextendedornsteinuhlenbeckop.hpp>
#include <ql/experimental/finitedifferences/fdmextoujumpop.hpp>
#include <ql/experimental/finitedifferences/fdmextoujumpsolver.hpp>
#include <ql/experimental/finitedifferences/fdmklugeextouop.hpp>
#include <ql/experimental/finitedifferences/fdmklugeextousolver.hpp>
#include <ql/experimental/finitedifferences/fdmsimple2dextousolver.hpp>
#include <ql/experimental/finitedifferences/fdmsimple3dextoujumpsolver.hpp>
#include <ql/experimental/finitedifferences/fdmspreadpayoffinnervalue.hpp>
#include <ql/experimental/finitedifferences/glued1dmesher.hpp>
#include <ql/experimental/processes/extendedornsteinuhlenbeckprocess.hpp>
#include <ql/experimental/processes/extouwithjumpsprocess.hpp>
#include <ql/experimental/processes/klugeextouprocess.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdmsolverdesc.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/date.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

namespace {

void emit(const char* name, Real v, bool comma = true) {
    std::cout << "  \"" << name << "\": " << v;
    if (comma) std::cout << ",";
    std::cout << "\n";
}

// ExtendedOU drift function b(t) = level (constant).
struct ConstantLevel {
    Real level;
    Real operator()(Real /*x*/) const { return level; }
};

// Block 1: Glued1dMesher
__attribute__((noinline))
void block_glued1d_mesher() {
    auto left = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 4);
    auto right = ext::make_shared<Uniform1dMesher>(1.0, 3.0, 3);
    Glued1dMesher glued(*left, *right);
    emit("glued_common_size", Real(glued.locations().size()));
    emit("glued_common_loc0", glued.locations()[0]);
    emit("glued_common_loc1", glued.locations()[1]);
    emit("glued_common_loc2", glued.locations()[2]);
    emit("glued_common_loc3", glued.locations()[3]);
    emit("glued_common_loc4", glued.locations()[4]);
    emit("glued_common_loc5", glued.locations()[5]);
    emit("glued_common_dplus2", glued.dplus(2));
    emit("glued_common_dminus3", glued.dminus(3));

    // No-overlap case: leftEnd < rightStart.
    auto left2 = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 3);
    auto right2 = ext::make_shared<Uniform1dMesher>(2.0, 4.0, 3);
    Glued1dMesher glued2(*left2, *right2);
    emit("glued_no_overlap_size", Real(glued2.locations().size()));
    emit("glued_no_overlap_loc2", glued2.locations()[2]);
    emit("glued_no_overlap_loc3", glued2.locations()[3]);
    emit("glued_no_overlap_dplus2", glued2.dplus(2));
}

// Block 2: FdmExtendedOrnsteinUhlenbeckOp
__attribute__((noinline))
void block_ou_op(const Handle<YieldTermStructure>& rTS) {
    Real speed = 1.0;
    Real sigma = 0.3;
    Real x0 = 0.0;
    ConstantLevel cl{0.0};
    std::function<Real(Real)> bfun = cl;
    auto process = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        speed, sigma, x0, bfun);

    auto m1 = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);

    FdmBoundaryConditionSet bcSet;
    FdmExtendedOrnsteinUhlenbeckOp op(
        mesher, process, rTS.currentLink(), bcSet, 0);

    op.setTime(0.0, 0.5);

    Array r(9, 1.0);
    Array out = op.apply(r);
    emit("ou_op_const_apply_0", out[0]);
    emit("ou_op_const_apply_3", out[3]);
    emit("ou_op_const_apply_4", out[4]);
    emit("ou_op_const_apply_5", out[5]);
    emit("ou_op_const_apply_8", out[8]);

    Array r2(9);
    for (Size i = 0; i < 9; ++i)
        r2[i] = mesher->locations(0)[i];
    Array out2 = op.apply(r2);
    emit("ou_op_lin_apply_3", out2[3]);
    emit("ou_op_lin_apply_4", out2[4]);
    emit("ou_op_lin_apply_5", out2[5]);

    Array r3(9);
    for (Size i = 0; i < 9; ++i)
        r3[i] = mesher->locations(0)[i] * mesher->locations(0)[i];
    Array out3 = op.apply(r3);
    emit("ou_op_quad_apply_3", out3[3]);
    emit("ou_op_quad_apply_4", out3[4]);
    emit("ou_op_quad_apply_5", out3[5]);

    Array solved = op.solve_splitting(0, r, 0.1);
    emit("ou_op_solve_splitting_4", solved[4]);
}

// Block 3: FdmExtOUJumpOp
__attribute__((noinline))
void block_jump_op(const Handle<YieldTermStructure>& rTS) {
    ConstantLevel cl{0.0};
    std::function<Real(Real)> bfun = cl;
    auto ouProcess = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        1.0, 0.3, 0.0, bfun);
    auto kluge = ext::make_shared<ExtOUWithJumpsProcess>(
        ouProcess, 0.0, 4.0, 2.0, 4.0);

    auto mx = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 7);
    auto my = ext::make_shared<Uniform1dMesher>(0.0, 2.0, 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);

    FdmBoundaryConditionSet bcSet;
    FdmExtOUJumpOp op(mesher, kluge, rTS.currentLink(), bcSet, 16);

    op.setTime(0.0, 0.5);

    const Size n = mesher->layout()->size();
    Array ones(n, 1.0);
    Array out = op.apply(ones);
    emit("jump_op_const_apply_0", out[0]);
    emit("jump_op_const_apply_size_4", out[n / 4]);
    emit("jump_op_const_apply_size_2", out[n / 2]);
    emit("jump_op_const_apply_size_3_4", out[3 * n / 4]);

    Array out_dir1 = op.apply_direction(1, ones);
    emit("jump_op_const_apply_dir1_0", out_dir1[0]);
    emit("jump_op_const_apply_dir1_size_2", out_dir1[n / 2]);

    Array out_mixed = op.apply_mixed(ones);
    emit("jump_op_const_apply_mixed_0", out_mixed[0]);
    emit("jump_op_const_apply_mixed_size_2", out_mixed[n / 2]);
}

// Block 4: FdmKlugeExtOUOp
__attribute__((noinline))
void block_kluge_op(const Handle<YieldTermStructure>& rTS) {
    ConstantLevel cl{0.0};
    std::function<Real(Real)> bfun = cl;
    auto extOU_a = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        1.0, 0.3, 0.0, bfun);
    auto kluge = ext::make_shared<ExtOUWithJumpsProcess>(
        extOU_a, 0.0, 4.0, 2.0, 4.0);

    ConstantLevel cl2{0.0};
    std::function<Real(Real)> bfun2 = cl2;
    auto extOU_b = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        0.5, 0.25, 0.0, bfun2);

    auto klugeExtOU = ext::make_shared<KlugeExtOUProcess>(
        0.4, kluge, extOU_b);

    auto mx = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 5);
    auto my = ext::make_shared<Uniform1dMesher>(0.0, 2.0, 4);
    auto mu = ext::make_shared<Uniform1dMesher>(-2.0, 2.0, 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my, mu);

    FdmBoundaryConditionSet bcSet;
    FdmKlugeExtOUOp op(mesher, klugeExtOU, rTS.currentLink(), bcSet, 16);

    op.setTime(0.0, 0.5);

    const Size n = mesher->layout()->size();
    Array ones(n, 1.0);
    Array out = op.apply(ones);
    emit("kluge_op_const_apply_0", out[0]);
    emit("kluge_op_const_apply_size_4", out[n / 4]);
    emit("kluge_op_const_apply_size_2", out[n / 2]);
    emit("kluge_op_const_apply_size_3_4", out[3 * n / 4]);

    Array r_u(n);
    for (const auto& iter : *mesher->layout()) {
        r_u[iter.index()] = mesher->location(iter, 2);
    }
    Array out_u = op.apply(r_u);
    emit("kluge_op_lin_apply_size_2", out_u[n / 2]);

    Array out_d2 = op.apply_direction(2, ones);
    emit("kluge_op_const_apply_dir2_size_2", out_d2[n / 2]);
}

// Block 5: FdmExpExtOUInnerValueCalculator
__attribute__((noinline))
void block_exp_ou_iv() {
    auto m = ext::make_shared<Uniform1dMesher>(
        std::log(50.0), std::log(150.0), 11);
    auto mesher = ext::make_shared<FdmMesherComposite>(m);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    FdmExpExtOUInnerValueCalculator calc(payoff, mesher);

    for (const auto& iter : *mesher->layout()) {
        const Size i = iter.index();
        const Real u = mesher->location(iter, 0);
        const Real spot = std::exp(u);
        if (i == 0) emit("exp_ou_iv_calc_0_spot", spot);
        if (i == 5) emit("exp_ou_iv_calc_5_spot", spot);
        if (i == 10) emit("exp_ou_iv_calc_10_spot", spot);
        if (i == 0) emit("exp_ou_iv_calc_0_iv", calc.innerValue(iter, 0.0));
        if (i == 5) emit("exp_ou_iv_calc_5_iv", calc.innerValue(iter, 0.0));
        if (i == 10) emit("exp_ou_iv_calc_10_iv", calc.innerValue(iter, 0.0));
    }
}

// Block 6: FdmSpreadPayoffInnerValue
__attribute__((noinline))
void block_spread_iv() {
    auto m1 = ext::make_shared<Uniform1dMesher>(
        std::log(50.0), std::log(150.0), 5);
    auto m2 = ext::make_shared<Uniform1dMesher>(
        std::log(50.0), std::log(150.0), 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1, m2);

    auto basePayoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0);
    auto spreadPayoff = ext::make_shared<SpreadBasketPayoff>(basePayoff);

    auto calc1 = ext::make_shared<FdmExpExtOUInnerValueCalculator>(
        basePayoff, mesher, ext::shared_ptr<
            FdmExpExtOUInnerValueCalculator::Shape>(), 0);
    auto calc2 = ext::make_shared<FdmExpExtOUInnerValueCalculator>(
        basePayoff, mesher, ext::shared_ptr<
            FdmExpExtOUInnerValueCalculator::Shape>(), 1);

    FdmSpreadPayoffInnerValue spreadCalc(spreadPayoff, calc1, calc2);

    for (const auto& iter : *mesher->layout()) {
        const Size i = iter.index();
        if (i == 0) {
            emit("spread_payoff_iv_0", spreadCalc.innerValue(iter, 0.0));
        }
        if (i == 12) {
            emit("spread_payoff_iv_12", spreadCalc.innerValue(iter, 0.0));
        }
        if (i == 24) {
            emit("spread_payoff_iv_24", spreadCalc.innerValue(iter, 0.0));
        }
        if (i == 4) {
            emit("spread_payoff_iv_4", spreadCalc.innerValue(iter, 0.0));
        }
        if (i == 20) {
            emit("spread_payoff_iv_20", spreadCalc.innerValue(iter, 0.0));
        }
    }
}

// Block 7: FdmExtOUJumpSolver
__attribute__((noinline))
void block_ext_ou_jump_solver(const Handle<YieldTermStructure>& rTS) {
    ConstantLevel cl{0.0};
    std::function<Real(Real)> bfun = cl;
    auto ouProcess = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        1.0, 0.3, 0.0, bfun);
    auto kluge = ext::make_shared<ExtOUWithJumpsProcess>(
        ouProcess, 0.0, 4.0, 2.0, 4.0);

    auto mx = ext::make_shared<Uniform1dMesher>(-3.0, 3.0, 25);
    auto my = ext::make_shared<Uniform1dMesher>(0.0, 3.0, 15);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 1.0);

    auto calc = ext::make_shared<FdmExpExtOUInnerValueCalculator>(
        payoff, mesher);

    FdmStepConditionComposite::Conditions stepConds;
    std::list<std::vector<Time>> stoppingTimes;
    auto conditions = ext::make_shared<FdmStepConditionComposite>(
        stoppingTimes, stepConds);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc solverDesc = {mesher, bcSet, conditions, calc,
                                 0.25, 25, 0};
    FdmExtOUJumpSolver solver(Handle<ExtOUWithJumpsProcess>(kluge),
                               rTS.currentLink(), solverDesc);
    emit("ext_ou_jump_solver_value_0_0", solver.valueAt(0.0, 0.0));
}

// Block 8: FdmSimple2dExtOUSolver
__attribute__((noinline))
void block_simple_2d_ext_ou_solver(const Handle<YieldTermStructure>& rTS) {
    ConstantLevel cl{0.0};
    std::function<Real(Real)> bfun = cl;
    auto process = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        1.0, 0.3, 0.0, bfun);

    auto mx = ext::make_shared<Uniform1dMesher>(-3.0, 3.0, 31);
    auto mt = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 11);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mt);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 1.0);
    auto calc = ext::make_shared<FdmExpExtOUInnerValueCalculator>(
        payoff, mesher);

    FdmStepConditionComposite::Conditions stepConds;
    std::list<std::vector<Time>> stoppingTimes;
    auto conditions = ext::make_shared<FdmStepConditionComposite>(
        stoppingTimes, stepConds);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc solverDesc = {mesher, bcSet, conditions, calc,
                                 0.25, 25, 0};
    FdmSimple2dExtOUSolver solver(
        Handle<ExtendedOrnsteinUhlenbeckProcess>(process),
        rTS.currentLink(), solverDesc);
    emit("simple_2d_ext_ou_solver_value_0_0", solver.valueAt(0.0, 0.0));
}

// Block 9: FdmSimple3dExtOUJumpSolver
__attribute__((noinline))
void block_simple_3d_ext_ou_jump_solver(const Handle<YieldTermStructure>& rTS) {
    ConstantLevel cl{0.0};
    std::function<Real(Real)> bfun = cl;
    auto ouProcess = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        1.0, 0.3, 0.0, bfun);
    auto kluge = ext::make_shared<ExtOUWithJumpsProcess>(
        ouProcess, 0.0, 4.0, 2.0, 4.0);

    auto mx = ext::make_shared<Uniform1dMesher>(-3.0, 3.0, 15);
    auto my = ext::make_shared<Uniform1dMesher>(0.0, 3.0, 9);
    auto mt = ext::make_shared<Uniform1dMesher>(0.0, 1.0, 11);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my, mt);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 1.0);
    auto calc = ext::make_shared<FdmExpExtOUInnerValueCalculator>(
        payoff, mesher);

    FdmStepConditionComposite::Conditions stepConds;
    std::list<std::vector<Time>> stoppingTimes;
    auto conditions = ext::make_shared<FdmStepConditionComposite>(
        stoppingTimes, stepConds);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc solverDesc = {mesher, bcSet, conditions, calc,
                                 0.25, 15, 0};
    FdmSimple3dExtOUJumpSolver solver(
        Handle<ExtOUWithJumpsProcess>(kluge),
        rTS.currentLink(), solverDesc);
    emit("simple_3d_ext_ou_jump_solver_value", solver.valueAt(0.0, 0.0, 0.0));
}

// Block 10: FdmKlugeExtOUSolver
__attribute__((noinline))
void block_kluge_ext_ou_solver(const Handle<YieldTermStructure>& rTS) {
    ConstantLevel cl{0.0};
    std::function<Real(Real)> bfun = cl;
    auto extOU_a = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        1.0, 0.3, 0.0, bfun);
    auto kluge = ext::make_shared<ExtOUWithJumpsProcess>(
        extOU_a, 0.0, 4.0, 2.0, 4.0);

    ConstantLevel cl2{0.0};
    std::function<Real(Real)> bfun2 = cl2;
    auto extOU_b = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        0.5, 0.25, 0.0, bfun2);

    auto klugeExtOU = ext::make_shared<KlugeExtOUProcess>(
        0.4, kluge, extOU_b);

    auto mx = ext::make_shared<Uniform1dMesher>(-3.0, 3.0, 13);
    auto my = ext::make_shared<Uniform1dMesher>(0.0, 3.0, 7);
    auto mu = ext::make_shared<Uniform1dMesher>(-3.0, 3.0, 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my, mu);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 1.0);
    auto calc = ext::make_shared<FdmExpExtOUInnerValueCalculator>(
        payoff, mesher);

    FdmStepConditionComposite::Conditions stepConds;
    std::list<std::vector<Time>> stoppingTimes;
    auto conditions = ext::make_shared<FdmStepConditionComposite>(
        stoppingTimes, stepConds);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc solverDesc = {mesher, bcSet, conditions, calc,
                                 0.25, 15, 0};
    FdmKlugeExtOUSolver<3> solver(
        Handle<KlugeExtOUProcess>(klugeExtOU),
        rTS.currentLink(), solverDesc);
    std::vector<Real> at = {0.0, 0.0, 0.0};
    emit("kluge_ext_ou_solver_value", solver.valueAt(at), false);
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    DayCounter dc = Actual365Fixed();
    Date today(15, January, 2024);

    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, 0.05, dc));

    block_glued1d_mesher();
    block_ou_op(rTS);
    block_jump_op(rTS);
    block_kluge_op(rTS);
    block_exp_ou_iv();
    block_spread_iv();
    block_ext_ou_jump_solver(rTS);
    block_simple_2d_ext_ou_solver(rTS);
    block_simple_3d_ext_ou_jump_solver(rTS);
    block_kluge_ext_ou_solver(rTS);

    std::cout << "}\n";
    return 0;
}
