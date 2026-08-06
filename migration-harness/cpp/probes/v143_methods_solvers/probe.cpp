// v1.43 coverage-closure probe: ql/methods/finitedifferences/solvers/.
//
// Pins the 13 solver classes of the FD `solvers/` cluster:
//
//   Fdm1DimSolver, Fdm2DimSolver, Fdm3DimSolver, FdmNdimSolver<N>,
//   FdmBlackScholesSolver, FdmSimple2dBSSolver, Fdm2dBlackScholesSolver,
//   FdmHestonSolver, FdmBatesSolver, FdmCIRSolver, FdmG2Solver,
//   FdmHullWhiteSolver, FdmHestonHullWhiteSolver.
//
// For every solver the probe emits, where the class exposes them:
//   * interpolateAt / valueAt at interior points AND at grid nodes,
//   * thetaAt,
//   * derivativeX / derivativeY / derivativeXX / derivativeYY / derivativeXY
//     (resp. deltaAt / gammaAt / meanVarianceDeltaAt / meanVarianceGammaAt).
//
// Scheme dispatch is pinned by re-running the same FdmSolverDesc under
// several FdmSchemeDesc values (Douglas / CrankNicolson / ImplicitEuler /
// ExplicitEuler / CraigSneyd / Hundsdorfer), so a solver that ignored the
// scheme argument could not pass.
//
// Cheap intermediates (mesher locations, avgInnerValue at nodes, one
// operator apply) are emitted too so that a mismatch in the rolled-back
// end value can be localised instead of hiding behind a LOOSE tolerance.
//
// Block M re-runs the 1-D and 2-D problems under a non-empty
// FdmSolverDesc::bcSet (FdmDirichletBoundary on one or both faces), which is
// what pins that the boundary-condition set reaches the scheme instead of
// being silently dropped.
//
// C++ parity (all @ v1.43, submodule 6b57206e0):
//   ql/methods/finitedifferences/solvers/fdm1dimsolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdm2dimsolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdm3dimsolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmndimsolver.hpp
//   ql/methods/finitedifferences/solvers/fdmblackscholessolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmsimple2dbssolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdm2dblackscholessolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmhestonsolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmbatessolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmcirsolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmg2solver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmhullwhitesolver.{hpp,cpp}
//   ql/methods/finitedifferences/solvers/fdmhestonhullwhitesolver.{hpp,cpp}

#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdm2dblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmg2op.hpp>
#include <ql/methods/finitedifferences/operators/fdmhestonhullwhiteop.hpp>
#include <ql/methods/finitedifferences/operators/fdmhestonop.hpp>
#include <ql/methods/finitedifferences/operators/fdmhullwhiteop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdm1dimsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdm2dblackscholessolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdm2dimsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdm3dimsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbatessolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmblackscholessolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmcirsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmg2solver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmhestonhullwhitesolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmhestonsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmhullwhitesolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmndimsolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmsimple2dbssolver.hpp>
#include <ql/methods/finitedifferences/solvers/fdmsolverdesc.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdirichletboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/processes/batesprocess.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/coxingersollrossprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/processes/hullwhiteprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
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

// Common market data (shared by every block).
struct Market {
    DayCounter dc;
    Date today;
    Handle<YieldTermStructure> rTS;
    Handle<YieldTermStructure> qTS;
    Handle<BlackVolTermStructure> volTS;
    Handle<Quote> s0;
};

Market makeMarket() {
    Market m;
    m.dc = Actual365Fixed();
    m.today = Date(15, January, 2024);
    m.rTS = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(m.today, 0.05, m.dc));
    m.qTS = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(m.today, 0.02, m.dc));
    m.volTS = Handle<BlackVolTermStructure>(
        ext::make_shared<BlackConstantVol>(m.today, NullCalendar(), 0.20, m.dc));
    m.s0 = Handle<Quote>(ext::make_shared<SimpleQuote>(100.0));
    return m;
}

ext::shared_ptr<GeneralizedBlackScholesProcess> makeBsProcess(const Market& m) {
    return ext::make_shared<BlackScholesMertonProcess>(m.s0, m.qTS, m.rTS, m.volTS);
}

ext::shared_ptr<FdmStepConditionComposite> emptyConditions() {
    FdmStepConditionComposite::Conditions stepConds;
    std::list<std::vector<Time> > stoppingTimes;
    return ext::make_shared<FdmStepConditionComposite>(stoppingTimes, stepConds);
}

// The six scheme descriptors the probe sweeps, with the suffix used in the
// emitted key names.
struct NamedScheme {
    const char* name;
    FdmSchemeDesc desc;
};

std::vector<NamedScheme> allSchemes() {
    return {{"douglas", FdmSchemeDesc::Douglas()},
            {"cn", FdmSchemeDesc::CrankNicolson()},
            {"implicit", FdmSchemeDesc::ImplicitEuler()},
            {"explicit", FdmSchemeDesc::ExplicitEuler()},
            {"craigsneyd", FdmSchemeDesc::CraigSneyd()},
            {"hundsdorfer", FdmSchemeDesc::Hundsdorfer()}};
}

// ---------------------------------------------------------------------------
// Block A: Fdm1DimSolver on a 1-D Black-Scholes operator.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdm1dimsolver(const Market& m) {
    const Real strike = 100.0;
    const Real xMin = std::log(50.0), xMax = std::log(150.0);
    const Size n = 25;

    auto m1 = ext::make_shared<Uniform1dMesher>(xMin, xMax, n);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    // --- cheap intermediates (TIGHT tier on the Python side) ---------------
    const Array& locs = mesher->locations(0);
    emit("a_mesher_loc_0", locs[0]);
    emit("a_mesher_loc_12", locs[12]);
    emit("a_mesher_loc_24", locs[24]);

    Array initialValues(mesher->layout()->size());
    for (const auto& iter : *mesher->layout()) {
        initialValues[iter.index()] = calc->avgInnerValue(iter, 1.0);
    }
    emit("a_init_0", initialValues[0]);
    emit("a_init_12", initialValues[12]);
    emit("a_init_18", initialValues[18]);
    emit("a_init_24", initialValues[24]);

    {
        FdmBlackScholesOp op(mesher, makeBsProcess(m), strike);
        op.setTime(0.0, 1.0);
        const Array applied = op.apply(initialValues);
        emit("a_op_apply_1", applied[1]);
        emit("a_op_apply_12", applied[12]);
        emit("a_op_apply_23", applied[23]);
    }

    // --- the solver, once per scheme ---------------------------------------
    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 0};

    const Real xNode = locs[12];
    emit("a_node_x", xNode);

    for (const auto& s : allSchemes()) {
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, makeBsProcess(m), strike);
        Fdm1DimSolver solver(desc, s.desc, op);
        const std::string p = std::string("a_") + s.name + "_";
        emit(p + "value_90", solver.interpolateAt(std::log(90.0)));
        emit(p + "value_100", solver.interpolateAt(std::log(100.0)));
        emit(p + "value_110", solver.interpolateAt(std::log(110.0)));
        emit(p + "value_node", solver.interpolateAt(xNode));
        emit(p + "theta_100", solver.thetaAt(std::log(100.0)));
        emit(p + "dx_100", solver.derivativeX(std::log(100.0)));
        emit(p + "dxx_100", solver.derivativeXX(std::log(100.0)));
    }
}

// ---------------------------------------------------------------------------
// Block B: FdmBlackScholesSolver (1-D wrapper).
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmblackscholessolver(const Market& m) {
    const Real strike = 100.0;
    auto m1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 25);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 0};

    for (const auto& s : allSchemes()) {
        FdmBlackScholesSolver solver(
            Handle<GeneralizedBlackScholesProcess>(makeBsProcess(m)),
            strike, desc, s.desc);
        const std::string p = std::string("b_") + s.name + "_";
        emit(p + "value_90", solver.valueAt(90.0));
        emit(p + "value_100", solver.valueAt(100.0));
        emit(p + "value_110", solver.valueAt(110.0));
        emit(p + "delta_100", solver.deltaAt(100.0));
        emit(p + "gamma_100", solver.gammaAt(100.0));
        emit(p + "theta_100", solver.thetaAt(100.0));
    }

    // Damping steps exercise the implicit-Euler smoothing branch.
    FdmSolverDesc damped = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 5};
    FdmBlackScholesSolver dampedSolver(
        Handle<GeneralizedBlackScholesProcess>(makeBsProcess(m)),
        strike, damped, FdmSchemeDesc::Douglas());
    emit("b_damped_value_100", dampedSolver.valueAt(100.0));
    emit("b_damped_delta_100", dampedSolver.deltaAt(100.0));
}

// ---------------------------------------------------------------------------
// Block C: FdmHullWhiteSolver (1-D, short-rate operator).
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmhullwhitesolver(const Market& m) {
    auto model = ext::make_shared<HullWhite>(m.rTS, 0.1, 0.01);
    auto m1 = ext::make_shared<Uniform1dMesher>(-0.05, 0.15, 21);
    auto mesher = ext::make_shared<FdmMesherComposite>(m1);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 1.0);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    emit("c_mesher_loc_0", mesher->locations(0)[0]);
    emit("c_mesher_loc_10", mesher->locations(0)[10]);
    {
        Array initialValues(mesher->layout()->size());
        for (const auto& iter : *mesher->layout()) {
            initialValues[iter.index()] = calc->avgInnerValue(iter, 1.0);
        }
        emit("c_init_0", initialValues[0]);
        emit("c_init_10", initialValues[10]);
        emit("c_init_20", initialValues[20]);
    }

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 50, 0};

    for (const auto& s : allSchemes()) {
        FdmHullWhiteSolver solver(Handle<HullWhite>(model), desc, s.desc);
        const std::string p = std::string("c_") + s.name + "_";
        emit(p + "value_0", solver.valueAt(0.0));
        emit(p + "value_005", solver.valueAt(0.05));
        emit(p + "value_01", solver.valueAt(0.1));
    }
}

// ---------------------------------------------------------------------------
// The shared 2-D Heston setup (blocks D, E, L).
// ---------------------------------------------------------------------------
struct HestonSetup {
    ext::shared_ptr<FdmMesherComposite> mesher;
    ext::shared_ptr<FdmInnerValueCalculator> calc;
    ext::shared_ptr<HestonProcess> process;
    FdmSolverDesc desc;
};

HestonSetup makeHestonSetup(const Market& m) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(60.0), std::log(160.0), 13);
    auto mv = ext::make_shared<Uniform1dMesher>(0.0, 0.4, 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mv);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);
    auto process = ext::make_shared<HestonProcess>(
        m.rTS, m.qTS, m.s0, 0.04, 2.0, 0.04, 0.4, -0.5);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 0.25, 400, 0};
    return {mesher, calc, process, desc};
}

// ---------------------------------------------------------------------------
// Block D: Fdm2DimSolver on FdmHestonOp.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdm2dimsolver(const Market& m) {
    const HestonSetup h = makeHestonSetup(m);

    emit("d_mesher_x_loc_0", h.mesher->locations(0)[0]);
    emit("d_mesher_x_loc_12", h.mesher->locations(0)[12]);
    emit("d_mesher_y_loc_0", h.mesher->locations(1)[0]);
    emit("d_mesher_y_loc_13", h.mesher->locations(1)[13]);
    {
        Array initialValues(h.mesher->layout()->size());
        for (const auto& iter : *h.mesher->layout()) {
            initialValues[iter.index()] = h.calc->avgInnerValue(iter, 0.25);
        }
        emit("d_init_0", initialValues[0]);
        emit("d_init_50", initialValues[50]);
        emit("d_init_100", initialValues[100]);

        FdmHestonOp op(h.mesher, h.process);
        op.setTime(0.0, 0.25);
        const Array applied = op.apply(initialValues);
        emit("d_op_apply_20", applied[20]);
        emit("d_op_apply_60", applied[60]);
    }

    const Real xNode = h.mesher->locations(0)[6];
    const Real yNode = h.mesher->locations(1)[13 * 2];
    emit("d_node_x", xNode);
    emit("d_node_y", yNode);

    for (const auto& s : allSchemes()) {
        auto op = ext::make_shared<FdmHestonOp>(h.mesher, h.process);
        Fdm2DimSolver solver(h.desc, s.desc, op);
        const std::string p = std::string("d_") + s.name + "_";
        emit(p + "value_100_004", solver.interpolateAt(std::log(100.0), 0.04));
        emit(p + "value_90_01", solver.interpolateAt(std::log(90.0), 0.10));
        emit(p + "value_node", solver.interpolateAt(xNode, yNode));
        emit(p + "theta_100_004", solver.thetaAt(std::log(100.0), 0.04));
        emit(p + "dx", solver.derivativeX(std::log(100.0), 0.04));
        emit(p + "dy", solver.derivativeY(std::log(100.0), 0.04));
        emit(p + "dxx", solver.derivativeXX(std::log(100.0), 0.04));
        emit(p + "dyy", solver.derivativeYY(std::log(100.0), 0.04));
        emit(p + "dxy", solver.derivativeXY(std::log(100.0), 0.04));
    }
}

// ---------------------------------------------------------------------------
// Block E: FdmHestonSolver.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmhestonsolver(const Market& m) {
    const HestonSetup h = makeHestonSetup(m);

    for (const auto& s : allSchemes()) {
        FdmHestonSolver solver(Handle<HestonProcess>(h.process), h.desc, s.desc);
        const std::string p = std::string("e_") + s.name + "_";
        emit(p + "value_100_004", solver.valueAt(100.0, 0.04));
        emit(p + "value_90_01", solver.valueAt(90.0, 0.10));
        emit(p + "delta", solver.deltaAt(100.0, 0.04));
        emit(p + "gamma", solver.gammaAt(100.0, 0.04));
        emit(p + "theta", solver.thetaAt(100.0, 0.04));
        emit(p + "mv_delta", solver.meanVarianceDeltaAt(100.0, 0.04));
        emit(p + "mv_gamma", solver.meanVarianceGammaAt(100.0, 0.04));
    }

    // Non-unit mixing factor changes the operator, not the solver plumbing —
    // pinned so the argument cannot be silently dropped.
    FdmHestonSolver mixed(Handle<HestonProcess>(h.process), h.desc,
                          FdmSchemeDesc::ExplicitEuler(),
                          Handle<FdmQuantoHelper>(),
                          ext::shared_ptr<LocalVolTermStructure>(), 0.75);
    emit("e_mixing075_value_100_004", mixed.valueAt(100.0, 0.04));
}

// ---------------------------------------------------------------------------
// Block F: FdmSimple2dBSSolver (1-direction operator on a 2-D mesh).
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmsimple2dbssolver(const Market& m) {
    const Real strike = 100.0;
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 21);
    auto ma = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 11);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, ma);
    // A max-basket payoff over both axes: FdmBlackScholesOp acts only on
    // direction 0, so a direction-0 calculator would leave the solution
    // constant in the second axis and the y-grid extraction untested.
    auto payoff = ext::make_shared<MaxBasketPayoff>(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, strike));
    auto calc = ext::make_shared<FdmLogBasketInnerValue>(payoff, mesher);

    emit("f_mesher_y_loc_21", mesher->locations(1)[21]);
    {
        Array initialValues(mesher->layout()->size());
        for (const auto& iter : *mesher->layout()) {
            initialValues[iter.index()] = calc->avgInnerValue(iter, 1.0);
        }
        emit("f_init_0", initialValues[0]);
        emit("f_init_100", initialValues[100]);
        emit("f_init_200", initialValues[200]);
    }

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 0};

    for (const auto& s : allSchemes()) {
        FdmSimple2dBSSolver solver(
            Handle<GeneralizedBlackScholesProcess>(makeBsProcess(m)),
            strike, desc, s.desc);
        const std::string p = std::string("f_") + s.name + "_";
        emit(p + "value_100_100", solver.valueAt(100.0, 100.0));
        emit(p + "value_90_110", solver.valueAt(90.0, 110.0));
        emit(p + "delta", solver.deltaAt(100.0, 100.0, 0.5));
        emit(p + "gamma", solver.gammaAt(100.0, 100.0, 0.5));
        emit(p + "theta", solver.thetaAt(100.0, 100.0));
    }
}

// ---------------------------------------------------------------------------
// Block G: Fdm2dBlackScholesSolver (genuine 2-asset operator).
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdm2dblackscholessolver(const Market& m) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(60.0), std::log(160.0), 13);
    auto my = ext::make_shared<Uniform1dMesher>(std::log(60.0), std::log(160.0), 13);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);
    // Rainbow (max-of-two) payoff: a single-direction calculator would keep
    // the solution constant along y and leave that axis untested.
    auto payoff = ext::make_shared<MaxBasketPayoff>(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0));
    auto calc = ext::make_shared<FdmLogBasketInnerValue>(payoff, mesher);

    emit("g_mesher_y_loc_13", mesher->locations(1)[13]);
    {
        Array initialValues(mesher->layout()->size());
        for (const auto& iter : *mesher->layout()) {
            initialValues[iter.index()] = calc->avgInnerValue(iter, 0.25);
        }
        emit("g_init_0", initialValues[0]);
        emit("g_init_84", initialValues[84]);
        emit("g_init_168", initialValues[168]);
    }

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 0.25, 300, 0};

    auto p1 = makeBsProcess(m);
    auto p2 = ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(ext::make_shared<SimpleQuote>(95.0)), m.qTS, m.rTS,
        Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(
            m.today, NullCalendar(), 0.25, m.dc)));

    for (const auto& s : allSchemes()) {
        Fdm2dBlackScholesSolver solver(
            Handle<GeneralizedBlackScholesProcess>(p1),
            Handle<GeneralizedBlackScholesProcess>(p2),
            0.30, desc, s.desc);
        const std::string p = std::string("g_") + s.name + "_";
        emit(p + "value_100_95", solver.valueAt(100.0, 95.0));
        emit(p + "value_110_90", solver.valueAt(110.0, 90.0));
        emit(p + "theta", solver.thetaAt(100.0, 95.0));
        emit(p + "delta_x", solver.deltaXat(100.0, 95.0));
        emit(p + "delta_y", solver.deltaYat(100.0, 95.0));
        emit(p + "gamma_x", solver.gammaXat(100.0, 95.0));
        emit(p + "gamma_y", solver.gammaYat(100.0, 95.0));
        emit(p + "gamma_xy", solver.gammaXYat(100.0, 95.0));
    }
}

// ---------------------------------------------------------------------------
// Block H: FdmCIRSolver.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmcirsolver(const Market& m) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(60.0), std::log(160.0), 13);
    auto mr = ext::make_shared<Uniform1dMesher>(0.01, 0.10, 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mr);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    auto cir = ext::make_shared<CoxIngersollRossProcess>(1.0, 0.1, 0.04, 0.04);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 0.25, 300, 0};

    for (const auto& s : allSchemes()) {
        FdmCIRSolver solver(Handle<CoxIngersollRossProcess>(cir),
                            Handle<GeneralizedBlackScholesProcess>(makeBsProcess(m)),
                            desc, s.desc, -0.3, 100.0);
        const std::string p = std::string("h_") + s.name + "_";
        emit(p + "value_100_004", solver.valueAt(100.0, 0.04));
        emit(p + "value_90_006", solver.valueAt(90.0, 0.06));
        emit(p + "delta", solver.deltaAt(100.0, 0.04));
        emit(p + "gamma", solver.gammaAt(100.0, 0.04));
        emit(p + "theta", solver.thetaAt(100.0, 0.04));
    }
}

// ---------------------------------------------------------------------------
// Block I: FdmG2Solver.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmg2solver(const Market& m) {
    auto model = ext::make_shared<G2>(m.rTS, 0.1, 0.01, 0.1, 0.01, -0.75);
    auto mx = ext::make_shared<Uniform1dMesher>(-0.1, 0.1, 11);
    auto my = ext::make_shared<Uniform1dMesher>(-0.1, 0.1, 11);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, my);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 1.0);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 0};

    for (const auto& s : allSchemes()) {
        FdmG2Solver solver(Handle<G2>(model), desc, s.desc);
        const std::string p = std::string("i_") + s.name + "_";
        emit(p + "value_0_0", solver.valueAt(0.0, 0.0));
        emit(p + "value_002_m003", solver.valueAt(0.02, -0.03));
    }
}

// ---------------------------------------------------------------------------
// Block J: FdmBatesSolver.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmbatessolver(const Market& m) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(60.0), std::log(160.0), 13);
    auto mv = ext::make_shared<Uniform1dMesher>(0.0, 0.4, 9);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mv);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    auto bates = ext::make_shared<BatesProcess>(
        m.rTS, m.qTS, m.s0, 0.04, 2.0, 0.04, 0.4, -0.5, 0.2, -0.1, 0.2);

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 0.25, 400, 0};

    for (const auto& s : allSchemes()) {
        FdmBatesSolver solver(Handle<BatesProcess>(bates), desc, s.desc);
        const std::string p = std::string("j_") + s.name + "_";
        emit(p + "value_100_004", solver.valueAt(100.0, 0.04));
        emit(p + "value_90_01", solver.valueAt(90.0, 0.10));
        emit(p + "delta", solver.deltaAt(100.0, 0.04));
        emit(p + "gamma", solver.gammaAt(100.0, 0.04));
        emit(p + "theta", solver.thetaAt(100.0, 0.04));
    }

    // Non-default integro integration order.
    FdmBatesSolver order8(Handle<BatesProcess>(bates), desc,
                          FdmSchemeDesc::ExplicitEuler(), 8);
    emit("j_order8_value_100_004", order8.valueAt(100.0, 0.04));
}

// ---------------------------------------------------------------------------
// Block K: Fdm3DimSolver + FdmHestonHullWhiteSolver.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_3dim(const Market& m) {
    auto mx = ext::make_shared<Uniform1dMesher>(std::log(70.0), std::log(140.0), 11);
    auto mv = ext::make_shared<Uniform1dMesher>(0.0, 0.3, 7);
    auto mr = ext::make_shared<Uniform1dMesher>(-0.05, 0.15, 5);
    auto mesher = ext::make_shared<FdmMesherComposite>(mx, mv, mr);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

    auto heston = ext::make_shared<HestonProcess>(
        m.rTS, m.qTS, m.s0, 0.04, 2.0, 0.04, 0.4, -0.5);
    auto hw = ext::make_shared<HullWhiteProcess>(m.rTS, 0.1, 0.01);

    emit("k_mesher_z_loc_0", mesher->locations(2)[0]);
    emit("k_mesher_z_loc_384", mesher->locations(2)[384]);
    {
        Array initialValues(mesher->layout()->size());
        for (const auto& iter : *mesher->layout()) {
            initialValues[iter.index()] = calc->avgInnerValue(iter, 0.25);
        }
        emit("k_init_0", initialValues[0]);
        emit("k_init_200", initialValues[200]);
    }

    FdmBoundaryConditionSet bcSet;
    FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 0.25, 400, 0};

    for (const auto& s : allSchemes()) {
        auto op = ext::make_shared<FdmHestonHullWhiteOp>(mesher, heston, hw, -0.2);
        Fdm3DimSolver solver(desc, s.desc, op);
        const std::string p = std::string("k3_") + s.name + "_";
        emit(p + "value", solver.interpolateAt(std::log(100.0), 0.04, 0.05));
        emit(p + "value_2", solver.interpolateAt(std::log(90.0), 0.10, 0.02));
        emit(p + "theta", solver.thetaAt(std::log(100.0), 0.04, 0.05));
    }

    for (const auto& s : allSchemes()) {
        FdmHestonHullWhiteSolver solver(Handle<HestonProcess>(heston),
                                        Handle<HullWhiteProcess>(hw),
                                        -0.2, desc, s.desc);
        const std::string p = std::string("k_") + s.name + "_";
        emit(p + "value", solver.valueAt(100.0, 0.04, 0.05));
        emit(p + "delta", solver.deltaAt(100.0, 0.04, 0.05, 1.0));
        emit(p + "gamma", solver.gammaAt(100.0, 0.04, 0.05, 1.0));
        emit(p + "theta", solver.thetaAt(100.0, 0.04, 0.05));
    }
}

// ---------------------------------------------------------------------------
// Block L: FdmNdimSolver<2> (multi-cubic interpolation instead of bicubic).
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_fdmndimsolver(const Market& m) {
    const HestonSetup h = makeHestonSetup(m);

    for (const auto& s : allSchemes()) {
        auto op = ext::make_shared<FdmHestonOp>(h.mesher, h.process);
        FdmNdimSolver<2> solver(h.desc, s.desc, op);
        const std::string p = std::string("l_") + s.name + "_";
        std::vector<Real> at1 = {std::log(100.0), 0.04};
        std::vector<Real> at2 = {std::log(90.0), 0.10};
        emit(p + "value_100_004", solver.interpolateAt(at1));
        emit(p + "value_90_01", solver.interpolateAt(at2));
        emit(p + "theta_100_004", solver.thetaAt(at1));
    }
}

// ---------------------------------------------------------------------------
// Block M: FdmSolverDesc::bcSet actually reaches the scheme.
//
// Every other block runs with an empty bcSet, which cannot distinguish a
// solver that threads solverDesc.bcSet through to FdmBackwardSolver from one
// that drops it on the floor. This block puts a Dirichlet condition on each
// face of the same 1-D and 2-D problems used by blocks A and D, so the
// answer moves iff the boundary set is honoured.
// ---------------------------------------------------------------------------
__attribute__((noinline))
void block_bcset(const Market& m) {
    // --- 1-D: knock the upper face of the log-spot mesh down to zero -------
    {
        const Real strike = 100.0;
        auto m1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 25);
        auto mesher = ext::make_shared<FdmMesherComposite>(m1);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);

        FdmBoundaryConditionSet bcSet;
        bcSet.push_back(ext::make_shared<FdmDirichletBoundary>(
            mesher, 0.0, 0, FdmDirichletBoundary::Upper));
        FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 0};

        for (const auto& s : allSchemes()) {
            auto op = ext::make_shared<FdmBlackScholesOp>(mesher, makeBsProcess(m), strike);
            Fdm1DimSolver solver(desc, s.desc, op);
            const std::string p = std::string("m1_") + s.name + "_";
            emit(p + "value_100", solver.interpolateAt(std::log(100.0)));
            emit(p + "value_140", solver.interpolateAt(std::log(140.0)));
            emit(p + "dx_100", solver.derivativeX(std::log(100.0)));
        }

        // Both faces pinned at once — two conditions in the set, so a port
        // that only ever applied bcSet[0] would miss the lower face.
        FdmBoundaryConditionSet both;
        both.push_back(ext::make_shared<FdmDirichletBoundary>(
            mesher, 0.0, 0, FdmDirichletBoundary::Lower));
        both.push_back(ext::make_shared<FdmDirichletBoundary>(
            mesher, 7.5, 0, FdmDirichletBoundary::Upper));
        FdmSolverDesc bothDesc = {mesher, both, emptyConditions(), calc, 1.0, 100, 0};
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, makeBsProcess(m), strike);
        Fdm1DimSolver solver(bothDesc, FdmSchemeDesc::Douglas(), op);
        emit("m1_both_value_100", solver.interpolateAt(std::log(100.0)));
        emit("m1_both_value_55", solver.interpolateAt(std::log(55.0)));
        emit("m1_both_value_145", solver.interpolateAt(std::log(145.0)));
    }

    // --- 2-D: pin the top of the variance axis of the Heston mesh ----------
    {
        const HestonSetup h = makeHestonSetup(m);
        FdmBoundaryConditionSet bcSet;
        bcSet.push_back(ext::make_shared<FdmDirichletBoundary>(
            h.mesher, 3.0, 1, FdmDirichletBoundary::Upper));
        FdmSolverDesc desc = {h.mesher, bcSet, emptyConditions(), h.calc, 0.25, 400, 0};

        for (const auto& s : allSchemes()) {
            auto op = ext::make_shared<FdmHestonOp>(h.mesher, h.process);
            Fdm2DimSolver solver(desc, s.desc, op);
            const std::string p = std::string("m2_") + s.name + "_";
            emit(p + "value_100_004", solver.interpolateAt(std::log(100.0), 0.04));
            emit(p + "value_90_03", solver.interpolateAt(std::log(90.0), 0.30));
            emit(p + "dy", solver.derivativeY(std::log(100.0), 0.04));
        }
    }

    // --- damping steps run under the same bcSet ---------------------------
    {
        const Real strike = 100.0;
        auto m1 = ext::make_shared<Uniform1dMesher>(std::log(50.0), std::log(150.0), 25);
        auto mesher = ext::make_shared<FdmMesherComposite>(m1);
        auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
        auto calc = ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0);
        FdmBoundaryConditionSet bcSet;
        bcSet.push_back(ext::make_shared<FdmDirichletBoundary>(
            mesher, 0.0, 0, FdmDirichletBoundary::Upper));
        FdmSolverDesc desc = {mesher, bcSet, emptyConditions(), calc, 1.0, 100, 5};
        auto op = ext::make_shared<FdmBlackScholesOp>(mesher, makeBsProcess(m), strike);
        Fdm1DimSolver solver(desc, FdmSchemeDesc::Douglas(), op);
        emit("m1_damped_value_100", solver.interpolateAt(std::log(100.0)));
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const Market m = makeMarket();

    block_fdm1dimsolver(m);
    block_fdmblackscholessolver(m);
    block_fdmhullwhitesolver(m);
    block_fdm2dimsolver(m);
    block_fdmhestonsolver(m);
    block_fdmsimple2dbssolver(m);
    block_fdm2dblackscholessolver(m);
    block_fdmcirsolver(m);
    block_fdmg2solver(m);
    block_fdmbatessolver(m);
    block_3dim(m);
    block_fdmndimsolver(m);
    block_bcset(m);

    // Trailing sentinel keeps every preceding entry comma-terminated.
    std::cout << "  \"_end\": 0\n}\n";
    return 0;
}
