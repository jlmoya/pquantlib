// migration-harness/cpp/probes/v143_experimental_extoufd/probe.cpp
//
// Reference values for the two extended-OU / Kluge FD engines @ v1.43:
//
//   FdExtOUJumpVanillaEngine   experimental/finitedifferences/fdextoujumpvanillaengine.hpp:38
//   FdKlugeExtOUSpreadEngine   experimental/finitedifferences/fdklugeextouspreadengine.hpp:41
//
// plus the two pieces they need that had never been exercised end to end:
//   ExtendedOrnsteinUhlenbeckProcess::expectation (all three Discretization
//   modes), and FdmStepConditionComposite::vanillaComposite.
//
// WHY THE PROCESS MOMENTS ARE PINNED FIRST
// ----------------------------------------
// Neither engine touches the payoff grid directly: FdmSimpleProcess1dMesher
// sizes direction 0 by calling process->evolve(0, x0, T, +-InvCDF(eps)), i.e.
// by ExtendedOrnsteinUhlenbeckProcess::expectation and ::stdDeviation. A port
// whose expectation is wrong produces a DIFFERENT MESH, and then every NPV
// below disagrees for a reason that has nothing to do with the engine. Section
// A therefore pins expectation/variance/stdDeviation directly, for all three
// Discretization values, with a non-constant b(t) so MidPoint, Trapezodial and
// GaussLobatto genuinely differ. (With constant b they coincide, which is
// exactly why a port can ship the wrong one and pass a constant-b test.)
//
// Note C++ spells the middle enumerator "Trapezodial"
// (extendedornsteinuhlenbeckprocess.hpp:44).
//
// GRID SIZES
// ----------
// Small on purpose. The C++ test-suite runs 25x200x50 (swingoption.cpp:226)
// and 5x200x50x20 (vpp.cpp:451); at those sizes the 3-D Python solver would
// dominate the test suite's runtime while proving nothing extra. The grids
// here are chosen so consecutive refinements move the NPV visibly -- see the
// *_conv triples -- which is a stronger check than one number: a port with a
// subtly wrong operator usually converges to a different limit.
//
// WHAT ELSE IS COVERED
// --------------------
//  * a shape curve on FdExtOUJumpVanillaEngine (the seasonal log-offset), and
//    the no-shape default, since the shape enters the INNER VALUE only;
//  * European and Bermudan exercise, so vanillaComposite's exercise dispatch
//    and its stopping-time contribution are both exercised;
//  * put as well as call;
//  * all four FdmSchemeDesc variants the engines are likely to be driven with
//    (Hundsdorfer default, Douglas, CraigSneyd, ModifiedCraigSneyd), because
//    the scheme is a constructor argument and a port that ignores it would
//    otherwise pass;
//  * for the spread engine: a genuine spark spread (weights 1, -heatRate) and
//    a reversed-sign basket, plus a case with BOTH shapes supplied.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/extoufd.json.

#include <ql/exercise.hpp>
#include <ql/experimental/finitedifferences/fdextoujumpvanillaengine.hpp>
#include <ql/experimental/finitedifferences/fdklugeextouspreadengine.hpp>
#include <ql/experimental/processes/extendedornsteinuhlenbeckprocess.hpp>
#include <ql/experimental/processes/extouwithjumpsprocess.hpp>
#include <ql/experimental/processes/klugeextouprocess.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

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

void emit_arr(const std::string& name, const std::vector<Real>& v) {
    sep();
    std::cout << "  \"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setprecision(17) << v[i];
    }
    std::cout << "]";
}

const Date TODAY(15, January, 2024);

// ==========================================================================
// SECTION A -- ExtendedOrnsteinUhlenbeckProcess moments
// ==========================================================================

// A deliberately non-constant, non-linear mean. Linear b makes Trapezodial
// exact and equal to GaussLobatto; constant b makes all three coincide.
Real bCurve(Real t) { return 0.75 + 0.4 * std::sin(2.0 * t) + 0.1 * t * t; }

void sectionProcess() {
    const Real speed = 1.6;
    const Real vol = 0.85;
    const Real x0 = 0.35;
    emit("proc_speed", speed);
    emit("proc_vol", vol);
    emit("proc_x0", x0);

    const std::vector<std::pair<Time, Time> > steps = {
        {0.0, 0.25}, {0.0, 1.0}, {0.5, 0.5}, {1.25, 2.0}, {0.0, 5.0}};
    std::vector<Real> t0s, dts;
    for (const auto& s : steps) {
        t0s.push_back(s.first);
        dts.push_back(s.second);
    }
    emit_arr("proc_t0", t0s);
    emit_arr("proc_dt", dts);

    // b(t) sampled on a grid, so a port can check the curve itself first.
    std::vector<Real> bts, bvals;
    for (Real t = 0.0; t <= 5.0 + 1e-12; t += 0.25) {
        bts.push_back(t);
        bvals.push_back(bCurve(t));
    }
    emit_arr("proc_b_t", bts);
    emit_arr("proc_b_value", bvals);

    struct Mode {
        const char* tag;
        ExtendedOrnsteinUhlenbeckProcess::Discretization d;
    };
    const Mode modes[] = {
        {"midpoint", ExtendedOrnsteinUhlenbeckProcess::MidPoint},
        {"trapezodial", ExtendedOrnsteinUhlenbeckProcess::Trapezodial},
        {"gausslobatto", ExtendedOrnsteinUhlenbeckProcess::GaussLobatto}};

    for (const auto& m : modes) {
        ExtendedOrnsteinUhlenbeckProcess p(speed, vol, x0, bCurve, m.d);
        std::vector<Real> exp_, var_, sd_, drift_, diff_;
        for (const auto& s : steps) {
            exp_.push_back(p.expectation(s.first, x0, s.second));
            var_.push_back(p.variance(s.first, x0, s.second));
            sd_.push_back(p.stdDeviation(s.first, x0, s.second));
            drift_.push_back(p.drift(s.first, x0));
            diff_.push_back(p.diffusion(s.first, x0));
        }
        emit_arr(std::string("proc_") + m.tag + "_expectation", exp_);
        emit_arr(std::string("proc_") + m.tag + "_variance", var_);
        emit_arr(std::string("proc_") + m.tag + "_stdDeviation", sd_);
        emit_arr(std::string("proc_") + m.tag + "_drift", drift_);
        emit_arr(std::string("proc_") + m.tag + "_diffusion", diff_);
    }

    // A constant-b control: all three modes MUST agree here. Pinned so a port
    // can localise a failure to "the b integral" rather than "the OU part".
    {
        ExtendedOrnsteinUhlenbeckProcess pm(
            speed, vol, x0, [](Real) { return 0.6; },
            ExtendedOrnsteinUhlenbeckProcess::MidPoint);
        ExtendedOrnsteinUhlenbeckProcess pt(
            speed, vol, x0, [](Real) { return 0.6; },
            ExtendedOrnsteinUhlenbeckProcess::Trapezodial);
        ExtendedOrnsteinUhlenbeckProcess pg(
            speed, vol, x0, [](Real) { return 0.6; },
            ExtendedOrnsteinUhlenbeckProcess::GaussLobatto);
        emit("proc_constb_midpoint", pm.expectation(0.3, x0, 1.4));
        emit("proc_constb_trapezodial", pt.expectation(0.3, x0, 1.4));
        emit("proc_constb_gausslobatto", pg.expectation(0.3, x0, 1.4));
    }
}

// ==========================================================================
// SECTION B -- FdExtOUJumpVanillaEngine
// ==========================================================================

typedef FdExtOUJumpVanillaEngine::Shape Shape;

// The Kluge power process from the C++ test-suite (swingoption.cpp:53-74),
// with the same numbers.
ext::shared_ptr<ExtOUWithJumpsProcess> createKlugeProcess() {
    const Real x0 = 3.0;
    const Real y0 = 0.0;
    const Real beta = 5.0;
    const Real eta = 2.0;
    const Real jumpIntensity = 1.0;
    const Real speed = 1.0;
    const Real volatility = 2.0;

    auto ouProcess = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        speed, volatility, x0, [x0](Real) { return x0; });
    return ext::make_shared<ExtOUWithJumpsProcess>(
        ouProcess, y0, beta, jumpIntensity, eta);
}

// Two things about this curve are deliberate, and both cost a probe revision.
//
//  1. The +0.15 offset. A first draft used a pure sin(2*pi*t), whose value at
//     t = 1 is -2.4e-16; a European option evaluates the inner value ONLY at
//     maturity, so that shape moved the NPV by nothing and the probe reported
//     the no-shape number to the last bit. A shape case a shape-ignoring port
//     passes is not a shape case.
//
//  2. The knots run to t = 1.5, past every maturity used below. C++
//     FdmExtOUJumpModelInnerValue::innerValue
//     (fdmextoujumpmodelinnervalue.hpp:46-56) does
//         std::lower_bound(shape->begin(), shape->end(),
//                          pair(t - sqrt(QL_EPSILON), 0.0))->second
//     with NO end() check. The European maturity here is 15-Jan-2025 from a
//     15-Jan-2024 reference under Actual365Fixed, i.e. t = 366/365 = 1.00274 —
//     past a curve that stopped at 1.0. lower_bound then returns end() and the
//     dereference reads past the vector. The second draft did exactly that and
//     the "shape" case printed the no-shape NPV bit for bit, because the slack
//     it read happened to be 0.0. Undefined behaviour yields no reference
//     value, so the curve is extended instead.
ext::shared_ptr<Shape> seasonalShape() {
    auto shape = ext::make_shared<Shape>();
    for (Size i = 0; i <= 18; ++i) {
        const Time t = i / 12.0;
        shape->emplace_back(t, 0.15 + 0.25 * std::sin(2.0 * M_PI * t));
    }
    return shape;
}

void emitJumpEngineCase(const std::string& tag,
                        Option::Type type,
                        Real strike,
                        const ext::shared_ptr<Exercise>& exercise,
                        Size tGrid,
                        Size xGrid,
                        Size yGrid,
                        const ext::shared_ptr<Shape>& shape,
                        const FdmSchemeDesc& schemeDesc,
                        Rate irRate) {
    auto process = createKlugeProcess();
    auto rTS = ext::make_shared<FlatForward>(TODAY, irRate, Actual365Fixed());

    VanillaOption option(ext::make_shared<PlainVanillaPayoff>(type, strike),
                         exercise);
    option.setPricingEngine(ext::make_shared<FdExtOUJumpVanillaEngine>(
        process, rTS, tGrid, xGrid, yGrid, shape, schemeDesc));

    emit(tag + "_NPV", option.NPV());
    emit_int(tag + "_tGrid", (long long)tGrid);
    emit_int(tag + "_xGrid", (long long)xGrid);
    emit_int(tag + "_yGrid", (long long)yGrid);
    emit_int(tag + "_type", (long long)type);
    emit(tag + "_strike", strike);
    emit(tag + "_irRate", irRate);
    emit_int(tag + "_hasShape", shape ? 1 : 0);
}

void sectionJumpEngine() {
    const Date maturity(15, January, 2025);
    auto european = ext::make_shared<EuropeanExercise>(maturity);

    // B1: the base European call, and two refinements of the same case so the
    // convergence direction is pinned as well as the value.
    emitJumpEngineCase("ext_call_c0", Option::Call, 30.0, european, 5, 20, 10,
                       ext::shared_ptr<Shape>(), FdmSchemeDesc::Hundsdorfer(),
                       0.1);
    emitJumpEngineCase("ext_call_c1", Option::Call, 30.0, european, 10, 40, 15,
                       ext::shared_ptr<Shape>(), FdmSchemeDesc::Hundsdorfer(),
                       0.1);
    emitJumpEngineCase("ext_call_c2", Option::Call, 30.0, european, 20, 60, 20,
                       ext::shared_ptr<Shape>(), FdmSchemeDesc::Hundsdorfer(),
                       0.1);

    // B2: put, deep OTM strike, zero rate.
    emitJumpEngineCase("ext_put", Option::Put, 15.0, european, 10, 40, 15,
                       ext::shared_ptr<Shape>(), FdmSchemeDesc::Hundsdorfer(),
                       0.0);

    // B3: with a seasonal shape curve -- the inner value becomes
    // payoff(exp(shape(t) + x + y)), so this must move the NPV.
    emitJumpEngineCase("ext_shape", Option::Call, 30.0, european, 10, 40, 15,
                       seasonalShape(), FdmSchemeDesc::Hundsdorfer(), 0.1);

    // B4: the scheme is a constructor argument; four of them.
    emitJumpEngineCase("ext_douglas", Option::Call, 30.0, european, 10, 40, 15,
                       ext::shared_ptr<Shape>(), FdmSchemeDesc::Douglas(), 0.1);
    emitJumpEngineCase("ext_craigsneyd", Option::Call, 30.0, european, 10, 40,
                       15, ext::shared_ptr<Shape>(),
                       FdmSchemeDesc::CraigSneyd(), 0.1);
    emitJumpEngineCase("ext_modcraigsneyd", Option::Call, 30.0, european, 10,
                       40, 15, ext::shared_ptr<Shape>(),
                       FdmSchemeDesc::ModifiedCraigSneyd(), 0.1);

    // B5: Bermudan exercise -- reaches vanillaComposite's Bermudan branch,
    // which contributes both a step condition AND stopping times. The second
    // variant adds the shape, so the inner value is sampled at four DIFFERENT
    // times and the shape INTERPOLATION is exercised, not just its endpoint.
    {
        std::vector<Date> dates = {Date(15, April, 2024), Date(15, July, 2024),
                                   Date(15, October, 2024), maturity};
        std::vector<Real> serials;
        for (const auto& d : dates)
            serials.push_back(Real(d.serialNumber()));
        emit_arr("ext_bermudan_dates", serials);
        emitJumpEngineCase("ext_bermudan", Option::Call, 30.0,
                           ext::make_shared<BermudanExercise>(dates), 10, 40,
                           15, ext::shared_ptr<Shape>(),
                           FdmSchemeDesc::Hundsdorfer(), 0.1);
        emitJumpEngineCase("ext_bermudan_shape", Option::Call, 30.0,
                           ext::make_shared<BermudanExercise>(dates), 10, 40,
                           15, seasonalShape(), FdmSchemeDesc::Hundsdorfer(),
                           0.1);
    }

    // B6: the shape curve's own knots, so a port can check the curve before
    // blaming the engine.
    {
        auto shape = seasonalShape();
        std::vector<Real> ts, vs;
        for (const auto& kv : *shape) {
            ts.push_back(kv.first);
            vs.push_back(kv.second);
        }
        emit_arr("ext_shape_t", ts);
        emit_arr("ext_shape_value", vs);
    }

    emit_int("ext_maturity_serial", (long long)maturity.serialNumber());
}

// ==========================================================================
// SECTION C -- FdKlugeExtOUSpreadEngine
// ==========================================================================

typedef FdKlugeExtOUSpreadEngine::GasShape GasShape;
typedef FdKlugeExtOUSpreadEngine::PowerShape PowerShape;

// vpp.cpp:205-234, same numbers.
ext::shared_ptr<KlugeExtOUProcess> createKlugeExtOUProcess() {
    const Real beta = 200;
    const Real eta = 1.0 / 0.2;
    const Real lambda = 4.0;
    const Real alpha = 7.0;
    const Real volatility_x = 1.4;
    const Real kappa = 4.45;
    const Real volatility_u = std::sqrt(1.3);
    const Real rho = 0.7;

    const Real x0 = 0.0;
    const Real y0 = 0.0;
    const Real u = 0.0;

    auto ouProcess = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        alpha, volatility_x, x0, [x0](Real) { return x0; });
    auto lnPowerProcess = ext::make_shared<ExtOUWithJumpsProcess>(
        ouProcess, y0, beta, lambda, eta);
    auto lnGasProcess = ext::make_shared<ExtendedOrnsteinUhlenbeckProcess>(
        kappa, volatility_u, u, [u](Real) { return u; });
    return ext::make_shared<KlugeExtOUProcess>(rho, lnPowerProcess,
                                               lnGasProcess);
}

void emitSpreadCase(const std::string& tag,
                    Real heatRate,
                    Size tGrid,
                    Size xGrid,
                    Size yGrid,
                    Size uGrid,
                    const ext::shared_ptr<GasShape>& gasShape,
                    const ext::shared_ptr<PowerShape>& powerShape,
                    const FdmSchemeDesc& schemeDesc,
                    const Date& maturityDate,
                    Rate irRate) {
    auto klugeOUProcess = createKlugeExtOUProcess();
    auto rTS = ext::make_shared<FlatForward>(TODAY, irRate, Actual365Fixed());

    Array spreadFactors(2);
    spreadFactors[0] = 1.0;
    spreadFactors[1] = -heatRate;
    auto basketPayoff = ext::make_shared<AverageBasketPayoff>(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 0.0), spreadFactors);

    BasketOption option(basketPayoff,
                        ext::make_shared<EuropeanExercise>(maturityDate));
    option.setPricingEngine(ext::make_shared<FdKlugeExtOUSpreadEngine>(
        klugeOUProcess, rTS, tGrid, xGrid, yGrid, uGrid, gasShape, powerShape,
        schemeDesc));

    emit(tag + "_NPV", option.NPV());
    emit(tag + "_heatRate", heatRate);
    emit_int(tag + "_tGrid", (long long)tGrid);
    emit_int(tag + "_xGrid", (long long)xGrid);
    emit_int(tag + "_yGrid", (long long)yGrid);
    emit_int(tag + "_uGrid", (long long)uGrid);
    emit(tag + "_irRate", irRate);
    emit_int(tag + "_maturity_serial", (long long)maturityDate.serialNumber());
    emit_int(tag + "_hasShapes", (gasShape || powerShape) ? 1 : 0);
}

// Knots to t = 1.5 for the same reason as seasonalShape: the longest maturity
// below is 366/365 years out, and C++ dereferences lower_bound's result
// unconditionally.
ext::shared_ptr<GasShape> flatShape(Real level) {
    auto shape = ext::make_shared<GasShape>();
    for (Size i = 0; i <= 6; ++i)
        shape->emplace_back(i / 4.0, level);
    return shape;
}

void sectionSpreadEngine() {
    const Date maturity(15, April, 2024);

    // C1: base spark spread + two refinements.
    emitSpreadCase("spr_c0", 2.0, 3, 10, 5, 6, ext::shared_ptr<GasShape>(),
                   ext::shared_ptr<PowerShape>(), FdmSchemeDesc::Hundsdorfer(),
                   maturity, 0.0);
    emitSpreadCase("spr_c1", 2.0, 5, 16, 6, 8, ext::shared_ptr<GasShape>(),
                   ext::shared_ptr<PowerShape>(), FdmSchemeDesc::Hundsdorfer(),
                   maturity, 0.0);
    emitSpreadCase("spr_c2", 2.0, 5, 24, 8, 12, ext::shared_ptr<GasShape>(),
                   ext::shared_ptr<PowerShape>(), FdmSchemeDesc::Hundsdorfer(),
                   maturity, 0.0);

    // C2: reversed weights -- gas minus power. Different payoff branch.
    emitSpreadCase("spr_neg", -0.5, 5, 16, 6, 8, ext::shared_ptr<GasShape>(),
                   ext::shared_ptr<PowerShape>(), FdmSchemeDesc::Hundsdorfer(),
                   maturity, 0.0);

    // C3: both shapes supplied (constant levels, different per leg), and a
    // non-zero discount rate.
    emitSpreadCase("spr_shapes", 2.0, 5, 16, 6, 8, flatShape(0.3),
                   flatShape(-0.2), FdmSchemeDesc::Hundsdorfer(), maturity,
                   0.05);

    // C4: alternative schemes.
    emitSpreadCase("spr_douglas", 2.0, 5, 16, 6, 8, ext::shared_ptr<GasShape>(),
                   ext::shared_ptr<PowerShape>(), FdmSchemeDesc::Douglas(),
                   maturity, 0.0);
    emitSpreadCase("spr_craigsneyd", 2.0, 5, 16, 6, 8,
                   ext::shared_ptr<GasShape>(), ext::shared_ptr<PowerShape>(),
                   FdmSchemeDesc::CraigSneyd(), maturity, 0.0);

    // C5: longer maturity.
    emitSpreadCase("spr_long", 2.0, 5, 16, 6, 8, ext::shared_ptr<GasShape>(),
                   ext::shared_ptr<PowerShape>(), FdmSchemeDesc::Hundsdorfer(),
                   Date(15, January, 2025), 0.0);
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = TODAY;
    std::cout << "{\n";
    emit_int("evaluationDate_serial", (long long)TODAY.serialNumber());
    sectionProcess();
    sectionJumpEngine();
    sectionSpreadEngine();
    std::cout << "\n}\n";
    return 0;
}
