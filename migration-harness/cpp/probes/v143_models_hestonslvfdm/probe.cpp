// migration-harness/cpp/probes/v143_models_hestonslvfdm/probe.cpp
//
// Reference values for @ v1.43:
//
//   HestonSLVFDMModel             ql/models/equity/hestonslvfdmmodel.hpp:73
//   HestonSLVFDMModel::LogEntry   ql/models/equity/hestonslvfdmmodel.hpp:87
//   HestonSLVFokkerPlanckFdmParams ql/models/equity/hestonslvfdmmodel.hpp:43
//
// THE FIXTURE
// -----------
// The C++ test-suite drives this model off a NoExceptLocalVolSurface built on
// a Bicubic-interpolated BlackVarianceSurface (hestonslvmodel.cpp:1555-1600).
// Reproducing that surface exactly in a second language means reproducing the
// whole implied-vol stack before anything about the SLV model is being
// tested. This probe instead feeds a FixedLocalVolSurface whose matrix is
// written out in closed form, so both languages start from bit-identical
// local vols and any disagreement is the model's.
//
// A CONFIGURATION THAT ACTUALLY CONVERGES
// ---------------------------------------
// localVolEpsProb is 1e-3, not the 1e-4 the C++ test-suite uses, and that is
// load-bearing rather than cosmetic. performCalculations seeds vMesher[0] with
// the DEGENERATE Predefined1dMesher(vector<Real>(vGrid, v0)) -- every node at
// v0, zero spacing -- and only replaces it at a LocalVolRNDCalculator rescale
// step (hestonslvfdmmodel.cpp:360-376). If the first rescale step is not 1,
// vMesher[1] stays degenerate, and the reshapePDF<Bilinear> at the first
// rescale interpolates FROM that mesh: Bilinear's yMin equals its yMax, every
// target v falls outside [v0, v0], the guard at line 183 zeroes the whole
// density, and rescalePDF then divides by zero. Everything downstream is NaN
// and BiCGstab reports "could not converge".
//
// This is not hypothetical. A sweep of 72 (xGrid, vGrid, tMaxStepsPerYear,
// tMinStepsPerYear, tStepNumberDecay, localVolEpsProb) combinations against
// this fixture found the correspondence exact: every configuration whose
// rescaleTimeSteps() begins at 1 calibrates, every configuration whose
// rescaleTimeSteps() begins at 2 throws "could not converge". localVolEpsProb
// controls it, because it is the threshold the step-1 probability-leak test
// compares against (localvolrndcalculator.cpp:273).
//
// So this probe pins a configuration inside the model's working envelope, and
// section A pins rescaleTimeSteps()[0] == 1 explicitly so a port cannot drift
// out of it without the test saying so.
//
// WHY SECTION A EXISTS
// --------------------
// performCalculations builds one variance mesher per RESCALE step of the
// LocalVolRNDCalculator (hestonslvfdmmodel.cpp:360-376) and reuses the
// previous mesher otherwise. Index 1 is special: unless the local-vol
// calculator rescaled at step 1, vMesher[1] is the degenerate
// Predefined1dMesher(vector<Real>(vGrid, v0)) seeded at index 0 — every node
// at v0, zero spacing. Both FdmHestonFwdOp (through FirstDerivativeOp) and
// reshapePDF (through Bilinear, whose yMin == yMax) then operate on a
// zero-width axis.
//
// Section A pins rescaleTimeSteps() and the endpoints of the first few
// meshers so a port can tell whether it is reproducing the C++ MESH before it
// starts comparing leverage values. A port whose LocalVolRNDCalculator
// rescales at different steps builds a different mesh sequence, and then
// nothing downstream can agree.
//
// GRID SIZES
// ----------
// Deliberately small (xGrid 21, vGrid 21, one year). The C++ test uses
// 51x151 over five years; at that size a Python port of the same algorithm
// would dominate a test suite's runtime without testing anything the small
// grid does not.
//
// Emits JSON on stdout; redirect to
// references/v143/models/hestonslvfdm.json.

#include <ql/math/integrals/discreteintegrals.hpp>
#include <ql/methods/finitedifferences/utilities/localvolrndcalculator.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/models/equity/hestonslvfdmmodel.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/fixedlocalvolsurface.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>

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

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
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

const Date TODAY(15, January, 2024);
const Date FINAL_DATE(15, January, 2025);

const Real S0 = 100.0;
const Rate R = 0.035;
const Rate Q = 0.01;
const Real KAPPA = 2.0;
const Real THETA = 0.09;
const Real RHO = -0.5;
const Real SIGMA = 0.4;
const Real V0 = 0.09;

const Size X_GRID = 31;
const Size V_GRID = 41;

// Closed-form local-vol matrix so both languages start from identical inputs.
Real localVolValue(Real k, Time t) {
    return 0.30 + 0.10 * (100.0 / k - 1.0) - 0.02 * t;
}

const std::vector<Time> LV_TIMES = {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};
const std::vector<Real> LV_STRIKES = {50.0, 75.0, 90.0, 100.0, 110.0, 130.0, 170.0};

ext::shared_ptr<FixedLocalVolSurface> makeLocalVol(const DayCounter& dc) {
    auto m = ext::make_shared<Matrix>(LV_STRIKES.size(), LV_TIMES.size());
    for (Size i = 0; i < LV_STRIKES.size(); ++i)
        for (Size j = 0; j < LV_TIMES.size(); ++j)
            (*m)[i][j] = localVolValue(LV_STRIKES[i], LV_TIMES[j]);
    auto lv = ext::make_shared<FixedLocalVolSurface>(
        TODAY, LV_TIMES, LV_STRIKES, m, dc);
    lv->enableExtrapolation(true);
    return lv;
}

struct Fixture {
    DayCounter dc;
    Handle<Quote> spot;
    Handle<YieldTermStructure> rTS, qTS;
    Handle<LocalVolTermStructure> localVol;
    Handle<HestonModel> hestonModel;
};

Fixture makeFixture() {
    Fixture f;
    f.dc = Actual365Fixed();
    f.spot = Handle<Quote>(ext::make_shared<SimpleQuote>(S0));
    f.rTS = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(TODAY, R, f.dc));
    f.qTS = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(TODAY, Q, f.dc));
    f.localVol = Handle<LocalVolTermStructure>(makeLocalVol(f.dc));
    auto process = ext::make_shared<HestonProcess>(
        f.rTS, f.qTS, f.spot, V0, KAPPA, THETA, SIGMA, RHO);
    f.hestonModel = Handle<HestonModel>(ext::make_shared<HestonModel>(process));
    return f;
}

HestonSLVFokkerPlanckFdmParams makeParams(
    Size predictionCorrectionSteps,
    FdmSquareRootFwdOp::TransformationType trafoType,
    FdmHestonGreensFct::Algorithm greens,
    const FdmSchemeDesc& schemeDesc) {
    HestonSLVFokkerPlanckFdmParams p = {
        X_GRID, V_GRID,
        /*tMaxStepsPerYear*/ 100, /*tMinStepsPerYear*/ 25,
        /*tStepNumberDecay*/ 100.0,
        /*nRannacherTimeSteps*/ 2,
        predictionCorrectionSteps,
        /*x0Density*/ 0.1, /*localVolEpsProb*/ 1e-3,
        /*maxIntegrationIterations*/ 10000,
        /*vLowerEps*/ 1e-5, /*vUpperEps*/ 1e-5, /*vMin*/ 0.0000025,
        /*v0Density*/ 1.0, /*vLowerBoundDensity*/ 0.1,
        /*vUpperBoundDensity*/ 0.9,
        /*leverageFctPropEps*/ 1e-5,
        greens, trafoType, schemeDesc};
    return p;
}

// ==========================================================================
// SECTION A -- the mesh sequence performCalculations is built on
// ==========================================================================

void sectionMesh(const Fixture& f) {
    const HestonSLVFokkerPlanckFdmParams params = makeParams(
        2, FdmSquareRootFwdOp::Log, FdmHestonGreensFct::ZeroCorrelation,
        FdmSchemeDesc::ModifiedCraigSneyd());

    const Time T = f.dc.yearFraction(TODAY, FINAL_DATE);
    emit("mesh_T", T);

    const Time maxDt = 1.0 / params.tMaxStepsPerYear;
    const Time minDt = 1.0 / params.tMinStepsPerYear;
    Time tIdx = 0.0;
    std::vector<Time> times(1, tIdx);
    while (tIdx < T) {
        const Real decayFactor = std::exp(-params.tStepNumberDecay * tIdx);
        const Time dt = maxDt * decayFactor + minDt * (1.0 - decayFactor);
        times.push_back(std::min(T, tIdx += dt));
    }
    emit_arr("mesh_raw_times", times);

    const auto timeGrid = ext::make_shared<TimeGrid>(times.begin(), times.end());
    std::vector<Real> gridTimes;
    for (Size i = 0; i < timeGrid->size(); ++i)
        gridTimes.push_back(timeGrid->at(i));
    emit_arr("mesh_grid_times", gridTimes);
    emit_int("mesh_grid_size", (long long)timeGrid->size());

    const LocalVolRNDCalculator localVolRND(
        f.spot.currentLink(), f.rTS.currentLink(), f.qTS.currentLink(),
        f.localVol.currentLink(), timeGrid, X_GRID, params.x0Density,
        params.localVolEpsProb, params.maxIntegrationIterations);

    const std::vector<Size> rescaleSteps = localVolRND.rescaleTimeSteps();
    std::vector<long long> rs;
    for (Size s : rescaleSteps)
        rs.push_back((long long)s);
    emit_iarr("mesh_rescale_steps", rs);

    // The x-mesher endpoints at each time step: the cheapest complete summary
    // of the mesh a port must reproduce before any density can agree.
    std::vector<Real> xFront, xBack;
    for (Size i = 0; i < timeGrid->size(); ++i) {
        const auto m = localVolRND.mesher(timeGrid->at(i));
        xFront.push_back(m->locations().front());
        xBack.push_back(m->locations().back());
    }
    emit_arr("mesh_x_front", xFront);
    emit_arr("mesh_x_back", xBack);

    // Full locations of the mesher at step 1, where the SLV model starts.
    {
        const auto m = localVolRND.mesher(timeGrid->at(1));
        std::vector<Real> locs(m->locations().begin(), m->locations().end());
        emit_arr("mesh_x1_locations", locs);
    }

    // The local-vol inputs, so a port can prove the surface agrees first.
    std::vector<Real> lvProbe;
    for (Time t : {0.0, 0.1, 0.25, 0.5, 1.0}) {
        for (Real k : {60.0, 80.0, 100.0, 125.0, 160.0}) {
            lvProbe.push_back(f.localVol->localVol(t, k, true));
        }
    }
    emit_arr("mesh_localVol_probe", lvProbe);
}

// ==========================================================================
// SECTION B -- the calibrated leverage function
// ==========================================================================

const std::vector<Time> L_TIMES = {0.05, 0.15, 0.3, 0.5, 0.75, 1.0};
const std::vector<Real> L_SPOTS = {70.0, 85.0, 100.0, 115.0, 140.0};

void emitLeverage(const std::string& tag,
                  const Fixture& f,
                  const HestonSLVFokkerPlanckFdmParams& params) {
    const HestonSLVFDMModel model(f.localVol, f.hestonModel, FINAL_DATE, params,
                                  /*logging*/ false);
    std::vector<Real> vals;
    std::string err;
    try {
        const auto leverage = model.leverageFunction();
        for (Time t : L_TIMES)
            for (Real s : L_SPOTS)
                vals.push_back(leverage->localVol(t, s, true));
        emit("mesh_maxTime_" + tag, leverage->maxTime());
    } catch (const std::exception& e) {
        err = e.what();
    }
    emit_arr(tag + "_leverage", vals);
    emit_str(tag + "_error", err);
}

void sectionLeverage(const Fixture& f) {
    emit_arr("lev_times", L_TIMES);
    emit_arr("lev_spots", L_SPOTS);

    // B1: the calibration proper, two predictor-corrector steps.
    emitLeverage("lev_log_pc2", f,
                 makeParams(2, FdmSquareRootFwdOp::Log,
                            FdmHestonGreensFct::ZeroCorrelation,
                            FdmSchemeDesc::ModifiedCraigSneyd()));

    // NOT PINNED: predictionCorretionSteps == 0. The inner loop body then never
    // executes, so no leverage column past index 1 is ever written -- and the
    // matrix those columns live in is `new Matrix(xGrid, timeGrid->size())`
    // (hestonslvfdmmodel.cpp:386), which QuantLib allocates with
    // `new Real[n]` and does NOT value-initialise. Only columns 0 and 1 get
    // filled (lines 389-390). Reading the surface past the second time point
    // therefore reads uninitialised memory: a draft of this probe pinned that
    // case and consecutive runs of the same binary printed
    // [0, 0, ..., 0] and then
    // [0.5017167319402513, 0.46517054407727826, 1.004168382136165, ...].
    // Undefined behaviour yields no reference value.

    // B3: the Plain transformation, and B4 Power.
    emitLeverage("lev_plain_pc2", f,
                 makeParams(2, FdmSquareRootFwdOp::Plain,
                            FdmHestonGreensFct::Gaussian,
                            FdmSchemeDesc::ModifiedCraigSneyd()));
    emitLeverage("lev_power_pc2", f,
                 makeParams(2, FdmSquareRootFwdOp::Power,
                            FdmHestonGreensFct::Gaussian,
                            FdmSchemeDesc::ModifiedCraigSneyd()));

    // B5: a different FD scheme, to prove schemeDesc is honoured. Note the
    // first nRannacherTimeSteps steps use implicit Euler regardless.
    emitLeverage("lev_log_hundsdorfer", f,
                 makeParams(2, FdmSquareRootFwdOp::Log,
                            FdmHestonGreensFct::ZeroCorrelation,
                            FdmSchemeDesc::Hundsdorfer()));
}

// ==========================================================================
// SECTION C -- LogEntry
// ==========================================================================

void sectionLogEntries(const Fixture& f) {
    const HestonSLVFokkerPlanckFdmParams params = makeParams(
        2, FdmSquareRootFwdOp::Log, FdmHestonGreensFct::ZeroCorrelation,
        FdmSchemeDesc::ModifiedCraigSneyd());

    // logging = false: the list must be EMPTY even though logEntries() does
    // the whole calibration anyway (it calls performCalculations directly).
    {
        const HestonSLVFDMModel quiet(f.localVol, f.hestonModel, FINAL_DATE,
                                      params, /*logging*/ false);
        long long n = -1;
        try {
            n = (long long)quiet.logEntries().size();
        } catch (const std::exception&) {
        }
        emit_int("log_quiet_count", n);
    }

    const HestonSLVFDMModel model(f.localVol, f.hestonModel, FINAL_DATE, params,
                                  /*logging*/ true);
    // LogEntry has const members, so std::list<LogEntry> is not assignable;
    // hold the model's own list by reference and record any failure instead.
    std::string logErr;
    static const std::list<HestonSLVFDMModel::LogEntry> kEmpty;
    const std::list<HestonSLVFDMModel::LogEntry>* entriesPtr = &kEmpty;
    try {
        entriesPtr = &model.logEntries();
    } catch (const std::exception& e) {
        logErr = e.what();
    }
    const std::list<HestonSLVFDMModel::LogEntry>& entries = *entriesPtr;
    emit_str("log_error", logErr);
    emit_int("log_count", (long long)entries.size());

    std::vector<Real> ts, probSums, probMins, probMaxs, integrals;
    std::vector<long long> dim0, dim1;
    std::vector<Real> xFront, xBack, vFront, vBack;
    for (const auto& e : entries) {
        ts.push_back(e.t);

        Real s = 0.0, mn = e.prob->front(), mx = e.prob->front();
        for (Real v : *e.prob) {
            s += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        probSums.push_back(s);
        probMins.push_back(mn);
        probMaxs.push_back(mx);

        const auto& dim = e.mesher->layout()->dim();
        dim0.push_back((long long)dim[0]);
        dim1.push_back((long long)dim[1]);

        const auto& xm = e.mesher->getFdm1dMeshers()[0]->locations();
        const auto& vm = e.mesher->getFdm1dMeshers()[1]->locations();
        xFront.push_back(xm.front());
        xBack.push_back(xm.back());
        vFront.push_back(vm.front());
        vBack.push_back(vm.back());

        // Marginal in x, integrated over v with the same rule the C++ test
        // uses (hestonslvmodel.cpp:1608-1612).
        const Array x(xm.begin(), xm.end());
        Real marg = 0.0;
        for (Size i = 0; i < vm.size(); ++i) {
            marg += DiscreteSimpsonIntegral()(
                x, Array(e.prob->begin() + i * X_GRID,
                         e.prob->begin() + (i + 1) * X_GRID));
        }
        integrals.push_back(marg);
    }
    emit_arr("log_t", ts);
    emit_arr("log_prob_sum", probSums);
    emit_arr("log_prob_min", probMins);
    emit_arr("log_prob_max", probMaxs);
    emit_arr("log_marginal_sum", integrals);
    emit_iarr("log_dim0", dim0);
    emit_iarr("log_dim1", dim1);
    emit_arr("log_x_front", xFront);
    emit_arr("log_x_back", xBack);
    emit_arr("log_v_front", vFront);
    emit_arr("log_v_back", vBack);

    // The full density of the LAST entry, which is the one that has been
    // through every step of the algorithm.
    if (!entries.empty()) {
        const auto& last = entries.back();
        std::vector<Real> p(last.prob->begin(), last.prob->end());
        emit_arr("log_last_prob", p);
    }
}

}  // namespace

int main() {
    Settings::instance().evaluationDate() = TODAY;
    std::cout << "{\n";
    emit_int("evaluationDate_serial", (long long)TODAY.serialNumber());
    emit_int("finalDate_serial", (long long)FINAL_DATE.serialNumber());
    emit("s0", S0);
    emit("r", R);
    emit("q", Q);
    emit("kappa", KAPPA);
    emit("theta", THETA);
    emit("rho", RHO);
    emit("sigma", SIGMA);
    emit("v0", V0);
    emit_int("xGrid", (long long)X_GRID);
    emit_int("vGrid", (long long)V_GRID);
    emit_arr("lv_times", LV_TIMES);
    emit_arr("lv_strikes", LV_STRIKES);

    const Fixture f = makeFixture();
    sectionMesh(f);
    sectionLeverage(f);
    sectionLogEntries(f);
    std::cout << "\n}\n";
    return 0;
}
