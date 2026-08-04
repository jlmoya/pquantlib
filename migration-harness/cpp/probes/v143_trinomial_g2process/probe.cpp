// migration-harness/cpp/probes/methods/v143_trinomial_g2process_probe.cpp
//
// Reference values for the two behavioural changes made in C++ QuantLib v1.43
// that this probe cross-validates:
//
//   1. TrinomialTree  -- gated dx floor (ql/methods/lattices/trinomialtree.cpp).
//      On grid steps much shorter than the longest step the natural spacing
//      v*sqrt(3) collapses and the node count explodes.  v1.43 caches the
//      per-step variances, forms dxFloor = sqrt(3 * max_i v2_i) and, on steps
//      with dt < 0.01 * dtMax, widens dx to max(dxNatural, dxFloor).  When --
//      and only when -- the floor actually widened dx, the branching
//      probabilities switch from the classical Hull-White / Clewlow form to a
//      general moment-matching form.
//
//   2. G2Process / G2ForwardProcess (ql/processes/g2process.{hpp,cpp}).
//      Both gain a trailing `const Handle<YieldTermStructure>&` constructor
//      argument plus termStructure(), phi(Time) and shortRate(Time,Real,Real).
//      The simulated state is shifted to (x + phi(t), y) so that
//      state[0] + state[1] == r(t).  With an empty handle the process must
//      degenerate to exactly the pre-v1.43 pair of zero-mean OU processes.
//
// Design notes
// ------------
//
// Processes.  The upstream regression test drives the tree with
// HullWhite::dynamics()->process(), which is literally
// OrnsteinUhlenbeckProcess(a, sigma) -- see hullwhite.hpp, HullWhite::Dynamics.
// This probe constructs that OU process directly, so the reference depends on
// the tree and on OU, not on the HullWhite model or on a yield curve.
//
// The "gate fires but dx is left unchanged" case needs a *time-dependent*
// diffusion, because for a process whose variance grows monotonically with dt
// (OU, Brownian motion) a shorter step always has a smaller variance and the
// floor therefore always widens dx.  Note the algebra: dxFloor is built from
// the maximum variance over *all* steps, so dxNatural_i <= dxFloor always, with
// equality exactly when step i is the variance argmax.  "Gate fires, dx
// unchanged" is therefore precisely the case where the short step carries the
// grid's largest variance, and whether max() returns dxNatural or dxFloor comes
// down to whether sqrt(3.0*v2) <= sqrt(v2)*sqrt(3.0) for that particular
// double.  To make that comparison reproducible rather than a coin flip
// between libm and the JVM, the case uses
// GeneralizedOrnsteinUhlenbeckProcess with speed == 0, which takes the
// `speed < sqrt(QL_EPSILON)` algebraic-limit branch: variance == vol*vol*dt and
// expectation == x0, i.e. no transcendental function is involved anywhere in
// the tree construction, so every quantity below is bit-reproducible.  The vol
// function has a spike on [0.75, 1.0] which makes the tiny step the argmax; the
// chosen numbers give sqrt(3.0*v2) == sqrt(v2)*sqrt(3.0) exactly, hence
// dxIsFloored == false while the gate fired.
//
// A separate isPositive case is included because v1.43 also introduced
// `tempBumped`: when isPositive pushes `temp` upward, |e| may exceed dx/2 and
// the resulting negative weights are deliberately exempted from the new
// QL_ENSURE.  That case genuinely produces a negative p2, so a port that
// applies the assertion unconditionally throws instead of returning values.
//
// Curve.  G2 is exercised against a non-flat InterpolatedZeroCurve<Linear> over
// eight explicit (date, continuously-compounded zero) pillars -- trivially
// reproducible in Java -- and against an empty handle.  Pillar dates are also
// emitted as serial numbers in `inputs` so the consuming test cannot drift.
//
// Tolerance note for consumers.  phi(t) reads the instantaneous forward via
// YieldTermStructure::forwardRate(t, t, ...), which is a 1e-4 finite difference
// of discount factors; a 1-ULP disagreement in exp() between libm and the JVM
// therefore shows up as ~2e-12 absolute in phi.  drift() then forward-differences
// phi with h = 1e-4, amplifying that by another 1e4.  phi, expectation and x0
// stay comfortably inside a 1e-8 relative band; the curve-mode drift does not,
// and the consuming test documents a derived absolute bound for it.  Everything
// that does not touch the curve (diffusion, stdDeviation, covariance, the whole
// empty-handle surface, and every trinomial-tree quantity) is exact arithmetic
// and is compared tightly.

#include <ql/version.hpp>

#include <ql/experimental/shortrate/generalizedornsteinuhlenbeckprocess.hpp>
#include <ql/math/array.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/matrix.hpp>
#include <ql/methods/lattices/trinomialtree.hpp>
#include <ql/processes/g2process.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/settings.hpp>
#include <ql/stochasticprocess.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/timegrid.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "common.hpp"

using namespace QuantLib;
using namespace jqml_harness;

namespace {

// ---------------------------------------------------------------------------
// TrinomialTree
// ---------------------------------------------------------------------------

// Hull-White dynamics process: OrnsteinUhlenbeckProcess(a, sigma) with x0 = 0.
const Real kA = 0.1;
const Real kSigma = 0.01;

ext::shared_ptr<StochasticProcess1D> ouProcess() {
    return ext::make_shared<OrnsteinUhlenbeckProcess>(kA, kSigma);
}

// Transcendental-free time-dependent-vol process (see design notes): speed == 0
// selects GeneralizedOrnsteinUhlenbeckProcess's algebraic-limit branch, so
// variance(t, ., dt) == vol(t)^2 * dt and expectation(t, x, dt) == x.
ext::shared_ptr<StochasticProcess1D> spikeProcess(Real x0) {
    auto speed = [](Time) { return 0.0; };
    auto vol = [](Time t) { return (t >= 0.75 && t <= 1.0) ? 0.5 : 0.01; };
    return ext::make_shared<GeneralizedOrnsteinUhlenbeckProcess>(speed, vol, x0, 0.0);
}

json jsonArray(const std::vector<Real>& v) {
    json a = json::array();
    for (Real x : v)
        a.push_back(x);
    return a;
}

// Everything an independent implementation of the constructor must reproduce:
// the grid it derived, the per-step dx (and the *unfloored* dx it would have
// used, so the consumer can positively assert the floor's on/off state), the
// j-extent of every branching, and the descendant index plus the three
// probabilities of every single node.
json describeTree(const ext::shared_ptr<StochasticProcess1D>& process, const TimeGrid& grid,
                  bool isPositive) {
    TrinomialTree tree(process, grid, isPositive);

    const Size nSteps = grid.size() - 1;
    const Real x0 = process->x0();

    json times = json::array();
    json dx = json::array();
    json dxNatural = json::array();
    json sizes = json::array();
    for (Size i = 0; i < grid.size(); ++i) {
        times.push_back(grid[i]);
        dx.push_back(tree.dx(i));
        sizes.push_back(static_cast<long>(tree.size(i)));
    }
    // dx_[0] is the constructor's dummy 0.0 entry; step i's dx lives at dx_[i+1].
    dxNatural.push_back(0.0);
    for (Size i = 0; i < nSteps; ++i) {
        const Real v2 = process->variance(grid[i], 0.0, grid.dt(i));
        // Same operation order as production (sqrt(v2) * sqrt(3.0), not
        // sqrt(3.0 * v2)) so an equality assertion against dx holds
        // bit-for-bit rather than by coincidence in the last bit.
        dxNatural.push_back(std::sqrt(v2) * std::sqrt(3.0));
    }

    json steps = json::array();
    Size maxNodes = 1;
    for (Size i = 0; i < nSteps; ++i) {
        // Branching is a private nested class, so jMin is recovered from the
        // public surface: underlying(i+1, 0) == x0 + jMin(i) * dx(i+1).
        const long jMin = std::lround((tree.underlying(i + 1, 0) - x0) / tree.dx(i + 1));
        const Size n = tree.size(i);           // number of nodes branching at step i
        const Size nNext = tree.size(i + 1);
        maxNodes = std::max(maxNodes, nNext);

        json nodes = json::array();
        for (Size index = 0; index < n; ++index) {
            json descendants = json::array();
            json probabilities = json::array();
            for (Size branch = 0; branch < 3; ++branch) {
                descendants.push_back(static_cast<long>(tree.descendant(i, index, branch)));
                probabilities.push_back(tree.probability(i, index, branch));
            }
            nodes.push_back(json{
                {"index", static_cast<long>(index)},
                {"underlying", tree.underlying(i, index)},
                {"descendants", descendants},
                {"probabilities", probabilities},
            });
        }

        steps.push_back(json{
            {"i", static_cast<long>(i)},
            {"t", grid[i]},
            {"dt", grid.dt(i)},
            {"dx", tree.dx(i + 1)},
            {"jMin", jMin},
            {"jMax", jMin + static_cast<long>(nNext) - 1},
            {"nodeCount", static_cast<long>(n)},
            {"nodes", nodes},
        });
    }

    // Underlyings of the terminal level, which no step's `nodes` block covers.
    json terminalUnderlying = json::array();
    for (Size index = 0; index < tree.size(nSteps); ++index)
        terminalUnderlying.push_back(tree.underlying(nSteps, index));

    return json{
        {"nSteps", static_cast<long>(nSteps)},
        {"x0", x0},
        {"times", times},
        {"dx", dx},
        {"dxNatural", dxNatural},
        {"sizes", sizes},
        {"maxNodes", static_cast<long>(maxNodes)},
        {"terminalUnderlying", terminalUnderlying},
        {"steps", steps},
    };
}

TimeGrid mandatoryGrid(const std::vector<Time>& times) {
    return TimeGrid(times.begin(), times.end());
}

// ---------------------------------------------------------------------------
// G2Process / G2ForwardProcess
// ---------------------------------------------------------------------------

const Real kG2a = 0.1;
const Real kG2sigma = 0.01;
const Real kG2b = 0.2;
const Real kG2eta = 0.013;
const Real kG2rho = -0.5;
const Time kFwdMeasureTime = 7.5;

const Date kToday(15, January, 2026);

std::vector<Date> curveDates() {
    return {
        Date(15, January, 2026), Date(15, July, 2026),    Date(15, January, 2027),
        Date(15, January, 2028), Date(15, January, 2031), Date(15, January, 2036),
        Date(15, January, 2046), Date(15, January, 2056),
    };
}

std::vector<Rate> curveZeros() {
    // Deliberately non-flat, and with a sign change in the slope at the long
    // end so phi'(t) is not constant.
    return {0.0180, 0.0215, 0.0245, 0.0290, 0.0335, 0.0360, 0.0372, 0.0368};
}

Handle<YieldTermStructure> zeroCurve() {
    return Handle<YieldTermStructure>(ext::make_shared<InterpolatedZeroCurve<Linear>>(
        curveDates(), curveZeros(), Actual365Fixed()));
}

const std::vector<Time>& phiTimes() {
    static const std::vector<Time> t = {0.0, 0.25, 0.75, 1.0, 2.0, 3.5, 5.0, 10.0, 20.0};
    return t;
}

// (t, z1, z2) triples for shortRate.
const std::vector<std::array<Real, 3>>& shortRateArgs() {
    static const std::vector<std::array<Real, 3>> v = {
        {0.5, -0.01, -0.002}, {0.5, -0.01, 0.004}, {0.5, 0.0, 0.0},
        {0.5, 0.005, 0.004},  {3.0, -0.01, 0.0},   {3.0, 0.005, -0.002},
    };
    return v;
}

// (t, z0, z1) triples for drift.
const std::vector<std::array<Real, 3>>& driftArgs() {
    static const std::vector<std::array<Real, 3>> v = {
        {0.25, 0.002, -0.003}, {1.0, 0.002, -0.003},  {1.0, -0.015, 0.006},
        {2.5, 0.030, 0.001},   {5.0, 0.002, -0.003},  {10.0, 0.041, -0.008},
    };
    return v;
}

// (t0, dt, z0[0], z0[1]) quadruples for expectation / stdDeviation / covariance.
const std::vector<std::array<Real, 4>>& evolutionArgs() {
    static const std::vector<std::array<Real, 4>> v = {
        {0.0, 0.25, 0.012, -0.004}, {0.25, 0.75, 0.012, -0.004}, {1.0, 1.0, 0.030, 0.002},
        {2.0, 3.0, 0.012, -0.004},  {5.0, 5.0, -0.006, 0.009},
    };
    return v;
}

json matrixJson(const Matrix& m) {
    json rows = json::array();
    for (Size i = 0; i < m.rows(); ++i) {
        json row = json::array();
        for (Size j = 0; j < m.columns(); ++j)
            row.push_back(m[i][j]);
        rows.push_back(row);
    }
    return rows;
}

json arrayJson(const Array& a) {
    json v = json::array();
    for (Size i = 0; i < a.size(); ++i)
        v.push_back(a[i]);
    return v;
}

// `Process` is G2Process or G2ForwardProcess; both expose the same surface
// apart from x0()/y0(), which only G2Process has.
template <class Process>
json scalarsJson(const Process& p, bool hasCurve) {
    json out{
        {"size", static_cast<long>(p.size())},
        {"initialValues", arrayJson(p.initialValues())},
        {"termStructureEmpty", p.termStructure().empty()},
        {"phiTimes", jsonArray(phiTimes())},
    };
    if (hasCurve) {
        json phis = json::array();
        for (Time t : phiTimes())
            phis.push_back(p.phi(t));
        out["phi"] = phis;
    } else {
        // phi() must throw without a curve; there is no value to pin.
        out["phi"] = nullptr;
    }
    json sr = json::array();
    for (const auto& a : shortRateArgs())
        sr.push_back(json{{"t", a[0]}, {"z1", a[1]}, {"z2", a[2]},
                          {"shortRate", p.shortRate(a[0], a[1], a[2])}});
    out["shortRate"] = sr;
    return out;
}

template <class Process>
json driftJson(const Process& p, bool hasCurve) {
    json rows = json::array();
    for (const auto& a : driftArgs()) {
        Array z(2);
        z[0] = a[1];
        z[1] = a[2];
        json row{{"t", a[0]}, {"z", json::array({a[1], a[2]})}, {"drift", arrayJson(p.drift(a[0], z))}};
        if (hasCurve) {
            // Pinned so the consumer can reconstruct the shift term and see how
            // small phi(t+h) - phi(t) is relative to the rounding noise in phi.
            row["phi"] = p.phi(a[0]);
            row["phiPlusH"] = p.phi(a[0] + 1.0e-4);
        }
        rows.push_back(row);
    }
    return rows;
}

template <class Process>
json evolutionJson(const Process& p) {
    json rows = json::array();
    for (const auto& a : evolutionArgs()) {
        const Time t0 = a[0];
        const Time dt = a[1];
        Array z0(2);
        z0[0] = a[2];
        z0[1] = a[3];
        rows.push_back(json{
            {"t0", t0},
            {"dt", dt},
            {"z0", json::array({a[2], a[3]})},
            {"expectation", arrayJson(p.expectation(t0, z0, dt))},
            {"diffusion", matrixJson(p.diffusion(t0, z0))},
            {"stdDeviation", matrixJson(p.stdDeviation(t0, z0, dt))},
            {"covariance", matrixJson(p.covariance(t0, z0, dt))},
        });
    }
    return rows;
}

json g2Inputs(bool hasCurve) {
    json in{{"a", kG2a},   {"sigma", kG2sigma}, {"b", kG2b},
            {"eta", kG2eta}, {"rho", kG2rho},   {"hasTermStructure", hasCurve}};
    if (hasCurve) {
        json serials = json::array();
        for (const Date& d : curveDates())
            serials.push_back(static_cast<long>(d.serialNumber()));
        in["curveDateSerials"] = serials;
        in["curveZeroRates"] = jsonArray(curveZeros());
        in["curveDayCounter"] = "Actual/365 (Fixed)";
        in["evaluationDateSerial"] = static_cast<long>(kToday.serialNumber());
    }
    return in;
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    ReferenceWriter out("methods/v143_trinomial_g2process", QL_VERSION,
                        "v143_trinomial_g2process_probe");

    // =====================================================================
    // TrinomialTree
    // =====================================================================

    // Uniform grid: dt == dtMax on every step, so the gate never fires and the
    // tree must be identical to v1.42.1.
    out.addCase("tree_uniform_ou",
                json{{"process", "OrnsteinUhlenbeckProcess"},
                     {"speed", kA},
                     {"vol", kSigma},
                     {"grid", "uniform"},
                     {"end", 3.0},
                     {"steps", 6},
                     {"isPositive", false}},
                describeTree(ouProcess(), TimeGrid(3.0, Size(6)), false));

    // Mildly non-uniform: quarterly pillars with 3-day weekend rolls. The
    // shortest step is 3/365 against dtMax = 0.25, a ratio of 0.033 -- above the
    // 0.01 gate, so the floor must stay off, exactly as the upstream comment
    // claims for "weekend rolls, 1-day mismatches".
    {
        const std::vector<Time> t = {0.25, 0.5, 0.5 + 3.0 / 365.0, 0.75, 1.0};
        out.addCase("tree_weekend_roll_ou",
                    json{{"process", "OrnsteinUhlenbeckProcess"},
                         {"speed", kA},
                         {"vol", kSigma},
                         {"grid", "mandatory"},
                         {"mandatoryTimes", jsonArray(t)},
                         {"isPositive", false}},
                    describeTree(ouProcess(), mandatoryGrid(t), false));
    }

    // The pathology that motivated the change: a 1ms mandatory gap after t=1.
    {
        const std::vector<Time> t = {1.0, 1.0 + 1.0e-3, 2.0, 3.0};
        out.addCase("tree_small_gap_ou",
                    json{{"process", "OrnsteinUhlenbeckProcess"},
                         {"speed", kA},
                         {"vol", kSigma},
                         {"grid", "mandatory"},
                         {"mandatoryTimes", jsonArray(t)},
                         {"isPositive", false}},
                    describeTree(ouProcess(), mandatoryGrid(t), false));
    }

    // Same pathology, but the grid's last step is deliberately *not* the
    // variance argmax (dt = 0.5 versus dtMax = 1.0). dxFloor is built from the
    // maximum variance over all steps; on every other grid in this probe the
    // final step happens to attain that maximum, so a `dxFloorVar = v2_i`
    // assignment where upstream writes `std::max(dxFloorVar, v2_i)` would go
    // unnoticed. Here the two differ visibly: 0.0164895 versus 0.0119475.
    {
        const std::vector<Time> t = {1.0, 1.0 + 1.0e-3, 2.0, 2.5};
        out.addCase("tree_small_gap_short_tail_ou",
                    json{{"process", "OrnsteinUhlenbeckProcess"},
                         {"speed", kA},
                         {"vol", kSigma},
                         {"grid", "mandatory"},
                         {"mandatoryTimes", jsonArray(t)},
                         {"isPositive", false}},
                    describeTree(ouProcess(), mandatoryGrid(t), false));
    }

    // Immediately either side of the 0.01 activation threshold. Loose
    // multipliers (0.005 / 0.02) would pass for any threshold in a wide range;
    // these two pin the boundary itself.
    {
        const std::vector<Time> t = {1.0, 1.0 + 0.0099, 2.0, 3.0};
        out.addCase("tree_threshold_below_ou",
                    json{{"process", "OrnsteinUhlenbeckProcess"},
                         {"speed", kA},
                         {"vol", kSigma},
                         {"grid", "mandatory"},
                         {"mandatoryTimes", jsonArray(t)},
                         {"gapRatio", 0.0099},
                         {"isPositive", false}},
                    describeTree(ouProcess(), mandatoryGrid(t), false));
    }
    {
        const std::vector<Time> t = {1.0, 1.0 + 0.0101, 2.0, 3.0};
        out.addCase("tree_threshold_above_ou",
                    json{{"process", "OrnsteinUhlenbeckProcess"},
                         {"speed", kA},
                         {"vol", kSigma},
                         {"grid", "mandatory"},
                         {"mandatoryTimes", jsonArray(t)},
                         {"gapRatio", 0.0101},
                         {"isPositive", false}},
                    describeTree(ouProcess(), mandatoryGrid(t), false));
    }

    // Gate fires (dt = 0.005 < 0.01 * dtMax) but the short step carries the
    // grid's largest variance, so dx is left at its natural value and the
    // classical probability branch must still be taken.
    {
        const std::vector<Time> t = {1.0, 1.005, 2.0, 3.0};
        out.addCase("tree_gate_fires_dx_unchanged",
                    json{{"process", "GeneralizedOrnsteinUhlenbeckProcess"},
                         {"speed", 0.0},
                         {"volBase", 0.01},
                         {"volSpike", 0.5},
                         {"volSpikeFrom", 0.75},
                         {"volSpikeTo", 1.0},
                         {"x0", 0.0},
                         {"level", 0.0},
                         {"grid", "mandatory"},
                         {"mandatoryTimes", jsonArray(t)},
                         {"isPositive", false}},
                    describeTree(spikeProcess(0.0), mandatoryGrid(t), false));
    }

    // Same grid and process but x0 = 0.05 and isPositive = true: at the spiked
    // step dx (0.0612) exceeds x0, so `temp` is bumped upward and the resulting
    // p2 is negative. v1.43 exempts bumped nodes from the new QL_ENSURE; an
    // implementation that asserts unconditionally throws here.
    {
        const std::vector<Time> t = {1.0, 1.005, 2.0, 3.0};
        out.addCase("tree_is_positive_bump",
                    json{{"process", "GeneralizedOrnsteinUhlenbeckProcess"},
                         {"speed", 0.0},
                         {"volBase", 0.01},
                         {"volSpike", 0.5},
                         {"volSpikeFrom", 0.75},
                         {"volSpikeTo", 1.0},
                         {"x0", 0.05},
                         {"level", 0.0},
                         {"grid", "mandatory"},
                         {"mandatoryTimes", jsonArray(t)},
                         {"isPositive", true}},
                    describeTree(spikeProcess(0.05), mandatoryGrid(t), true));
    }

    // =====================================================================
    // G2Process
    // =====================================================================

    {
        const Handle<YieldTermStructure> curve = zeroCurve();
        G2Process withCurve(kG2a, kG2sigma, kG2b, kG2eta, kG2rho, curve);
        G2Process noCurve(kG2a, kG2sigma, kG2b, kG2eta, kG2rho);

        {
            json j = scalarsJson(withCurve, true);
            j["x0"] = withCurve.x0();
            j["y0"] = withCurve.y0();
            out.addCase("g2_curve_scalars", g2Inputs(true), j);
        }
        out.addCase("g2_curve_drift", g2Inputs(true), driftJson(withCurve, true));
        out.addCase("g2_curve_evolution", g2Inputs(true), evolutionJson(withCurve));

        // The property the state reshaping exists for: starting from
        // initialValues(), E[z1(t) + z2(t)] == phi(t) == the curve-implied
        // short-rate expectation.
        {
            const Array iv = withCurve.initialValues();
            json times = json::array();
            json phis = json::array();
            json sums = json::array();
            json exps = json::array();
            for (Time t : phiTimes()) {
                if (t == 0.0)
                    continue;
                const Array e = withCurve.expectation(0.0, iv, t);
                times.push_back(t);
                phis.push_back(withCurve.phi(t));
                sums.push_back(e[0] + e[1]);
                exps.push_back(arrayJson(e));
            }
            out.addCase("g2_curve_short_rate_identity", g2Inputs(true),
                        json{{"initialValues", arrayJson(iv)},
                             {"times", times},
                             {"phi", phis},
                             {"expectation", exps},
                             {"expectationSum", sums}});
        }

        {
            json j = scalarsJson(noCurve, false);
            j["x0"] = noCurve.x0();
            j["y0"] = noCurve.y0();
            out.addCase("g2_empty_scalars", g2Inputs(false), j);
        }
        out.addCase("g2_empty_drift", g2Inputs(false), driftJson(noCurve, false));
        out.addCase("g2_empty_evolution", g2Inputs(false), evolutionJson(noCurve));
    }

    // =====================================================================
    // G2ForwardProcess
    // =====================================================================

    {
        const Handle<YieldTermStructure> curve = zeroCurve();
        G2ForwardProcess withCurve(kG2a, kG2sigma, kG2b, kG2eta, kG2rho, curve);
        G2ForwardProcess noCurve(kG2a, kG2sigma, kG2b, kG2eta, kG2rho);
        // ForwardMeasureProcess::T_ is not default-initialised upstream, so it
        // is always set explicitly here and in the consuming test.
        withCurve.setForwardMeasureTime(kFwdMeasureTime);
        noCurve.setForwardMeasureTime(kFwdMeasureTime);

        json inCurve = g2Inputs(true);
        inCurve["forwardMeasureTime"] = kFwdMeasureTime;
        json inEmpty = g2Inputs(false);
        inEmpty["forwardMeasureTime"] = kFwdMeasureTime;

        out.addCase("g2fwd_curve_scalars", inCurve, scalarsJson(withCurve, true));
        out.addCase("g2fwd_curve_drift", inCurve, driftJson(withCurve, true));
        out.addCase("g2fwd_curve_evolution", inCurve, evolutionJson(withCurve));

        out.addCase("g2fwd_empty_scalars", inEmpty, scalarsJson(noCurve, false));
        out.addCase("g2fwd_empty_drift", inEmpty, driftJson(noCurve, false));
        out.addCase("g2fwd_empty_evolution", inEmpty, evolutionJson(noCurve));
    }

    out.write();
    return 0;
}
