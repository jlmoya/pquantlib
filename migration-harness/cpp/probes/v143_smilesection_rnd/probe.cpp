// migration-harness/cpp/probes/v143_smilesection_rnd/probe.cpp
//
// Reference values for SmileSectionRNDCalculator, new in C++ QuantLib v1.43
// (ql/methods/finitedifferences/utilities/smilesectionrndcalculator.{hpp,cpp}).
//
// It derives the risk-neutral terminal density implied by a SmileSection via
// Breeden-Litzenberger, and implements RiskNeutralDensityCalculator:
//
//     Real pdf(Real x, Time t) const override;     // x = ln(S)
//     Real cdf(Real x, Time t) const override;
//     Real invcdf(Real p, Time t) const override;
//     Real pdf(Real x) const;                      // t = smile->exerciseTime()
//     Real cdf(Real x) const;
//     Real invcdf(Real p) const;
//
// Constructor:
//     SmileSectionRNDCalculator(ext::shared_ptr<SmileSection> smile,
//                               Size nStrikes = 200,
//                               Real nStd = 5.0);
//
// Behaviour a port has to reproduce, none of it obvious from the signatures
// ------------------------------------------------------------------------
//  1. pdf() and cdf() are DIRECT smile evaluations and never touch the strike
//     grid, so nStrikes / nStd are irrelevant to them:
//         pdf(x, t) = exp(x) * smile->density(exp(x), 1.0)          [gap 1e-4]
//         cdf(x, t) = 1 - smile->digitalOptionPrice(exp(x), Call, 1.0)  [gap 1e-5]
//     Both of those SmileSection helpers are finite differences of
//     SmileSection::optionPrice with the DEFAULT gaps; a port that uses a
//     different gap, or an analytic derivative, will not match in the wings.
//     Case `*_grid_independent_pdf_cdf` pins this independence explicitly.
//  2. Only invcdf() builds the grid, lazily, once, in initialize():
//         forward  = smile->atmLevel()             (must not be Null<Real>())
//         sigmaAtm = smile->volatility(forward)
//         logStd   = sigmaAtm * sqrt(T)
//         kMin     = max(forward * exp(-nStd * logStd), QL_EPSILON)
//         kMax     = forward * exp( nStd * logStd)
//         K_i      = kMin + (kMax - kMin) * i / (nStrikes - 1),  i in [0, nStrikes)
//         cdf_i    = clamp(1 - digitalOptionPrice(K_i, Call, 1.0), 0, 1)
//     then a RUNNING MAXIMUM is applied and points whose monotonised cdf gains
//     <= 1e-12 over the previous kept point are DROPPED, so the abscissa handed
//     to the spline is strictly increasing and generally shorter than nStrikes.
//     Fewer than 4 surviving points is a hard error.
//  3. The quantile function is a MonotonicCubicNaturalSpline over
//     (cdf_i -> K_i), i.e. cdf is the abscissa and strike the ordinate.
//     invcdf(p) requires 0 < p < 1, then CLAMPS p into
//     [cdf_.front(), cdf_.back()] and returns log(spline(p)).
//     Consequently invcdf(1e-12) returns exactly log(kMin_kept) and
//     invcdf(1 - 1e-12) returns exactly log(kMax_kept) -- the sharpest
//     available pin on the grid construction, captured by `*_grid_endpoints`.
//  4. checkTime() requires close_enough(t, smile->exerciseTime()); the 2-arg
//     overloads are not a way to reprice at another maturity.
//  5. Ordering inside invcdf() is checkTime -> initialize -> p-range check.
//     So on a smile with a Null atm level, invcdf(-1.0) reports the missing
//     atmLevel, NOT the invalid probability. `invcdf_atm_check_precedes_p_check`
//     pins that ordering.
//
// Smiles used
// -----------
//   flat       FlatSmileSection(2026-03-01, 0.20, Actual365Fixed, 2025-03-01, atm=100)
//   svi1       SviSmileSection(T=1, fwd=100, {0.04, 0.10, 0.30, -0.40,  0.00})
//   svi2       SviSmileSection(T=1, fwd= 96, {0.02, 0.08, 0.25, -0.30,  0.00})
//   sviSteep   SviSmileSection(T=1, fwd=100, {0.03, 0.25, 0.15, -0.75, -0.10})
//
// svi1/svi2 are the two parameter sets the v1.43 test-suite uses in
// testGaussianCopulaSpreadEngineSVI, so the reference values here line up with
// the spread-engine probe. sviSteep adds a strongly skewed, off-centre smile
// where a wrong wing discretisation shows up first. Forwards are exact decimals
// so no input value has to be reproduced through a transcendental function.
//
// The offsets go out to +-1.2 in log-moneyness, i.e. roughly +-6 ATM standard
// deviations, well outside the nStd = 5 grid, because that is where a wrong
// finite-difference gap or a wrong extrapolation first becomes visible.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/smilesection/rnd.json. Nothing else may be printed.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/qldefines.hpp>

#include <ql/errors.hpp>
#include <ql/experimental/volatility/svismilesection.hpp>
#include <ql/methods/finitedifferences/utilities/smilesectionrndcalculator.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/atmsmilesection.hpp>
#include <ql/termstructures/volatility/flatsmilesection.hpp>
#include <ql/termstructures/volatility/smilesection.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON emitter (this harness has no nlohmann dependency).
// ---------------------------------------------------------------------------
std::string num(Real v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

class Obj {
  public:
    Obj& n(const std::string& k, Real v) { return put(k, num(v)); }
    Obj& i(const std::string& k, long long v) { return put(k, std::to_string(v)); }
    Obj& s(const std::string& k, const std::string& v) { return put(k, "\"" + v + "\""); }
    Obj& b(const std::string& k, bool v) { return put(k, v ? "true" : "false"); }
    Obj& obj(const std::string& k, const Obj& v) { return put(k, v.str()); }
    std::string str() const { return "{" + body_ + "}"; }

  private:
    Obj& put(const std::string& k, const std::string& v) {
        if (!body_.empty())
            body_ += ", ";
        body_ += "\"" + k + "\": " + v;
        return *this;
    }
    std::string body_;
};

std::vector<std::pair<std::string, std::string>> gCases;

void addCase(const std::string& name, const Obj& inputs, const Obj& expected) {
    gCases.emplace_back(name, "{\"inputs\": " + inputs.str() +
                                  ", \"expected\": " + expected.str() + "}");
}

void emitDocument() {
    std::cout << "{\n";
    for (std::size_t i = 0; i < gCases.size(); ++i)
        std::cout << "  \"" << gCases[i].first << "\": " << gCases[i].second
                  << (i + 1 < gCases.size() ? "," : "") << "\n";
    std::cout << "}\n";
}

// 2025-03-01 -> 2026-03-01 is exactly 365 days, so Actual365Fixed gives an
// exercise time of exactly 1.0 and the date-based and time-based smile
// constructors agree bit for bit.
const Date kToday(1, March, 2025);
const Date kMaturity(1, March, 2026);
const Time kT = 1.0;

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

using Smile = ext::shared_ptr<SmileSection>;

Smile flatSmile(Real atm) {
    return ext::make_shared<FlatSmileSection>(kMaturity, 0.20, dayCounter(), kToday, atm);
}

Smile flatSmileWithoutAtm() {
    return ext::make_shared<FlatSmileSection>(kMaturity, 0.20, dayCounter(), kToday);
}

Smile sviSmile(Real forward, const std::vector<Real>& params) {
    return ext::make_shared<SviSmileSection>(kT, forward, params);
}

struct SmileSpec {
    const char* tag;
    Smile smile;
    Real forward;
};

std::vector<SmileSpec> smiles() {
    return {
        {"flat", flatSmile(100.0), 100.0},
        {"svi1", sviSmile(100.0, {0.04, 0.10, 0.30, -0.40, 0.0}), 100.0},
        {"svi2", sviSmile(96.0, {0.02, 0.08, 0.25, -0.30, 0.0}), 96.0},
        {"svi_steep", sviSmile(100.0, {0.03, 0.25, 0.15, -0.75, -0.10}), 100.0},
    };
}

struct Offset {
    const char* tag;
    Real value;
};

const std::vector<Offset>& offsets() {
    static const std::vector<Offset> v = {
        {"m120", -1.2}, {"m090", -0.9}, {"m060", -0.6}, {"m030", -0.3},
        {"m010", -0.1}, {"000", 0.0},   {"p010", 0.1},  {"p030", 0.3},
        {"p060", 0.6},  {"p090", 0.9},  {"p120", 1.2},
    };
    return v;
}

struct Prob {
    const char* tag;
    Real value;
};

// Includes 1e-12 and 1 - 1e-12, which land on the clamp to the grid endpoints,
// and 1e-6 / 0.9999 which sit outside the region the nStd = 5 grid resolves
// well -- exactly where a wrong discretisation first shows up.
const std::vector<Prob>& probabilities() {
    static const std::vector<Prob> v = {
        {"p1em12", 1e-12},   {"p1em6", 1e-6},   {"p1em4", 1e-4},
        {"p1em3", 1e-3},     {"p001", 0.01},    {"p005", 0.05},
        {"p010", 0.10},      {"p025", 0.25},    {"p050", 0.50},
        {"p075", 0.75},      {"p090", 0.90},    {"p095", 0.95},
        {"p099", 0.99},      {"p0999", 0.999},  {"p09999", 0.9999},
        {"p1m1em12", 1.0 - 1e-12},
    };
    return v;
}

Obj baseInputs(const SmileSpec& s, Size nStrikes, Real nStd) {
    Obj o;
    o.s("smile", s.tag)
        .n("forward", s.forward)
        .n("exercise_time", s.smile->exerciseTime())
        .i("n_strikes", static_cast<long long>(nStrikes))
        .n("n_std", nStd);
    return o;
}

std::string name(const std::string& a, const std::string& b) { return a + "_" + b; }

// ---------------------------------------------------------------------------
// pdf / cdf over a log-moneyness ladder. Grid parameters are irrelevant here,
// so the default calculator is used.
// ---------------------------------------------------------------------------
void emitPdfCdf(const SmileSpec& s) {
    const SmileSectionRNDCalculator calc(s.smile);
    const Real logFwd = std::log(s.forward);
    const Time t = s.smile->exerciseTime();

    for (const Offset& o : offsets()) {
        const Real x = logFwd + o.value;
        Obj in = baseInputs(s, 200, 5.0);
        in.n("log_forward", logFwd).n("offset", o.value).n("x", x)
            .n("strike", std::exp(x)).n("t", t);

        Obj ex;
        ex.n("pdf", calc.pdf(x, t))
            .n("cdf", calc.cdf(x, t))
            .n("smile_volatility", s.smile->volatility(std::exp(x)));

        addCase(name(s.tag, std::string("pdf_cdf_") + o.tag), in, ex);
    }
}

// ---------------------------------------------------------------------------
// invcdf over a probability ladder, for an arbitrary (nStrikes, nStd) grid.
// ---------------------------------------------------------------------------
void emitInvCdf(const SmileSpec& s, Size nStrikes, Real nStd, const std::string& prefix,
                const std::vector<Prob>& probs) {
    const SmileSectionRNDCalculator calc(s.smile, nStrikes, nStd);
    const Time t = s.smile->exerciseTime();
    for (const Prob& p : probs) {
        Obj in = baseInputs(s, nStrikes, nStd);
        in.n("p", p.value).n("t", t);
        const Real x = calc.invcdf(p.value, t);
        Obj ex;
        ex.n("invcdf", x).n("strike", std::exp(x));
        addCase(name(prefix, std::string("invcdf_") + p.tag), in, ex);
    }
}

// ---------------------------------------------------------------------------
// The grid initialize() builds, reconstructed from the public formula, plus the
// two clamped endpoints of invcdf. If a port's kMin/kMax or its dedup rule
// differ, `invcdf_at_p_min` / `invcdf_at_p_max` diverge immediately.
// ---------------------------------------------------------------------------
void emitGridEndpoints(const SmileSpec& s, Size nStrikes, Real nStd,
                       const std::string& prefix) {
    const SmileSectionRNDCalculator calc(s.smile, nStrikes, nStd);
    const Time t = s.smile->exerciseTime();
    const Real forward = s.smile->atmLevel();
    const Real sigmaAtm = s.smile->volatility(forward);
    const Real logStd = sigmaAtm * std::sqrt(t);
    const Real kMin = std::max(forward * std::exp(-nStd * logStd), Real(QL_EPSILON));
    const Real kMax = forward * std::exp(nStd * logStd);

    const Real invAtPMin = calc.invcdf(1e-12, t);
    const Real invAtPMax = calc.invcdf(1.0 - 1e-12, t);

    Obj ex;
    ex.n("exercise_time", t)
        .n("atm_level", forward)
        .n("sigma_atm", sigmaAtm)
        .n("log_std", logStd)
        .n("k_min", kMin)
        .n("k_max", kMax)
        .n("cdf_at_k_min", calc.cdf(std::log(kMin), t))
        .n("cdf_at_k_max", calc.cdf(std::log(kMax), t))
        .n("invcdf_at_p_min", invAtPMin)
        .n("invcdf_at_p_max", invAtPMax)
        .n("strike_at_p_min", std::exp(invAtPMin))
        .n("strike_at_p_max", std::exp(invAtPMax));

    addCase(name(prefix, "grid_endpoints"), baseInputs(s, nStrikes, nStd), ex);
}

// ---------------------------------------------------------------------------
// The 1-arg overloads must equal the 2-arg ones at t = smile->exerciseTime().
// ---------------------------------------------------------------------------
void emitOverloadAgreement(const SmileSpec& s) {
    const SmileSectionRNDCalculator calc(s.smile);
    const Time t = s.smile->exerciseTime();
    const Real x = std::log(s.forward) + 0.15;
    const Real p = 0.3;

    Obj in = baseInputs(s, 200, 5.0);
    in.n("x", x).n("p", p).n("t", t);

    Obj ex;
    ex.n("pdf_1arg", calc.pdf(x))
        .n("pdf_2arg", calc.pdf(x, t))
        .n("cdf_1arg", calc.cdf(x))
        .n("cdf_2arg", calc.cdf(x, t))
        .n("invcdf_1arg", calc.invcdf(p))
        .n("invcdf_2arg", calc.invcdf(p, t));

    addCase(name(s.tag, "overloads_agree"), in, ex);
}

// ---------------------------------------------------------------------------
// pdf / cdf must be identical across wildly different grid parameters, because
// neither ever calls initialize().
// ---------------------------------------------------------------------------
void emitGridIndependence(const SmileSpec& s) {
    const SmileSectionRNDCalculator wide(s.smile, 1000, 6.0);
    const SmileSectionRNDCalculator narrow(s.smile, 4, 0.25);
    const Time t = s.smile->exerciseTime();
    const Real x = std::log(s.forward) - 0.4;

    Obj gridA;
    gridA.i("n_strikes", 1000).n("n_std", 6.0);
    Obj gridB;
    gridB.i("n_strikes", 4).n("n_std", 0.25);

    Obj in = baseInputs(s, 200, 5.0);
    in.n("x", x).n("t", t).obj("grid_a", gridA).obj("grid_b", gridB);

    Obj ex;
    ex.n("pdf_grid_a", wide.pdf(x, t))
        .n("pdf_grid_b", narrow.pdf(x, t))
        .n("cdf_grid_a", wide.cdf(x, t))
        .n("cdf_grid_b", narrow.cdf(x, t));

    addCase(name(s.tag, "grid_independent_pdf_cdf"), in, ex);
}

// ---------------------------------------------------------------------------
// Guard clauses. Both the fact that C++ throws and the message it throws are
// pinned: the substring lives in the case inputs, so a port can assert on the
// same wording without this file having to embed a full, build-dependent
// exception string.
// ---------------------------------------------------------------------------
struct ThrowInfo {
    bool threw = false;
    bool found = false;
};

template <class F>
ThrowInfo probeThrow(F&& f, const std::string& needle) {
    try {
        f();
        return ThrowInfo{false, false};
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        return ThrowInfo{true, msg.find(needle) != std::string::npos};
    }
}

void emitGuard(const std::string& caseName, const std::string& needle,
               const ThrowInfo& info, Obj extraInputs = Obj()) {
    Obj in = extraInputs;
    in.s("expected_message_substring", needle);
    Obj ex;
    ex.b("throws", info.threw).b("message_contains_substring", info.found);
    addCase(caseName, in, ex);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    const std::vector<SmileSpec> all = smiles();
    const SmileSpec& flat = all[0];
    const SmileSpec& svi1 = all[1];

    // -----------------------------------------------------------------------
    // Per-smile: pdf/cdf ladder, invcdf ladder, grid endpoints, overloads,
    // grid independence -- all on the DEFAULT calculator (nStrikes 200, nStd 5).
    // -----------------------------------------------------------------------
    for (const SmileSpec& s : all) {
        emitPdfCdf(s);
        emitInvCdf(s, 200, 5.0, s.tag, probabilities());
        emitGridEndpoints(s, 200, 5.0, s.tag);
        emitOverloadAgreement(s);
        emitGridIndependence(s);
    }

    // -----------------------------------------------------------------------
    // Non-default grids. Only invcdf is affected, and it is affected a lot:
    //   nStd = 2   -> grid spans only +-2 ATM std, so extreme p clamp hard
    //   nStrikes 8 -> coarse spline, large and very characteristic bias
    //   nStrikes 4 -> the documented minimum
    //   1000/6.0   -> fine and wide, the converged end of the scale
    // -----------------------------------------------------------------------
    const std::vector<Prob> shortLadder = {
        {"p1em12", 1e-12}, {"p1em3", 1e-3},  {"p005", 0.05},          {"p050", 0.50},
        {"p095", 0.95},    {"p0999", 0.999}, {"p1m1em12", 1.0 - 1e-12},
    };
    const std::vector<Prob> midLadder = {
        {"p001", 0.01}, {"p010", 0.10}, {"p025", 0.25}, {"p050", 0.50},
        {"p075", 0.75}, {"p090", 0.90}, {"p099", 0.99},
    };
    const std::vector<Prob> tinyLadder = {
        {"p025", 0.25}, {"p050", 0.50}, {"p075", 0.75},
    };

    emitInvCdf(svi1, 200, 2.0, "svi1_nstd2", shortLadder);
    emitGridEndpoints(svi1, 200, 2.0, "svi1_nstd2");

    emitInvCdf(svi1, 8, 5.0, "svi1_nstrikes8", midLadder);
    emitGridEndpoints(svi1, 8, 5.0, "svi1_nstrikes8");

    emitInvCdf(svi1, 4, 5.0, "svi1_nstrikes4", tinyLadder);
    emitGridEndpoints(svi1, 4, 5.0, "svi1_nstrikes4");

    emitInvCdf(flat, 4, 5.0, "flat_nstrikes4", tinyLadder);
    emitInvCdf(flat, 1000, 6.0, "flat_nstrikes1000_nstd6", shortLadder);
    emitGridEndpoints(flat, 1000, 6.0, "flat_nstrikes1000_nstd6");

    // A very narrow grid (nStd = 0.5) makes the clamp dominate: every p outside
    // [cdf(kMin), cdf(kMax)] collapses onto the same two strikes.
    emitInvCdf(flat, 200, 0.5, "flat_nstd05", shortLadder);
    emitGridEndpoints(flat, 200, 0.5, "flat_nstd05");

    // -----------------------------------------------------------------------
    // AtmSmileSection is the documented way to supply a missing atm level.
    // Wrapping the atm-less flat smile at 100 must reproduce the direct
    // FlatSmileSection(atm = 100) results exactly.
    // -----------------------------------------------------------------------
    {
        const Smile wrapped = ext::make_shared<AtmSmileSection>(flatSmileWithoutAtm(), 100.0);
        const SmileSectionRNDCalculator wrappedCalc(wrapped);
        const SmileSectionRNDCalculator directCalc(flat.smile);
        const Time t = flat.smile->exerciseTime();
        const Real x = std::log(100.0) + 0.25;

        Obj in;
        in.s("smile", "AtmSmileSection(FlatSmileSection(vol=0.20, atm=Null), 100)")
            .n("forward", 100.0)
            .n("x", x)
            .n("p", 0.65)
            .n("t", t)
            .i("n_strikes", 200)
            .n("n_std", 5.0);

        Obj ex;
        ex.n("wrapped_exercise_time", wrapped->exerciseTime())
            .n("wrapped_atm_level", wrapped->atmLevel())
            .n("pdf_wrapped", wrappedCalc.pdf(x, t))
            .n("pdf_direct", directCalc.pdf(x, t))
            .n("cdf_wrapped", wrappedCalc.cdf(x, t))
            .n("cdf_direct", directCalc.cdf(x, t))
            .n("invcdf_wrapped", wrappedCalc.invcdf(0.65, t))
            .n("invcdf_direct", directCalc.invcdf(0.65, t));

        addCase("flat_via_atmsmilesection_matches_direct", in, ex);
    }

    // -----------------------------------------------------------------------
    // Constructor guards.
    // -----------------------------------------------------------------------
    emitGuard("ctor_rejects_null_smile", "null SmileSection",
              probeThrow([] { SmileSectionRNDCalculator c{Smile()}; }, "null SmileSection"),
              Obj().s("smile", "null"));

    emitGuard("ctor_rejects_n_strikes_3", "at least 4 strikes required",
              probeThrow([&] { SmileSectionRNDCalculator c{flat.smile, 3, 5.0}; },
                         "at least 4 strikes required"),
              Obj().s("smile", "flat").i("n_strikes", 3).n("n_std", 5.0));

    // nStrikes == 4 is the documented minimum and must be accepted, so this
    // case is expected to report throws = false.
    emitGuard("ctor_accepts_n_strikes_4", "at least 4 strikes required",
              probeThrow([&] { SmileSectionRNDCalculator c{flat.smile, 4, 5.0}; },
                         "at least 4 strikes required"),
              Obj().s("smile", "flat").i("n_strikes", 4).n("n_std", 5.0)
                  .s("note", "4 is the documented minimum; the ctor must accept it"));

    emitGuard("ctor_rejects_n_std_zero", "nStd must be positive",
              probeThrow([&] { SmileSectionRNDCalculator c{flat.smile, 200, 0.0}; },
                         "nStd must be positive"),
              Obj().s("smile", "flat").i("n_strikes", 200).n("n_std", 0.0));

    emitGuard("ctor_rejects_n_std_negative", "nStd must be positive",
              probeThrow([&] { SmileSectionRNDCalculator c{flat.smile, 200, -1.0}; },
                         "nStd must be positive"),
              Obj().s("smile", "flat").i("n_strikes", 200).n("n_std", -1.0));

    // -----------------------------------------------------------------------
    // invcdf probability range: strictly inside (0, 1).
    // -----------------------------------------------------------------------
    emitGuard("invcdf_rejects_p_zero", "p must be in (0, 1)",
              probeThrow([&] {
                  SmileSectionRNDCalculator c(flat.smile);
                  c.invcdf(0.0, kT);
              }, "p must be in (0, 1)"),
              Obj().s("smile", "flat").n("p", 0.0).n("t", kT));

    emitGuard("invcdf_rejects_p_one", "p must be in (0, 1)",
              probeThrow([&] {
                  SmileSectionRNDCalculator c(flat.smile);
                  c.invcdf(1.0, kT);
              }, "p must be in (0, 1)"),
              Obj().s("smile", "flat").n("p", 1.0).n("t", kT));

    emitGuard("invcdf_rejects_p_negative", "p must be in (0, 1)",
              probeThrow([&] {
                  SmileSectionRNDCalculator c(flat.smile);
                  c.invcdf(-0.25, kT);
              }, "p must be in (0, 1)"),
              Obj().s("smile", "flat").n("p", -0.25).n("t", kT));

    // -----------------------------------------------------------------------
    // checkTime(): every entry point rejects a time other than the smile's own
    // exercise time (compared with close_enough, not ==).
    // -----------------------------------------------------------------------
    emitGuard("pdf_rejects_time_mismatch", "does not match smile exercise time",
              probeThrow([&] {
                  SmileSectionRNDCalculator c(flat.smile);
                  c.pdf(std::log(100.0), 0.5);
              }, "does not match smile exercise time"),
              Obj().s("smile", "flat").n("t", 0.5).n("smile_exercise_time", kT));

    emitGuard("cdf_rejects_time_mismatch", "does not match smile exercise time",
              probeThrow([&] {
                  SmileSectionRNDCalculator c(flat.smile);
                  c.cdf(std::log(100.0), 0.5);
              }, "does not match smile exercise time"),
              Obj().s("smile", "flat").n("t", 0.5).n("smile_exercise_time", kT));

    emitGuard("invcdf_rejects_time_mismatch", "does not match smile exercise time",
              probeThrow([&] {
                  SmileSectionRNDCalculator c(flat.smile);
                  c.invcdf(0.5, 2.0);
              }, "does not match smile exercise time"),
              Obj().s("smile", "flat").n("t", 2.0).n("smile_exercise_time", kT));

    // -----------------------------------------------------------------------
    // Missing atm level. invcdf reports it from initialize(); cdf/pdf hit the
    // *SmileSection*'s own requirement inside optionPrice() instead, so the two
    // messages differ. This mirrors the v1.43 test-suite case
    // testSmileSectionRNDMissingAtmLevel in riskneutraldensitycalculator.cpp.
    // -----------------------------------------------------------------------
    emitGuard("invcdf_rejects_missing_atm_level", "wrap with AtmSmileSection",
              probeThrow([] {
                  SmileSectionRNDCalculator c(flatSmileWithoutAtm());
                  c.invcdf(0.5, kT);
              }, "wrap with AtmSmileSection"),
              Obj().s("smile", "FlatSmileSection(vol=0.20, atm=Null)").n("p", 0.5).n("t", kT));

    emitGuard("cdf_rejects_missing_atm_level", "smile section must provide atm level",
              probeThrow([] {
                  SmileSectionRNDCalculator c(flatSmileWithoutAtm());
                  c.cdf(std::log(100.0), kT);
              }, "smile section must provide atm level"),
              Obj().s("smile", "FlatSmileSection(vol=0.20, atm=Null)").n("t", kT));

    emitGuard("pdf_rejects_missing_atm_level", "smile section must provide atm level",
              probeThrow([] {
                  SmileSectionRNDCalculator c(flatSmileWithoutAtm());
                  c.pdf(std::log(100.0), kT);
              }, "smile section must provide atm level"),
              Obj().s("smile", "FlatSmileSection(vol=0.20, atm=Null)").n("t", kT));

    // invcdf runs checkTime -> initialize -> p-range check, so on an atm-less
    // smile an out-of-range p is reported as the atmLevel failure.
    {
        const ThrowInfo atmInfo = probeThrow([] {
            SmileSectionRNDCalculator c(flatSmileWithoutAtm());
            c.invcdf(-1.0, kT);
        }, "wrap with AtmSmileSection");
        const ThrowInfo pInfo = probeThrow([] {
            SmileSectionRNDCalculator c(flatSmileWithoutAtm());
            c.invcdf(-1.0, kT);
        }, "p must be in (0, 1)");

        Obj in;
        in.s("smile", "FlatSmileSection(vol=0.20, atm=Null)")
            .n("p", -1.0)
            .n("t", kT)
            .s("atm_substring", "wrap with AtmSmileSection")
            .s("p_range_substring", "p must be in (0, 1)");

        Obj ex;
        ex.b("throws", atmInfo.threw)
            .b("message_contains_atm_substring", atmInfo.found)
            .b("message_contains_p_range_substring", pInfo.found);

        addCase("invcdf_atm_check_precedes_p_check", in, ex);
    }

    emitDocument();
    return 0;
}
