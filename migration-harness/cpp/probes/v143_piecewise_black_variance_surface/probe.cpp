// migration-harness/cpp/probes/v143_piecewise_black_variance_surface/probe.cpp
//
// Reference values for PiecewiseBlackVarianceSurface
// (ql/termstructures/volatility/equityfx/piecewiseblackvariancesurface.{hpp,cpp}).
//
// A BlackVarianceTermStructure carrying one SmileSection per tenor and
// interpolating linearly in TOTAL VARIANCE between them. v1.43 gave it a
// smileSectionImpl override, which is what makes it the surface a smile-aware
// engine (GaussianCopulaSpreadEngine) can actually be driven by: at a tenor it
// hands back the ORIGINAL section object rather than a re-interpolation.
//
// What has to be pinned, and why
// ------------------------------
//  1. blackVarianceImpl has four regimes, and three of them are easy to get
//     subtly wrong:
//         t == 0            -> 0 exactly
//         t <= times[0]     -> var_0(K) * t / times[0]      (ramp from origin)
//         t >= times[n-1]   -> var_{n-1}(K) * t / times[n-1] (FLAT VOL, not
//                              flat variance -- variance keeps growing)
//         otherwise         -> linear in total variance between the brackets
//     A port that interpolates in VOLATILITY instead of variance, or that
//     freezes the variance past the last tenor, matches at the tenors and
//     nowhere else; the `*_variance_*` ladders below run between and beyond
//     them for exactly that reason.
//  2. smileSectionImpl returns smileSections_[i] when close_enough(t, times[i])
//     and otherwise falls through to BlackVolTermStructure::smileSectionImpl,
//     whose adapter reads back through blackVol. So AT a tenor the smile is the
//     parametric SVI shape (and its atmLevel is the SVI forward); BETWEEN
//     tenors it is an adapter whose atmLevel is Null<Real>(). Both halves are
//     recorded: `*_smile_at_tenor_*` and `*_smile_between_tenors`.
//  3. The strike range checked in sectionVariance is the SECTION's, not the
//     surface's -- the surface itself reports [QL_MIN_REAL, QL_MAX_REAL], so
//     the only way a strike is ever rejected is through the section. With an
//     SVI section (minStrike 0) a negative strike must therefore throw unless
//     extrapolation is enabled. `rejects_strike_below_section_min` /
//     `allows_strike_below_section_min_when_extrapolating` pin the pair.
//  4. Constructor guards: empty dates, dates/sections length mismatch, a first
//     date at or before the reference date, and unsorted dates.
//
// Not covered here: the static makeFromGrid factory. It builds one
// InterpolatedSmileSection<Linear> per column of a vol matrix as a migration
// path for BlackVarianceSurface-shaped input, and is not ported.
//
// Smiles used
// -----------
//   svi1  SviSmileSection(T, fwd=100, {0.04, 0.10, 0.30, -0.40,  0.00})
//   svi2  SviSmileSection(T, fwd= 96, {0.02, 0.08, 0.25, -0.30,  0.00})
//   flat  FlatSmileSection(T, 0.20, atm=100)
//
// svi1/svi2 are the parameter sets the v1.43 test-suite uses in
// testGaussianCopulaSpreadEngineSVI, so the sections here are the same objects
// the spread-engine probe drives its setups D and E with.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/piecewise/black/variance/surface.json. Nothing else may be
// printed.

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ql/errors.hpp>
#include <ql/experimental/volatility/svismilesection.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/piecewiseblackvariancesurface.hpp>
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

// 2025-03-01 -> 2026-03-01 is exactly 365 days and -> 2027-03-01 exactly 730,
// so with Actual365Fixed the tenor times are exactly 1.0 and 2.0.
const Date kToday(1, March, 2025);
const Date kT1(1, March, 2026);
const Date kT2(1, March, 2027);

const DayCounter& dayCounter() {
    static const DayCounter dc = Actual365Fixed();
    return dc;
}

using Smile = ext::shared_ptr<SmileSection>;

const std::vector<Real> kSvi1 = {0.04, 0.10, 0.30, -0.40, 0.0};
const std::vector<Real> kSvi2 = {0.02, 0.08, 0.25, -0.30, 0.0};

Smile sviSmile(Time t, Real forward, const std::vector<Real>& params) {
    return ext::make_shared<SviSmileSection>(t, forward, params);
}

Smile flatSmile(Time t, Real atm) {
    return ext::make_shared<FlatSmileSection>(t, 0.20, dayCounter(), atm);
}

using Surface = ext::shared_ptr<PiecewiseBlackVarianceSurface>;

// Single tenor at 1y: the shape setups D and E of the spread-engine probe use.
Surface singleTenorSurface(Real forward, const std::vector<Real>& params) {
    return ext::make_shared<PiecewiseBlackVarianceSurface>(
        kToday, kT1, sviSmile(1.0, forward, params), dayCounter());
}

// Two tenors at 1y and 2y, deliberately DIFFERENT SVI shapes so that
// interpolating in variance and interpolating in volatility give different
// answers between them.
Surface twoTenorSurface() {
    const std::vector<Date> dates = {kT1, kT2};
    const std::vector<Smile> smiles = {sviSmile(1.0, 100.0, kSvi1),
                                       sviSmile(2.0, 96.0, kSvi2)};
    return ext::make_shared<PiecewiseBlackVarianceSurface>(
        kToday, dates, smiles, dayCounter());
}

Obj baseInputs(const char* setup) {
    Obj o;
    o.s("setup", setup)
        .s("today", "2025-03-01")
        .s("day_counter", "Actual365Fixed");
    return o;
}

struct Sample {
    const char* tag;
    Time t;
};

// Times chosen to hit every regime: 0, inside the ramp, exactly on each tenor,
// strictly between the tenors, and beyond the last one.
const std::vector<Sample>& times() {
    static const std::vector<Sample> v = {
        {"t000", 0.0},   {"t025", 0.25}, {"t050", 0.5}, {"t100", 1.0},
        {"t125", 1.25},  {"t150", 1.5},  {"t175", 1.75}, {"t200", 2.0},
        {"t250", 2.5},   {"t400", 4.0},
    };
    return v;
}

const std::vector<Real>& strikes() {
    static const std::vector<Real> v = {60.0, 80.0, 96.0, 100.0, 120.0, 160.0};
    return v;
}

void emitVarianceLadder(const char* setup, const Surface& s) {
    for (const Sample& sample : times()) {
        for (Real k : strikes()) {
            Obj in = baseInputs(setup);
            in.n("t", sample.t).n("strike", k);
            Obj ex;
            ex.n("black_variance", s->blackVariance(sample.t, k, true))
                .n("black_vol", s->blackVol(sample.t, k, true));
            std::ostringstream name;
            name << setup << "_variance_" << sample.tag << "_k"
                 << static_cast<long long>(k);
            addCase(name.str(), in, ex);
        }
    }
}

// The smile view. At a tenor the section IS the SVI object, so its atmLevel is
// the SVI forward and its volatility is the SVI volatility. Off-tenor the base
// adapter takes over and its atmLevel is Null<Real>().
void emitSmileSection(const std::string& caseName, const char* setup,
                      const Surface& s, Time t, bool expectTenorSection) {
    const auto section = s->smileSection(t, true);
    Obj in = baseInputs(setup);
    in.n("t", t).b("expect_tenor_section", expectTenorSection);
    Obj ex;
    ex.n("exercise_time", section->exerciseTime())
        .b("atm_level_is_null", section->atmLevel() == Null<Real>())
        .n("volatility_at_80", section->volatility(80.0))
        .n("volatility_at_100", section->volatility(100.0))
        .n("volatility_at_120", section->volatility(120.0))
        .n("variance_at_100", section->variance(100.0));
    if (section->atmLevel() != Null<Real>())
        ex.n("atm_level", section->atmLevel());
    addCase(caseName, in, ex);
}

struct ThrowInfo {
    bool threw = false;
};

template <class F>
ThrowInfo probeThrow(F&& f) {
    try {
        f();
        return ThrowInfo{false};
    } catch (const std::exception&) {
        return ThrowInfo{true};
    }
}

void emitGuard(const std::string& caseName, const std::string& note,
               const ThrowInfo& info) {
    Obj in;
    in.s("note", note);
    Obj ex;
    ex.b("throws", info.threw);
    addCase(caseName, in, ex);
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;

    const Surface single = singleTenorSurface(100.0, kSvi1);
    const Surface twoTenor = twoTenorSurface();

    // -----------------------------------------------------------------------
    // Structure.
    // -----------------------------------------------------------------------
    {
        Obj ex;
        ex.n("max_time_single", single->maxTime())
            .n("max_time_two_tenor", twoTenor->maxTime())
            .b("min_strike_is_ql_min_real", single->minStrike() == QL_MIN_REAL)
            .b("max_strike_is_ql_max_real", single->maxStrike() == QL_MAX_REAL)
            .s("day_counter", single->dayCounter().name());
        addCase("structure", baseInputs("both"), ex);
    }

    // -----------------------------------------------------------------------
    // blackVariance / blackVol across every regime.
    // -----------------------------------------------------------------------
    emitVarianceLadder("single", single);
    emitVarianceLadder("two_tenor", twoTenor);

    // -----------------------------------------------------------------------
    // The smile view.
    // -----------------------------------------------------------------------
    emitSmileSection("single_smile_at_tenor_1y", "single", single, 1.0, true);
    emitSmileSection("single_smile_between_tenors", "single", single, 0.5, false);
    emitSmileSection("two_tenor_smile_at_tenor_1y", "two_tenor", twoTenor, 1.0, true);
    emitSmileSection("two_tenor_smile_at_tenor_2y", "two_tenor", twoTenor, 2.0, true);
    emitSmileSection("two_tenor_smile_between_tenors", "two_tenor", twoTenor, 1.5, false);

    // The section handed back at a tenor must be the SAME object that was put
    // in -- not a copy, and not a re-interpolation.
    {
        const auto section = sviSmile(1.0, 100.0, kSvi1);
        const auto surface = ext::make_shared<PiecewiseBlackVarianceSurface>(
            kToday, kT1, section, dayCounter());
        Obj ex;
        ex.b("same_object", surface->smileSection(1.0, true) == section);
        addCase("smile_at_tenor_is_the_input_section", baseInputs("single"), ex);
    }

    // -----------------------------------------------------------------------
    // Strike range: enforced by the SECTION, not the surface.
    // -----------------------------------------------------------------------
    emitGuard("rejects_strike_below_section_min",
              "SVI section minStrike is 0; the surface's own range is unbounded",
              probeThrow([&] { single->blackVariance(1.0, -1.0, false); }));
    {
        const Surface s = singleTenorSurface(100.0, kSvi1);
        s->enableExtrapolation();
        const ThrowInfo info = probeThrow([&] { s->blackVariance(1.0, -1.0, false); });
        emitGuard("allows_strike_below_section_min_when_extrapolating",
                  "enableExtrapolation() lifts the section's own strike check",
                  info);
    }

    // A FlatSmileSection reports an unbounded strike range, so the same strike
    // is fine there -- which shows the check really is the section's.
    {
        const auto s = ext::make_shared<PiecewiseBlackVarianceSurface>(
            kToday, kT1, flatSmile(1.0, 100.0), dayCounter());
        const ThrowInfo info = probeThrow([&] { s->blackVariance(1.0, -1.0, false); });
        emitGuard("flat_section_accepts_negative_strike",
                  "FlatSmileSection minStrike is QL_MIN_REAL - shift", info);
    }

    // -----------------------------------------------------------------------
    // Constructor guards.
    // -----------------------------------------------------------------------
    emitGuard("ctor_rejects_empty_dates", "at least one date is required",
              probeThrow([] {
                  PiecewiseBlackVarianceSurface s(kToday, std::vector<Date>{},
                                                  std::vector<Smile>{}, dayCounter());
              }));

    emitGuard("ctor_rejects_length_mismatch", "2 dates, 1 smile section",
              probeThrow([] {
                  PiecewiseBlackVarianceSurface s(
                      kToday, std::vector<Date>{kT1, kT2},
                      std::vector<Smile>{sviSmile(1.0, 100.0, kSvi1)}, dayCounter());
              }));

    emitGuard("ctor_rejects_first_date_at_reference",
              "first date must be strictly after the reference date",
              probeThrow([] {
                  PiecewiseBlackVarianceSurface s(
                      kToday, std::vector<Date>{kToday},
                      std::vector<Smile>{sviSmile(1.0, 100.0, kSvi1)}, dayCounter());
              }));

    emitGuard("ctor_rejects_unsorted_dates", "dates must be sorted and unique",
              probeThrow([] {
                  PiecewiseBlackVarianceSurface s(
                      kToday, std::vector<Date>{kT2, kT1},
                      std::vector<Smile>{sviSmile(2.0, 96.0, kSvi2),
                                         sviSmile(1.0, 100.0, kSvi1)},
                      dayCounter());
              }));

    emitGuard("ctor_rejects_duplicate_dates", "dates must be unique",
              probeThrow([] {
                  PiecewiseBlackVarianceSurface s(
                      kToday, std::vector<Date>{kT1, kT1},
                      std::vector<Smile>{sviSmile(1.0, 100.0, kSvi1),
                                         sviSmile(1.0, 96.0, kSvi2)},
                      dayCounter());
              }));

    emitDocument();
    return 0;
}
