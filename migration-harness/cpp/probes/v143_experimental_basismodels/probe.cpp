// migration-harness/cpp/probes/v143_experimental_basismodels/probe.cpp
//
// Reference values for the two ql/experimental/basismodels smile sections
// @ v1.43:
//
//   TenorOptionletVTS::TenorOptionletSmileSection  (tenoroptionletvts.hpp:43-66,
//                                                   .cpp:51-107)
//   TenorSwaptionVTS::TenorSwaptionSmileSection    (tenorswaptionvts.hpp:39-64,
//                                                   .cpp:37-135)
//
// WHY THIS PROBE LOOKS THE WAY IT DOES
// ------------------------------------
// Both classes are *rescalers*: they take a smile section off a base-tenor
// volatility structure, shift/scale the strike, evaluate the base smile there,
// and scale the resulting normal vol back. Concretely
//
//   optionlet:  vol(K)^2 = sum_ij  rho_ij v_i v_j volBase_i(K_i) volBase_j(K_j)
//               with     K_i = (K - (F_targ - (sum v) F_base_i)) / (sum v)
//   swaption:   vol(K)   = A * volBase((K - C) / A)
//               with     A = annuityScaling (1+lambda),
//                        C = swapRateTarg - (1+lambda) swapRateBase
//
// If the base structure is FLAT IN STRIKE, none of the strike arithmetic is
// observable: volBase_i(K_i) == volBase_i(anything), so a port that got the
// strike transform completely wrong would still match. A flat base therefore
// pins only the *aggregation* (the v_i / rho_ij sum, and the A scaling).
//
// So each class is probed twice:
//
//   Blocks A/C  flat-in-strike Normal base. Pins the aggregation, the
//               atmLevel/exerciseTime/shift/volatilityType metadata, and the
//               VTS -> smileSection delegation.
//
//   Blocks B/D  a *smiley* Normal base. With a strike-dependent base the
//               strike transform becomes observable, and minStrike() /
//               maxStrike() become finite numbers instead of +-QL_MAX_REAL.
//
// The bases are the analytic quadratic fixtures QuadOptionletVol /
// QuadSwaptionVol defined below (b = c = 0 gives the flat case). Their
// smileSectionImpl and volatilityImpl return the SAME closed form, so the
// fixture carries no interpolation of its own and can be mirrored exactly in
// Python. That also matters because PQuantLib's OptionletVolatilityStructure
// has no smileSection() accessor, so its TenorOptionletSmileSection reads the
// base vol through baseVTS->volatility(fixingDate, K); a fixture where the two
// agree keeps that substitution provably neutral (see the report).
//
// NOT used as a base: ConstantOptionletVolatility. Its smileSectionImpl drops
// the volatility type (constantoptionletvol.cpp:74-84 builds a FlatSmileSection
// without passing volatilityType()/displacement(), unlike its swaption twin
// swaptionconstantvol.cpp:82-96 which does pass them). The section therefore
// reports ShiftedLognormal with a Null<Real> atmLevel, so the
// volatility(K, Normal, 0.0) call in TenorOptionletSmileSection::volatilityImpl
// takes the type-conversion branch and aborts with "smile section must provide
// atm level". Block G pins that abort as observed behaviour of v1.43.
//
// Block E pins TenorOptionletVTS::TwoParameterCorrelation, including the fact
// that its interpolation lookups use the DEFAULT allowExtrapolation=false and
// therefore throw outside the parameter grid.
//
// Emits JSON on stdout; redirect to
// references/v143/experimental/basismodels.json.

#include <ql/currencies/europe.hpp>
#include <ql/experimental/basismodels/tenoroptionletvts.hpp>
#include <ql/experimental/basismodels/tenorswaptionvts.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/iborindex.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/volatility/optionlet/optionletvolatilitystructure.hpp>
#include <ql/termstructures/volatility/smilesection.hpp>
#include <ql/termstructures/volatility/swaption/swaptionconstantvol.hpp>
#include <ql/termstructures/volatility/swaption/swaptionvolstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

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

void emit_int(const std::string& name, long long v) {
    sep();
    std::cout << "  \"" << name << "\": " << v;
}

void emit_bool(const std::string& name, bool v) {
    sep();
    std::cout << "  \"" << name << "\": " << (v ? "true" : "false");
}

void emit_str(const std::string& name, const std::string& v) {
    sep();
    std::cout << "  \"" << name << "\": \"" << v << "\"";
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

// --------------------------------------------------------------------------
// Shared market data
// --------------------------------------------------------------------------

const Date today(15, January, 2024);
const Actual365Fixed dcA365;
const Actual360 dc360;
const NullCalendar ncal;
const TARGET tcal;

// the strike grid used everywhere: below / at / above the 3% ATM level
const std::vector<Rate> kGrid = {-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08};

Handle<YieldTermStructure> flatCurve(Real r) {
    return Handle<YieldTermStructure>(ext::make_shared<FlatForward>(today, r, dcA365));
}

// --------------------------------------------------------------------------
// Analytic quadratic fixtures
//
//   optionlet:  vol(t, K)      = (a0 + aT t)         + b (K - k0) + c (K - k0)^2
//   swaption:   vol(t, len, K) = (a0 + aT t + aL len) + b (K - k0) + c (K - k0)^2
//
// Normal type, zero shift, finite min/max strike. b = c = 0 gives the
// flat-in-strike case.
// --------------------------------------------------------------------------

const Real kQuadK0 = 0.03;         // expansion point
const Real kQuadMinStrike = -0.02;
const Real kQuadMaxStrike = 0.12;

Real quadVol(Real a, Real b, Real c, Rate strike) {
    Real x = strike - kQuadK0;
    return a + b * x + c * x * x;
}

class QuadSmileSection : public SmileSection {
  public:
    QuadSmileSection(Time t, Real a, Real b, Real c, const DayCounter& dc)
    : SmileSection(t, dc, Normal, 0.0), a_(a), b_(b), c_(c) {}

    Real minStrike() const override { return kQuadMinStrike; }
    Real maxStrike() const override { return kQuadMaxStrike; }
    Real atmLevel() const override { return Null<Real>(); }

  protected:
    Volatility volatilityImpl(Rate strike) const override { return quadVol(a_, b_, c_, strike); }

  private:
    Real a_, b_, c_;
};

class QuadOptionletVol : public OptionletVolatilityStructure {
  public:
    QuadOptionletVol(Real a0, Real aT, Real b, Real c)
    : OptionletVolatilityStructure(today, ncal, ModifiedFollowing, dcA365), a0_(a0), aT_(aT),
      b_(b), c_(c) {}

    Date maxDate() const override { return Date::maxDate(); }
    Rate minStrike() const override { return kQuadMinStrike; }
    Rate maxStrike() const override { return kQuadMaxStrike; }
    VolatilityType volatilityType() const override { return Normal; }

  protected:
    ext::shared_ptr<SmileSection> smileSectionImpl(Time t) const override {
        return ext::make_shared<QuadSmileSection>(t, a0_ + aT_ * t, b_, c_, dayCounter());
    }
    Volatility volatilityImpl(Time t, Rate strike) const override {
        return quadVol(a0_ + aT_ * t, b_, c_, strike);
    }

  private:
    Real a0_, aT_, b_, c_;
};

class QuadSwaptionVol : public SwaptionVolatilityStructure {
  public:
    QuadSwaptionVol(Real a0, Real aT, Real aL, Real b, Real c)
    : SwaptionVolatilityStructure(today, tcal, ModifiedFollowing, dcA365),
      maxSwapTenor_(100, Years), a0_(a0), aT_(aT), aL_(aL), b_(b), c_(c) {}

    Date maxDate() const override { return Date::maxDate(); }
    Rate minStrike() const override { return kQuadMinStrike; }
    Rate maxStrike() const override { return kQuadMaxStrike; }
    const Period& maxSwapTenor() const override { return maxSwapTenor_; }
    VolatilityType volatilityType() const override { return Normal; }

  protected:
    ext::shared_ptr<SmileSection> smileSectionImpl(Time t, Time len) const override {
        return ext::make_shared<QuadSmileSection>(t, a0_ + aT_ * t + aL_ * len, b_, c_,
                                                  dayCounter());
    }
    Volatility volatilityImpl(Time t, Time len, Rate strike) const override {
        return quadVol(a0_ + aT_ * t + aL_ * len, b_, c_, strike);
    }

  private:
    Period maxSwapTenor_;
    Real a0_, aT_, aL_, b_, c_;
};

// flat-in-strike and smiley parameter sets
ext::shared_ptr<OptionletVolatilityStructure> flatOptionletVol() {
    return ext::make_shared<QuadOptionletVol>(0.0070, 0.0, 0.0, 0.0);
}
ext::shared_ptr<OptionletVolatilityStructure> smileyOptionletVol() {
    return ext::make_shared<QuadOptionletVol>(0.0060, 0.0002, -0.010, 0.30);
}
ext::shared_ptr<SwaptionVolatilityStructure> smileySwaptionVol() {
    return ext::make_shared<QuadSwaptionVol>(0.0080, 0.0002, 0.0001, -0.010, 0.30);
}

// --------------------------------------------------------------------------
// TenorOptionletVTS construction helpers
// --------------------------------------------------------------------------

ext::shared_ptr<IborIndex> makeIbor(const std::string& name,
                                    const Period& tenor,
                                    const Handle<YieldTermStructure>& curve) {
    return ext::make_shared<IborIndex>(name, tenor, 2, EURCurrency(), ncal, ModifiedFollowing,
                                       false, dc360, curve);
}

// TwoParameterCorrelation over a two-point linear grid. The Interpolation
// objects must outlive the correlation, so they are heap-allocated and the
// x/y vectors are leaked deliberately (probe process, single shot).
ext::shared_ptr<TenorOptionletVTS::CorrelationStructure>
makeCorrelation(Real t0, Real t1, Real rho0, Real rho1, Real beta0, Real beta1) {
    auto* xs = new std::vector<Real>{t0, t1};
    auto* rhos = new std::vector<Real>{rho0, rho1};
    auto* betas = new std::vector<Real>{beta0, beta1};
    ext::shared_ptr<Interpolation> rhoInf(
        new LinearInterpolation(xs->begin(), xs->end(), rhos->begin()));
    ext::shared_ptr<Interpolation> beta(
        new LinearInterpolation(xs->begin(), xs->end(), betas->begin()));
    return ext::make_shared<TenorOptionletVTS::TwoParameterCorrelation>(rhoInf, beta);
}

// Dump every SmileSection observable for one section.
void emitSection(const std::string& prefix, const ext::shared_ptr<SmileSection>& sec) {
    emit(prefix + "_exercise_time", sec->exerciseTime());
    emit(prefix + "_atm_level", sec->atmLevel());
    emit(prefix + "_min_strike", sec->minStrike());
    emit(prefix + "_max_strike", sec->maxStrike());
    emit(prefix + "_shift", sec->shift());
    emit_int(prefix + "_vol_type_is_normal", sec->volatilityType() == Normal ? 1 : 0);

    std::vector<Real> vols;
    vols.reserve(kGrid.size());
    for (Rate k : kGrid)
        vols.push_back(sec->volatility(k));
    emit_arr(prefix + "_vols", vols);

    // variance couples the vol to the exercise time
    emit(prefix + "_variance_atm", sec->variance(0.03));

    // the boundary strikes the section itself advertises
    emit(prefix + "_vol_at_min_strike", sec->volatility(sec->minStrike()));
    emit(prefix + "_vol_at_max_strike", sec->volatility(sec->maxStrike()));
}

// --------------------------------------------------------------------------
// Block A - TenorOptionletSmileSection over a FLAT-in-strike Normal base
// --------------------------------------------------------------------------
void block_optionlet_flat() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto base3m = makeIbor("Base3M", Period(3, Months), curve);
    auto targ6m = makeIbor("Targ6M", Period(6, Months), curve);

    Handle<OptionletVolatilityStructure> baseVol(flatOptionletVol());

    auto corr = makeCorrelation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1);
    TenorOptionletVTS vts(baseVol, base3m, targ6m, corr);

    const std::vector<Time> times = {1.0, 2.5, 5.0};
    for (Size i = 0; i < times.size(); ++i) {
        std::string p = "optflat_t" + std::to_string(i);
        emit(p + "_option_time", times[i]);
        emitSection(p, vts.smileSection(times[i], true));
        // the VTS must delegate to the section
        emit(p + "_vts_vol_atm", vts.volatility(times[i], 0.03, true));
        emit(p + "_vts_black_variance_atm", vts.blackVariance(times[i], 0.03, true));
    }
    emit_int("optflat_vts_vol_type_is_normal", vts.volatilityType() == Normal ? 1 : 0);
    emit("optflat_vts_min_strike", vts.minStrike());
    emit("optflat_vts_max_strike", vts.maxStrike());
    emit_int("optflat_vts_max_date_serial", (long long)vts.maxDate().serialNumber());
}

// --------------------------------------------------------------------------
// Block B - TenorOptionletSmileSection over a SMILEY Normal base
// --------------------------------------------------------------------------
void block_optionlet_smiley() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto base3m = makeIbor("Base3M", Period(3, Months), curve);
    auto targ6m = makeIbor("Targ6M", Period(6, Months), curve);

    Handle<OptionletVolatilityStructure> baseVol(smileyOptionletVol());

    // constant-parameter correlation
    auto corrFlat = makeCorrelation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1);
    TenorOptionletVTS vtsFlatCorr(baseVol, base3m, targ6m, corrFlat);

    const std::vector<Time> times = {1.0, 5.0, 12.0};
    for (Size i = 0; i < times.size(); ++i) {
        std::string p = "optsm_t" + std::to_string(i);
        emit(p + "_option_time", times[i]);
        emitSection(p, vtsFlatCorr.smileSection(times[i], true));
        emit(p + "_vts_vol_atm", vtsFlatCorr.volatility(times[i], 0.03, true));
    }

    // term-structure-of-correlation: rhoInf and beta both vary with start time,
    // so the interpolated lookup at startTimeBase_[i] is observable.
    auto corrTerm = makeCorrelation(0.0, 30.0, 0.2, 0.8, 0.05, 0.50);
    TenorOptionletVTS vtsTermCorr(baseVol, base3m, targ6m, corrTerm);
    for (Size i = 0; i < times.size(); ++i) {
        std::string p = "optsmc_t" + std::to_string(i);
        emitSection(p, vtsTermCorr.smileSection(times[i], true));
    }

    // a 12M target over a 3M base: four base FRAs instead of two, so the
    // double sum has six cross terms rather than one.
    auto targ12m = makeIbor("Targ12M", Period(12, Months), curve);
    TenorOptionletVTS vts4(baseVol, base3m, targ12m, corrTerm);
    emitSection("optsm4_t0", vts4.smileSection(5.0, true));

    // a TARGET-calendar variant: the internal base schedule now rolls off
    // business days, which moves every fixing date and year fraction.
    //
    // # C++ parity note: optionTime is 4.0, not 5.0. TenorOptionletSmileSection
    // turns optionTime into an exercise date by raw day arithmetic --
    // referenceDate + round(optionTime / oneDayAsYear) -- with NO calendar
    // adjustment (tenoroptionletvts.cpp:57-60). 2024-01-15 + 1825d = Saturday
    // 2029-01-13, and targIndex_->fixing() then rejects it as an invalid fixing
    // date. 4.0 -> 1460d -> Friday 2028-01-14, which is a TARGET business day.
    // The fragility is C++'s; it is reproduced, not fixed.
    auto base3mT = ext::make_shared<Euribor3M>(curve);
    auto targ6mT = ext::make_shared<Euribor6M>(curve);
    TenorOptionletVTS vtsCal(baseVol, base3mT, targ6mT, corrTerm);
    emitSection("optsmcal_t0", vtsCal.smileSection(4.0, true));

    // and the weekend-exercise-date failure itself, pinned.
    bool weekendThrew = false;
    try {
        (void)vtsCal.smileSection(5.0, true);
    } catch (const std::exception&) {
        weekendThrew = true;
    }
    emit_bool("optsmcal_weekend_exercise_throws", weekendThrew);
}

// --------------------------------------------------------------------------
// Block C - TenorSwaptionSmileSection over a FLAT Normal base
// --------------------------------------------------------------------------
void block_swaption_flat() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto base6m = ext::make_shared<Euribor6M>(curve);
    auto targ3m = ext::make_shared<Euribor3M>(curve);

    Handle<SwaptionVolatilityStructure> baseVol(ext::make_shared<ConstantSwaptionVolatility>(
        today, tcal, ModifiedFollowing, 0.0090, dcA365, Normal));

    TenorSwaptionVTS vts(baseVol, curve, base6m, targ3m, Period(1, Years), Period(1, Years),
                         Thirty360(Thirty360::BondBasis), Thirty360(Thirty360::BondBasis));

    const std::vector<Time> optT = {5.0, 2.0};
    const std::vector<Time> swpL = {10.0, 5.0};
    for (Size i = 0; i < optT.size(); ++i) {
        std::string p = "swpflat_c" + std::to_string(i);
        emit(p + "_option_time", optT[i]);
        emit(p + "_swap_length", swpL[i]);
        emitSection(p, vts.smileSection(optT[i], swpL[i], true));
        emit(p + "_vts_vol_atm", vts.volatility(optT[i], swpL[i], 0.03, true));
        emit(p + "_vts_black_variance_atm", vts.blackVariance(optT[i], swpL[i], 0.03, true));
    }
    emit_int("swpflat_vts_vol_type_is_normal", vts.volatilityType() == Normal ? 1 : 0);
    emit("swpflat_vts_min_strike", vts.minStrike());
    emit("swpflat_vts_max_strike", vts.maxStrike());
    emit_int("swpflat_vts_max_date_serial", (long long)vts.maxDate().serialNumber());
    emit_int("swpflat_vts_max_swap_tenor_years", (long long)vts.maxSwapTenor().length());
}

// --------------------------------------------------------------------------
// Block D - TenorSwaptionSmileSection over a SMILEY Normal base
// --------------------------------------------------------------------------
void block_swaption_smiley() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto base6m = ext::make_shared<Euribor6M>(curve);
    auto targ3m = ext::make_shared<Euribor3M>(curve);

    Handle<SwaptionVolatilityStructure> baseVol(smileySwaptionVol());

    TenorSwaptionVTS vts(baseVol, curve, base6m, targ3m, Period(1, Years), Period(1, Years),
                         Thirty360(Thirty360::BondBasis), Thirty360(Thirty360::BondBasis));

    const std::vector<Time> optT = {5.0, 2.0, 10.0};
    const std::vector<Time> swpL = {10.0, 5.0, 2.0};
    for (Size i = 0; i < optT.size(); ++i) {
        std::string p = "swpsm_c" + std::to_string(i);
        emit(p + "_option_time", optT[i]);
        emit(p + "_swap_length", swpL[i]);
        emitSection(p, vts.smileSection(optT[i], swpL[i], true));
        emit(p + "_vts_vol_atm", vts.volatility(optT[i], swpL[i], 0.03, true));
    }

    // a semi-annual target fixed leg + Actual/360 target fixed day count:
    // both feed finlSwap only, so they move annuityScaling_ and swapRateFinl_
    // (i.e. atmLevel) without touching lambda_.
    TenorSwaptionVTS vts2(baseVol, curve, base6m, targ3m, Period(1, Years), Period(6, Months),
                          Thirty360(Thirty360::BondBasis), Actual360());
    emitSection("swpsm2_c0", vts2.smileSection(5.0, 10.0, true));

    // base 3M / target 6M - the reverse basis direction.
    TenorSwaptionVTS vts3(baseVol, curve, targ3m, base6m, Period(1, Years), Period(1, Years),
                          Thirty360(Thirty360::BondBasis), Thirty360(Thirty360::BondBasis));
    emitSection("swpsm3_c0", vts3.smileSection(5.0, 10.0, true));

    // # C++ parity note: a NON-INTEGRAL swap length. The maturity date is
    // built as ``((BigInteger)swapLength * 12.0) * Months``
    // (tenorswaptionvts.cpp:48-49) -- the cast to BigInteger binds tighter than
    // the multiplication, so swapLength is truncated to whole YEARS first:
    // 7.5 -> 7 -> 84 months, NOT 90. Only the base smile section itself sees
    // the raw 7.5. Pinned here because a port that writes int(7.5 * 12) builds
    // a different swap and cannot match these numbers.
    emitSection("swpsmlen_c0", vts.smileSection(5.0, 7.5, true));
    emitSection("swpsmlen_c1", vts.smileSection(5.0, 7.0, true));
    emitSection("swpsmlen_c2", vts.smileSection(5.0, 7.9999, true));
}

// --------------------------------------------------------------------------
// Block E - TwoParameterCorrelation itself
// --------------------------------------------------------------------------
void block_correlation() {
    auto corr = makeCorrelation(0.0, 10.0, 0.2, 0.8, 0.05, 0.50);
    const std::vector<std::pair<Real, Real>> pts = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0},  {2.5, 7.5},
        {5.0, 5.0}, {5.0, 5.25}, {7.5, 2.5}, {10.0, 0.0},
    };
    std::vector<Real> vals;
    vals.reserve(pts.size());
    for (const auto& pr : pts)
        vals.push_back((*corr)(pr.first, pr.second));
    emit_arr("corr_values", vals);

    // C++ TwoParameterCorrelation::operator() uses Interpolation::operator()
    // with the DEFAULT allowExtrapolation=false, so a start time outside the
    // parameter grid is an error, not an extrapolation.
    bool threwLow = false;
    try {
        (void)(*corr)(-1.0, 0.0);
    } catch (const std::exception&) {
        threwLow = true;
    }
    bool threwHigh = false;
    try {
        (void)(*corr)(10.5, 0.0);
    } catch (const std::exception&) {
        threwHigh = true;
    }
    // only start1 is looked up; start2 is used only through |start2 - start1|
    bool threwSecond = false;
    try {
        (void)(*corr)(5.0, 1.0e6);
    } catch (const std::exception&) {
        threwSecond = true;
    }
    emit_bool("corr_throws_below_grid", threwLow);
    emit_bool("corr_throws_above_grid", threwHigh);
    emit_bool("corr_throws_on_second_arg", threwSecond);
    emit("corr_far_second_arg_value", (*corr)(5.0, 1.0e6));

    // and the same, reached through the smile section: a grid that does not
    // cover the base FRA start times must make volatilityImpl throw.
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto base3m = makeIbor("Base3M", Period(3, Months), curve);
    auto targ6m = makeIbor("Targ6M", Period(6, Months), curve);
    Handle<OptionletVolatilityStructure> baseVol(smileyOptionletVol());
    auto narrow = makeCorrelation(0.0, 1.0, 0.2, 0.8, 0.05, 0.50);
    TenorOptionletVTS vtsNarrow(baseVol, base3m, targ6m, narrow);
    bool sectionThrew = false;
    try {
        (void)vtsNarrow.smileSection(5.0, true)->volatility(0.03);
    } catch (const std::exception&) {
        sectionThrew = true;
    }
    emit_bool("corr_section_throws_off_grid", sectionThrew);
}

// --------------------------------------------------------------------------
// Block F - the frequency precondition on TenorOptionletVTS
// --------------------------------------------------------------------------
void block_precondition() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto base6m = makeIbor("Base6M", Period(6, Months), curve);
    auto targ3m = makeIbor("Targ3M", Period(3, Months), curve);
    Handle<OptionletVolatilityStructure> baseVol(smileyOptionletVol());
    auto corr = makeCorrelation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1);
    bool threw = false;
    try {
        TenorOptionletVTS vts(baseVol, base6m, targ3m, corr);
    } catch (const std::exception&) {
        threw = true;
    }
    emit_bool("opt_ctor_throws_on_bad_frequency", threw);
}

// --------------------------------------------------------------------------
// Block G - the ConstantOptionletVolatility base defect
//
// ConstantOptionletVolatility::smileSectionImpl (constantoptionletvol.cpp:74-84)
// builds its FlatSmileSection WITHOUT forwarding volatilityType() /
// displacement(), so the section always claims ShiftedLognormal with a
// Null<Real> atmLevel -- even when the surface itself was constructed with
// type = Normal. TenorOptionletSmileSection::volatilityImpl asks that section
// for volatility(K, Normal, 0.0); the types disagree, the conversion branch is
// taken, and SmileSection::volatility (smilesection.cpp:119-143) aborts on the
// null atmLevel.
//
// Consequence: TenorOptionletVTS cannot be layered on a
// ConstantOptionletVolatility at all in v1.43. Pinned here so the Python port
// has the C++ verdict on record.
// --------------------------------------------------------------------------
void block_constant_optionlet_base() {
    Handle<YieldTermStructure> curve = flatCurve(0.03);
    auto base3m = makeIbor("Base3M", Period(3, Months), curve);
    auto targ6m = makeIbor("Targ6M", Period(6, Months), curve);
    auto corr = makeCorrelation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1);

    for (int i = 0; i < 2; ++i) {
        VolatilityType type = (i == 0) ? Normal : ShiftedLognormal;
        Handle<OptionletVolatilityStructure> baseVol(ext::make_shared<ConstantOptionletVolatility>(
            today, ncal, ModifiedFollowing, 0.0070, dcA365, type));
        // the surface reports the type it was built with ...
        emit_int(std::string("cov_") + (i == 0 ? "normal" : "sln") + "_surface_is_normal",
                 baseVol->volatilityType() == Normal ? 1 : 0);
        // ... but its smile section does not.
        emit_int(std::string("cov_") + (i == 0 ? "normal" : "sln") + "_section_is_normal",
                 baseVol->smileSection(5.0, true)->volatilityType() == Normal ? 1 : 0);

        TenorOptionletVTS vts(baseVol, base3m, targ6m, corr);
        bool threw = false;
        try {
            (void)vts.smileSection(5.0, true)->volatility(0.03);
        } catch (const std::exception&) {
            threw = true;
        }
        emit_bool(std::string("cov_") + (i == 0 ? "normal" : "sln") + "_section_vol_throws",
                  threw);
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = today;
    std::cout << "{\n";

    emit_int("meta_eval_date_serial", (long long)today.serialNumber());

    block_optionlet_flat();
    block_optionlet_smiley();
    block_swaption_flat();
    block_swaption_smiley();
    block_correlation();
    block_precondition();
    block_constant_optionlet_base();

    std::cout << "\n}\n";
    return 0;
}
