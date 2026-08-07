// migration-harness/cpp/probes/v143_ts_blackvoldelta/probe.cpp
//
// Pins QuantLib v1.43 BlackVolatilitySurfaceDelta
// (ql/termstructures/volatility/equityfx/blackvolsurfacedelta.{hpp,cpp}).
//
// The class is a BlackVolatilityTermStructure whose market data is quoted in
// DELTA space: one BlackVarianceCurve per delta column, and at query time the
// deltas are converted to strikes with a BlackDeltaCalculator before an
// InterpolatedSmileSection (or FlatSmileSection) is built over the resulting
// (strike, stdDev) pairs.  Three things can silently go wrong in a port and are
// therefore pinned separately:
//
//  1. the delta -> strike inversion itself, per DeltaType and per AtmType.
//     Section "strike_from_delta" reproduces blackVolSmile()'s own arithmetic
//     (blackvolsurfacedelta.cpp:120-158) with a standalone BlackDeltaCalculator
//     so a wrong convention shows up as a strike, not as a smeared vol.
//  2. the smile assembly: strikes are de-duplicated through a std::map with a
//     close()-based comparator (cpp:115-116) and the *first* delta to produce a
//     given strike wins.  minStrike()/maxStrike() of the returned SmileSection
//     expose the resulting pillar set.
//  3. the switchTenor split: below switchTime_ the short-term
//     delta/ATM conventions apply, at or above it the long-term ones
//     (cpp:103-111).  Probed with a 1Y switch and deliberately different
//     conventions on the two sides.
//
// Plus: all four SmileInterpolationMethods, flatStrikeExtrapolation on/off,
// the strike == 0 ATM short-circuit (cpp:219-227), time extrapolation past the
// last pillar (cpp:216-217) and every constructor precondition.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/blackvoldelta.json.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ql/math/matrix.hpp>
#include <ql/pricingengines/blackdeltacalculator.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackvolsurfacedelta.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

// --- fixed market ---------------------------------------------------------

const Date kRefDate(15, June, 2023);
const Real kSpot = 1.25;
const Rate kDomesticRate = 0.03;
const Rate kForeignRate = 0.01;

std::vector<Date> expiryDates() {
    return {Date(17, July, 2023), Date(15, September, 2023), Date(15, December, 2023),
            Date(17, June, 2024), Date(16, June, 2025)};
}

std::vector<Real> putDeltas() { return {-0.10, -0.25}; }
std::vector<Real> callDeltas() { return {0.25, 0.10}; }

// rows = expiries, columns = [put -10, put -25, ATM, call 25, call 10].
// Deliberately asymmetric (skew + smile) so a transposed or mis-ordered matrix
// cannot reproduce these numbers.
Matrix volMatrix() {
    Matrix m(5, 5);
    const Real data[5][5] = {{0.1420, 0.1235, 0.1150, 0.1198, 0.1330},
                             {0.1385, 0.1215, 0.1140, 0.1186, 0.1305},
                             {0.1350, 0.1198, 0.1132, 0.1175, 0.1284},
                             {0.1312, 0.1180, 0.1125, 0.1166, 0.1262},
                             {0.1290, 0.1168, 0.1120, 0.1160, 0.1248}};
    for (Size i = 0; i < 5; ++i)
        for (Size j = 0; j < 5; ++j)
            m[i][j] = data[i][j];
    return m;
}

const Real kStrikes[] = {0.90, 1.05, 1.15, 1.25, 1.30, 1.45, 1.60, 1.90};
// 2.4 sits past the last pillar (732/365 = 2.0055) so the time-extrapolation
// branch is exercised; 0.02 sits before the first (32/365 = 0.0877).
const Time kTimes[] = {0.02, 0.0876712328767123, 0.25, 0.5, 1.0, 1.5, 2.0054794520547945, 2.4};

Handle<Quote> spotHandle() {
    return Handle<Quote>(ext::make_shared<SimpleQuote>(kSpot));
}

Handle<YieldTermStructure> flatCurve(Rate r) {
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(kRefDate, r, Actual365Fixed()));
}

ext::shared_ptr<BlackVolatilitySurfaceDelta>
makeSurface(DeltaVolQuote::DeltaType deltaType,
            DeltaVolQuote::AtmType atmType,
            ext::optional<DeltaVolQuote::DeltaType> atmDeltaType,
            BlackVolatilitySurfaceDelta::SmileInterpolationMethod im,
            bool flatStrikeExtrapolation,
            bool hasAtm = true,
            const Period& switchTenor = 0 * Days,
            DeltaVolQuote::DeltaType longTermDeltaType = DeltaVolQuote::DeltaType::Fwd,
            DeltaVolQuote::AtmType longTermAtmType = DeltaVolQuote::AtmType::AtmDeltaNeutral,
            ext::optional<DeltaVolQuote::DeltaType> longTermAtmDeltaType = ext::nullopt) {
    Matrix m = volMatrix();
    if (!hasAtm) {
        // drop the ATM column (index 2) -> 4 columns
        Matrix m4(5, 4);
        for (Size i = 0; i < 5; ++i) {
            m4[i][0] = m[i][0];
            m4[i][1] = m[i][1];
            m4[i][2] = m[i][3];
            m4[i][3] = m[i][4];
        }
        m = m4;
    }
    return ext::make_shared<BlackVolatilitySurfaceDelta>(
        kRefDate, expiryDates(), putDeltas(), callDeltas(), hasAtm, m, Actual365Fixed(),
        TARGET(), spotHandle(), flatCurve(kDomesticRate), flatCurve(kForeignRate), deltaType,
        atmType, atmDeltaType, im, flatStrikeExtrapolation,
        BlackVolTimeExtrapolation::FlatVolatility, switchTenor, longTermDeltaType,
        longTermAtmType, longTermAtmDeltaType);
}

// --- JSON helpers ---------------------------------------------------------

void emitArray(const std::string& name, const std::vector<Real>& v, const char* indent,
               bool last = false) {
    std::cout << indent << "\"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << v[i];
    std::cout << "]" << (last ? "\n" : ",\n");
}

// Grid of blackVol / blackVariance over kTimes x kStrikes, flattened row-major
// (time-major).  extrapolate = true throughout: 2.4 is past maxDate and the
// strike range is unbounded anyway (minStrike()=0, maxStrike()=QL_MAX_REAL).
void emitSurfaceBlock(const std::string& name,
                      const ext::shared_ptr<BlackVolatilitySurfaceDelta>& s, bool last) {
    std::cout << "    \"" << name << "\": {\n";

    std::vector<Real> vols, vars, atmVols;
    for (Time t : kTimes) {
        for (Real k : kStrikes) {
            vols.push_back(s->blackVol(t, k, true));
            vars.push_back(s->blackVariance(t, k, true));
        }
        // strike == 0 short-circuit (blackvolsurfacedelta.cpp:219-227)
        atmVols.push_back(s->blackVol(t, 0.0, true));
    }
    emitArray("black_vol", vols, "      ");
    emitArray("black_variance", vars, "      ");
    emitArray("black_vol_at_zero_strike", atmVols, "      ");

    // SmileSection view: pillar strikes surviving the close()-based map plus
    // the section's own atmLevel (= the ATM column vol, 1.0 when hasAtm_ is
    // false — blackvolsurfacedelta.cpp:118/134-135).
    std::vector<Real> smileMin, smileMax, smileAtm, smileVol;
    for (Time t : kTimes) {
        ext::shared_ptr<SmileSection> smile = s->blackVolSmile(t);
        smileMin.push_back(smile->minStrike());
        smileMax.push_back(smile->maxStrike());
        smileAtm.push_back(smile->atmLevel());
        for (Real k : kStrikes)
            smileVol.push_back(smile->volatility(k));
    }
    emitArray("smile_min_strike", smileMin, "      ");
    emitArray("smile_max_strike", smileMax, "      ");
    emitArray("smile_atm_level", smileAtm, "      ");
    emitArray("smile_volatility", smileVol, "      ", true);

    std::cout << "    }" << (last ? "\n" : ",\n");
}

std::string tryConstruct(const std::function<void()>& f) {
    try {
        f();
        return "";
    } catch (const std::exception& e) {
        return std::string(e.what());
    }
}

void emitString(const std::string& name, const std::string& v, const char* indent,
                bool last = false) {
    std::cout << indent << "\"" << name << "\": \"";
    for (char c : v) {
        if (c == '"' || c == '\\')
            std::cout << '\\' << c;
        else if (c == '\n')
            std::cout << "\\n";
        else
            std::cout << c;
    }
    std::cout << "\"" << (last ? "\n" : ",\n");
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    // The surface takes an explicit reference date, but Settings is pinned so
    // nothing downstream can pick up "today".
    Settings::instance().evaluationDate() = kRefDate;

    std::cout << "{\n";

    // --- setup echo --------------------------------------------------------
    std::cout << "  \"setup\": {\n";
    std::cout << "    \"reference_date_serial\": " << kRefDate.serialNumber() << ",\n";
    std::cout << "    \"spot\": " << kSpot << ",\n";
    std::cout << "    \"domestic_rate\": " << kDomesticRate << ",\n";
    std::cout << "    \"foreign_rate\": " << kForeignRate << ",\n";
    std::cout << "    \"expiry_date_serials\": [";
    {
        std::vector<Date> ds = expiryDates();
        for (Size i = 0; i < ds.size(); ++i)
            std::cout << (i ? ", " : "") << ds[i].serialNumber();
    }
    std::cout << "],\n";
    {
        std::vector<Real> ts;
        Actual365Fixed dc;
        for (const Date& d : expiryDates())
            ts.push_back(dc.yearFraction(kRefDate, d));
        emitArray("expiry_times", ts, "    ");
    }
    emitArray("put_deltas", putDeltas(), "    ");
    emitArray("call_deltas", callDeltas(), "    ");
    {
        std::vector<Real> flat;
        Matrix m = volMatrix();
        for (Size i = 0; i < m.rows(); ++i)
            for (Size j = 0; j < m.columns(); ++j)
                flat.push_back(m[i][j]);
        emitArray("vol_matrix_row_major", flat, "    ");
    }
    {
        std::vector<Real> ks(std::begin(kStrikes), std::end(kStrikes));
        emitArray("strikes", ks, "    ");
    }
    {
        std::vector<Real> ts(std::begin(kTimes), std::end(kTimes));
        emitArray("times", ts, "    ", true);
    }
    std::cout << "  },\n";

    // --- 1. the delta -> strike inversion, standalone -----------------------
    //
    // Mirrors blackVolSmile()'s own arithmetic: per column i the vol comes from
    // interpolators_[i]->blackVol(t, 1, true), then a BlackDeltaCalculator with
    // stdDev = vol * sqrt(t) inverts the delta (cpp:120-158).  Emitted per
    // DeltaType so a port that swaps Spot for Fwd, or forgets the premium
    // adjustment, fails on a strike rather than on a vol three steps later.
    std::cout << "  \"strike_from_delta\": {\n";
    {
        const DeltaVolQuote::DeltaType dts[] = {
            DeltaVolQuote::DeltaType::Spot, DeltaVolQuote::DeltaType::Fwd,
            DeltaVolQuote::DeltaType::PaSpot, DeltaVolQuote::DeltaType::PaFwd};
        const char* dtNames[] = {"Spot", "Fwd", "PaSpot", "PaFwd"};

        ext::shared_ptr<BlackVolatilitySurfaceDelta> s =
            makeSurface(DeltaVolQuote::DeltaType::Spot, DeltaVolQuote::AtmType::AtmDeltaNeutral,
                        ext::nullopt, BlackVolatilitySurfaceDelta::SmileInterpolationMethod::Linear,
                        false);

        // the per-column vols the surface itself reads (column order:
        // put -10, put -25, ATM, call 25, call 10)
        std::vector<Real> colVols;
        for (Time t : kTimes)
            for (Size col = 0; col < 5; ++col) {
                // reproduce interpolators_[col] as an independent
                // BlackVarianceCurve over the same column
                std::vector<Volatility> vols;
                Matrix m = volMatrix();
                for (Size row = 0; row < 5; ++row)
                    vols.push_back(m[row][col]);
                BlackVarianceCurve bvc(kRefDate, expiryDates(), vols, Actual365Fixed(), false,
                                       BlackVolTimeExtrapolation::FlatVolatility);
                colVols.push_back(bvc.blackVol(t, 1, true));
            }
        emitArray("column_vols", colVols, "    ");

        for (Size di = 0; di < 4; ++di) {
            std::vector<Real> strikes;
            for (Time t : kTimes) {
                const Real sqrtT = std::sqrt(t);
                const DiscountFactor dD = flatCurve(kDomesticRate)->discount(t);
                const DiscountFactor dF = flatCurve(kForeignRate)->discount(t);
                Matrix m = volMatrix();
                for (Size col = 0; col < 5; ++col) {
                    std::vector<Volatility> vols;
                    for (Size row = 0; row < 5; ++row)
                        vols.push_back(m[row][col]);
                    BlackVarianceCurve bvc(kRefDate, expiryDates(), vols, Actual365Fixed(), false,
                                           BlackVolTimeExtrapolation::FlatVolatility);
                    const Real vol = bvc.blackVol(t, 1, true);
                    const Option::Type ot = (col < 2) ? Option::Put : Option::Call;
                    BlackDeltaCalculator bdc(ot, dts[di], kSpot, dD, dF, vol * sqrtT);
                    if (col == 0)
                        strikes.push_back(bdc.strikeFromDelta(-0.10));
                    else if (col == 1)
                        strikes.push_back(bdc.strikeFromDelta(-0.25));
                    else if (col == 2)
                        strikes.push_back(Null<Real>()); // ATM handled below
                    else if (col == 3)
                        strikes.push_back(bdc.strikeFromDelta(0.25));
                    else
                        strikes.push_back(bdc.strikeFromDelta(0.10));
                }
            }
            emitArray(std::string("delta_type_") + dtNames[di], strikes, "    ");
        }

        // ATM strikes, per (atmDeltaType, atmType) pair.  AtmPutCall50 is only
        // legal for DeltaType::Fwd (blackdeltacalculator.cpp), so it appears
        // only in the Fwd row.
        const DeltaVolQuote::AtmType ats[] = {
            DeltaVolQuote::AtmType::AtmSpot,     DeltaVolQuote::AtmType::AtmFwd,
            DeltaVolQuote::AtmType::AtmDeltaNeutral, DeltaVolQuote::AtmType::AtmVegaMax,
            DeltaVolQuote::AtmType::AtmGammaMax, DeltaVolQuote::AtmType::AtmPutCall50};
        const char* atNames[] = {"AtmSpot",     "AtmFwd",      "AtmDeltaNeutral",
                                 "AtmVegaMax",  "AtmGammaMax", "AtmPutCall50"};
        for (Size di = 0; di < 4; ++di) {
            for (Size ai = 0; ai < 6; ++ai) {
                if (ats[ai] == DeltaVolQuote::AtmType::AtmPutCall50 &&
                    dts[di] != DeltaVolQuote::DeltaType::Fwd)
                    continue;
                std::vector<Real> strikes;
                for (Time t : kTimes) {
                    const Real sqrtT = std::sqrt(t);
                    const DiscountFactor dD = flatCurve(kDomesticRate)->discount(t);
                    const DiscountFactor dF = flatCurve(kForeignRate)->discount(t);
                    std::vector<Volatility> vols;
                    Matrix m = volMatrix();
                    for (Size row = 0; row < 5; ++row)
                        vols.push_back(m[row][2]);
                    BlackVarianceCurve bvc(kRefDate, expiryDates(), vols, Actual365Fixed(), false,
                                           BlackVolTimeExtrapolation::FlatVolatility);
                    const Real vol = bvc.blackVol(t, 1, true);
                    BlackDeltaCalculator bdc(Option::Put, dts[di], kSpot, dD, dF, vol * sqrtT);
                    strikes.push_back(bdc.atmStrike(ats[ai]));
                }
                emitArray(std::string("atm_") + dtNames[di] + "_" + atNames[ai], strikes, "    ");
            }
        }

        // boundary / throw cases for strikeFromDelta.  Spot delta must satisfy
        // |delta| <= fDiscount; forward delta |delta| <= 1.
        {
            const Time t = 1.0;
            const DiscountFactor dD = flatCurve(kDomesticRate)->discount(t);
            const DiscountFactor dF = flatCurve(kForeignRate)->discount(t);
            BlackDeltaCalculator spotPut(Option::Put, DeltaVolQuote::DeltaType::Spot, kSpot, dD, dF,
                                         0.12);
            emitString("throw_spot_delta_out_of_range",
                       tryConstruct([&]() { spotPut.strikeFromDelta(-1.5); }), "    ");
            BlackDeltaCalculator fwdPut(Option::Put, DeltaVolQuote::DeltaType::Fwd, kSpot, dD, dF,
                                        0.12);
            emitString("throw_fwd_delta_out_of_range",
                       tryConstruct([&]() { fwdPut.strikeFromDelta(-1.5); }), "    ");
            emitString("throw_incoherent_option_type_and_delta",
                       tryConstruct([&]() { fwdPut.strikeFromDelta(0.25); }), "    ");
            BlackDeltaCalculator fwdCall(Option::Call, DeltaVolQuote::DeltaType::Fwd, kSpot, dD, dF,
                                         0.12);
            emitString("throw_atm_putcall50_needs_fwd_delta",
                       tryConstruct([&]() {
                           BlackDeltaCalculator spotCall(Option::Call,
                                                         DeltaVolQuote::DeltaType::Spot, kSpot, dD,
                                                         dF, 0.12);
                           spotCall.atmStrike(DeltaVolQuote::AtmType::AtmPutCall50);
                       }),
                       "    ", true);
        }
    }
    std::cout << "  },\n";

    // --- 2. surfaces --------------------------------------------------------
    std::cout << "  \"surfaces\": {\n";
    {
        using SIM = BlackVolatilitySurfaceDelta::SmileInterpolationMethod;

        emitSurfaceBlock("spot_dn_linear",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::Linear, false),
                         false);
        emitSurfaceBlock("spot_dn_natural_cubic",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::NaturalCubic, false),
                         false);
        emitSurfaceBlock("spot_dn_financial_cubic",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::FinancialCubic, false),
                         false);
        emitSurfaceBlock("spot_dn_cubic_spline",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::CubicSpline, false),
                         false);
        emitSurfaceBlock("spot_dn_linear_flat_extrap",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::Linear, true),
                         false);
        emitSurfaceBlock("fwd_atmfwd_linear",
                         makeSurface(DeltaVolQuote::DeltaType::Fwd,
                                     DeltaVolQuote::AtmType::AtmFwd, ext::nullopt, SIM::Linear,
                                     false),
                         false);
        emitSurfaceBlock("paspot_dn_linear",
                         makeSurface(DeltaVolQuote::DeltaType::PaSpot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::Linear, false),
                         false);
        emitSurfaceBlock("pafwd_dn_linear",
                         makeSurface(DeltaVolQuote::DeltaType::PaFwd,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::Linear, false),
                         false);
        // atmDeltaType override: delta columns read Spot, the ATM column Fwd.
        emitSurfaceBlock("spot_dn_atmdelta_fwd_linear",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral,
                                     DeltaVolQuote::DeltaType::Fwd, SIM::Linear, false),
                         false);
        // no ATM column -> atmLevel of the smile stays 1.0 and blackVol(t, 0)
        // falls back to the forward (cpp:118, 224-226).
        emitSurfaceBlock("no_atm_spot_linear",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::Linear, false, false),
                         false);
        // switchTenor = 1Y: short end Spot/AtmDeltaNeutral, long end
        // Fwd/AtmFwd.  switchTime_ = timeFromReference(TARGET-advanced 1Y).
        emitSurfaceBlock("switch_1y_spot_to_fwd",
                         makeSurface(DeltaVolQuote::DeltaType::Spot,
                                     DeltaVolQuote::AtmType::AtmDeltaNeutral, ext::nullopt,
                                     SIM::Linear, false, true, 1 * Years,
                                     DeltaVolQuote::DeltaType::Fwd,
                                     DeltaVolQuote::AtmType::AtmFwd, ext::nullopt),
                         false);

        // ATM column only -> exactly one strike survives the map, so
        // blackVolSmile returns a FlatSmileSection built as
        // stdDevs.front()/sqrtT with NO atmLevel (cpp:173-175), i.e. the
        // section's atmLevel() is Null<Rate>().  Emitted last so the
        // one-strike branch is pinned by name.
        {
            std::vector<Real> none;
            Matrix atmOnly(5, 1);
            Matrix m = volMatrix();
            for (Size i = 0; i < 5; ++i)
                atmOnly[i][0] = m[i][2];
            ext::shared_ptr<BlackVolatilitySurfaceDelta> s =
                ext::make_shared<BlackVolatilitySurfaceDelta>(
                    kRefDate, expiryDates(), none, none, true, atmOnly, Actual365Fixed(), TARGET(),
                    spotHandle(), flatCurve(kDomesticRate), flatCurve(kForeignRate),
                    DeltaVolQuote::DeltaType::Spot, DeltaVolQuote::AtmType::AtmDeltaNeutral,
                    ext::nullopt, SIM::Linear, false);
            emitSurfaceBlock("atm_only_flat_smile", s, true);
        }
    }
    std::cout << "  },\n";

    // --- 3. inspectors ------------------------------------------------------
    {
        ext::shared_ptr<BlackVolatilitySurfaceDelta> s =
            makeSurface(DeltaVolQuote::DeltaType::Spot, DeltaVolQuote::AtmType::AtmDeltaNeutral,
                        ext::nullopt, BlackVolatilitySurfaceDelta::SmileInterpolationMethod::Linear,
                        false);
        std::cout << "  \"inspectors\": {\n";
        std::cout << "    \"max_date_serial\": " << s->maxDate().serialNumber() << ",\n";
        std::cout << "    \"min_strike\": " << s->minStrike() << ",\n";
        std::cout << "    \"max_strike\": " << s->maxStrike() << ",\n";
        std::cout << "    \"day_counter\": \"" << s->dayCounter().name() << "\",\n";
        std::cout << "    \"calendar\": \"" << s->calendar().name() << "\",\n";
        std::cout << "    \"reference_date_serial\": " << s->referenceDate().serialNumber()
                  << ",\n";
        std::vector<Real> atm;
        for (Time t : kTimes)
            atm.push_back(s->atmLevel(t));
        emitArray("atm_level", atm, "    ");
        // switchTime_ for the 1Y surface: TARGET().advance(refDate, 1Y) under
        // the surface's Following convention, then timeFromReference.
        std::cout << "    \"switch_tenor_1y_date_serial\": "
                  << s->optionDateFromTenor(1 * Years).serialNumber() << ",\n";
        std::cout << "    \"switch_tenor_1y_time\": "
                  << s->timeFromReference(s->optionDateFromTenor(1 * Years)) << "\n";
        std::cout << "  },\n";
    }

    // --- 4. constructor preconditions --------------------------------------
    std::cout << "  \"throws\": {\n";
    {
        using SIM = BlackVolatilitySurfaceDelta::SmileInterpolationMethod;
        emitString("single_date", tryConstruct([&]() {
                       std::vector<Date> d{Date(17, July, 2023)};
                       Matrix m(1, 5, 0.12);
                       BlackVolatilitySurfaceDelta(kRefDate, d, putDeltas(), callDeltas(), true, m,
                                                   Actual365Fixed(), TARGET(), spotHandle(),
                                                   flatCurve(kDomesticRate),
                                                   flatCurve(kForeignRate));
                   }),
                   "    ");
        emitString("date_before_reference", tryConstruct([&]() {
                       std::vector<Date> d{Date(1, June, 2023), Date(17, July, 2023)};
                       Matrix m(2, 5, 0.12);
                       BlackVolatilitySurfaceDelta(kRefDate, d, putDeltas(), callDeltas(), true, m,
                                                   Actual365Fixed(), TARGET(), spotHandle(),
                                                   flatCurve(kDomesticRate),
                                                   flatCurve(kForeignRate));
                   }),
                   "    ");
        emitString("unsorted_dates", tryConstruct([&]() {
                       std::vector<Date> d{Date(17, July, 2023), Date(1, July, 2023)};
                       Matrix m(2, 5, 0.12);
                       BlackVolatilitySurfaceDelta(kRefDate, d, putDeltas(), callDeltas(), true, m,
                                                   Actual365Fixed(), TARGET(), spotHandle(),
                                                   flatCurve(kDomesticRate),
                                                   flatCurve(kForeignRate));
                   }),
                   "    ");
        emitString("wrong_columns", tryConstruct([&]() {
                       Matrix m(5, 4, 0.12);
                       BlackVolatilitySurfaceDelta(kRefDate, expiryDates(), putDeltas(),
                                                   callDeltas(), true, m, Actual365Fixed(),
                                                   TARGET(), spotHandle(),
                                                   flatCurve(kDomesticRate),
                                                   flatCurve(kForeignRate));
                   }),
                   "    ");
        emitString("wrong_rows", tryConstruct([&]() {
                       Matrix m(4, 5, 0.12);
                       BlackVolatilitySurfaceDelta(kRefDate, expiryDates(), putDeltas(),
                                                   callDeltas(), true, m, Actual365Fixed(),
                                                   TARGET(), spotHandle(),
                                                   flatCurve(kDomesticRate),
                                                   flatCurve(kForeignRate));
                   }),
                   "    ");
        emitString("no_deltas_at_all", tryConstruct([&]() {
                       std::vector<Real> none;
                       Matrix m(5, 0);
                       BlackVolatilitySurfaceDelta(kRefDate, expiryDates(), none, none, false, m,
                                                   Actual365Fixed(), TARGET(), spotHandle(),
                                                   flatCurve(kDomesticRate),
                                                   flatCurve(kForeignRate));
                   }),
                   "    ", true);
        (void)sizeof(SIM);
    }
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
