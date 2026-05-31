// Phase 11 W6-B cluster probe: experimental volatility surfaces.
//
// Captures reference values for:
//
//   * ExtendedBlackVarianceCurve — Black variance curve modelled from
//     Quote-backed vols. Probes blackVol / blackVariance at the input
//     pillar dates (must reproduce the input vols) and at intermediate
//     times (internal-coherence checkpoints).
//
//   * ExtendedBlackVarianceSurface — 2-D variant over (date, strike).
//     Probes blackVol / blackVariance at the known grid pillars and at
//     an interior (t, K) point.
//
//   * AbcdAtmVolCurve — ATM (no-smile) IR vol curve fitted by the Abcd
//     parametric form. Probes atmVol at the option tenors (the k-adjusted
//     fit reproduces the input ATM vols exactly), the fitted (a,b,c,d),
//     and atmVol at an interpolated time.
//
//   * SabrVolSurface — interpolated SABR vol surface from market vol
//     spreads. Probes blackVol(expiry, strike) at the reference option
//     tenors and a couple of strikes near the ATM forward.
//
// C++ parity:
//   ql/experimental/volatility/extendedblackvariancecurve.{hpp,cpp}
//   ql/experimental/volatility/extendedblackvariancesurface.{hpp,cpp}
//   ql/experimental/volatility/abcdatmvolcurve.{hpp,cpp}
//   ql/experimental/volatility/sabrvolsurface.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/experimental/volatility/abcdatmvolcurve.hpp>
#include <ql/experimental/volatility/extendedblackvariancecurve.hpp>
#include <ql/experimental/volatility/extendedblackvariancesurface.hpp>
#include <ql/experimental/volatility/sabrvolsurface.hpp>
#include <ql/handle.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/math/matrix.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/smilesection.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

void print_vec(const std::vector<Real>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]";
}

// Reference bilinear interpolation on a (times x strikes) variance grid.
// The C++ ExtendedBlackVarianceSurface::setVariances() has an
// out-of-bounds bug (it loops times_.size()+1 columns into a matrix with
// only dates+1 columns and indexes the flat quote vector as
// i*times_.size()+j-1 instead of i*nDates+(j-1)), so the class aborts on
// construction. We therefore compute the surface reference values inline
// using the *documented correct* variance grid + Bilinear interpolation
// (the same z[strike_idx][time_idx] convention the Python port uses).
// times/strikes are the FULL grids (times includes the t=0 anchor).
Real bilinear_variance(const std::vector<Time>& times,
                       const std::vector<Real>& strikes,
                       const Matrix& variances,  // (nStrikes x nTimes)
                       Time t, Real strike) {
    // Locate the time bracket.
    Size jx = 0;
    while (jx + 2 < times.size() && times[jx + 1] < t) ++jx;
    Size jx1 = std::min<Size>(jx + 1, times.size() - 1);
    // Locate the strike bracket.
    Size iy = 0;
    while (iy + 2 < strikes.size() && strikes[iy + 1] < strike) ++iy;
    Size iy1 = std::min<Size>(iy + 1, strikes.size() - 1);

    Real t0 = times[jx], t1 = times[jx1];
    Real k0 = strikes[iy], k1 = strikes[iy1];
    Real z00 = variances[iy][jx], z01 = variances[iy][jx1];
    Real z10 = variances[iy1][jx], z11 = variances[iy1][jx1];

    Real ut = (t1 != t0) ? (t - t0) / (t1 - t0) : 0.0;
    Real uk = (k1 != k0) ? (strike - k0) / (k1 - k0) : 0.0;
    return (1 - ut) * (1 - uk) * z00 + ut * (1 - uk) * z01 +
           (1 - ut) * uk * z10 + ut * uk * z11;
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);

    // Pin a fixed evaluation date so the floating-reference structures
    // (AbcdAtmVolCurve, SabrVolSurface) are deterministic.
    Date today(15, May, 2026);
    Settings::instance().evaluationDate() = today;

    Calendar cal = TARGET();
    DayCounter dc = Actual365Fixed();

    std::cout << "{\n";

    // ===================================================================
    // ExtendedBlackVarianceCurve
    // ===================================================================
    {
        std::vector<Date> dates = {
            today + Period(6, Months),
            today + Period(1, Years),
            today + Period(2, Years),
            today + Period(3, Years),
        };
        std::vector<Volatility> vols = {0.20, 0.22, 0.25, 0.27};
        std::vector<Handle<Quote>> volQuotes;
        for (Volatility v : vols)
            volQuotes.emplace_back(
                ext::shared_ptr<Quote>(new SimpleQuote(v)));

        ExtendedBlackVarianceCurve curve(today, dates, volQuotes, dc, true);

        // black vol/variance at the pillar dates (should reproduce inputs).
        std::vector<Real> pillar_times;
        std::vector<Real> pillar_vols;
        std::vector<Real> pillar_variances;
        for (const Date& d : dates) {
            Time t = dc.yearFraction(today, d);
            pillar_times.push_back(t);
            pillar_vols.push_back(curve.blackVol(d, 0.0, true));
            pillar_variances.push_back(curve.blackVariance(d, 0.0, true));
        }

        // interpolated checkpoints (internal coherence).
        std::vector<Real> interp_times = {0.75, 1.5, 2.5};
        std::vector<Real> interp_vols;
        std::vector<Real> interp_variances;
        for (Real t : interp_times) {
            interp_vols.push_back(curve.blackVol(t, 0.0, true));
            interp_variances.push_back(curve.blackVariance(t, 0.0, true));
        }
        // extrapolated point beyond last pillar.
        Real extrap_t = 4.0;
        Real extrap_var = curve.blackVariance(extrap_t, 0.0, true);

        std::cout << "  \"extended_black_variance_curve\": {\n";
        std::cout << "    \"input_vols\": ";
        print_vec(std::vector<Real>(vols.begin(), vols.end()));
        std::cout << ",\n";
        std::cout << "    \"pillar_times\": ";
        print_vec(pillar_times);
        std::cout << ",\n";
        std::cout << "    \"pillar_vols\": ";
        print_vec(pillar_vols);
        std::cout << ",\n";
        std::cout << "    \"pillar_variances\": ";
        print_vec(pillar_variances);
        std::cout << ",\n";
        std::cout << "    \"interp_times\": ";
        print_vec(interp_times);
        std::cout << ",\n";
        std::cout << "    \"interp_vols\": ";
        print_vec(interp_vols);
        std::cout << ",\n";
        std::cout << "    \"interp_variances\": ";
        print_vec(interp_variances);
        std::cout << ",\n";
        std::cout << "    \"extrap_t\": " << extrap_t << ",\n";
        std::cout << "    \"extrap_variance\": " << extrap_var << "\n";
        std::cout << "  },\n";
    }

    // ===================================================================
    // ExtendedBlackVarianceSurface
    //
    // NOTE: QuantLib v1.42.1's ExtendedBlackVarianceSurface aborts on
    // construction (out-of-bounds in setVariances()), so we cannot drive
    // the C++ class directly. We compute the reference values with the
    // *documented correct* variance grid (variance[i][j] = t[j]*vol[i][j-1]^2,
    // anchor column 0 = 0) + Bilinear interpolation. The Python port
    // implements this corrected behaviour and is validated against these
    // values. See the Python module docstring for the divergence note.
    // ===================================================================
    {
        std::vector<Date> dates = {
            today + Period(1, Years),
            today + Period(2, Years),
        };
        std::vector<Real> strikes = {90.0, 100.0, 110.0};
        // vol[strike_i][date_j] grid (rows = strikes, cols = dates).
        std::vector<std::vector<Volatility>> volGrid = {
            {0.22, 0.24},  // strike 90:  1Y, 2Y
            {0.20, 0.22},  // strike 100: 1Y, 2Y
            {0.21, 0.23},  // strike 110: 1Y, 2Y
        };

        Size nStrikes = strikes.size();
        Size nDates = dates.size();
        // times grid with the t=0 anchor prepended.
        std::vector<Time> times(nDates + 1, 0.0);
        for (Size j = 1; j <= nDates; ++j)
            times[j] = dc.yearFraction(today, dates[j - 1]);

        // variance grid (nStrikes x nDates+1), column 0 = 0.
        Matrix variances(nStrikes, nDates + 1, 0.0);
        for (Size i = 0; i < nStrikes; ++i)
            for (Size j = 1; j <= nDates; ++j)
                variances[i][j] = times[j] * volGrid[i][j - 1] * volGrid[i][j - 1];

        // Pillars: vol at each (date, strike) -> reproduces the input vols.
        std::vector<Real> pillar_vols;
        std::vector<Real> pillar_variances;
        for (Size i = 0; i < nStrikes; ++i) {
            for (Size j = 0; j < nDates; ++j) {
                Time t = times[j + 1];
                Real var = bilinear_variance(times, strikes, variances, t, strikes[i]);
                pillar_variances.push_back(var);
                pillar_vols.push_back(std::sqrt(var / t));
            }
        }

        // Interior point.
        Real interior_t = 1.5;
        Real interior_k = 95.0;
        Real interior_var =
            bilinear_variance(times, strikes, variances, interior_t, interior_k);
        Real interior_vol = std::sqrt(interior_var / interior_t);

        // flat row vector of input vols for the Python test.
        std::vector<Real> volRowMajor;
        for (Size i = 0; i < nStrikes; ++i)
            for (Size j = 0; j < nDates; ++j)
                volRowMajor.push_back(volGrid[i][j]);

        std::cout << "  \"extended_black_variance_surface\": {\n";
        std::cout << "    \"strikes\": ";
        print_vec(strikes);
        std::cout << ",\n";
        std::cout << "    \"pillar_times\": [" << times[1] << ", " << times[2] << "],\n";
        std::cout << "    \"input_vols_rowmajor\": ";
        print_vec(volRowMajor);
        std::cout << ",\n";
        std::cout << "    \"pillar_vols\": ";
        print_vec(pillar_vols);
        std::cout << ",\n";
        std::cout << "    \"pillar_variances\": ";
        print_vec(pillar_variances);
        std::cout << ",\n";
        std::cout << "    \"interior_t\": " << interior_t << ",\n";
        std::cout << "    \"interior_k\": " << interior_k << ",\n";
        std::cout << "    \"interior_vol\": " << interior_vol << ",\n";
        std::cout << "    \"interior_variance\": " << interior_var << "\n";
        std::cout << "  },\n";
    }

    // ===================================================================
    // AbcdAtmVolCurve
    // ===================================================================
    {
        std::vector<Period> optionTenors = {
            Period(1, Years),
            Period(2, Years),
            Period(3, Years),
            Period(5, Years),
            Period(7, Years),
            Period(10, Years),
        };
        std::vector<Volatility> atmVols = {0.15, 0.18, 0.20, 0.22, 0.21, 0.19};
        std::vector<Handle<Quote>> volQuotes;
        for (Volatility v : atmVols)
            volQuotes.emplace_back(
                ext::shared_ptr<Quote>(new SimpleQuote(v)));

        AbcdAtmVolCurve curve(2, cal, optionTenors, volQuotes,
                              std::vector<bool>(1, true), Following, dc);

        std::vector<Real> opt_times;
        std::vector<Real> fit_vols;
        for (const Period& p : optionTenors) {
            Date d = curve.optionDateFromTenor(p);
            Time t = dc.yearFraction(curve.referenceDate(), d);
            opt_times.push_back(t);
            // k-adjusted fit reproduces the input ATM vols exactly.
            fit_vols.push_back(curve.atmVol(p, true));
        }

        // interpolated ATM vol at an intermediate tenor.
        Real interp_vol_4y = curve.atmVol(Period(4, Years), true);

        std::cout << "  \"abcd_atm_vol_curve\": {\n";
        std::cout << "    \"input_vols\": ";
        print_vec(std::vector<Real>(atmVols.begin(), atmVols.end()));
        std::cout << ",\n";
        std::cout << "    \"option_times\": ";
        print_vec(opt_times);
        std::cout << ",\n";
        std::cout << "    \"fit_vols\": ";
        print_vec(fit_vols);
        std::cout << ",\n";
        std::cout << "    \"a\": " << curve.a() << ",\n";
        std::cout << "    \"b\": " << curve.b() << ",\n";
        std::cout << "    \"c\": " << curve.c() << ",\n";
        std::cout << "    \"d\": " << curve.d() << ",\n";
        std::cout << "    \"rms_error\": " << curve.rmsError() << ",\n";
        std::cout << "    \"interp_vol_4y\": " << interp_vol_4y << "\n";
        std::cout << "  },\n";
    }

    // ===================================================================
    // SabrVolSurface
    // ===================================================================
    {
        // A flat-forward yield curve to back the index forecast.
        Handle<YieldTermStructure> yts(
            ext::shared_ptr<YieldTermStructure>(
                new FlatForward(today, 0.03, dc)));
        ext::shared_ptr<InterestRateIndex> index(new Euribor6M(yts));

        // ATM Black vol curve: an ExtendedBlackVarianceCurve up-cast to
        // BlackAtmVolCurve... but ExtendedBlackVarianceCurve is a
        // BlackVarianceTermStructure, not a BlackAtmVolCurve. Use an
        // AbcdAtmVolCurve as the ATM curve (it IS a BlackAtmVolCurve).
        // The Abcd ATM curve fits 4 params (a,b,c,d), so it needs >= 5
        // option tenors.
        std::vector<Period> atmTenors = {
            Period(1, Years), Period(2, Years), Period(3, Years),
            Period(5, Years), Period(7, Years), Period(10, Years),
        };
        std::vector<Volatility> atmVols = {0.20, 0.22, 0.235, 0.25, 0.245, 0.24};
        std::vector<Handle<Quote>> atmQuotes;
        for (Volatility v : atmVols)
            atmQuotes.emplace_back(
                ext::shared_ptr<Quote>(new SimpleQuote(v)));
        Handle<BlackAtmVolCurve> atmCurve(
            ext::shared_ptr<BlackAtmVolCurve>(
                new AbcdAtmVolCurve(2, cal, atmTenors, atmQuotes,
                                    std::vector<bool>(1, true), Following, dc)));

        std::vector<Period> optionTenors = {
            Period(1, Years), Period(2, Years), Period(5, Years),
        };
        // SABR fits 4 free params (alpha, beta, nu, rho), so we need at
        // least 5 strike spreads to over-determine the smile slice.
        std::vector<Spread> atmRateSpreads = {-0.02, -0.01, 0.0, 0.01, 0.02};
        // volSpreads_[tenor][strikeSpread] — additive vol spreads.
        std::vector<std::vector<Handle<Quote>>> volSpreads;
        // Hand-tuned smile: convex (smile up away from ATM).
        std::vector<std::vector<Real>> smile = {
            {0.012, 0.005, 0.0, 0.004, 0.010},  // 1Y
            {0.014, 0.006, 0.0, 0.005, 0.012},  // 2Y
            {0.016, 0.007, 0.0, 0.006, 0.014},  // 5Y
        };
        for (const auto& row : smile) {
            std::vector<Handle<Quote>> qrow;
            for (Real s : row)
                qrow.emplace_back(
                    ext::shared_ptr<Quote>(new SimpleQuote(s)));
            volSpreads.push_back(qrow);
        }

        SabrVolSurface surface(index, atmCurve, optionTenors,
                               atmRateSpreads, volSpreads);

        // Black smile vol at the option tenors. The experimental
        // BlackVolSurface exposes vols via smileSection(date)->volatility(K),
        // not a blackVol() accessor. Probe at the ATM forward and at
        // forward +/- 1% strikes.
        std::vector<Real> bv_times;
        std::vector<Real> bv_forwards;
        std::vector<Real> bv_atm;
        std::vector<Real> bv_down;
        std::vector<Real> bv_up;
        for (const Period& p : optionTenors) {
            Date d = surface.optionDateFromTenor(p);
            Time t = dc.yearFraction(surface.referenceDate(), d);
            Real fwd = index->fixing(d, true);
            ext::shared_ptr<SmileSection> smile = surface.smileSection(d, true);
            bv_times.push_back(t);
            bv_forwards.push_back(fwd);
            bv_atm.push_back(smile->volatility(fwd));
            bv_down.push_back(smile->volatility(fwd - 0.01));
            bv_up.push_back(smile->volatility(fwd + 0.01));
        }

        std::cout << "  \"sabr_vol_surface\": {\n";
        std::cout << "    \"atm_rate_spreads\": ";
        print_vec(std::vector<Real>(atmRateSpreads.begin(), atmRateSpreads.end()));
        std::cout << ",\n";
        std::cout << "    \"bv_times\": ";
        print_vec(bv_times);
        std::cout << ",\n";
        std::cout << "    \"bv_forwards\": ";
        print_vec(bv_forwards);
        std::cout << ",\n";
        std::cout << "    \"bv_atm\": ";
        print_vec(bv_atm);
        std::cout << ",\n";
        std::cout << "    \"bv_down\": ";
        print_vec(bv_down);
        std::cout << ",\n";
        std::cout << "    \"bv_up\": ";
        print_vec(bv_up);
        std::cout << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
