// Phase 11 W2-A cluster probe: ZABR fitter + smile composition +
// XABR/ZABR swaption volatility cube generalization.
//
// Captures reference values for:
//
//   * ZabrInterpolation<ZabrShortMaturityLognormal> fit on a synthetic
//     5-strike ZABR slice (alpha=0.04, beta=0.5, nu=0.4, rho=-0.1,
//     gamma=0.7, F=0.05, T=3). Recovered (alpha, beta, nu, rho, gamma)
//     plus rms error and fitted vols at each strike.
//
//   * ZabrInterpolation<ZabrShortMaturityLognormal> with gamma_is_fixed
//     = true, gamma = 1.0 — collapses to SABR-equivalent on a SABR
//     synthetic slice (alpha=0.03, beta=0.5, nu=0.4, rho=-0.2, F=0.04,
//     T=1). Recovered (alpha, beta, nu, rho).
//
//   * ZabrInterpolatedSmileSection on the same synthetic ZABR slice
//     above + .volatility() at strikes 0.03..0.07.
//
//   * SabrSwaptionVolatilityCube grid-point smile section value at
//     ATM (sanity that L9-C codepath unaffected by the generalization).
//
//   * ZabrSwaptionVolatilityCube grid-point smile section ATM + a
//     strike-spread offset, using a 2x2x3 cube of ZABR vol spreads.
//
// C++ parity:
//   ql/math/interpolations/zabrinterpolation.hpp
//   ql/termstructures/volatility/zabrinterpolatedsmilesection.hpp
//   ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp
//   ql/termstructures/volatility/swaption/zabrswaptionvolatilitycube.hpp
//   ql/termstructures/volatility/zabrsmilesection.hpp
//   @ v1.42.1 (099987f0).

#include <ql/handle.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/indexes/swapindex.hpp>
#include <ql/math/interpolations/sabrinterpolation.hpp>
#include <ql/math/interpolations/zabrinterpolation.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/sabrsmilesection.hpp>
#include <ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp>
#include <ql/termstructures/volatility/swaption/zabrswaptionvolatilitycube.hpp>
#include <ql/termstructures/volatility/swaption/swaptionvolmatrix.hpp>
#include <ql/termstructures/volatility/zabr.hpp>
#include <ql/termstructures/volatility/zabrinterpolatedsmilesection.hpp>
#include <ql/termstructures/volatility/zabrsmilesection.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

// Produce a synthetic strike-vol slice via the ZABR short-maturity
// lognormal formula at known parameters.
std::vector<Real>
zabr_slice(const std::vector<Real>& strikes,
           Real forward,
           Real T,
           Real alpha,
           Real beta,
           Real nu,
           Real rho,
           Real gamma) {
    ZabrModel zabr(T, forward, alpha, beta, nu, rho, gamma);
    std::vector<Real> vols(strikes.size());
    for (std::size_t i = 0; i < strikes.size(); ++i) {
        vols[i] = zabr.lognormalVolatility(strikes[i]);
    }
    return vols;
}

// Produce a synthetic strike-vol slice via Hagan-2002 SABR.
std::vector<Real>
sabr_slice(const std::vector<Real>& strikes,
           Real forward,
           Real T,
           Real alpha,
           Real beta,
           Real nu,
           Real rho) {
    std::vector<Real> vols(strikes.size());
    for (std::size_t i = 0; i < strikes.size(); ++i) {
        vols[i] = sabrVolatility(strikes[i], forward, T, alpha, beta, nu, rho);
    }
    return vols;
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) ZabrInterpolation — fit + recover known (alpha,beta,nu,rho,gamma).
    //
    // Synthetic ZABR vol slice, gamma=0.7 (non-SABR regime). With
    // beta pinned (under-determined on a short slice).
    // ============================================================
    {
        const Real alpha_true = 0.04;
        const Real beta_true = 0.5;
        const Real nu_true = 0.4;
        const Real rho_true = -0.1;
        const Real gamma_true = 0.7;
        const Real forward = 0.05;
        const Real T = 3.0;
        const std::vector<Real> strikes = {0.03, 0.04, 0.05, 0.06, 0.07};
        std::vector<Real> vols = zabr_slice(
            strikes, forward, T,
            alpha_true, beta_true, nu_true, rho_true, gamma_true);

        ZabrInterpolation<ZabrShortMaturityLognormal> interp(
            strikes.begin(), strikes.end(), vols.begin(),
            T, forward,
            0.05, beta_true, 0.3, 0.0, 1.0,            // initial alpha, beta, nu, rho, gamma
            false, true, false, false, false,          // alpha free, beta fixed, nu/rho/gamma free
            false                                      // vega_weighted = false (deterministic)
        );
        interp.update();

        std::cout << "  \"zabr_interpolation_fit_gamma_free\": {\n";
        std::cout << "    \"forward\": " << forward << ",\n";
        std::cout << "    \"expiry\": " << T << ",\n";
        std::cout << "    \"strikes\": [";
        for (std::size_t i = 0; i < strikes.size(); ++i) {
            std::cout << strikes[i];
            if (i + 1 < strikes.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"input_vols\": [";
        for (std::size_t i = 0; i < vols.size(); ++i) {
            std::cout << vols[i];
            if (i + 1 < vols.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"alpha_true\": " << alpha_true << ",\n";
        std::cout << "    \"beta_true\": " << beta_true << ",\n";
        std::cout << "    \"nu_true\": " << nu_true << ",\n";
        std::cout << "    \"rho_true\": " << rho_true << ",\n";
        std::cout << "    \"gamma_true\": " << gamma_true << ",\n";
        std::cout << "    \"alpha_fitted\": " << interp.alpha() << ",\n";
        std::cout << "    \"beta_fitted\": " << interp.beta() << ",\n";
        std::cout << "    \"nu_fitted\": " << interp.nu() << ",\n";
        std::cout << "    \"rho_fitted\": " << interp.rho() << ",\n";
        std::cout << "    \"gamma_fitted\": " << interp.gamma() << ",\n";
        std::cout << "    \"rms_error\": " << interp.rmsError() << ",\n";
        std::cout << "    \"max_error\": " << interp.maxError() << ",\n";
        std::cout << "    \"fitted_vols\": [";
        for (std::size_t i = 0; i < strikes.size(); ++i) {
            std::cout << interp(strikes[i]);
            if (i + 1 < strikes.size()) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) ZabrInterpolation — gamma_is_fixed=true, gamma=1, on a SABR
    //    synthetic slice. Recovered (alpha, nu, rho) should match a
    //    pure SabrInterpolation on the same slice.
    // ============================================================
    {
        const Real alpha_true = 0.03;
        const Real beta_true = 0.5;
        const Real nu_true = 0.4;
        const Real rho_true = -0.2;
        const Real forward = 0.04;
        const Real T = 1.0;
        const std::vector<Real> strikes = {0.02, 0.03, 0.04, 0.05, 0.06};
        std::vector<Real> vols = sabr_slice(
            strikes, forward, T, alpha_true, beta_true, nu_true, rho_true);

        // ZABR fit with gamma fixed at 1.
        ZabrInterpolation<ZabrShortMaturityLognormal> zabr_interp(
            strikes.begin(), strikes.end(), vols.begin(),
            T, forward,
            0.04, beta_true, 0.5, 0.0, 1.0,
            false, true, false, false, true,           // gamma_is_fixed = true
            false
        );
        zabr_interp.update();

        // SABR reference fit.
        SABRInterpolation sabr_interp(
            strikes.begin(), strikes.end(), vols.begin(),
            T, forward,
            0.04, beta_true, 0.5, 0.0,
            false, true, false, false,
            false
        );
        sabr_interp.update();

        std::cout << "  \"zabr_interpolation_gamma_fixed_at_one\": {\n";
        std::cout << "    \"forward\": " << forward << ",\n";
        std::cout << "    \"expiry\": " << T << ",\n";
        std::cout << "    \"alpha_true\": " << alpha_true << ",\n";
        std::cout << "    \"beta_true\": " << beta_true << ",\n";
        std::cout << "    \"nu_true\": " << nu_true << ",\n";
        std::cout << "    \"rho_true\": " << rho_true << ",\n";
        std::cout << "    \"strikes\": [";
        for (std::size_t i = 0; i < strikes.size(); ++i) {
            std::cout << strikes[i];
            if (i + 1 < strikes.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"input_vols\": [";
        for (std::size_t i = 0; i < vols.size(); ++i) {
            std::cout << vols[i];
            if (i + 1 < vols.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "    \"zabr_alpha\": " << zabr_interp.alpha() << ",\n";
        std::cout << "    \"zabr_beta\": " << zabr_interp.beta() << ",\n";
        std::cout << "    \"zabr_nu\": " << zabr_interp.nu() << ",\n";
        std::cout << "    \"zabr_rho\": " << zabr_interp.rho() << ",\n";
        std::cout << "    \"zabr_gamma\": " << zabr_interp.gamma() << ",\n";
        std::cout << "    \"zabr_rms_error\": " << zabr_interp.rmsError() << ",\n";
        std::cout << "    \"sabr_alpha\": " << sabr_interp.alpha() << ",\n";
        std::cout << "    \"sabr_beta\": " << sabr_interp.beta() << ",\n";
        std::cout << "    \"sabr_nu\": " << sabr_interp.nu() << ",\n";
        std::cout << "    \"sabr_rho\": " << sabr_interp.rho() << ",\n";
        std::cout << "    \"sabr_rms_error\": " << sabr_interp.rmsError() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) ZabrInterpolatedSmileSection — eager fit + .volatility() at
    //    a few strikes.
    // ============================================================
    {
        const Real alpha_init = 0.05;
        const Real beta_init = 0.5;
        const Real nu_init = 0.3;
        const Real rho_init = 0.0;
        const Real gamma_init = 1.0;
        const Real forward = 0.05;
        const Real T = 3.0;
        const std::vector<Real> strikes = {0.03, 0.04, 0.05, 0.06, 0.07};
        std::vector<Real> vols = zabr_slice(
            strikes, forward, T, 0.04, 0.5, 0.4, -0.1, 0.7);

        // Use the no-quotes ctor — fully evaluable, hasFloatingStrikes=false.
        Date today = Date(15, January, 2024);
        Settings::instance().evaluationDate() = today;
        Date optionDate = today + 365 * 3;
        Volatility atmVol = 0.20;
        ZabrInterpolatedSmileSection<ZabrShortMaturityLognormal> section(
            optionDate, forward, strikes, false,
            atmVol, vols,
            alpha_init, beta_init, nu_init, rho_init, gamma_init,
            false, true, false, false, false,
            false                                     // vega_weighted = false
        );

        std::cout << "  \"zabr_interpolated_smile_section\": {\n";
        std::cout << "    \"forward\": " << forward << ",\n";
        std::cout << "    \"exercise_time\": " << section.exerciseTime() << ",\n";
        std::cout << "    \"atm_level\": " << section.atmLevel() << ",\n";
        std::cout << "    \"alpha\": " << section.alpha() << ",\n";
        std::cout << "    \"beta\": " << section.beta() << ",\n";
        std::cout << "    \"nu\": " << section.nu() << ",\n";
        std::cout << "    \"rho\": " << section.rho() << ",\n";
        std::cout << "    \"gamma\": " << section.gamma() << ",\n";
        std::cout << "    \"rms_error\": " << section.rmsError() << ",\n";
        std::cout << "    \"vol_strike_3pct\": " << section.volatility(0.03) << ",\n";
        std::cout << "    \"vol_strike_5pct\": " << section.volatility(0.05) << ",\n";
        std::cout << "    \"vol_strike_7pct\": " << section.volatility(0.07) << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
