// migration-harness/cpp/probes/v143_eqfx_hestonblackvol/probe.cpp
//
// Pins HestonBlackVolSurface — the Black vol surface implied by a Heston
// model, obtained by pricing a European vanilla under Heston and inverting
// the Black formula with Brent.
//
// What the cases discriminate:
//
//   * The OTM option-type switch. blackVolImpl prices a Put when
//     fwd > strike and a Call otherwise, so the inversion always runs on the
//     out-of-the-money side. A port that always priced a Call would agree at
//     high strikes and disagree in the low wing — where the Call is deep ITM,
//     its price is dominated by intrinsic value, and Brent's root is badly
//     conditioned. Strikes span 40..250 around a forward near 100, so both
//     branches are exercised at every maturity.
//   * The forward, fwd = s0 * dq(t) / dr(t), with r != q so the switch does
//     NOT happen at the spot. Strike 100 is below the forward at every
//     maturity here (r = 4% > q = 2%), so a port that compared against SPOT
//     instead of the FORWARD picks the wrong option type there.
//   * The npv <= 0 fallback to sqrt(theta). At t = 0.05 with strike 250 the
//     Fourier price underflows to <= 0 and the surface returns the long-run
//     Heston vol instead of failing.
//   * blackVariance vs blackVol (variance = vol^2 * t, so they cannot both be
//     right if the maturity is mishandled) and the atmLevel forward, pinned
//     separately.
//
// Two parameter sets: one comfortably Feller-satisfying, one violating Feller
// (2 kappa theta < sigma^2) where the density has a spike at the origin and
// the Fourier integrand is hardest — that is where a weaker quadrature shows.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/eqfx/hestonblackvol.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <ql/errors.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/hestonblackvolsurface.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kRef(15, June, 2026);
const Actual365Fixed kDc;

const std::vector<std::pair<std::string, Time>> kTimes = {
    {"t0p05", 0.05}, {"t0p25", 0.25}, {"t1p0", 1.0}, {"t3p0", 3.0}, {"t10p0", 10.0}};

const std::vector<std::pair<std::string, Real>> kStrikes = {
    {"k40", 40.0},   {"k60", 60.0},   {"k80", 80.0},  {"k100", 100.0},
    {"k120", 120.0}, {"k160", 160.0}, {"k250", 250.0}};

ext::shared_ptr<HestonModel> makeModel(
    Real v0, Real kappa, Real theta, Real sigma, Real rho) {
    Handle<YieldTermStructure> rf(
        ext::make_shared<FlatForward>(kRef, 0.04, kDc, Continuous, Annual));
    Handle<YieldTermStructure> div(
        ext::make_shared<FlatForward>(kRef, 0.02, kDc, Continuous, Annual));
    Handle<Quote> s0(ext::make_shared<SimpleQuote>(100.0));
    return ext::make_shared<HestonModel>(
        ext::make_shared<HestonProcess>(rf, div, s0, v0, kappa, theta, sigma, rho));
}

void emitCase(const std::string& key, const ext::shared_ptr<HestonModel>& model,
              bool first) {
    if (!first)
        std::cout << ",\n";
    // Braces, not parens: `Type s(Handle<T>(x));` is the most vexing parse.
    HestonBlackVolSurface s{Handle<HestonModel>(model)};

    std::cout << "  \"" << key << "\": {\n"
              << "    \"reference_date\": " << s.referenceDate().serialNumber() << ",\n"
              << "    \"min_strike\": " << s.minStrike() << ",\n"
              << "    \"max_strike\": " << s.maxStrike() << ",\n"
              << "    \"theta\": " << model->theta() << ",\n"
              << "    \"atm_level\": {\n";
    bool f = true;
    for (const auto& tp : kTimes) {
        if (!f) std::cout << ",\n";
        f = false;
        std::cout << "      \"" << tp.first << "\": " << s.atmLevel(tp.second);
    }
    std::cout << "\n    },\n    \"vol\": {\n";
    f = true;
    for (const auto& tp : kTimes) {
        for (const auto& kp : kStrikes) {
            if (!f) std::cout << ",\n";
            f = false;
            std::cout << "      \"" << tp.first << "_" << kp.first << "\": ";
            try {
                std::cout << s.blackVol(tp.second, kp.second, true);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
    }
    std::cout << "\n    },\n    \"variance\": {\n";
    f = true;
    for (const auto& tp : kTimes) {
        for (const auto& kp : kStrikes) {
            if (!f) std::cout << ",\n";
            f = false;
            std::cout << "      \"" << tp.first << "_" << kp.first << "\": ";
            try {
                std::cout << s.blackVariance(tp.second, kp.second, true);
            } catch (const Error&) {
                std::cout << "{\"raises\": true}";
            }
        }
    }
    std::cout << "\n    }\n  }";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    // Feller satisfied: 2 * 2.5 * 0.05 = 0.25 > 0.4^2 = 0.16.
    emitCase("feller_ok", makeModel(0.04, 2.5, 0.05, 0.4, -0.6), true);
    // Feller violated: 2*0.5*0.04 = 0.04 < 0.6^2 = 0.36.
    emitCase("feller_violated", makeModel(0.06, 0.5, 0.04, 0.6, -0.75), false);
    std::cout << "\n}\n";
    return 0;
}
