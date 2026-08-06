// migration-harness/cpp/probes/v143_zabr_fdsection/probe.cpp
//
// Pins the four evaluation arms of QuantLib::ZabrSmileSection<Evaluation>
// (ql/termstructures/volatility/zabrsmilesection.hpp).
//
// The four arms are template tag types, and the tags do far more than pick a
// formula — they select entirely different init/init2/init3 chains:
//
//  * ShortMaturityLognormal — no strike grid at all; volatilityImpl clamps the
//    strike at 1e-6 and returns model_->lognormalVolatility(k); optionPrice
//    falls through to SmileSection::optionPrice, i.e. Black with that vol.
//
//  * ShortMaturityNormal — optionPrice is bachelierBlackFormula with
//    model_->normalVolatility(k)*sqrt(T), but volatilityImpl does NOT return
//    that normal vol: it back-solves an implied LOGNORMAL vol from the
//    Bachelier price via blackFormulaImpliedStdDev, using Call above the
//    forward and Put below it, and swallows any exception into 0.0. A port
//    that returns normalVolatility(k) from volatility() is wrong by roughly
//    the ratio of the two vol conventions, which the `normal` block exposes at
//    every strike.
//
//  * LocalVolatility — builds a refined strike grid (each consecutive pair of
//    moneyness strikes gets fdRefinement interior points), prices the WHOLE
//    grid with one ZabrModel::fdPrice vector call, prepends (0.0, forward),
//    fits a natural cubic spline with extrapolation ON, and precomputes an
//    exponential right-tail from a one-sided 1e-5 finite difference at the
//    last strike: a_ = c0'/c0, b_ = log(c0) + a_*Kmax, used as exp(-a_ K + b_)
//    beyond Kmax. Put prices come from call-put parity against that same
//    curve. volatilityImpl then routes through the ShortMaturityNormal
//    back-solve, but on the FD price — so the reported vol depends on the
//    whole FD pipeline.
//
//  * FullFd — identical grid and interpolation as LocalVolatility, but each
//    grid point is priced with its own 2-D Hundsdorfer PDE via
//    ZabrModel::fullFdPrice.
//
// Strikes are probed inside the grid, at the ATM point, and past the last grid
// strike (0.6 = 20x forward for the default moneyness table) so the
// exponential extrapolation branch is exercised rather than the spline.
//
// The FullFd block deliberately uses a short custom moneyness vector and
// fdRefinement = 1: the C++ test-suite itself drops fdRefinement to 2 "to
// speed up the test", and each grid point here is a full 100x100 PDE solve.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/zabr/fdsection.json.

#include <iomanip>
#include <iostream>
#include <vector>

#include <ql/termstructures/volatility/zabrsmilesection.hpp>

using namespace QuantLib;

namespace {

// Same model as the C++ ZabrTests consistency test.
const Real kTau = 5.0;
const Real kForward = 0.03;
const std::vector<Real> kParams = {0.08, 0.70, 0.20, -0.30, 1.0};

// 0.0005 is far below the grid's first strike (0.01*F = 0.0003 is the first,
// so 0.0005 sits between the first two refined points); 0.03 is ATM; 0.6 is
// exactly the last default-grid strike (20*F) and 0.9 is past it, where the
// exponential tail takes over from the spline.
const Real kStrikes[] = {0.0005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.10, 0.30, 0.60, 0.90};
const Size kNumStrikes = sizeof(kStrikes) / sizeof(kStrikes[0]);

template <class Evaluation>
void emitSection(const char* key,
                 const std::vector<Real>& moneyness,
                 Size fdRefinement,
                 bool trailingComma) {
    ZabrSmileSection<Evaluation> s(kTau, kForward, kParams, moneyness, fdRefinement);

    std::vector<Real> strikes, vols, calls, puts;
    for (Size i = 0; i < kNumStrikes; ++i) {
        strikes.push_back(kStrikes[i]);
        vols.push_back(s.volatility(kStrikes[i]));
        calls.push_back(s.optionPrice(kStrikes[i], Option::Call, 1.0));
        puts.push_back(s.optionPrice(kStrikes[i], Option::Put, 1.0));
    }

    std::cout << "  \"" << key << "\": {\n";
    std::cout << "    \"exercise_time\": " << s.exerciseTime() << ",\n";
    std::cout << "    \"atm_level\": " << s.atmLevel() << ",\n";
    std::cout << "    \"min_strike\": " << s.minStrike() << ",\n";
    std::cout << "    \"strikes\": [";
    for (Size i = 0; i < strikes.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << strikes[i];
    std::cout << "],\n";
    std::cout << "    \"volatility\": [";
    for (Size i = 0; i < vols.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << vols[i];
    std::cout << "],\n";
    std::cout << "    \"call_price\": [";
    for (Size i = 0; i < calls.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << calls[i];
    std::cout << "],\n";
    std::cout << "    \"put_price\": [";
    for (Size i = 0; i < puts.size(); ++i)
        std::cout << (i != 0U ? ", " : "") << puts[i];
    std::cout << "]\n";
    std::cout << "  }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const std::vector<Real> defaultMoneyness;
    emitSection<ZabrShortMaturityLognormal>("short_maturity_lognormal", defaultMoneyness, 5, true);
    emitSection<ZabrShortMaturityNormal>("short_maturity_normal", defaultMoneyness, 5, true);
    emitSection<ZabrLocalVolatility>("local_volatility", defaultMoneyness, 5, true);

    // Small grid: 5 moneyness points + 1 interior point per gap = 9 PDE solves.
    const std::vector<Real> smallMoneyness = {0.5, 0.75, 1.0, 1.5, 2.0};
    emitSection<ZabrLocalVolatility>("local_volatility_small_grid", smallMoneyness, 1, true);
    emitSection<ZabrFullFd>("full_fd_small_grid", smallMoneyness, 1, false);

    std::cout << "}\n";
    return 0;
}
