// Phase 11 W6-A cluster probe: NoArbSABR + SVI smile families.
//
// Captures reference values for:
//
//   * SVI raw total-variance parameterization (sviTotalVariance) and
//     SviSmileSection.volatility(K) at Doust/SEB canonical params.
//
//   * NoArbSabrModel: absorptionProbability, optionPrice,
//     digitalOptionPrice, density at the Doust figure-3 params, plus
//     NoArbSabrSmileSection.volatility(K) (implied vol from model price).
//
//   * D0Interpolator absorption-matrix probe at the explicit table
//     checkpoints from test-suite/noarbsabr.cpp (integer absorptions).
//
//   * Hagan-vs-NoArbSABR consistency option prices (the noarb model
//     should match the closed-form Hagan SABR for near-zero absorption).
//
// It ALSO dumps the 1,209,600-entry absorption-probability table
// (QuantLib::detail::sabrabsprob) to a little-endian uint32 binary file
// so the Python port can mmap-load it as a compact data asset (the C++
// table is a 7.9 MB source literal that cannot be hand-ported).
//
// C++ parity:
//   ql/experimental/volatility/noarbsabr.{hpp,cpp}
//   ql/experimental/volatility/noarbsabrsmilesection.{hpp,cpp}
//   ql/experimental/volatility/sviinterpolation.hpp
//   ql/experimental/volatility/svismilesection.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/experimental/volatility/noarbsabr.hpp>
#include <ql/experimental/volatility/noarbsabrsmilesection.hpp>
#include <ql/experimental/volatility/sviinterpolation.hpp>
#include <ql/experimental/volatility/svismilesection.hpp>
#include <ql/termstructures/volatility/sabrsmilesection.hpp>

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace QuantLib::detail {
// The absorption table is declared extern "C" in noarbsabr.hpp; pull it
// in here so we can serialize it.
extern "C" const unsigned long sabrabsprob[1209600];
}

namespace {

void print_vec(const std::vector<Real>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]";
}

// Dump sabrabsprob to a little-endian uint32 binary blob.
void dump_absprob_table(const char* path) {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        std::cerr << "WARN: could not open " << path << " for absprob dump\n";
        return;
    }
    for (std::size_t i = 0; i < 1209600; ++i) {
        std::uint32_t v = static_cast<std::uint32_t>(QuantLib::detail::sabrabsprob[i]);
        unsigned char bytes[4];
        bytes[0] = static_cast<unsigned char>(v & 0xFFu);
        bytes[1] = static_cast<unsigned char>((v >> 8) & 0xFFu);
        bytes[2] = static_cast<unsigned char>((v >> 16) & 0xFFu);
        bytes[3] = static_cast<unsigned char>((v >> 24) & 0xFFu);
        std::fwrite(bytes, 1, 4, f);
    }
    std::fclose(f);
    std::cerr << "wrote absprob table -> " << path << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    // Optional argv[1]: path to dump the absorption table binary.
    if (argc > 1) {
        dump_absprob_table(argv[1]);
    }

    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) SVI raw total variance + smile section.
    //    SEB test params (test-suite/svivolatility.cpp).
    // ============================================================
    {
        const Real a = -0.0666;
        const Real b = 0.229;
        const Real sigma = 0.337;
        const Real rho = 0.439;
        const Real m = 0.193;
        const Real forward = 123.45;
        const Time tte = 11.0 / 365.0;

        // At strike = forward*exp(m), log-moneyness k = m and the SVI
        // total variance collapses to a + b*sigma.
        const Real strike_atm_m = forward * std::exp(m);

        // sviTotalVariance directly at several log-moneyness points.
        std::vector<Real> ks{-0.5, -0.2, 0.0, m, 0.3, 0.6};
        std::vector<Real> tvs;
        for (Real k : ks)
            tvs.push_back(detail::sviTotalVariance(a, b, sigma, rho, m, k));

        SviSmileSection svi(tte, forward, {a, b, sigma, rho, m});
        std::vector<Real> strikes{60.0, 100.0, forward, 150.0, 200.0};
        std::vector<Real> svi_vols;
        std::vector<Real> svi_vars;
        for (Real K : strikes) {
            svi_vols.push_back(svi.volatility(K));
            svi_vars.push_back(svi.variance(K));
        }

        std::cout << "  \"svi\": {\n";
        std::cout << "    \"a\": " << a << ",\n";
        std::cout << "    \"b\": " << b << ",\n";
        std::cout << "    \"sigma\": " << sigma << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"m\": " << m << ",\n";
        std::cout << "    \"forward\": " << forward << ",\n";
        std::cout << "    \"tte\": " << tte << ",\n";
        std::cout << "    \"k_points\": ";
        print_vec(ks);
        std::cout << ",\n";
        std::cout << "    \"total_variance\": ";
        print_vec(tvs);
        std::cout << ",\n";
        std::cout << "    \"strike_atm_m\": " << strike_atm_m << ",\n";
        std::cout << "    \"variance_at_atm_m\": " << svi.variance(strike_atm_m) << ",\n";
        std::cout << "    \"a_plus_b_sigma\": " << (a + b * sigma) << ",\n";
        std::cout << "    \"strikes\": ";
        print_vec(strikes);
        std::cout << ",\n";
        std::cout << "    \"volatility\": ";
        print_vec(svi_vols);
        std::cout << ",\n";
        std::cout << "    \"variance\": ";
        print_vec(svi_vars);
        std::cout << "\n  },\n";
    }

    // ============================================================
    // 2) NoArbSABR model + smile section (Doust figure 3 params).
    // ============================================================
    {
        const Real tau = 1.0;
        const Real beta = 0.5;
        const Real alpha = 0.026;
        const Real rho = -0.1;
        const Real nu = 0.4;
        const Real f = 0.0488;

        NoArbSabrModel model(tau, f, alpha, beta, nu, rho);
        SabrSmileSection sabr(tau, f, {alpha, beta, nu, rho});
        NoArbSabrSmileSection noarb(tau, f, {alpha, beta, nu, rho});

        std::vector<Real> strikes{0.01, 0.02, 0.0488, 0.07, 0.1, 0.13};
        std::vector<Real> opt_prices;
        std::vector<Real> dig_prices;
        std::vector<Real> densities;
        std::vector<Real> noarb_vols;
        std::vector<Real> sabr_vols;
        std::vector<Real> sabr_opt_prices;
        for (Real K : strikes) {
            opt_prices.push_back(model.optionPrice(K));
            dig_prices.push_back(model.digitalOptionPrice(K));
            densities.push_back(model.density(K));
            noarb_vols.push_back(noarb.volatility(K));
            sabr_vols.push_back(sabr.volatility(K));
            sabr_opt_prices.push_back(sabr.optionPrice(K));
        }

        std::cout << "  \"noarbsabr\": {\n";
        std::cout << "    \"tau\": " << tau << ",\n";
        std::cout << "    \"beta\": " << beta << ",\n";
        std::cout << "    \"alpha\": " << alpha << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"nu\": " << nu << ",\n";
        std::cout << "    \"forward\": " << f << ",\n";
        std::cout << "    \"absorption_probability\": " << model.absorptionProbability() << ",\n";
        std::cout << "    \"numerical_forward\": " << model.numericalForward() << ",\n";
        std::cout << "    \"strikes\": ";
        print_vec(strikes);
        std::cout << ",\n";
        std::cout << "    \"option_price\": ";
        print_vec(opt_prices);
        std::cout << ",\n";
        std::cout << "    \"digital_option_price\": ";
        print_vec(dig_prices);
        std::cout << ",\n";
        std::cout << "    \"density\": ";
        print_vec(densities);
        std::cout << ",\n";
        std::cout << "    \"noarb_volatility\": ";
        print_vec(noarb_vols);
        std::cout << ",\n";
        std::cout << "    \"sabr_volatility\": ";
        print_vec(sabr_vols);
        std::cout << ",\n";
        std::cout << "    \"sabr_option_price\": ";
        print_vec(sabr_opt_prices);
        std::cout << "\n  },\n";
    }

    // ============================================================
    // 3) D0Interpolator absorption-matrix checkpoints.
    //    From test-suite/noarbsabr.cpp testAbsorptionMatrix.
    //    Each row: (sigmaI, beta, rho, nu, tau, expected_absorptions).
    //    D0Interpolator returns d0 = absProb fraction; absorptions =
    //    d0 * nsim (nsim = 2.5e6). We report the raw d0 so the Python
    //    port can compare both the fraction and the implied count.
    // ============================================================
    {
        // forward is irrelevant (cancels in sigmaI mapping); use 0.03.
        const Real forward = 0.03;
        struct Row { Real sigmaI, beta, rho, nu, tau; long absorptions; };
        std::vector<Row> rows{
            {1.0, 0.01, 0.75, 0.1, 0.25, 60342},
            {0.8, 0.01, 0.75, 0.1, 0.25, 12148},
            {0.05, 0.01, 0.75, 0.1, 0.25, 0},
            {1.0, 0.01, 0.75, 0.1, 10.0, 1890509},
            {0.8, 0.01, 0.75, 0.1, 10.0, 1740233},
            {0.05, 0.01, 0.75, 0.1, 10.0, 0},
            {1.0, 0.01, 0.75, 0.1, 30.0, 2174176},
            {0.8, 0.01, 0.75, 0.1, 30.0, 2090672},
            {0.05, 0.01, 0.75, 0.1, 30.0, 31},
            {0.35, 0.10, -0.75, 0.1, 0.25, 0},
            {0.35, 0.10, -0.75, 0.1, 14.75, 1087841},
            {0.35, 0.10, -0.75, 0.1, 30.0, 1406569},
            {0.24, 0.90, 0.50, 0.8, 1.25, 27},
            {0.24, 0.90, 0.50, 0.8, 25.75, 167541},
            {0.05, 0.90, -0.75, 0.8, 2.0, 17},
            {0.05, 0.90, -0.75, 0.8, 30.0, 42100},
        };
        std::cout << "  \"d0_checkpoints\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const Row& r = rows[i];
            const Real alpha = r.sigmaI / std::pow(forward, r.beta - 1.0);
            QuantLib::detail::D0Interpolator d(forward, r.tau, alpha, r.beta, r.nu, r.rho);
            const Real d0 = d();
            std::cout << "    {\"sigmaI\": " << r.sigmaI << ", \"beta\": " << r.beta
                      << ", \"rho\": " << r.rho << ", \"nu\": " << r.nu
                      << ", \"tau\": " << r.tau << ", \"forward\": " << forward
                      << ", \"alpha\": " << alpha
                      << ", \"d0\": " << d0
                      << ", \"absorptions\": " << r.absorptions << "}";
            if (i + 1 < rows.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "  ]\n";
    }

    std::cout << "}\n";
    return 0;
}
