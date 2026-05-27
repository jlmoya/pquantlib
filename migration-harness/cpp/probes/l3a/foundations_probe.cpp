// L3-A foundations mega-probe.
//
// Captures reference values for the L3-A foundations layer (the
// Phase 3 pilot):
//
//   * Payoff hierarchy: PlainVanilla, CashOrNothing, AssetOrNothing,
//     Gap, SuperFund, SuperShare — evaluated at several spot prices.
//   * blackFormula (lognormal): Call and Put at a representative input
//     grid; checks at zero stdDev edge case as well.
//   * blackFormulaStdDevDerivative (Black vega/sqrt(T)).
//   * blackFormulaVolDerivative (Black vega).
//   * blackFormulaImpliedStdDev roundtrip (newton-safe based solver).
//   * bachelierBlackFormula at the same grid.
//   * bachelierBlackFormulaImpliedVol roundtrip.
//   * bachelierBlackFormulaStdDevDerivative.
//
// C++ parity:
//   ql/instruments/payoffs.{hpp,cpp},
//   ql/pricingengines/blackformula.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/instruments/payoffs.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/option.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // -----------------------------------------------------------------
    // Payoffs
    // -----------------------------------------------------------------
    {
        std::cout << "  \"payoffs\": {\n";

        // PlainVanillaPayoff
        {
            PlainVanillaPayoff call(Option::Call, 100.0);
            PlainVanillaPayoff put(Option::Put, 100.0);
            std::cout << "    \"plain_vanilla\": {\n";
            std::cout << "      \"name\": \"" << call.name() << "\",\n";
            std::cout << "      \"call_at_120\": " << call(120.0) << ",\n";
            std::cout << "      \"call_at_100\": " << call(100.0) << ",\n";
            std::cout << "      \"call_at_80\": " << call(80.0) << ",\n";
            std::cout << "      \"put_at_120\": " << put(120.0) << ",\n";
            std::cout << "      \"put_at_100\": " << put(100.0) << ",\n";
            std::cout << "      \"put_at_80\": " << put(80.0) << "\n";
            std::cout << "    },\n";
        }

        // CashOrNothingPayoff
        {
            CashOrNothingPayoff call(Option::Call, 100.0, 1.0);
            CashOrNothingPayoff put(Option::Put, 100.0, 1.0);
            std::cout << "    \"cash_or_nothing\": {\n";
            std::cout << "      \"name\": \"" << call.name() << "\",\n";
            std::cout << "      \"call_at_120\": " << call(120.0) << ",\n";
            std::cout << "      \"call_at_100\": " << call(100.0) << ",\n";
            std::cout << "      \"call_at_80\": " << call(80.0) << ",\n";
            std::cout << "      \"put_at_120\": " << put(120.0) << ",\n";
            std::cout << "      \"put_at_100\": " << put(100.0) << ",\n";
            std::cout << "      \"put_at_80\": " << put(80.0) << "\n";
            std::cout << "    },\n";
        }

        // AssetOrNothingPayoff
        {
            AssetOrNothingPayoff call(Option::Call, 100.0);
            AssetOrNothingPayoff put(Option::Put, 100.0);
            std::cout << "    \"asset_or_nothing\": {\n";
            std::cout << "      \"name\": \"" << call.name() << "\",\n";
            std::cout << "      \"call_at_120\": " << call(120.0) << ",\n";
            std::cout << "      \"call_at_100\": " << call(100.0) << ",\n";
            std::cout << "      \"call_at_80\": " << call(80.0) << ",\n";
            std::cout << "      \"put_at_120\": " << put(120.0) << ",\n";
            std::cout << "      \"put_at_100\": " << put(100.0) << ",\n";
            std::cout << "      \"put_at_80\": " << put(80.0) << "\n";
            std::cout << "    },\n";
        }

        // GapPayoff
        {
            GapPayoff call(Option::Call, 100.0, 110.0);
            GapPayoff put(Option::Put, 100.0, 90.0);
            std::cout << "    \"gap\": {\n";
            std::cout << "      \"name\": \"" << call.name() << "\",\n";
            std::cout << "      \"call_at_120\": " << call(120.0) << ",\n";
            std::cout << "      \"call_at_100\": " << call(100.0) << ",\n";
            std::cout << "      \"call_at_80\": " << call(80.0) << ",\n";
            std::cout << "      \"put_at_120\": " << put(120.0) << ",\n";
            std::cout << "      \"put_at_100\": " << put(100.0) << ",\n";
            std::cout << "      \"put_at_80\": " << put(80.0) << "\n";
            std::cout << "    },\n";
        }

        // SuperFundPayoff (strike=100, second_strike=120) — payout
        // price/strike if strike <= price < second_strike, else 0.
        {
            SuperFundPayoff sf(100.0, 120.0);
            std::cout << "    \"super_fund\": {\n";
            std::cout << "      \"name\": \"" << sf.name() << "\",\n";
            std::cout << "      \"at_90\": " << sf(90.0) << ",\n";
            std::cout << "      \"at_100\": " << sf(100.0) << ",\n";
            std::cout << "      \"at_110\": " << sf(110.0) << ",\n";
            std::cout << "      \"at_120\": " << sf(120.0) << ",\n";
            std::cout << "      \"at_130\": " << sf(130.0) << "\n";
            std::cout << "    },\n";
        }

        // SuperSharePayoff (strike=100, second_strike=120, cash=1.0)
        {
            SuperSharePayoff ss(100.0, 120.0, 1.0);
            std::cout << "    \"super_share\": {\n";
            std::cout << "      \"name\": \"" << ss.name() << "\",\n";
            std::cout << "      \"at_90\": " << ss(90.0) << ",\n";
            std::cout << "      \"at_100\": " << ss(100.0) << ",\n";
            std::cout << "      \"at_110\": " << ss(110.0) << ",\n";
            std::cout << "      \"at_120\": " << ss(120.0) << ",\n";
            std::cout << "      \"at_130\": " << ss(130.0) << "\n";
            std::cout << "    }\n";
        }

        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // blackFormula (lognormal Black-76)
    //
    // Inputs:
    //   forward = 100, std_dev = 0.20*sqrt(1.0) = 0.20 (i.e. vol 20%
    //   over 1 yr), discount = 0.95.
    // -----------------------------------------------------------------
    {
        const double F = 100.0;
        const double sigma_sqrt_T = 0.20;
        const double df = 0.95;

        std::cout << "  \"black_formula\": {\n";
        std::cout << "    \"forward\": " << F << ",\n";
        std::cout << "    \"std_dev\": " << sigma_sqrt_T << ",\n";
        std::cout << "    \"discount\": " << df << ",\n";

        // ATM call & put
        double call_atm = blackFormula(Option::Call, 100.0, F, sigma_sqrt_T, df);
        double put_atm  = blackFormula(Option::Put,  100.0, F, sigma_sqrt_T, df);
        std::cout << "    \"call_k100\": " << call_atm << ",\n";
        std::cout << "    \"put_k100\": " << put_atm << ",\n";

        // ITM call (K=80), OTM call (K=120)
        std::cout << "    \"call_k80\": " << blackFormula(Option::Call, 80.0, F, sigma_sqrt_T, df) << ",\n";
        std::cout << "    \"call_k120\": " << blackFormula(Option::Call, 120.0, F, sigma_sqrt_T, df) << ",\n";

        // Put-call parity: c - p = df * (F - K)  ⇒ at ATM, c == p.
        std::cout << "    \"put_k80\": " << blackFormula(Option::Put, 80.0, F, sigma_sqrt_T, df) << ",\n";
        std::cout << "    \"put_k120\": " << blackFormula(Option::Put, 120.0, F, sigma_sqrt_T, df) << ",\n";

        // Zero std_dev edge: returns intrinsic * discount
        std::cout << "    \"call_zero_stddev\": " << blackFormula(Option::Call, 80.0, F, 0.0, df) << ",\n";
        std::cout << "    \"put_zero_stddev\": " << blackFormula(Option::Put, 120.0, F, 0.0, df) << ",\n";

        // Default discount = 1.0, no displacement
        std::cout << "    \"call_k100_no_df\": " << blackFormula(Option::Call, 100.0, F, sigma_sqrt_T, 1.0) << ",\n";

        // Displacement
        std::cout << "    \"call_k100_shift10\": "
                  << blackFormula(Option::Call, 100.0, F, sigma_sqrt_T, 1.0, 10.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // blackFormulaStdDevDerivative + blackFormulaVolDerivative
    // -----------------------------------------------------------------
    {
        const double F = 100.0;
        const double K = 100.0;
        const double sigma_sqrt_T = 0.20;
        const double df = 0.95;
        const double T = 1.0;

        double std_dev_deriv = blackFormulaStdDevDerivative(K, F, sigma_sqrt_T, df);
        double vol_deriv = blackFormulaVolDerivative(K, F, sigma_sqrt_T, T, df);
        std::cout << "  \"black_derivatives\": {\n";
        std::cout << "    \"std_dev_derivative\": " << std_dev_deriv << ",\n";
        std::cout << "    \"vol_derivative\": " << vol_deriv << ",\n";
        std::cout << "    \"vol_derivative_check\": " << std_dev_deriv * std::sqrt(T) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // blackFormulaImpliedStdDev roundtrip
    // -----------------------------------------------------------------
    {
        const double F = 100.0;
        const double K = 100.0;
        const double sigma_sqrt_T = 0.20;
        const double df = 0.95;
        double price = blackFormula(Option::Call, K, F, sigma_sqrt_T, df);
        double implied = blackFormulaImpliedStdDev(Option::Call, K, F, price, df);
        std::cout << "  \"black_implied_std_dev\": {\n";
        std::cout << "    \"price\": " << price << ",\n";
        std::cout << "    \"implied\": " << implied << ",\n";
        std::cout << "    \"original\": " << sigma_sqrt_T << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // bachelierBlackFormula (normal model)
    // -----------------------------------------------------------------
    {
        const double F = 100.0;
        const double sigma_sqrt_T = 5.0; // absolute vol, e.g. 5% of F
        const double df = 0.95;

        std::cout << "  \"bachelier_black_formula\": {\n";
        std::cout << "    \"forward\": " << F << ",\n";
        std::cout << "    \"std_dev\": " << sigma_sqrt_T << ",\n";
        std::cout << "    \"discount\": " << df << ",\n";

        std::cout << "    \"call_k100\": " << bachelierBlackFormula(Option::Call, 100.0, F, sigma_sqrt_T, df) << ",\n";
        std::cout << "    \"put_k100\": " << bachelierBlackFormula(Option::Put, 100.0, F, sigma_sqrt_T, df) << ",\n";

        std::cout << "    \"call_k80\": " << bachelierBlackFormula(Option::Call, 80.0, F, sigma_sqrt_T, df) << ",\n";
        std::cout << "    \"call_k120\": " << bachelierBlackFormula(Option::Call, 120.0, F, sigma_sqrt_T, df) << ",\n";
        std::cout << "    \"put_k80\": " << bachelierBlackFormula(Option::Put, 80.0, F, sigma_sqrt_T, df) << ",\n";
        std::cout << "    \"put_k120\": " << bachelierBlackFormula(Option::Put, 120.0, F, sigma_sqrt_T, df) << ",\n";

        std::cout << "    \"call_zero_stddev\": " << bachelierBlackFormula(Option::Call, 80.0, F, 0.0, df) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // bachelierBlackFormulaStdDevDerivative
    // -----------------------------------------------------------------
    {
        const double F = 100.0;
        const double K = 100.0;
        const double sigma_sqrt_T = 5.0;
        const double df = 0.95;
        std::cout << "  \"bachelier_std_dev_derivative\": "
                  << bachelierBlackFormulaStdDevDerivative(K, F, sigma_sqrt_T, df) << ",\n";
    }

    // -----------------------------------------------------------------
    // bachelierBlackFormulaImpliedVol roundtrip
    // -----------------------------------------------------------------
    {
        const double F = 100.0;
        const double K = 100.0;
        const double T = 1.0;
        const double vol = 5.0;            // absolute vol
        const double sigma_sqrt_T = vol * std::sqrt(T);
        const double df = 0.95;
        double price = bachelierBlackFormula(Option::Call, K, F, sigma_sqrt_T, df);
        double implied = bachelierBlackFormulaImpliedVol(Option::Call, K, F, T, price, df);
        std::cout << "  \"bachelier_implied_vol\": {\n";
        std::cout << "    \"price\": " << price << ",\n";
        std::cout << "    \"implied_vol\": " << implied << ",\n";
        std::cout << "    \"original_vol\": " << vol << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
