// L4-B cluster probe — short-rate models.
//
// Captures reference values for the L4-B short-rate-model layer:
//
//   * OrnsteinUhlenbeckProcess (drift/diffusion/expectation/std-deviation/
//     variance) at known (a, sigma, x0, level) — Vasicek's underlying.
//   * CoxIngersollRossProcess (drift/diffusion/expectation/variance/evolve)
//     under Feller condition.
//   * Vasicek (r0=0.05, a=0.1, b=0.05, sigma=0.01) discount_bond at multi
//     (t, T) pairs; A(t, T) + B(t, T) probes; discountBondOption call/put.
//   * HullWhite (a=0.1, sigma=0.01) fit to FlatForward 5% curve;
//     discount_bond round-trip + discountBondOption call/put.
//   * CoxIngersollRoss (r0=0.05, theta=0.06, k=0.5, sigma=0.1) [Feller OK]
//     discount_bond + A + B + discountBondOption.
//   * ExtendedCoxIngersollRoss fit + discount_bond.
//
// C++ parity:
//   ql/processes/ornsteinuhlenbeckprocess.{hpp,cpp},
//   ql/processes/coxingersollrossprocess.{hpp,cpp},
//   ql/models/shortrate/onefactormodel.{hpp,cpp},
//   ql/models/shortrate/onefactormodels/vasicek.{hpp,cpp},
//   ql/models/shortrate/onefactormodels/hullwhite.{hpp,cpp},
//   ql/models/shortrate/onefactormodels/coxingersollross.{hpp,cpp},
//   ql/models/shortrate/onefactormodels/extendedcoxingersollross.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/handle.hpp>
#include <ql/models/shortrate/onefactormodels/coxingersollross.hpp>
#include <ql/models/shortrate/onefactormodels/extendedcoxingersollross.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/onefactormodels/vasicek.hpp>
#include <ql/processes/coxingersollrossprocess.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // -----------------------------------------------------------------
    // OrnsteinUhlenbeckProcess
    // -----------------------------------------------------------------
    {
        // OU dx = a*(level - x)dt + sigma dW with a=0.5, sigma=0.02,
        // x0=0.03, level=0.05.
        Real a = 0.5;
        Real sigma = 0.02;
        Real x0 = 0.03;
        Real level = 0.05;
        OrnsteinUhlenbeckProcess p(a, sigma, x0, level);

        std::cout << "  \"ornstein_uhlenbeck\": {\n";
        std::cout << "    \"x0\": " << p.x0() << ",\n";
        std::cout << "    \"speed\": " << p.speed() << ",\n";
        std::cout << "    \"volatility\": " << p.volatility() << ",\n";
        std::cout << "    \"level\": " << p.level() << ",\n";
        std::cout << "    \"drift_at_0_03\": " << p.drift(0.0, 0.03) << ",\n";
        std::cout << "    \"drift_at_0_08\": " << p.drift(1.0, 0.08) << ",\n";
        std::cout << "    \"diffusion_at_0_03\": " << p.diffusion(0.0, 0.03) << ",\n";
        std::cout << "    \"expectation_dt_1\": " << p.expectation(0.0, 0.03, 1.0) << ",\n";
        std::cout << "    \"variance_dt_1\": " << p.variance(0.0, 0.03, 1.0) << ",\n";
        std::cout << "    \"variance_dt_5\": " << p.variance(0.0, 0.03, 5.0) << ",\n";
        std::cout << "    \"std_deviation_dt_1\": " << p.stdDeviation(0.0, 0.03, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // OU with near-zero speed (algebraic-limit branch)
    {
        Real a = 1e-12;
        Real sigma = 0.02;
        OrnsteinUhlenbeckProcess p(a, sigma, 0.03, 0.05);
        std::cout << "  \"ornstein_uhlenbeck_zero_speed\": {\n";
        std::cout << "    \"variance_dt_1\": " << p.variance(0.0, 0.03, 1.0) << ",\n";
        std::cout << "    \"variance_dt_5\": " << p.variance(0.0, 0.03, 5.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // CoxIngersollRossProcess
    // -----------------------------------------------------------------
    {
        // CIR dx = k*(theta - x)dt + sigma sqrt(x) dW; Feller ok (2*k*theta > sigma^2).
        Real k = 0.5;
        Real theta = 0.06;
        Real sigma = 0.1;
        Real x0 = 0.05;
        CoxIngersollRossProcess p(k, sigma, x0, theta);
        std::cout << "  \"cox_ingersoll_ross\": {\n";
        std::cout << "    \"x0\": " << p.x0() << ",\n";
        std::cout << "    \"speed\": " << p.speed() << ",\n";
        std::cout << "    \"volatility\": " << p.volatility() << ",\n";
        std::cout << "    \"level\": " << p.level() << ",\n";
        std::cout << "    \"drift_at_x0\": " << p.drift(0.0, x0) << ",\n";
        std::cout << "    \"diffusion_at_x0\": " << p.diffusion(0.0, x0) << ",\n";
        std::cout << "    \"expectation_dt_1\": " << p.expectation(0.0, x0, 1.0) << ",\n";
        std::cout << "    \"variance_dt_1\": " << p.variance(0.0, x0, 1.0) << ",\n";
        std::cout << "    \"variance_dt_5\": " << p.variance(0.0, x0, 5.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // Vasicek
    // -----------------------------------------------------------------
    {
        Vasicek v(0.05, 0.1, 0.05, 0.01, 0.0);
        std::cout << "  \"vasicek\": {\n";
        std::cout << "    \"r0\": " << v.r0() << ",\n";
        std::cout << "    \"a\": " << v.a() << ",\n";
        std::cout << "    \"b\": " << v.b() << ",\n";
        std::cout << "    \"sigma\": " << v.sigma() << ",\n";
        std::cout << "    \"lambda\": " << v.lambda() << ",\n";
        std::cout << "    \"discount_t1\": " << v.discount(1.0) << ",\n";
        std::cout << "    \"discount_t5\": " << v.discount(5.0) << ",\n";
        std::cout << "    \"discount_t10\": " << v.discount(10.0) << ",\n";
        std::cout << "    \"discount_bond_0_1\": " << v.discountBond(0.0, 1.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_0_5\": " << v.discountBond(0.0, 5.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_0_10\": " << v.discountBond(0.0, 10.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_1_5_at_r03\": " << v.discountBond(1.0, 5.0, 0.03) << ",\n";
        std::cout << "    \"discount_bond_1_5_at_r07\": " << v.discountBond(1.0, 5.0, 0.07) << ",\n";
        std::cout << "    \"discount_bond_option_call\": " << v.discountBondOption(Option::Call, 0.85, 1.0, 5.0) << ",\n";
        std::cout << "    \"discount_bond_option_put\": " << v.discountBondOption(Option::Put, 0.85, 1.0, 5.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // HullWhite — fits a FlatForward(5%) curve
    // -----------------------------------------------------------------
    {
        Date today(15, May, 2026);
        DayCounter dc = Actual365Fixed();
        Handle<Quote> rate(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(today, rate, dc, Continuous, Annual));
        HullWhite hw(curve, 0.1, 0.01);

        std::cout << "  \"hull_white\": {\n";
        std::cout << "    \"a\": " << hw.a() << ",\n";
        std::cout << "    \"sigma\": " << hw.sigma() << ",\n";
        // The HullWhite term-structure-fitting parameter ensures that
        // discount(t) round-trips to curve->discount(t) for any t.
        std::cout << "    \"curve_discount_1\": " << curve->discount(1.0) << ",\n";
        std::cout << "    \"curve_discount_5\": " << curve->discount(5.0) << ",\n";
        std::cout << "    \"curve_discount_10\": " << curve->discount(10.0) << ",\n";
        std::cout << "    \"model_discount_1\": " << hw.discount(1.0) << ",\n";
        std::cout << "    \"model_discount_5\": " << hw.discount(5.0) << ",\n";
        std::cout << "    \"model_discount_10\": " << hw.discount(10.0) << ",\n";
        std::cout << "    \"discount_bond_0_1_at_r05\": " << hw.discountBond(0.0, 1.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_0_5_at_r05\": " << hw.discountBond(0.0, 5.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_1_5_at_r05\": " << hw.discountBond(1.0, 5.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_1_5_at_r03\": " << hw.discountBond(1.0, 5.0, 0.03) << ",\n";
        std::cout << "    \"discount_bond_option_call\": " << hw.discountBondOption(Option::Call, 0.85, 1.0, 5.0) << ",\n";
        std::cout << "    \"discount_bond_option_put\": " << hw.discountBondOption(Option::Put, 0.85, 1.0, 5.0) << ",\n";
        std::cout << "    \"discount_bond_option_3args_call\": " << hw.discountBondOption(Option::Call, 0.85, 1.0, 1.5, 5.0) << ",\n";
        std::cout << "    \"discount_bond_option_3args_put\": " << hw.discountBondOption(Option::Put, 0.85, 1.0, 1.5, 5.0) << "\n";
        std::cout << "  },\n";
    }

    // HullWhite::convexityBias static
    {
        Real cb = HullWhite::convexityBias(99.0, 0.5, 0.75, 0.01, 0.1);
        std::cout << "  \"hull_white_convexity_bias\": " << cb << ",\n";
    }

    // -----------------------------------------------------------------
    // CoxIngersollRoss (Feller-OK params)
    // -----------------------------------------------------------------
    {
        // 2 * k * theta = 2*0.5*0.06 = 0.06, sigma^2 = 0.01, so Feller ok.
        CoxIngersollRoss cir(0.05, 0.06, 0.5, 0.1, true);
        std::cout << "  \"cox_ingersoll_ross_model\": {\n";
        std::cout << "    \"discount_t1\": " << cir.discount(1.0) << ",\n";
        std::cout << "    \"discount_t5\": " << cir.discount(5.0) << ",\n";
        std::cout << "    \"discount_t10\": " << cir.discount(10.0) << ",\n";
        std::cout << "    \"discount_bond_0_1_at_r05\": " << cir.discountBond(0.0, 1.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_0_5_at_r05\": " << cir.discountBond(0.0, 5.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_1_5_at_r03\": " << cir.discountBond(1.0, 5.0, 0.03) << ",\n";
        std::cout << "    \"discount_bond_option_call\": " << cir.discountBondOption(Option::Call, 0.85, 1.0, 5.0) << ",\n";
        std::cout << "    \"discount_bond_option_put\": " << cir.discountBondOption(Option::Put, 0.85, 1.0, 5.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // ExtendedCoxIngersollRoss — fits FlatForward(5%)
    // -----------------------------------------------------------------
    {
        Date today(15, May, 2026);
        DayCounter dc = Actual365Fixed();
        Handle<Quote> rate(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(today, rate, dc, Continuous, Annual));
        ExtendedCoxIngersollRoss ecir(curve, 0.06, 0.5, 0.1, 0.05, true);
        std::cout << "  \"extended_cox_ingersoll_ross\": {\n";
        std::cout << "    \"model_discount_1\": " << ecir.discount(1.0) << ",\n";
        std::cout << "    \"model_discount_5\": " << ecir.discount(5.0) << ",\n";
        std::cout << "    \"model_discount_10\": " << ecir.discount(10.0) << ",\n";
        std::cout << "    \"discount_bond_0_1_at_r05\": " << ecir.discountBond(0.0, 1.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_0_5_at_r05\": " << ecir.discountBond(0.0, 5.0, 0.05) << ",\n";
        std::cout << "    \"discount_bond_1_5_at_r03\": " << ecir.discountBond(1.0, 5.0, 0.03) << ",\n";
        std::cout << "    \"discount_bond_option_call\": " << ecir.discountBondOption(Option::Call, 0.85, 1.0, 5.0) << ",\n";
        std::cout << "    \"discount_bond_option_put\": " << ecir.discountBondOption(Option::Put, 0.85, 1.0, 5.0) << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
