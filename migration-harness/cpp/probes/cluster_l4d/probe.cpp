// L4-D cluster probe: two-factor + multi-process.
//
// Emits cross-validation reference values for:
//   * G2Process: drift/diffusion/expectation/std_dev/covariance at known
//     (a=0.1, sigma=0.01, b=0.1, eta=0.01, rho=-0.75); origin x0=(0,0).
//   * G2ForwardProcess: expectation/std_dev/drift at T=10y, t0=0, dt=1.
//   * HullWhiteForwardProcess (FlatForward 5% Act/365F curve): drift,
//     expectation, std_dev at t0=0, dt=1, x0=0.
//   * CoxIngersollRossProcess(k=0.5, sigma=0.1, x0=0.05, level=0.04):
//     drift / diffusion / expectation / variance / stdDeviation.
//   * OrnsteinUhlenbeckProcess(speed=0.3, vol=0.05, x0=0.02, level=0.0):
//     drift / diffusion / expectation / variance / stdDeviation.
//   * G2 short-rate model on a FlatForward(5%) Act/365F curve at
//     params (0.1, 0.01, 0.1, 0.01, -0.75):
//       - discount_bond(now, maturity, x, y) on a small grid;
//       - discount_bond_option (Black-formula closed form) call/put;
//       - V(t), A(t,T), B(x,t) inspectable closed-forms.
//   * G2.swaption: 5y x 5y annual payer at fixed=5% — segment integral
//     with 200 intervals, range=5*sigma.
//
// C++ parity:
//   ql/processes/g2process.{hpp,cpp},
//   ql/processes/hullwhiteprocess.{hpp,cpp},
//   ql/processes/coxingersollrossprocess.{hpp,cpp},
//   ql/processes/ornsteinuhlenbeckprocess.{hpp,cpp},
//   ql/models/shortrate/twofactormodels/g2.{hpp,cpp}
//   @ v1.42.1 (099987f0).

#include <ql/handle.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/math/array.hpp>
#include <ql/math/matrix.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/processes/coxingersollrossprocess.hpp>
#include <ql/processes/g2process.hpp>
#include <ql/processes/hullwhiteprocess.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace QuantLib;

namespace {

void emitArray(const Array& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

void emitMatrix(const Matrix& m) {
    std::cout << "[";
    for (Size i = 0; i < m.rows(); ++i) {
        std::cout << "[";
        for (Size j = 0; j < m.columns(); ++j) {
            std::cout << m[i][j];
            if (j + 1 < m.columns()) std::cout << ", ";
        }
        std::cout << "]";
        if (i + 1 < m.rows()) std::cout << ", ";
    }
    std::cout << "]";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ---------------------------------------------------------------
    // OrnsteinUhlenbeckProcess(speed=0.3, vol=0.05, x0=0.02, level=0.0)
    // ---------------------------------------------------------------
    {
        OrnsteinUhlenbeckProcess ou(0.3, 0.05, 0.02, 0.0);
        std::cout << "  \"ornstein_uhlenbeck\": {\n";
        std::cout << "    \"x0\": " << ou.x0() << ",\n";
        std::cout << "    \"speed\": " << ou.speed() << ",\n";
        std::cout << "    \"volatility\": " << ou.volatility() << ",\n";
        std::cout << "    \"level\": " << ou.level() << ",\n";
        std::cout << "    \"drift_t1_x0_02\": " << ou.drift(1.0, 0.02) << ",\n";
        std::cout << "    \"diffusion_t1_x0_02\": " << ou.diffusion(1.0, 0.02) << ",\n";
        std::cout << "    \"expectation_t0_x0_02_dt1\": "
                  << ou.expectation(0.0, 0.02, 1.0) << ",\n";
        std::cout << "    \"variance_t0_x0_02_dt1\": "
                  << ou.variance(0.0, 0.02, 1.0) << ",\n";
        std::cout << "    \"stdDeviation_t0_x0_02_dt1\": "
                  << ou.stdDeviation(0.0, 0.02, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // CoxIngersollRossProcess(k=0.5, vol=0.1, x0=0.05, level=0.04)
    // ---------------------------------------------------------------
    {
        CoxIngersollRossProcess cir(0.5, 0.1, 0.05, 0.04);
        std::cout << "  \"cir\": {\n";
        std::cout << "    \"x0\": " << cir.x0() << ",\n";
        std::cout << "    \"speed\": " << cir.speed() << ",\n";
        std::cout << "    \"volatility\": " << cir.volatility() << ",\n";
        std::cout << "    \"level\": " << cir.level() << ",\n";
        std::cout << "    \"drift_t1_x0_05\": " << cir.drift(1.0, 0.05) << ",\n";
        std::cout << "    \"diffusion_t1_x0_05\": " << cir.diffusion(1.0, 0.05) << ",\n";
        std::cout << "    \"expectation_t0_x0_05_dt1\": "
                  << cir.expectation(0.0, 0.05, 1.0) << ",\n";
        std::cout << "    \"variance_t0_x0_05_dt1\": "
                  << cir.variance(0.0, 0.05, 1.0) << ",\n";
        std::cout << "    \"stdDeviation_t0_x0_05_dt1\": "
                  << cir.stdDeviation(0.0, 0.05, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // G2Process(a=0.1, sigma=0.01, b=0.1, eta=0.01, rho=-0.75)
    // ---------------------------------------------------------------
    {
        G2Process g2p(0.1, 0.01, 0.1, 0.01, -0.75);
        Array x0(2);
        x0[0] = 0.0; x0[1] = 0.0;
        std::cout << "  \"g2_process\": {\n";
        std::cout << "    \"size\": " << g2p.size() << ",\n";
        std::cout << "    \"initial_values\": ";
        emitArray(g2p.initialValues());
        std::cout << ",\n";
        std::cout << "    \"drift_t1_origin\": ";
        emitArray(g2p.drift(1.0, x0));
        std::cout << ",\n";
        std::cout << "    \"diffusion_t1_origin\": ";
        emitMatrix(g2p.diffusion(1.0, x0));
        std::cout << ",\n";
        std::cout << "    \"expectation_t0_origin_dt1\": ";
        emitArray(g2p.expectation(0.0, x0, 1.0));
        std::cout << ",\n";
        std::cout << "    \"std_deviation_t0_origin_dt1\": ";
        emitMatrix(g2p.stdDeviation(0.0, x0, 1.0));
        std::cout << ",\n";
        std::cout << "    \"covariance_t0_origin_dt1\": ";
        emitMatrix(g2p.covariance(0.0, x0, 1.0));
        std::cout << ",\n";
        // Repeat at a non-zero state to sanity-check OU coupling.
        Array xS(2);
        xS[0] = 0.01; xS[1] = -0.005;
        std::cout << "    \"drift_t2_x_001_neg_0005\": ";
        emitArray(g2p.drift(2.0, xS));
        std::cout << ",\n";
        std::cout << "    \"a\": " << g2p.a() << ",\n";
        std::cout << "    \"sigma\": " << g2p.sigma() << ",\n";
        std::cout << "    \"b\": " << g2p.b() << ",\n";
        std::cout << "    \"eta\": " << g2p.eta() << ",\n";
        std::cout << "    \"rho\": " << g2p.rho() << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // G2ForwardProcess at T=10
    // ---------------------------------------------------------------
    {
        G2ForwardProcess g2f(0.1, 0.01, 0.1, 0.01, -0.75);
        g2f.setForwardMeasureTime(10.0);
        Array x0(2);
        x0[0] = 0.0; x0[1] = 0.0;
        std::cout << "  \"g2_forward_process\": {\n";
        std::cout << "    \"size\": " << g2f.size() << ",\n";
        std::cout << "    \"forward_T\": " << g2f.getForwardMeasureTime() << ",\n";
        std::cout << "    \"drift_t1_origin\": ";
        emitArray(g2f.drift(1.0, x0));
        std::cout << ",\n";
        std::cout << "    \"diffusion_t1_origin\": ";
        emitMatrix(g2f.diffusion(1.0, x0));
        std::cout << ",\n";
        std::cout << "    \"expectation_t0_origin_dt1\": ";
        emitArray(g2f.expectation(0.0, x0, 1.0));
        std::cout << ",\n";
        std::cout << "    \"std_deviation_t0_origin_dt1\": ";
        emitMatrix(g2f.stdDeviation(0.0, x0, 1.0));
        std::cout << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // HullWhiteForwardProcess (FlatForward 5% Act/365F)
    // ---------------------------------------------------------------
    Date refDate(15, June, 2026);
    Settings::instance().evaluationDate() = refDate;
    DayCounter dc = Actual365Fixed();
    Handle<YieldTermStructure> flatCurve(
        ext::shared_ptr<YieldTermStructure>(
            new FlatForward(refDate, 0.05, dc, Continuous, Annual)));

    {
        HullWhiteForwardProcess hwf(flatCurve, 0.1, 0.01);
        hwf.setForwardMeasureTime(10.0);
        std::cout << "  \"hull_white_forward_process\": {\n";
        std::cout << "    \"a\": " << hwf.a() << ",\n";
        std::cout << "    \"sigma\": " << hwf.sigma() << ",\n";
        std::cout << "    \"forward_T\": " << hwf.getForwardMeasureTime() << ",\n";
        std::cout << "    \"x0\": " << hwf.x0() << ",\n";
        std::cout << "    \"alpha_at_1\": " << hwf.alpha(1.0) << ",\n";
        std::cout << "    \"alpha_at_5\": " << hwf.alpha(5.0) << ",\n";
        std::cout << "    \"B_t1_T10\": " << hwf.B(1.0, 10.0) << ",\n";
        std::cout << "    \"M_T_s0_t1_T10\": " << hwf.M_T(0.0, 1.0, 10.0) << ",\n";
        // Drift/expectation/variance at t=1, x=hwf.x0()
        Real x = hwf.x0();
        std::cout << "    \"drift_t1_x0\": " << hwf.drift(1.0, x) << ",\n";
        std::cout << "    \"diffusion_t1_x0\": " << hwf.diffusion(1.0, x) << ",\n";
        std::cout << "    \"expectation_t0_x0_dt1\": " << hwf.expectation(0.0, x, 1.0) << ",\n";
        std::cout << "    \"variance_t0_x0_dt1\": " << hwf.variance(0.0, x, 1.0) << ",\n";
        std::cout << "    \"stdDeviation_t0_x0_dt1\": "
                  << hwf.stdDeviation(0.0, x, 1.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // G2 short-rate model
    // ---------------------------------------------------------------
    {
        G2 model(flatCurve, 0.1, 0.01, 0.1, 0.01, -0.75);
        std::cout << "  \"g2\": {\n";
        std::cout << "    \"a\": " << model.a() << ",\n";
        std::cout << "    \"sigma\": " << model.sigma() << ",\n";
        std::cout << "    \"b\": " << model.b() << ",\n";
        std::cout << "    \"eta\": " << model.eta() << ",\n";
        std::cout << "    \"rho\": " << model.rho() << ",\n";
        std::cout << "    \"discount_t1\": " << model.discount(1.0) << ",\n";
        std::cout << "    \"discount_t5\": " << model.discount(5.0) << ",\n";
        std::cout << "    \"discount_t10\": " << model.discount(10.0) << ",\n";
        // discountBond(now=0.5, maturity=5.5, x=0.01, y=-0.005)
        std::cout << "    \"discount_bond_0_5_5_5_x001_yneg0005\": "
                  << model.discountBond(0.5, 5.5, 0.01, -0.005) << ",\n";
        // discountBond(0, 5, 0, 0) — equals discount(5) (curve-implied).
        std::cout << "    \"discount_bond_0_5_origin\": "
                  << model.discountBond(0.0, 5.0, 0.0, 0.0) << ",\n";
        // discountBond via factors Array
        Array factors(2);
        factors[0] = 0.005; factors[1] = -0.002;
        std::cout << "    \"discount_bond_factors_0_3_x_005_yneg002\": "
                  << model.discountBond(0.0, 3.0, factors) << ",\n";
        // discountBondOption: call, strike=0.95, maturity=1, bondMaturity=3
        std::cout << "    \"dbo_call_K0_95_T1_TB3\": "
                  << model.discountBondOption(Option::Call, 0.95, 1.0, 3.0) << ",\n";
        std::cout << "    \"dbo_put_K0_95_T1_TB3\": "
                  << model.discountBondOption(Option::Put, 0.95, 1.0, 3.0) << "\n";
        std::cout << "  },\n";
    }

    // ---------------------------------------------------------------
    // G2.swaption: 5y x 5y annual payer at fixed=5%
    // ---------------------------------------------------------------
    {
        G2 model(flatCurve, 0.1, 0.01, 0.1, 0.01, -0.75);

        // Build a manual Swaption::arguments matching a payer 5x5
        // annual swap. start=5y from refDate; pay dates at 6..10y.
        Swaption::arguments args;
        args.type = Swap::Payer;
        args.nominal = 1.0;
        // floatingResetDates[0] is used as the swap start date.
        args.floatingResetDates.push_back(refDate + 5 * 365);
        for (Size i = 1; i <= 5; ++i) {
            args.fixedPayDates.push_back(refDate + (5 + i) * 365);
        }
        Real fixedRate = 0.05;
        Real range = 5.0;
        Size intervals = 200;
        Real swPrice = model.swaption(args, fixedRate, range, intervals);
        std::cout << "  \"g2_swaption\": {\n";
        std::cout << "    \"start_y\": 5.0,\n";
        std::cout << "    \"tenor_y\": 5.0,\n";
        std::cout << "    \"fixed_rate\": " << fixedRate << ",\n";
        std::cout << "    \"range\": " << range << ",\n";
        std::cout << "    \"intervals\": " << intervals << ",\n";
        std::cout << "    \"price\": " << swPrice << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
