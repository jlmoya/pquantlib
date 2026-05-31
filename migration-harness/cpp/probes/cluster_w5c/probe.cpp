// Phase 11 W5-C cluster probe: experimental FD operators + engines.
//
// Captures reference values for:
//
//   * FdmZabrOp — 2D ZABR linear operator (apply on test vector under
//     uniform-grid mesher with known (forward, vol) locations and
//     known (alpha, beta, nu, rho, gamma) parameters).
//
//   * FdmDupire1dOp — 1D Dupire local-volatility operator (apply on
//     test vector with known sigma(x) curve).
//
//   * FdmOrnsteinUhlenbeckOp — 1D OU operator (apply on test vector
//     under uniform mesher with known (speed, vol, level) parameters).
//
//   * FdOrnsteinUhlenbeckVanillaEngine — NPV of a European call under
//     pure OU dynamics (no drift to log-spot conversion — the OU level
//     is the "spot" itself).
//
//   * FdmExtOUJumpModelInnerValue — innerValue at known grid coords
//     and known shape curve (validates the exp(f + x + y) payoff
//     evaluation).
//
// C++ parity:
//   ql/experimental/finitedifferences/fdmzabrop.hpp
//   ql/experimental/finitedifferences/fdmdupire1dop.hpp
//   ql/methods/finitedifferences/operators/fdmornsteinuhlenbeckop.hpp
//   ql/experimental/finitedifferences/fdornsteinuhlenbeckvanillaengine.hpp
//   ql/experimental/finitedifferences/fdmextoujumpmodelinnervalue.hpp
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/experimental/finitedifferences/fdmdupire1dop.hpp>
#include <ql/experimental/finitedifferences/fdmextoujumpmodelinnervalue.hpp>
#include <ql/experimental/finitedifferences/fdmzabrop.hpp>
#include <ql/experimental/finitedifferences/fdornsteinuhlenbeckvanillaengine.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniformgridmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/fdmornsteinuhlenbeckop.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

namespace {

// Print a single-pass-comma list of doubles to stdout in JSON array
// form, e.g.  print_array({1.0, 2.5, 3.0}) -> "[1, 2.5, 3]".
void print_array(const Array& a) {
    std::cout << "[";
    for (Size i = 0; i < a.size(); ++i) {
        std::cout << a[i];
        if (i + 1 < a.size()) std::cout << ", ";
    }
    std::cout << "]";
}

void print_vec(const std::vector<Real>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]";
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // 1) FdmZabrOp apply — uniform 2D grid (forward x vol).
    //
    // Build a uniform mesher on (forward in [0.02, 0.08], vol in
    // [0.01, 0.05]) with 5x4 grid, ZABR params, and apply the
    // operator to a test vector u[i] = sin(2 * forward + 3 * vol).
    // ============================================================
    {
        const Real beta = 0.5;
        const Real nu = 0.4;
        const Real rho = -0.2;
        const Real gamma = 0.7;

        const Size nFwd = 5;
        const Size nVol = 4;

        std::vector<Size> dim{nFwd, nVol};
        ext::shared_ptr<FdmLinearOpLayout> layout(new FdmLinearOpLayout(dim));
        std::vector<std::pair<Real, Real>> bounds{
            std::make_pair(0.02, 0.08),
            std::make_pair(0.01, 0.05),
        };
        ext::shared_ptr<FdmMesher> mesher(
            new UniformGridMesher(layout, bounds));

        FdmZabrOp op(mesher, beta, nu, rho, gamma);

        // Test vector u[k] = sin(2*F[k] + 3*V[k]).
        Array u(layout->size());
        const Array F = mesher->locations(0);
        const Array V = mesher->locations(1);
        for (Size k = 0; k < u.size(); ++k) {
            u[k] = std::sin(2.0 * F[k] + 3.0 * V[k]);
        }
        const Array out_full = op.apply(u);
        const Array out_dx = op.apply_direction(0, u);
        const Array out_dy = op.apply_direction(1, u);
        const Array out_mixed = op.apply_mixed(u);

        std::cout << "  \"fdm_zabr_op_apply\": {\n";
        std::cout << "    \"beta\": " << beta << ",\n";
        std::cout << "    \"nu\": " << nu << ",\n";
        std::cout << "    \"rho\": " << rho << ",\n";
        std::cout << "    \"gamma\": " << gamma << ",\n";
        std::cout << "    \"n_fwd\": " << nFwd << ",\n";
        std::cout << "    \"n_vol\": " << nVol << ",\n";
        std::cout << "    \"fwd_min\": " << bounds[0].first << ",\n";
        std::cout << "    \"fwd_max\": " << bounds[0].second << ",\n";
        std::cout << "    \"vol_min\": " << bounds[1].first << ",\n";
        std::cout << "    \"vol_max\": " << bounds[1].second << ",\n";
        std::cout << "    \"u\": ";
        print_array(u);
        std::cout << ",\n";
        std::cout << "    \"forward_locations\": ";
        print_array(F);
        std::cout << ",\n";
        std::cout << "    \"vol_locations\": ";
        print_array(V);
        std::cout << ",\n";
        std::cout << "    \"apply\": ";
        print_array(out_full);
        std::cout << ",\n";
        std::cout << "    \"apply_dx\": ";
        print_array(out_dx);
        std::cout << ",\n";
        std::cout << "    \"apply_dy\": ";
        print_array(out_dy);
        std::cout << ",\n";
        std::cout << "    \"apply_mixed\": ";
        print_array(out_mixed);
        std::cout << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) FdmDupire1dOp apply — uniform 1D mesher with known local-vol
    //    array. Local vol grid: sigma[i] = 0.20 + 0.01 * i (5 points).
    // ============================================================
    {
        const Size n = 5;
        std::vector<Size> dim{n};
        ext::shared_ptr<FdmLinearOpLayout> layout(new FdmLinearOpLayout(dim));
        std::vector<std::pair<Real, Real>> bounds{
            std::make_pair(80.0, 120.0),
        };
        ext::shared_ptr<FdmMesher> mesher(
            new UniformGridMesher(layout, bounds));

        Array local_vol(n);
        for (Size i = 0; i < n; ++i) {
            local_vol[i] = 0.20 + 0.01 * Real(i);
        }
        FdmDupire1dOp op(mesher, local_vol);

        Array u(n);
        const Array S = mesher->locations(0);
        for (Size i = 0; i < n; ++i) {
            u[i] = (S[i] - 100.0) * (S[i] - 100.0);  // quadratic test func.
        }
        const Array out = op.apply(u);

        std::cout << "  \"fdm_dupire_1d_op_apply\": {\n";
        std::cout << "    \"local_vol\": ";
        print_array(local_vol);
        std::cout << ",\n";
        std::cout << "    \"n\": " << n << ",\n";
        std::cout << "    \"s_min\": " << bounds[0].first << ",\n";
        std::cout << "    \"s_max\": " << bounds[0].second << ",\n";
        std::cout << "    \"u\": ";
        print_array(u);
        std::cout << ",\n";
        std::cout << "    \"locations\": ";
        print_array(S);
        std::cout << ",\n";
        std::cout << "    \"apply\": ";
        print_array(out);
        std::cout << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) FdmOrnsteinUhlenbeckOp apply — uniform 1D mesher with known
    //    (speed=0.5, vol=0.10, level=0.05) and reference rate=2%.
    //    Used to validate the OU PDE operator. The op is built once,
    //    then set_time(0, T) is called before apply.
    // ============================================================
    {
        const Real speed = 0.5;
        const Real vol = 0.10;
        const Real level = 0.05;
        const Real x0 = 0.05;
        const Size n = 7;
        const Real x_min = -0.05;
        const Real x_max = 0.15;
        const Real T = 0.5;
        const Real flat_rate = 0.02;

        ext::shared_ptr<OrnsteinUhlenbeckProcess> ou_process(
            new OrnsteinUhlenbeckProcess(speed, vol, x0, level));

        const Date today = Date(15, May, 2026);
        Settings::instance().evaluationDate() = today;
        DayCounter dc = Actual365Fixed();

        ext::shared_ptr<YieldTermStructure> rTS(
            new FlatForward(today, flat_rate, dc));

        std::vector<Size> dim{n};
        ext::shared_ptr<FdmLinearOpLayout> layout(new FdmLinearOpLayout(dim));
        std::vector<std::pair<Real, Real>> bounds{
            std::make_pair(x_min, x_max),
        };
        ext::shared_ptr<FdmMesher> mesher(
            new UniformGridMesher(layout, bounds));

        FdmOrnsteinUhlenbeckOp op(mesher, ou_process, rTS, 0);
        op.setTime(0.0, T);

        Array u(n);
        const Array X = mesher->locations(0);
        for (Size i = 0; i < n; ++i) {
            u[i] = std::sin(50.0 * X[i]);
        }
        const Array out = op.apply(u);

        std::cout << "  \"fdm_ornstein_uhlenbeck_op_apply\": {\n";
        std::cout << "    \"speed\": " << speed << ",\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"level\": " << level << ",\n";
        std::cout << "    \"x0\": " << x0 << ",\n";
        std::cout << "    \"n\": " << n << ",\n";
        std::cout << "    \"x_min\": " << x_min << ",\n";
        std::cout << "    \"x_max\": " << x_max << ",\n";
        std::cout << "    \"t1\": 0.0,\n";
        std::cout << "    \"t2\": " << T << ",\n";
        std::cout << "    \"flat_rate\": " << flat_rate << ",\n";
        std::cout << "    \"u\": ";
        print_array(u);
        std::cout << ",\n";
        std::cout << "    \"locations\": ";
        print_array(X);
        std::cout << ",\n";
        std::cout << "    \"apply\": ";
        print_array(out);
        std::cout << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 4) FdOrnsteinUhlenbeckVanillaEngine — NPV of a European call.
    //    OU dynamics: x0=0.05, speed=0.5, sigma=0.10, level=0.05.
    //    Strike=0.05, T=0.5y, r=2%, default grid (t=100, x=100).
    // ============================================================
    {
        const Real speed = 0.5;
        const Real vol = 0.10;
        const Real level = 0.05;
        const Real x0 = 0.05;
        const Real strike = 0.05;
        const Real T_years = 0.5;
        const Real flat_rate = 0.02;
        const Size t_grid = 100;
        const Size x_grid = 100;

        const Date today = Date(15, May, 2026);
        Settings::instance().evaluationDate() = today;
        DayCounter dc = Actual365Fixed();
        const Date expiry = today + Period(Size(T_years * 365), Days);

        ext::shared_ptr<OrnsteinUhlenbeckProcess> ou_process(
            new OrnsteinUhlenbeckProcess(speed, vol, x0, level));
        ext::shared_ptr<YieldTermStructure> rTS(
            new FlatForward(today, flat_rate, dc));

        ext::shared_ptr<StrikedTypePayoff> payoff(
            new PlainVanillaPayoff(Option::Call, strike));
        ext::shared_ptr<Exercise> exercise(new EuropeanExercise(expiry));

        VanillaOption opt(payoff, exercise);
        ext::shared_ptr<PricingEngine> engine(
            new FdOrnsteinUhlenbeckVanillaEngine(
                ou_process, rTS, t_grid, x_grid));
        opt.setPricingEngine(engine);

        const Real npv_call = opt.NPV();

        // Also test a put.
        ext::shared_ptr<StrikedTypePayoff> put_payoff(
            new PlainVanillaPayoff(Option::Put, strike));
        VanillaOption put_opt(put_payoff, exercise);
        put_opt.setPricingEngine(engine);
        const Real npv_put = put_opt.NPV();

        // And an OTM call to validate the long tail.
        ext::shared_ptr<StrikedTypePayoff> otm_payoff(
            new PlainVanillaPayoff(Option::Call, 0.08));
        VanillaOption otm_opt(otm_payoff, exercise);
        otm_opt.setPricingEngine(engine);
        const Real npv_otm_call = otm_opt.NPV();

        const Real exact_T = dc.yearFraction(today, expiry);

        std::cout << "  \"fd_ornstein_uhlenbeck_vanilla_engine_npv\": {\n";
        std::cout << "    \"speed\": " << speed << ",\n";
        std::cout << "    \"vol\": " << vol << ",\n";
        std::cout << "    \"level\": " << level << ",\n";
        std::cout << "    \"x0\": " << x0 << ",\n";
        std::cout << "    \"strike\": " << strike << ",\n";
        std::cout << "    \"T_years_input\": " << T_years << ",\n";
        std::cout << "    \"T_years_actual\": " << exact_T << ",\n";
        std::cout << "    \"flat_rate\": " << flat_rate << ",\n";
        std::cout << "    \"t_grid\": " << t_grid << ",\n";
        std::cout << "    \"x_grid\": " << x_grid << ",\n";
        std::cout << "    \"npv_call_atm\": " << npv_call << ",\n";
        std::cout << "    \"npv_put_atm\": " << npv_put << ",\n";
        std::cout << "    \"npv_call_otm_0_08\": " << npv_otm_call << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 5) FdmExtOUJumpModelInnerValue — innerValue at known shape curve
    //    and known mesh coords. shape(t) = piecewise constant.
    // ============================================================
    {
        const Size nx = 4;
        const Size ny = 3;
        const Real x_min = -1.0;
        const Real x_max = 1.0;
        const Real y_min = -0.5;
        const Real y_max = 0.5;

        std::vector<Size> dim{nx, ny};
        ext::shared_ptr<FdmLinearOpLayout> layout(new FdmLinearOpLayout(dim));
        std::vector<std::pair<Real, Real>> bounds{
            std::make_pair(x_min, x_max),
            std::make_pair(y_min, y_max),
        };
        ext::shared_ptr<FdmMesher> mesher(
            new UniformGridMesher(layout, bounds));

        const Real strike = 3.0;
        ext::shared_ptr<Payoff> payoff(
            new PlainVanillaPayoff(Option::Call, strike));

        typedef FdmExtOUJumpModelInnerValue::Shape Shape;
        ext::shared_ptr<Shape> shape(new Shape);
        shape->push_back({0.25, 1.0});
        shape->push_back({1.00, 1.5});

        FdmExtOUJumpModelInnerValue calc(payoff, mesher, shape);
        const Real t_test = 0.5;

        // Iterate the mesh.
        std::vector<Real> inner_values;
        for (auto iter = layout->begin(); iter != layout->end(); ++iter) {
            inner_values.push_back(calc.innerValue(iter, t_test));
        }

        // Locations.
        const Array X = mesher->locations(0);
        const Array Y = mesher->locations(1);

        std::cout << "  \"fdm_ext_ou_jump_model_inner_value\": {\n";
        std::cout << "    \"nx\": " << nx << ",\n";
        std::cout << "    \"ny\": " << ny << ",\n";
        std::cout << "    \"strike\": " << strike << ",\n";
        std::cout << "    \"t_test\": " << t_test << ",\n";
        std::cout << "    \"shape\": [[0.25, 1.0], [1.0, 1.5]],\n";
        std::cout << "    \"x_locations\": ";
        print_array(X);
        std::cout << ",\n";
        std::cout << "    \"y_locations\": ";
        print_array(Y);
        std::cout << ",\n";
        std::cout << "    \"inner_values\": ";
        print_vec(inner_values);
        std::cout << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
