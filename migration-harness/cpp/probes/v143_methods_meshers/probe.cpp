// v1.43 methods coverage: the finite-difference 1-D meshers.
//
// ql/methods/finitedifferences/meshers/{concentrating1dmesher,
// predefined1dmesher, exponentialjump1dmesher, fdmcev1dmesher,
// fdmblackscholesmesher, fdmblackscholesmultistrikemesher,
// fdmhestonvariancemesher}.{hpp,cpp} @ v1.43 (6b57206e0).
//
// For every mesher the full locations / dplus / dminus triple is emitted, so
// a wrong node placement cannot hide behind an aggregate. dplus at the last
// node and dminus at the first are Null<Real>() in C++ (== a huge sentinel);
// they are emitted as the JSON string "null_real" so the Python side has to
// answer NaN there rather than a number.
//
// Concentrating1dMesher is emitted for the plain sinh transform, the
// require-c-point variant (which bends the parameterisation through a
// three-knot linear map), the degenerate c-point-at-the-edge cases, the
// no-c-point fallback, and the multi-critical-point ODE constructor with one
// and with two required points.
//
// Settings::evaluationDate() is deliberately NOT set (see cluster_w5a's
// note); all term structures carry an explicit reference date.

#include <ql/methods/finitedifferences/meshers/concentrating1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/exponentialjump1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmultistrikemesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmcev1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmhestonvariancemesher.hpp>
#include <ql/methods/finitedifferences/meshers/predefined1dmesher.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/localconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/utilities/null.hpp>

#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace QuantLib;

namespace {

    bool g_first = true;

    std::string fmt(Real x) {
        if (x == Null<Real>())
            return "\"null_real\"";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", x);
        return buf;
    }

    void emit_cells(const std::string& key, const std::vector<std::string>& v) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": [";
        for (Size i = 0; i < v.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << v[i];
        }
        std::cout << "]";
    }

    void emit1(const std::string& key, Real v) {
        if (!g_first) std::cout << ",\n";
        g_first = false;
        std::cout << "  \"" << key << "\": " << fmt(v);
    }

    // locations + dplus + dminus, under <prefix>_{loc,dplus,dminus}.
    void emit_mesher(const std::string& prefix, const Fdm1dMesher& m) {
        std::vector<std::string> loc, dp, dm;
        for (Size i = 0; i < m.size(); ++i) {
            loc.push_back(fmt(m.location(i)));
            dp.push_back(fmt(m.dplus(i)));
            dm.push_back(fmt(m.dminus(i)));
        }
        emit_cells(prefix + "_loc", loc);
        emit_cells(prefix + "_dplus", dp);
        emit_cells(prefix + "_dminus", dm);
    }

    const Date REF_DATE(15, May, 2026);
    const DayCounter DC = Actual365Fixed();

    Handle<YieldTermStructure> flat(Rate r) {
        return Handle<YieldTermStructure>(
            ext::make_shared<FlatForward>(REF_DATE, r, DC));
    }

}

int main() {
    std::cout.precision(17);
    std::cout << "{\n";

    // ------------------------------------------------------------------
    // Predefined1dMesher — the trivial case, pins the Null<Real> sentinels.
    // ------------------------------------------------------------------
    {
        const std::vector<Real> x{-2.0, -0.5, 0.0, 0.25, 3.0, 7.5};
        emit_mesher("predef", Predefined1dMesher(x));
    }

    // ------------------------------------------------------------------
    // Concentrating1dMesher — single critical point.
    // ------------------------------------------------------------------
    {
        // no critical point at all -> uniform interior
        emit_mesher("conc_none",
                    Concentrating1dMesher(-1.0, 3.0, 9));
        // critical point in the interior, not required on the grid
        emit_mesher("conc_mid",
                    Concentrating1dMesher(-1.0, 3.0, 9, std::make_pair(0.5, 0.1)));
        // same, but the node is forced onto the critical point
        emit_mesher("conc_mid_req",
                    Concentrating1dMesher(-1.0, 3.0, 9, std::make_pair(0.5, 0.1), true));
        // dense concentration
        emit_mesher("conc_dense",
                    Concentrating1dMesher(-1.0, 3.0, 15, std::make_pair(0.0, 0.005), true));
        // critical point exactly at the start / end: the require-c-point
        // branch skips the middle knot (close(cPoint, start/end)).
        emit_mesher("conc_at_start",
                    Concentrating1dMesher(-1.0, 3.0, 9, std::make_pair(-1.0, 0.1), true));
        emit_mesher("conc_at_end",
                    Concentrating1dMesher(-1.0, 3.0, 9, std::make_pair(3.0, 0.1), true));
        // odd/even size, to pin the lround(z0*(size-1)) clamp
        emit_mesher("conc_n10_req",
                    Concentrating1dMesher(0.0, 1.0, 10, std::make_pair(0.97, 0.05), true));
    }

    // ------------------------------------------------------------------
    // Concentrating1dMesher — multi-critical-point ODE constructor.
    // ------------------------------------------------------------------
    {
        std::vector<std::tuple<Real, Real, bool> > cp1{
            std::make_tuple(0.5, 0.05, false)};
        emit_mesher("conc_ode_1", Concentrating1dMesher(-1.0, 3.0, 11, cp1));

        std::vector<std::tuple<Real, Real, bool> > cp2{
            std::make_tuple(0.0, 0.05, true),
            std::make_tuple(1.5, 0.1, true)};
        emit_mesher("conc_ode_2req", Concentrating1dMesher(-1.0, 3.0, 13, cp2));

        std::vector<std::tuple<Real, Real, bool> > cp3{
            std::make_tuple(-0.25, 0.03, true),
            std::make_tuple(0.75, 0.08, false),
            std::make_tuple(2.0, 0.05, true)};
        emit_mesher("conc_ode_3", Concentrating1dMesher(-1.0, 3.0, 17, cp3));
    }

    // ------------------------------------------------------------------
    // ExponentialJump1dMesher — grid plus the four density/distribution
    // members (the t->inf pair and the finite-t pair).
    // ------------------------------------------------------------------
    {
        const ExponentialJump1dMesher m(12, 4.0, 1.0, 5.0);
        emit_mesher("expjump", m);

        const Real xs[] = {0.01, 0.05, 0.2, 0.5, 1.0, 2.0};
        std::vector<std::string> dens, dist, densT, distT;
        for (Real x : xs) {
            dens.push_back(fmt(m.jumpSizeDensity(x)));
            dist.push_back(fmt(m.jumpSizeDistribution(x)));
            densT.push_back(fmt(m.jumpSizeDensity(x, 1.5)));
            distT.push_back(fmt(m.jumpSizeDistribution(x, 1.5)));
        }
        emit_cells("expjump_density", dens);
        emit_cells("expjump_distribution", dist);
        emit_cells("expjump_density_t1.5", densT);
        emit_cells("expjump_distribution_t1.5", distT);

        // a second parameter set with beta/jumpIntensity swapped roles
        emit_mesher("expjump_b",
                    ExponentialJump1dMesher(8, 2.0, 0.5, 3.0, 1e-2));
    }

    // ------------------------------------------------------------------
    // FdmCEV1dMesher — the uniform-helper branch and the concentrating one,
    // plus the beta<0 lower-bound special case.
    // ------------------------------------------------------------------
    {
        emit_mesher("cev_uniform",
                    FdmCEV1dMesher(11, 100.0, 0.3, 0.3, 1.0));
        emit_mesher("cev_conc",
                    FdmCEV1dMesher(11, 100.0, 0.3, 0.3, 1.0, 1e-4, 1.5,
                                   std::make_pair(100.0, 0.1)));
        emit_mesher("cev_betaneg",
                    FdmCEV1dMesher(11, 1.0, 0.5, -1.0, 1.0));
        emit_mesher("cev_beta15",
                    FdmCEV1dMesher(9, 100.0, 0.5, 1.5, 0.5));
    }

    // ------------------------------------------------------------------
    // FdmBlackScholesMultiStrikeMesher.
    // ------------------------------------------------------------------
    {
        const auto spot = ext::make_shared<SimpleQuote>(100.0);
        const auto process = ext::make_shared<BlackScholesMertonProcess>(
            Handle<Quote>(spot), flat(0.02), flat(0.05),
            Handle<BlackVolTermStructure>(ext::make_shared<BlackConstantVol>(
                REF_DATE, NullCalendar(), 0.25, DC)));

        const std::vector<Real> strikes{80.0, 100.0, 130.0};
        emit_mesher("bsmulti",
                    FdmBlackScholesMultiStrikeMesher(13, process, 1.0, strikes));
        emit_mesher("bsmulti_conc",
                    FdmBlackScholesMultiStrikeMesher(13, process, 1.0, strikes,
                                                     1e-4, 1.5,
                                                     std::make_pair(100.0, 0.1)));
        const std::vector<Real> wide{20.0, 100.0, 400.0};
        emit_mesher("bsmulti_wide",
                    FdmBlackScholesMultiStrikeMesher(11, process, 2.0, wide));
    }

    // ------------------------------------------------------------------
    // FdmHestonVarianceMesher / FdmHestonLocalVolatilityVarianceMesher.
    // ------------------------------------------------------------------
    {
        const auto spot = ext::make_shared<SimpleQuote>(100.0);
        const auto process = ext::make_shared<HestonProcess>(
            flat(0.05), flat(0.02), Handle<Quote>(spot),
            0.09, 1.0, 0.09, 0.4, -0.75);

        {
            const FdmHestonVarianceMesher m(9, process, 1.0);
            emit_mesher("hestonvar", m);
            emit1("hestonvar_volaEstimate", m.volaEstimate());
        }
        {
            const FdmHestonVarianceMesher m(13, process, 2.5, 6, 1e-3);
            emit_mesher("hestonvar_b", m);
            emit1("hestonvar_b_volaEstimate", m.volaEstimate());
        }
        {
            // mixingFactor != 1 rescales sigma throughout
            const FdmHestonVarianceMesher m(9, process, 1.0, 10, 1e-4, 1.7);
            emit_mesher("hestonvar_mix", m);
            emit1("hestonvar_mix_volaEstimate", m.volaEstimate());
        }
        {
            // Heavily Feller-violating parameters (2*kappa*theta << sigma^2,
            // df = 0.0056): the chi-square quantiles crowd towards zero and the
            // minVStep floor becomes the binding constraint on most nodes.
            // NOTE: this does NOT reach the `catch (const Error&)` fallback —
            // QuantLib's inverse chi-square still converges here. No parameter
            // set was found that makes it throw, so that defensive branch is
            // ported but not pinned.
            const auto p2 = ext::make_shared<HestonProcess>(
                flat(0.05), flat(0.02), Handle<Quote>(spot),
                0.04, 0.05, 0.04, 1.2, -0.3);
            const FdmHestonVarianceMesher m(9, p2, 1.0);
            emit_mesher("hestonvar_lowfeller", m);
            emit1("hestonvar_lowfeller_volaEstimate", m.volaEstimate());
        }
        {
            // local-volatility variant, with and without a leverage function
            const FdmHestonLocalVolatilityVarianceMesher m0(
                9, process, ext::shared_ptr<LocalVolTermStructure>(), 1.0);
            emit_mesher("hestonlv_none", m0);
            emit1("hestonlv_none_volaEstimate", m0.volaEstimate());

            const auto lv = ext::make_shared<LocalConstantVol>(REF_DATE, 0.8, DC);
            const FdmHestonLocalVolatilityVarianceMesher m1(9, process, lv, 1.0);
            emit_mesher("hestonlv_flat", m1);
            emit1("hestonlv_flat_volaEstimate", m1.volaEstimate());
        }
    }

    // ------------------------------------------------------------------
    // FdmBlackScholesMesher — the cPoint branch.
    //
    // fdmblackscholesmesher.cpp picks Concentrating1dMesher when
    // cPoint.first is set AND log(cPoint.first) lies inside [xMin, xMax];
    // otherwise Uniform1dMesher. Both arms are emitted here, plus the two
    // ways the guard can reject a cPoint (Null, and out of range), because
    // dropping the branch entirely still reproduces the uniform arm.
    // ------------------------------------------------------------------
    {
        const auto spot = ext::make_shared<SimpleQuote>(100.0);
        const auto vol = Handle<BlackVolTermStructure>(
            ext::make_shared<BlackConstantVol>(REF_DATE, NullCalendar(), 0.25, DC));
        const auto process = ext::make_shared<BlackScholesMertonProcess>(
            Handle<Quote>(spot), flat(0.02), flat(0.05), vol);
        const Real nullReal = Null<Real>();

        // no cPoint at all -> uniform
        emit_mesher("bsm_uniform",
                    FdmBlackScholesMesher(11, process, 1.0, 100.0));
        // cPoint at the strike, inside the range -> concentrating
        emit_mesher("bsm_conc",
                    FdmBlackScholesMesher(11, process, 1.0, 100.0,
                                          nullReal, nullReal, 0.0001, 1.5,
                                          std::make_pair(100.0, 0.1)));
        // a much denser concentration, to separate the two arms further
        emit_mesher("bsm_conc_dense",
                    FdmBlackScholesMesher(15, process, 1.0, 100.0,
                                          nullReal, nullReal, 0.0001, 1.5,
                                          std::make_pair(90.0, 0.01)));
        // cPoint outside [xMin, xMax] -> guard rejects it, uniform again
        emit_mesher("bsm_conc_out_of_range",
                    FdmBlackScholesMesher(11, process, 1.0, 100.0,
                                          nullReal, nullReal, 0.0001, 1.5,
                                          std::make_pair(1.0e-6, 0.1)));
        // explicit x bounds + cPoint, so the guard is exercised against
        // overridden bounds rather than the vol-derived ones
        emit_mesher("bsm_conc_bounded",
                    FdmBlackScholesMesher(11, process, 1.0, 100.0,
                                          std::log(50.0), std::log(150.0),
                                          0.0001, 1.5,
                                          std::make_pair(120.0, 0.05)));
        // spotAdjustment shifts the forward the bounds are built from
        emit_mesher("bsm_spot_adj",
                    FdmBlackScholesMesher(11, process, 1.0, 100.0,
                                          nullReal, nullReal, 0.0001, 1.5,
                                          std::make_pair(nullReal, nullReal),
                                          DividendSchedule(),
                                          ext::shared_ptr<FdmQuantoHelper>(),
                                          -5.0));
    }

    std::cout << "\n}" << std::endl;
    return 0;
}
