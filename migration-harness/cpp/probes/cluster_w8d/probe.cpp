// Phase 11 W8-D cluster probe: arithmetic-average OIS (deprecated, carved
// out) + CLV models + generalized short-rate + basis/xccy rate helpers.
//
// Captures reference values for:
//
//   * GeneralizedOrnsteinUhlenbeckProcess — expectation / stdDeviation /
//     variance / drift / diffusion, at both constant coefficients (==
//     plain OU) and time-varying (piecewise) coefficients.
//   * GeneralizedHullWhite (analytic-fitting constructor) — A(t,T),
//     B(t,T), discount(t), discountBondOption. At constant reversion+vol
//     these match the classical Hull-White model.
//   * NormalCLVModel — cdf / invCDF / collocationPointsX / collocationPointsY
//     / g(t,x) mapping function.
//   * SquareRootCLVModel — cdf / invCDF / collocationPointsX /
//     collocationPointsY / g(t,x).
//   * GBSMRNDCalculator — pdf / cdf / invcdf (foundation for the CLV
//     models).
//   * IborIborBasisSwapRateHelper / OvernightIborBasisSwapRateHelper —
//     impliedQuote (basis roundtrip).
//   * ConstNotional / MtM CrossCurrencyBasisSwapRateHelper +
//     ConstNotionalCrossCurrencySwapRateHelper — impliedQuote.
//
// C++ parity (all @ v1.42.1, 099987f0):
//   ql/experimental/shortrate/generalizedornsteinuhlenbeckprocess.{hpp,cpp}
//   ql/experimental/shortrate/generalizedhullwhite.{hpp,cpp}
//   ql/experimental/models/normalclvmodel.{hpp,cpp}
//   ql/experimental/models/squarerootclvmodel.{hpp,cpp}
//   ql/methods/finitedifferences/utilities/gbsmrndcalculator.{hpp,cpp}
//   ql/experimental/termstructures/basisswapratehelpers.{hpp,cpp}
//   ql/experimental/termstructures/crosscurrencyratehelpers.{hpp,cpp}

#include <ql/experimental/shortrate/generalizedornsteinuhlenbeckprocess.hpp>
#include <ql/experimental/shortrate/generalizedhullwhite.hpp>
#include <ql/experimental/models/normalclvmodel.hpp>
#include <ql/experimental/models/squarerootclvmodel.hpp>
#include <ql/experimental/termstructures/basisswapratehelpers.hpp>
#include <ql/experimental/termstructures/crosscurrencyratehelpers.hpp>
#include <ql/methods/finitedifferences/utilities/gbsmrndcalculator.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include <ql/processes/squarerootprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/piecewiseyieldcurve.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/indexes/ibor/usdlibor.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

static std::ostream& kv(std::ostream& os, const std::string& key, Real v, bool comma = true) {
    os << "  \"" << key << "\": " << std::setprecision(17) << v << (comma ? ",\n" : "\n");
    return os;
}

int main() {
    std::ofstream out("references/cluster/w8d.json");
    out << "{\n";

    // ============================================================
    // GeneralizedOrnsteinUhlenbeckProcess
    // ============================================================
    {
        // Constant coefficients: speed=0.3, vol=0.15 -> must equal plain OU.
        const Real a = 0.3, sig = 0.15, x0 = 0.05, level = 0.04;
        GeneralizedOrnsteinUhlenbeckProcess g(
            [a](Time) { return a; }, [sig](Time) { return sig; }, x0, level);
        OrnsteinUhlenbeckProcess ou(a, sig, x0, level);

        kv(out, "gou_const_x0", g.x0());
        kv(out, "gou_const_speed", g.speed(0.7));
        kv(out, "gou_const_vol", g.volatility(0.7));
        kv(out, "gou_const_level", g.level());
        kv(out, "gou_const_drift", g.drift(0.7, 0.06));
        kv(out, "gou_const_diffusion", g.diffusion(0.7, 0.06));
        kv(out, "gou_const_expectation", g.expectation(0.0, x0, 1.5));
        kv(out, "gou_const_variance", g.variance(0.0, x0, 1.5));
        kv(out, "gou_const_stddev", g.stdDeviation(0.0, x0, 1.5));
        kv(out, "ou_expectation", ou.expectation(0.0, x0, 1.5));
        kv(out, "ou_variance", ou.variance(0.0, x0, 1.5));

        // Time-varying coefficients: linear-in-t speed and vol.
        GeneralizedOrnsteinUhlenbeckProcess gv(
            [](Time t) { return 0.2 + 0.1 * t; },
            [](Time t) { return 0.10 + 0.05 * t; }, x0, level);
        kv(out, "gou_tv_speed_t2", gv.speed(2.0));
        kv(out, "gou_tv_vol_t2", gv.volatility(2.0));
        kv(out, "gou_tv_drift", gv.drift(2.0, 0.06));
        kv(out, "gou_tv_diffusion", gv.diffusion(2.0, 0.06));
        kv(out, "gou_tv_expectation", gv.expectation(2.0, x0, 0.5));
        kv(out, "gou_tv_variance", gv.variance(2.0, x0, 0.5));
        kv(out, "gou_tv_stddev", gv.stdDeviation(2.0, x0, 0.5));
        // small-speed algebraic limit
        GeneralizedOrnsteinUhlenbeckProcess gs(
            [](Time) { return 0.0; }, [](Time) { return 0.2; }, x0, level);
        kv(out, "gou_smallspeed_variance", gs.variance(0.0, x0, 2.0));
    }

    // ============================================================
    // GeneralizedHullWhite (analytic-fitting) vs classical Hull-White
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2024);
        DayCounter dc = Actual365Fixed();
        Handle<YieldTermStructure> yts(
            ext::make_shared<FlatForward>(0, TARGET(), 0.04, dc));

        const Real a = 0.1, sigma = 0.012;
        GeneralizedHullWhite ghw(yts, a, sigma);
        HullWhite hw(yts, a, sigma);

        // NB: GHW::dynamics() deliberately throws ("use HWdynamics()"), so
        // OneFactorAffineModel::discount(Time) (which calls dynamics()) is
        // NOT usable on GHW. The A/B path feeds discountBond / discountBondOption
        // which ARE public and analytic — those are the parity anchors.

        // discountBond(t, T, rate) — public on AffineModel.
        kv(out, "ghw_discountbond_2_5_r", ghw.discountBond(2.0, 5.0, 0.045));
        kv(out, "hw_discountbond_2_5_r", hw.discountBond(2.0, 5.0, 0.045));

        // discountBondOption — only valid under HW; matches classical HW.
        kv(out, "ghw_dbo_call", ghw.discountBondOption(Option::Call, 0.85, 2.0, 5.0));
        kv(out, "hw_dbo_call", hw.discountBondOption(Option::Call, 0.85, 2.0, 5.0));
        kv(out, "ghw_dbo_put", ghw.discountBondOption(Option::Put, 0.85, 2.0, 5.0));
        kv(out, "hw_dbo_put", hw.discountBondOption(Option::Put, 0.85, 2.0, 5.0));

        // Piecewise (time-varying) GHW: 2 reversion + 2 vol pillars.
        std::vector<Date> speedStruct = {Date(15, January, 2024), Date(15, January, 2026)};
        std::vector<Date> volStruct = {Date(15, January, 2024), Date(15, January, 2026)};
        std::vector<Real> speeds = {0.1, 0.2};
        std::vector<Real> vols = {0.010, 0.015};
        GeneralizedHullWhite ghwpw(yts, speedStruct, volStruct, speeds, vols);
        kv(out, "ghwpw_discountbond_2_5_r", ghwpw.discountBond(2.0, 5.0, 0.045));
        kv(out, "ghwpw_dbo_call", ghwpw.discountBondOption(Option::Call, 0.85, 2.0, 5.0));
    }

    // ============================================================
    // GBSMRNDCalculator + NormalCLVModel
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2024);
        DayCounter dc = Actual365Fixed();
        Date ref(15, January, 2024);
        Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
        Handle<YieldTermStructure> rf(
            ext::make_shared<FlatForward>(ref, 0.05, dc));
        Handle<YieldTermStructure> div(
            ext::make_shared<FlatForward>(ref, 0.02, dc));
        Handle<BlackVolTermStructure> vol(
            ext::make_shared<BlackConstantVol>(ref, TARGET(), 0.25, dc));
        auto bsProcess = ext::make_shared<GeneralizedBlackScholesProcess>(
            spot, div, rf, vol);

        GBSMRNDCalculator rnd(bsProcess);
        kv(out, "rnd_cdf_t1_k100", rnd.cdf(100.0, 1.0));
        kv(out, "rnd_cdf_t1_k120", rnd.cdf(120.0, 1.0));
        kv(out, "rnd_pdf_t1_k100", rnd.pdf(100.0, 1.0));
        kv(out, "rnd_invcdf_t1_q05", rnd.invcdf(0.5, 1.0));
        kv(out, "rnd_invcdf_t1_q09", rnd.invcdf(0.9, 1.0));

        auto ouProcess = ext::make_shared<OrnsteinUhlenbeckProcess>(
            0.5, 0.2, 0.1, 0.0);
        std::vector<Date> maturities = {
            ref + Period(6, Months), ref + Period(1, Years),
            ref + Period(2, Years)};
        const Size lagrangeOrder = 5;
        NormalCLVModel clv(bsProcess, ouProcess, maturities, lagrangeOrder);

        kv(out, "nclv_cdf_1y_k100", clv.cdf(ref + Period(1, Years), 100.0));
        kv(out, "nclv_invcdf_1y_q05", clv.invCDF(ref + Period(1, Years), 0.5));

        Array cx = clv.collocationPointsX(ref + Period(1, Years));
        Array cy = clv.collocationPointsY(ref + Period(1, Years));
        for (Size i = 0; i < cx.size(); ++i)
            kv(out, "nclv_cx_1y_" + std::to_string(i), cx[i]);
        for (Size i = 0; i < cy.size(); ++i)
            kv(out, "nclv_cy_1y_" + std::to_string(i), cy[i]);

        auto g = clv.g();
        const Time t1 = bsProcess->time(ref + Period(1, Years));
        kv(out, "nclv_g_1y_x0", g(t1, 0.1));
        kv(out, "nclv_g_1y_xp", g(t1, 0.3));
        kv(out, "nclv_g_1y_xn", g(t1, -0.1));
        // interpolated maturity between 6M and 1Y
        const Time tmid = bsProcess->time(ref + Period(9, Months));
        kv(out, "nclv_g_9m_x0", g(tmid, 0.05));
    }

    // ============================================================
    // SquareRootCLVModel
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2024);
        DayCounter dc = Actual365Fixed();
        Date ref(15, January, 2024);
        Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
        Handle<YieldTermStructure> rf(
            ext::make_shared<FlatForward>(ref, 0.05, dc));
        Handle<YieldTermStructure> div(
            ext::make_shared<FlatForward>(ref, 0.02, dc));
        Handle<BlackVolTermStructure> vol(
            ext::make_shared<BlackConstantVol>(ref, TARGET(), 0.25, dc));
        auto bsProcess = ext::make_shared<GeneralizedBlackScholesProcess>(
            spot, div, rf, vol);

        // Square-root (CIR) kernel: b (mean), a (speed), sigma, x0.
        const Real kappa = 1.0, theta = 0.09, sigma = 0.2, x0 = 0.09;
        auto sqrtProcess = ext::make_shared<SquareRootProcess>(
            theta, kappa, sigma, x0);

        std::vector<Date> maturities = {
            ref + Period(1, Years), ref + Period(2, Years)};
        const Size lagrangeOrder = 5;
        SquareRootCLVModel clv(bsProcess, sqrtProcess, maturities, lagrangeOrder);

        kv(out, "sclv_cdf_1y_k100", clv.cdf(ref + Period(1, Years), 100.0));
        kv(out, "sclv_invcdf_1y_q05", clv.invCDF(ref + Period(1, Years), 0.5));

        Array cx = clv.collocationPointsX(ref + Period(1, Years));
        Array cy = clv.collocationPointsY(ref + Period(1, Years));
        for (Size i = 0; i < cx.size(); ++i)
            kv(out, "sclv_cx_1y_" + std::to_string(i), cx[i]);
        for (Size i = 0; i < cy.size(); ++i)
            kv(out, "sclv_cy_1y_" + std::to_string(i), cy[i]);

        auto g = clv.g();
        const Time t1 = bsProcess->time(ref + Period(1, Years));
        kv(out, "sclv_g_1y_x", g(t1, cx[2]));
    }

    // ============================================================
    // IborIborBasisSwapRateHelper roundtrip
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2024);
        DayCounter dc = Actual365Fixed();
        // Forecast curves for the two indices + an exogenous discount curve.
        Handle<YieldTermStructure> discount(
            ext::make_shared<FlatForward>(0, TARGET(), 0.030, dc));
        Handle<YieldTermStructure> baseFwd(
            ext::make_shared<FlatForward>(0, TARGET(), 0.032, dc));
        Handle<YieldTermStructure> otherFwd(
            ext::make_shared<FlatForward>(0, TARGET(), 0.035, dc));

        auto baseIdx = ext::make_shared<Euribor>(Period(3, Months), baseFwd);
        auto otherIdx = ext::make_shared<Euribor>(Period(6, Months), otherFwd);

        Handle<Quote> basis(ext::make_shared<SimpleQuote>(0.0010));
        IborIborBasisSwapRateHelper h(
            basis, Period(5, Years), 2, TARGET(), ModifiedFollowing, false,
            baseIdx, otherIdx, discount, /*bootstrapBaseCurve=*/false);
        // bootstrapBaseCurve=false -> otherIndex's forecast curve is being
        // solved for; feed it the known otherFwd curve and the implied quote
        // should reproduce the input basis. setTermStructure is access-narrowed
        // to private on the concrete helper, so call via the RateHelper base.
        static_cast<RateHelper&>(h).setTermStructure(otherFwd.currentLink().get());
        kv(out, "iibasis_implied_quote", h.impliedQuote());
    }

    // ============================================================
    // OvernightIborBasisSwapRateHelper roundtrip
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2024);
        DayCounter dc = Actual365Fixed();
        Handle<YieldTermStructure> discount(
            ext::make_shared<FlatForward>(0, UnitedStates(UnitedStates::GovernmentBond), 0.030, dc));
        Handle<YieldTermStructure> onFwd(
            ext::make_shared<FlatForward>(0, UnitedStates(UnitedStates::GovernmentBond), 0.031, dc));
        Handle<YieldTermStructure> iborFwd(
            ext::make_shared<FlatForward>(0, UnitedStates(UnitedStates::GovernmentBond), 0.034, dc));

        auto onIdx = ext::make_shared<Sofr>(onFwd);
        auto iborIdx = ext::make_shared<USDLibor>(Period(3, Months), iborFwd);

        Handle<Quote> basis(ext::make_shared<SimpleQuote>(0.0015));
        OvernightIborBasisSwapRateHelper h(
            basis, Period(5, Years), 2,
            UnitedStates(UnitedStates::GovernmentBond), ModifiedFollowing, false,
            onIdx, iborIdx, discount);
        // bootstraps otherIndex (ibor) forecast curve; feed it iborFwd.
        static_cast<RateHelper&>(h).setTermStructure(iborFwd.currentLink().get());
        kv(out, "onibasis_implied_quote", h.impliedQuote());
    }

    // ============================================================
    // CrossCurrency basis swap rate helpers
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2024);
        DayCounter dc = Actual365Fixed();
        // EUR (base) vs USD (quote). Collateral in USD (quote ccy).
        Handle<YieldTermStructure> eurFwd(
            ext::make_shared<FlatForward>(0, TARGET(), 0.025, dc));
        Handle<YieldTermStructure> usdFwd(
            ext::make_shared<FlatForward>(0, TARGET(), 0.040, dc));
        Handle<YieldTermStructure> collateral(
            ext::make_shared<FlatForward>(0, TARGET(), 0.038, dc));

        auto eurIdx = ext::make_shared<Euribor>(Period(3, Months), eurFwd);
        auto usdIdx = ext::make_shared<USDLibor>(Period(3, Months), usdFwd);

        Handle<Quote> basis(ext::make_shared<SimpleQuote>(-0.0020));

        ConstNotionalCrossCurrencyBasisSwapRateHelper cn(
            basis, Period(5, Years), 2, TARGET(), ModifiedFollowing, false,
            eurIdx, usdIdx, collateral,
            /*isFxBaseCurrencyCollateralCurrency=*/false,
            /*isBasisOnFxBaseCurrencyLeg=*/true,
            Quarterly, 0);
        // isFxBaseCurrencyCollateralCurrency=false -> collateral discounts the
        // quote (USD) leg; the bootstrapped curve discounts the base (EUR) leg.
        cn.setTermStructure(eurFwd.currentLink().get());
        kv(out, "xccy_const_implied_quote", cn.impliedQuote());

        MtMCrossCurrencyBasisSwapRateHelper mtm(
            basis, Period(5, Years), 2, TARGET(), ModifiedFollowing, false,
            eurIdx, usdIdx, collateral,
            /*isFxBaseCurrencyCollateralCurrency=*/false,
            /*isBasisOnFxBaseCurrencyLeg=*/true,
            /*isFxBaseCurrencyLegResettable=*/true,
            Quarterly, 0);
        mtm.setTermStructure(eurFwd.currentLink().get());
        kv(out, "xccy_mtm_implied_quote", mtm.impliedQuote());

        ConstNotionalCrossCurrencySwapRateHelper fixfloat(
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.02)),
            Period(5, Years), 2, TARGET(), ModifiedFollowing, false,
            Annual, Thirty360(Thirty360::BondBasis),
            usdIdx, collateral,
            /*collateralOnFixedLeg=*/true, 0);
        // collateralOnFixedLeg=true -> collateral discounts the fixed leg;
        // the bootstrapped curve discounts the USD floating leg.
        fixfloat.setTermStructure(usdFwd.currentLink().get());
        kv(out, "xccy_fixfloat_implied_quote", fixfloat.impliedQuote(), false);
    }

    out << "}\n";
    out.close();
    std::cout << "Wrote references/cluster/w8d.json\n";
    return 0;
}
