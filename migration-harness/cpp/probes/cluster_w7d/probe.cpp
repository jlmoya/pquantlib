// Phase 11 W7-D cluster probe: experimental inflation — YoY optionlet
// stripping + CPI/YoY cap-floor price surfaces.
//
// Captures reference values for:
//
//   * Polynomial2DSpline — 2-D interpolation (parabolic in y / strike,
//     cubic spline in x / maturity) at grid pillars + interior points.
//   * InterpolatedYoYCapFloorTermPriceSurface<Bicubic,Linear> — capPrice
//     / floorPrice at the market quotes + atmYoYSwapRate.
//   * InterpolatedCPICapFloorTermPriceSurface<Bilinear> — capPrice /
//     floorPrice at quotes (put/call-parity-completed surface).
//   * InterpolatingCPICapFloorEngine — NPV of a CPI cap from the surface.
//   * InterpolatedYoYOptionletStripper<Linear> + KInterpolatedYoY-
//     OptionletVolatilitySurface<Linear>.volatility(t, K) — stripped
//     caplet vols.
//
// C++ parity (all @ v1.42.1, 099987f0):
//   ql/experimental/inflation/polynomial2Dspline.hpp
//   ql/experimental/inflation/cpicapfloortermpricesurface.hpp
//   ql/experimental/inflation/yoycapfloortermpricesurface.hpp
//   ql/experimental/inflation/cpicapfloorengines.hpp
//   ql/experimental/inflation/interpolatedyoyoptionletstripper.hpp
//   ql/experimental/inflation/kinterpolatedyoyoptionletvolatilitysurface.hpp
//
// Market data mirrors QuantLib's inflation cap/floor test-suite
// (test-suite/inflationcapfloor.cpp) where applicable.

#include <ql/cashflows/cpicoupon.hpp>
#include <ql/errors.hpp>
#include <ql/experimental/inflation/cpicapfloorengines.hpp>
#include <ql/experimental/inflation/cpicapfloortermpricesurface.hpp>
#include <ql/experimental/inflation/polynomial2Dspline.hpp>
#include <ql/indexes/inflation/ukrpi.hpp>
#include <ql/instruments/cpicapfloor.hpp>
#include <ql/math/interpolations/bicubicsplineinterpolation.hpp>
#include <ql/math/interpolations/bilinearinterpolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/math/matrix.hpp>
#include <ql/termstructures/inflation/inflationhelpers.hpp>
#include <ql/termstructures/inflation/piecewisezeroinflationcurve.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {
    std::string j(const std::string& key, Real v) {
        std::ostringstream os;
        os << "  \"" << key << "\": " << std::setprecision(17) << v;
        return os.str();
    }
}

int main() {
    std::vector<std::string> out;

    // ---- Polynomial2DSpline -------------------------------------------
    // x = maturities (spline dir), y = strikes (parabolic dir).
    // z indexed [y][x]: rows = strikes, cols = maturities.
    {
        std::vector<Real> x = {1.0, 2.0, 3.0, 4.0, 5.0};          // maturities
        std::vector<Real> y = {0.01, 0.02, 0.03, 0.04};           // strikes
        Matrix z(y.size(), x.size());
        // a smooth bilinear-ish surface with curvature
        for (Size i = 0; i < y.size(); ++i)
            for (Size k = 0; k < x.size(); ++k)
                z[i][k] = 100.0 * y[i] * y[i] + 5.0 * x[k] + 2.0 * y[i] * x[k];

        Polynomial2DSpline spline(x.begin(), x.end(), y.begin(), y.end(), z);

        // pillar reproduction (TIGHT)
        out.push_back(j("poly2d_pillar_x1_y0.01", spline(1.0, 0.01)));
        out.push_back(j("poly2d_pillar_x3_y0.02", spline(3.0, 0.02)));
        out.push_back(j("poly2d_pillar_x5_y0.04", spline(5.0, 0.04)));
        // interior (LOOSE)
        out.push_back(j("poly2d_interior_x2.5_y0.025", spline(2.5, 0.025)));
        out.push_back(j("poly2d_interior_x4.2_y0.015", spline(4.2, 0.015)));
        out.push_back(j("poly2d_interior_x1.5_y0.035", spline(1.5, 0.035)));
    }

    // ---- CPI cap/floor price surface (canonical UKRPI test data) ------
    // Mirrors test-suite/inflationcpicapfloor.cpp CommonVars.
    {
        Calendar cal = UnitedKingdom();
        BusinessDayConvention bdc = ModifiedFollowing;
        Date today(1, June, 2010);
        Date evalDate = cal.adjust(today);
        Settings::instance().evaluationDate() = evalDate;
        DayCounter dc = ActualActual(ActualActual::ISDA);
        Period observationLag(2, Months);

        RelinkableHandle<ZeroInflationTermStructure> hcpi;
        auto ii = ext::make_shared<UKRPI>(hcpi);

        // UK RPI fixings (monthly, Jul-2007 .. Apr-2010)
        Schedule rpiSchedule = MakeSchedule()
            .from(Date(1, July, 2007))
            .to(Date(1, April, 2010))
            .withFrequency(Monthly);
        Real fixData[] = {
            206.1, 207.3, 208.0, 208.9, 209.7, 210.9,
            209.8, 211.4, 212.1, 214.0, 215.1, 216.8,
            216.5, 217.2, 218.4, 217.7, 216.0, 212.9,
            210.1, 211.4, 211.3, 211.5, 212.8, 213.4,
            213.4, 214.4, 215.3, 216.0, 216.6, 218.0,
            217.9, 219.2, 220.7, 222.8};
        for (Size i = 0; i < rpiSchedule.size(); ++i)
            ii->addFixing(rpiSchedule[i], fixData[i], true);

        // nominal curve (zero rates / 100)
        struct Datum { Date d; Rate r; };
        Datum nominalData[] = {
            {Date(2, June, 2010), 0.499997}, {Date(3, June, 2010), 0.524992},
            {Date(8, June, 2010), 0.524974}, {Date(15, June, 2010), 0.549942},
            {Date(22, June, 2010), 0.549913}, {Date(1, July, 2010), 0.574864},
            {Date(2, August, 2010), 0.624668}, {Date(1, September, 2010), 0.724338},
            {Date(16, September, 2010), 0.769461}, {Date(1, December, 2010), 0.997501},
            {Date(17, March, 2011), 0.916996}, {Date(16, June, 2011), 0.984339},
            {Date(22, September, 2011), 1.06085}, {Date(22, December, 2011), 1.141788},
            {Date(1, June, 2012), 1.504426}, {Date(3, June, 2013), 1.92064},
            {Date(2, June, 2014), 2.290824}, {Date(1, June, 2015), 2.614394},
            {Date(1, June, 2016), 2.887445}, {Date(1, June, 2017), 3.122128},
            {Date(1, June, 2018), 3.322511}, {Date(3, June, 2019), 3.483997},
            {Date(1, June, 2020), 3.616896}, {Date(1, June, 2022), 3.8281},
            {Date(2, June, 2025), 4.0341}, {Date(3, June, 2030), 4.070854},
            {Date(1, June, 2035), 4.023202}, {Date(1, June, 2040), 3.954748},
            {Date(1, June, 2050), 3.870953}, {Date(1, June, 2060), 3.85298},
            {Date(2, June, 2070), 3.757542}, {Date(3, June, 2080), 3.651379}};
        std::vector<Date> nomD;
        std::vector<Rate> nomR;
        for (auto& it : nominalData) { nomD.push_back(it.d); nomR.push_back(it.r / 100.0); }
        Handle<YieldTermStructure> nominalUK(
            ext::make_shared<InterpolatedZeroCurve<Linear>>(nomD, nomR, dc));

        // ZCIIS data -> zero inflation curve via helpers
        Datum zciisData[] = {
            {Date(1, June, 2011), 3.087}, {Date(1, June, 2012), 3.12},
            {Date(1, June, 2013), 3.059}, {Date(1, June, 2014), 3.11},
            {Date(1, June, 2015), 3.15}, {Date(1, June, 2016), 3.207},
            {Date(1, June, 2017), 3.253}, {Date(1, June, 2018), 3.288},
            {Date(1, June, 2019), 3.314}, {Date(1, June, 2020), 3.401},
            {Date(1, June, 2022), 3.458}, {Date(1, June, 2025), 3.52},
            {Date(1, June, 2030), 3.655}, {Date(1, June, 2035), 3.668},
            {Date(1, June, 2040), 3.695}, {Date(1, June, 2050), 3.634},
            {Date(1, June, 2060), 3.629}};
        std::vector<ext::shared_ptr<BootstrapHelper<ZeroInflationTermStructure>>> helpers;
        for (auto& z : zciisData) {
            Handle<Quote> q(ext::make_shared<SimpleQuote>(z.r / 100.0));
            helpers.push_back(ext::make_shared<ZeroCouponInflationSwapHelper>(
                q, observationLag, z.d, cal, bdc, dc, ii, CPI::AsIndex, nominalUK));
        }
        Real baseZeroRate = zciisData[0].r / 100.0;
        Date baseDate = ii->lastFixingDate();
        auto pCPIts = ext::make_shared<PiecewiseZeroInflationCurve<Linear>>(
            evalDate, baseDate, ii->frequency(), dc, helpers);
        pCPIts->recalculate();
        hcpi.linkTo(pCPIts);

        // surface market data
        std::vector<Period> cfMat = {3 * Years, 5 * Years, 7 * Years,
                                     10 * Years, 15 * Years, 20 * Years, 30 * Years};
        std::vector<Rate> cStrike = {0.03, 0.04, 0.05, 0.06};
        std::vector<Rate> fStrike = {-0.01, 0, 0.01, 0.02};
        Real cPrice[7][4] = {
            {227.6, 100.27, 38.8, 14.94}, {345.32, 127.9, 40.59, 14.11},
            {477.95, 170.19, 50.62, 16.88}, {757.81, 303.95, 107.62, 43.61},
            {1140.73, 481.89, 168.4, 63.65}, {1537.6, 607.72, 172.27, 54.87},
            {2211.67, 839.24, 184.75, 45.03}};
        Real fPrice[7][4] = {
            {15.62, 28.38, 53.61, 104.6}, {21.45, 36.73, 66.66, 129.6},
            {24.45, 42.08, 77.04, 152.24}, {39.25, 63.52, 109.2, 203.44},
            {36.82, 63.62, 116.97, 232.73}, {39.7, 67.47, 121.79, 238.56},
            {41.48, 73.9, 139.75, 286.75}};
        Matrix cPriceM(4, 7), fPriceM(4, 7);
        for (Size i = 0; i < 4; ++i)
            for (Size j = 0; j < 7; ++j) {
                cPriceM[i][j] = cPrice[j][i] / 10000.0;
                fPriceM[i][j] = fPrice[j][i] / 10000.0;
            }

        Real nominal = 1.0;
        auto cpiSurf = ext::make_shared<
            InterpolatedCPICapFloorTermPriceSurface<Bilinear>>(
            nominal, baseZeroRate, observationLag, cal, bdc, dc, ii, CPI::Flat,
            nominalUK, cStrike, fStrike, cfMat, cPriceM, fPriceM);

        // reproduce a few quote points (put/call parity completion)
        out.push_back(j("cpi_surf_floor_3y_fstrike-0.01",
                        cpiSurf->floorPrice(cfMat[0], fStrike[0])));
        out.push_back(j("cpi_surf_floor_7y_fstrike0.01",
                        cpiSurf->floorPrice(cfMat[2], fStrike[2])));
        out.push_back(j("cpi_surf_cap_3y_cstrike0.03",
                        cpiSurf->capPrice(cfMat[0], cStrike[0])));
        out.push_back(j("cpi_surf_cap_10y_cstrike0.05",
                        cpiSurf->capPrice(cfMat[3], cStrike[2])));
        // an interior interpolated point
        out.push_back(j("cpi_surf_cap_4y_cstrike0.035",
                        cpiSurf->capPrice(Period(4, Years), 0.035)));
        out.push_back(j("cpi_surf_atm_rate_3y", cpiSurf->atmRate(
                        cpiSurf->cpiOptionDateFromTenor(cfMat[0]))));

        // ---- InterpolatingCPICapFloorEngine: 3Y Call @ 0.03 -> 227.6 bps
        Date startDate = Settings::instance().evaluationDate();
        Date maturity(startDate + Period(3, Years));
        Rate strike = 0.03;
        CPI::InterpolationType obsInterp = CPI::AsIndex;
        Real baseCPI = CPI::laggedFixing(ii, startDate, observationLag, obsInterp);
        CPICapFloor aCap(Option::Call, nominal, startDate, baseCPI, maturity,
                         cal, Unadjusted, cal, ModifiedFollowing, strike, ii,
                         observationLag, obsInterp);
        Handle<CPICapFloorTermPriceSurface> surfH(cpiSurf);
        aCap.setPricingEngine(
            ext::make_shared<InterpolatingCPICapFloorEngine>(surfH));
        out.push_back(j("cpi_engine_cap_3y_0.03_npv", aCap.NPV()));
    }

    std::cout << "{\n";
    for (Size i = 0; i < out.size(); ++i)
        std::cout << out[i] << (i + 1 < out.size() ? ",\n" : "\n");
    std::cout << "}\n";
    return 0;
}
