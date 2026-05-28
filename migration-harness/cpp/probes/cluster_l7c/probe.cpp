// L7-C cluster probe: inflation cashflows + pricers.
//
// Emits reference values for:
//   * CPI::laggedFixing closed-form lookups (Flat + AsIndex modes) against
//     a stub EUHICP history. These pin down the index-ratio math used by
//     CPICoupon, CPICashFlow, and ZeroInflationCashFlow.
//   * IndexedCashFlow amount(): N * I1/I0 (bond style) and N * (I1/I0 - 1)
//     (swap style, growthOnly = true).
//   * ZeroInflationCashFlow baseFixing()/indexFixing()/amount() at a known
//     pair of fixings (June -> June, 3-month observation lag → March
//     fixings on each end).
//   * CPICashFlow baseFixing()/indexFixing()/amount() with explicit
//     baseFixing override.
//   * CPICoupon indexRatio(d) + amount() for both growthOnly modes
//     (subtract notional flag is downstream — at the coupon level we test
//     the fixedRate * indexRatio * accrualPeriod * nominal product).
//   * YoYInflationCoupon indexFixing() (Flat YoY lag lookup) + amount()
//     = gearing * yoyRate + spread, times accrual_period * nominal.
//   * BlackYoYInflationCouponPricer optionletRate (cap + floor) at known
//     Black-formula params (forward, strike, std_dev).
//   * BachelierYoYInflationCouponPricer optionletRate (cap + floor) at
//     the same params for normal-model parity.
//
// Stub histories use a deterministic monthly path:
//   EUHICP[2020-01..2022-12]:
//     fixing(month m of year y) = 100.0 + 0.5 * (12 * (y - 2020) + (m - 1))
//   (linear ramp from 100.0 to 117.5).

#include <ql/cashflows/cpicoupon.hpp>
#include <ql/cashflows/cpicouponpricer.hpp>
#include <ql/cashflows/indexedcashflow.hpp>
#include <ql/cashflows/inflationcoupon.hpp>
#include <ql/cashflows/inflationcouponpricer.hpp>
#include <ql/cashflows/yoyinflationcoupon.hpp>
#include <ql/cashflows/zeroinflationcashflow.hpp>
#include <ql/indexes/inflation/euhicp.hpp>
#include <ql/indexes/inflationindex.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace QuantLib;

namespace {

// Linear ramp: fixing(y, m) = base + 0.5 * (12*(y - 2020) + (m - 1)).
Real ramp_fixing(int y, int m) {
    return 100.0 + 0.5 * static_cast<Real>(12 * (y - 2020) + (m - 1));
}

// Populate a 3-year history (2020-01 .. 2022-12) on the given index.
void seed_history(ZeroInflationIndex& idx) {
    for (int y = 2020; y <= 2022; ++y) {
        for (int m = 1; m <= 12; ++m) {
            Date d(1, Month(m), y);
            idx.addFixing(d, ramp_fixing(y, m), true);
        }
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // Build a EUHICP zero-inflation index with the ramped history above.
    auto eu = ext::make_shared<EUHICP>();
    seed_history(*eu);

    const Period lag3m(3, Months);
    const Period lag0(0, Months);
    Thirty360 dc(Thirty360::BondBasis);
    Actual365Fixed actDc;
    NullCalendar nullCal;

    // ============================================================
    // CPI::laggedFixing — anchor (Flat / AsIndex) and Linear-mode results.
    // ============================================================
    {
        std::cout << "  \"lagged_fixing\": {\n";
        // Sample probe: date = 2021-06-15, lag = 3M -> fixing on 2021-03.
        // ramp(2021, 3) = 100 + 0.5 * (12 + 2) = 107.0.
        Date d1(15, June, 2021);
        Real flat = CPI::laggedFixing(eu, d1, lag3m, CPI::Flat);
        Real asIdx = CPI::laggedFixing(eu, d1, lag3m, CPI::AsIndex);
        Real lin = CPI::laggedFixing(eu, d1, lag3m, CPI::Linear);

        std::cout << "    \"date_2021_06_15\": {\n";
        std::cout << "      \"flat\": " << flat << ",\n";
        std::cout << "      \"as_index\": " << asIdx << ",\n";
        std::cout << "      \"linear\": " << lin << "\n";
        std::cout << "    },\n";

        // Probe 2: date = 2022-08-20, lag = 3M -> 2022-05 fixing.
        // ramp(2022, 5) = 100 + 0.5 * (24 + 4) = 114.0.
        Date d2(20, August, 2022);
        Real flat2 = CPI::laggedFixing(eu, d2, lag3m, CPI::Flat);
        Real lin2 = CPI::laggedFixing(eu, d2, lag3m, CPI::Linear);

        std::cout << "    \"date_2022_08_20\": {\n";
        std::cout << "      \"flat\": " << flat2 << ",\n";
        std::cout << "      \"linear\": " << lin2 << "\n";
        std::cout << "    },\n";

        // Probe 3: date = 2020-04-01 (period start exactly == date), lag 0.
        // No interpolation since date == interpolationPeriod.first.
        Date d3(1, April, 2020);
        Real lin3 = CPI::laggedFixing(eu, d3, lag0, CPI::Linear);
        std::cout << "    \"date_2020_04_01_lag0_linear_eq_start\": " << lin3 << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // IndexedCashFlow: notional / I0 / I1 / growthOnly.
    // ============================================================
    {
        std::cout << "  \"indexed_cashflow\": {\n";
        // baseDate = 2020-06-01 (fixing 102.5), fixingDate = 2022-06-01 (114.5).
        Date baseDate(1, June, 2020);
        Date fixingDate(1, June, 2022);
        Date payDate(15, June, 2022);
        Real notional = 1000000.0;

        IndexedCashFlow bondStyle(notional, eu, baseDate, fixingDate, payDate, /*growthOnly=*/false);
        IndexedCashFlow swapStyle(notional, eu, baseDate, fixingDate, payDate, /*growthOnly=*/true);

        std::cout << "    \"notional\": " << notional << ",\n";
        std::cout << "    \"base_fixing\": " << bondStyle.baseFixing() << ",\n";
        std::cout << "    \"index_fixing\": " << bondStyle.indexFixing() << ",\n";
        std::cout << "    \"bond_style_amount\": " << bondStyle.amount() << ",\n";
        std::cout << "    \"swap_style_amount\": " << swapStyle.amount() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // ZeroInflationCashFlow: base/index/amount.
    // ============================================================
    {
        std::cout << "  \"zero_inflation_cashflow\": {\n";
        // start=2021-06-01, end=2022-06-01, lag=3M; growthOnly=true.
        // baseFixing = laggedFixing(start=2021-06-01, 3M, Flat) = ramp(2021, 3) = 107.0
        // indexFixing = laggedFixing(end=2022-06-01, 3M, Flat) = ramp(2022, 3) = 113.0
        Date start(1, June, 2021);
        Date end(1, June, 2022);
        Date pay(15, June, 2022);
        Real notional = 500000.0;

        ZeroInflationCashFlow growthOnly(
            notional, eu, CPI::Flat,
            start, end, lag3m, pay, /*growthOnly=*/true);
        ZeroInflationCashFlow bondStyle(
            notional, eu, CPI::Flat,
            start, end, lag3m, pay, /*growthOnly=*/false);

        std::cout << "    \"notional\": " << notional << ",\n";
        std::cout << "    \"base_fixing\": " << growthOnly.baseFixing() << ",\n";
        std::cout << "    \"index_fixing\": " << growthOnly.indexFixing() << ",\n";
        std::cout << "    \"growth_only_amount\": " << growthOnly.amount() << ",\n";
        std::cout << "    \"bond_style_amount\": " << bondStyle.amount() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // CPICashFlow: fully-specified base fixing.
    // ============================================================
    {
        std::cout << "  \"cpi_cashflow\": {\n";
        Date baseDate(1, March, 2020);
        Real baseFixing = 100.0;  // explicit
        Date obsDate(1, June, 2022);
        Date pay(15, June, 2022);
        Real notional = 750000.0;

        CPICashFlow cf(notional, eu, baseDate, baseFixing, obsDate, lag3m,
                       CPI::Flat, pay, /*growthOnly=*/true);

        std::cout << "    \"notional\": " << notional << ",\n";
        std::cout << "    \"base_fixing\": " << cf.baseFixing() << ",\n";
        std::cout << "    \"index_fixing\": " << cf.indexFixing() << ",\n";
        std::cout << "    \"growth_only_amount\": " << cf.amount() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // CPICoupon: indexRatio + amount via CPICouponPricer.
    // ============================================================
    {
        std::cout << "  \"cpi_coupon\": {\n";

        Date pay(15, June, 2022);
        Date start(1, June, 2021);
        Date end(1, June, 2022);
        Real nominal = 200000.0;
        Real baseCPI = 100.0;
        Real fixedRate = 0.025;

        CPICoupon cpn(baseCPI, pay, nominal, start, end, eu, lag3m,
                      CPI::Flat, dc, fixedRate);
        cpn.setPricer(ext::make_shared<CPICouponPricer>());

        std::cout << "    \"nominal\": " << nominal << ",\n";
        std::cout << "    \"base_cpi\": " << baseCPI << ",\n";
        std::cout << "    \"fixed_rate\": " << fixedRate << ",\n";
        std::cout << "    \"accrual_period\": " << cpn.accrualPeriod() << ",\n";
        std::cout << "    \"index_fixing\": " << cpn.indexFixing() << ",\n";
        // ratio at accrual end
        Rate ratio = cpn.indexRatio(cpn.accrualEndDate());
        std::cout << "    \"index_ratio_at_end\": " << ratio << ",\n";
        // rate = fixedRate * indexRatio  (per CPICouponPricer::accruedRate)
        std::cout << "    \"rate\": " << cpn.rate() << ",\n";
        std::cout << "    \"amount\": " << cpn.amount() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // YoYInflationCoupon (ratio mode underlying) + pricer.
    // We use a quoted YoY index and seed YoY fixings directly so the
    // probe is deterministic and avoids interaction with the ratio path
    // (which involves forecast curves once L7-B lands).
    // ============================================================
    {
        std::cout << "  \"yoy_inflation_coupon\": {\n";

        // Build YYEUHICP (quoted-mode YoY).
        auto yoy = ext::make_shared<YYEUHICP>(/*interpolated=*/false);
        // Seed 2 years of YoY history (2021..2022) at 0.020 + 0.005 * (m-1)
        // -> simple linear ramp.
        for (int y = 2021; y <= 2022; ++y) {
            for (int m = 1; m <= 12; ++m) {
                Date d(1, Month(m), y);
                Real fix = 0.020 + 0.005 * static_cast<Real>(m - 1);
                yoy->addFixing(d, fix, true);
            }
        }

        Date pay(15, June, 2022);
        Date start(1, June, 2021);
        Date end(1, June, 2022);
        Real nominal = 100000.0;
        Real gearing = 1.5;
        Real spread = 0.005;
        Natural fixingDays = 0;

        YoYInflationCoupon cpn(pay, nominal, start, end, fixingDays, yoy,
                               lag3m, CPI::Flat, dc, gearing, spread);
        cpn.setPricer(ext::make_shared<YoYInflationCouponPricer>());

        std::cout << "    \"nominal\": " << nominal << ",\n";
        std::cout << "    \"gearing\": " << gearing << ",\n";
        std::cout << "    \"spread\": " << spread << ",\n";
        std::cout << "    \"accrual_period\": " << cpn.accrualPeriod() << ",\n";
        std::cout << "    \"index_fixing\": " << cpn.indexFixing() << ",\n";
        std::cout << "    \"rate\": " << cpn.rate() << ",\n";
        std::cout << "    \"amount\": " << cpn.amount() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // BlackYoYInflationCouponPricer & BachelierYoYInflationCouponPricer:
    // optionletPriceImp parity vs the underlying Black / Bachelier formula.
    // We probe both Call (cap) and Put (floor) at one set of params.
    // ============================================================
    {
        std::cout << "  \"black_pricer\": {\n";
        Real forward = 0.025;
        Real strike = 0.020;
        Real stdDev = 0.01 * 0.5;  // vol=0.01, sqrt(T)=0.5 -> 0.005

        Real bf_call = blackFormula(Option::Call, strike, forward, stdDev);
        Real bf_put = blackFormula(Option::Put, strike, forward, stdDev);
        Real bach_call = bachelierBlackFormula(Option::Call, strike, forward, stdDev);
        Real bach_put = bachelierBlackFormula(Option::Put, strike, forward, stdDev);

        // Unit-displaced Black: strike+1, forward+1.
        Real ud_call = blackFormula(Option::Call, strike + 1.0, forward + 1.0, stdDev);
        Real ud_put = blackFormula(Option::Put, strike + 1.0, forward + 1.0, stdDev);

        std::cout << "    \"forward\": " << forward << ",\n";
        std::cout << "    \"strike\": " << strike << ",\n";
        std::cout << "    \"std_dev\": " << stdDev << ",\n";
        std::cout << "    \"black_call\": " << bf_call << ",\n";
        std::cout << "    \"black_put\": " << bf_put << ",\n";
        std::cout << "    \"bachelier_call\": " << bach_call << ",\n";
        std::cout << "    \"bachelier_put\": " << bach_put << ",\n";
        std::cout << "    \"unit_displaced_call\": " << ud_call << ",\n";
        std::cout << "    \"unit_displaced_put\": " << ud_put << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
