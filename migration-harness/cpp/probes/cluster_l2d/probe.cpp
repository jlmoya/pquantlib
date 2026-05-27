// L2-D mega-probe: InterestRate algebra, fixed/ibor/overnight coupons,
// CashFlows npv/bps/irr/duration/convexity for a flat-curve bond.

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/overnightindexedcoupon.hpp>
#include <ql/cashflows/overnightindexedcouponpricer.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/indexes/ibor/eonia.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/interestrate.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    Actual360 act360;
    Actual365Fixed act365;
    NullCalendar nullCal;

    // ============================================================
    // InterestRate.compoundFactor and equivalentRate
    // ============================================================
    {
        std::cout << "  \"interest_rate\": {\n";

        // t=1.0, r=0.05 across compoundings
        double r = 0.05;
        double t = 1.0;

        InterestRate ir_simple(r, act365, Simple, Annual);
        InterestRate ir_annual(r, act365, Compounded, Annual);
        InterestRate ir_semi(r, act365, Compounded, Semiannual);
        InterestRate ir_cont(r, act365, Continuous, Annual);

        std::cout << "    \"r\": " << r << ",\n";
        std::cout << "    \"t\": " << t << ",\n";
        std::cout << "    \"cf_simple\": "    << ir_simple.compoundFactor(t)  << ",\n";
        std::cout << "    \"cf_annual\": "    << ir_annual.compoundFactor(t)  << ",\n";
        std::cout << "    \"cf_semi\": "      << ir_semi.compoundFactor(t)    << ",\n";
        std::cout << "    \"cf_continuous\": "<< ir_cont.compoundFactor(t)    << ",\n";

        // discount factors == 1 / compound factors
        std::cout << "    \"df_simple\": "    << ir_simple.discountFactor(t)  << ",\n";
        std::cout << "    \"df_annual\": "    << ir_annual.discountFactor(t)  << ",\n";
        std::cout << "    \"df_semi\": "      << ir_semi.discountFactor(t)    << ",\n";
        std::cout << "    \"df_continuous\": "<< ir_cont.discountFactor(t)    << ",\n";

        // equivalent rate roundtrip: annual @ 5% -> semi -> back to annual
        InterestRate ir_eq_semi = ir_annual.equivalentRate(Compounded, Semiannual, t);
        InterestRate ir_eq_back = ir_eq_semi.equivalentRate(Compounded, Annual, t);

        std::cout << "    \"eq_annual_to_semi\": " << ir_eq_semi.rate() << ",\n";
        std::cout << "    \"eq_back_to_annual\": " << ir_eq_back.rate() << ",\n";

        // impliedRate: from a compound factor of 1.05, recover 5% annual / 1y
        InterestRate ir_implied = InterestRate::impliedRate(1.05, act365, Compounded, Annual, 1.0);
        std::cout << "    \"implied_5pct\": " << ir_implied.rate() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // FixedRateCoupon: 100,000 @ 5% Act/360, Jan1->Jul1 2026
    // ============================================================
    {
        Date d1(1, January, 2026);
        Date d2(1, July, 2026);
        // Amount = nominal * (compoundFactor - 1)
        // compoundFactor with Simple/Annual: 1 + r*tau
        InterestRate ir(0.05, act360, Simple, Annual);
        FixedRateCoupon frc(d2, 100000.0, ir, d1, d2);

        std::cout << "  \"fixed_rate_coupon\": {\n";
        std::cout << "    \"payment_date_serial\": " << d2.serialNumber() << ",\n";
        std::cout << "    \"start_serial\": " << d1.serialNumber() << ",\n";
        std::cout << "    \"end_serial\": " << d2.serialNumber() << ",\n";
        std::cout << "    \"nominal\": 100000,\n";
        std::cout << "    \"rate\": 0.05,\n";
        std::cout << "    \"accrual_period\": " << frc.accrualPeriod() << ",\n";
        std::cout << "    \"accrual_days\": " << frc.accrualDays() << ",\n";
        std::cout << "    \"amount\": " << frc.amount() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // SimpleCashFlow: predetermined 1234.56 on Jul 1, 2026
    // ============================================================
    {
        Date d(1, July, 2026);
        SimpleCashFlow scf(1234.56, d);
        std::cout << "  \"simple_cash_flow\": {\n";
        std::cout << "    \"amount\": " << scf.amount() << ",\n";
        std::cout << "    \"date_serial\": " << scf.date().serialNumber() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // CashFlows: 4 fixed cashflows + flat 5% discount curve
    // ============================================================
    {
        // Build a 2y semiannual bond: 4 coupons of 100,000*0.05*0.5 = 2,500 each,
        // plus 100,000 redemption at maturity.
        // Settlement = Jan 1, 2026.
        Settings::instance().evaluationDate() = Date(1, January, 2026);

        std::vector<Date> dates = {
            Date(1, July, 2026),
            Date(1, January, 2027),
            Date(1, July, 2027),
            Date(1, January, 2028),
        };

        Leg leg;
        for (size_t i = 0; i < dates.size(); ++i) {
            Date start = (i == 0) ? Date(1, January, 2026) : dates[i - 1];
            Date end = dates[i];
            leg.push_back(ext::shared_ptr<CashFlow>(
                new FixedRateCoupon(end, 100000.0, 0.05, act360, start, end)));
        }
        // redemption at maturity
        leg.push_back(ext::shared_ptr<CashFlow>(
            new SimpleCashFlow(100000.0, Date(1, January, 2028))));

        // Flat 5% Annual/Act365 yield curve at settlement
        Date settle(1, January, 2026);
        Handle<YieldTermStructure> curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(settle, 0.05, act365, Compounded, Annual)));

        double npv_curve = CashFlows::npv(leg, **curve, false, settle);
        double bps_curve = CashFlows::bps(leg, **curve, false, settle);

        std::cout << "  \"bond_flat_curve\": {\n";
        std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
        std::cout << "    \"npv\": " << npv_curve << ",\n";
        std::cout << "    \"bps\": " << bps_curve << ",\n";

        // With YTM = 5% (Compounded, Annual) -> compute NPV via InterestRate flavour
        InterestRate y(0.05, act365, Compounded, Annual);
        double npv_y = CashFlows::npv(leg, y, false, settle);
        double simple_d = CashFlows::duration(leg, y, Duration::Simple, false, settle);
        double mod_d    = CashFlows::duration(leg, y, Duration::Modified, false, settle);
        double mac_d    = CashFlows::duration(leg, y, Duration::Macaulay, false, settle);
        double conv     = CashFlows::convexity(leg, y, false, settle);

        std::cout << "    \"npv_at_ytm5\": " << npv_y << ",\n";
        std::cout << "    \"simple_duration\": " << simple_d << ",\n";
        std::cout << "    \"modified_duration\": " << mod_d << ",\n";
        std::cout << "    \"macaulay_duration\": " << mac_d << ",\n";
        std::cout << "    \"convexity\": " << conv << ",\n";

        // IRR: given the curve NPV, recover an annual yield
        double irr = CashFlows::yield(leg, npv_y, act365, Compounded, Annual,
                                      false, settle, settle, 1.0e-12, 100, 0.04);
        std::cout << "    \"irr\": " << irr << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // OvernightIndexedCoupon: 1mo period at flat 4% Eonia/Act360
    // ============================================================
    {
        // Build a flat 4% curve, attach to Eonia, build a 1mo coupon.
        Settings::instance().evaluationDate() = Date(15, January, 2026);

        Date d1(15, January, 2026);
        Date d2(15, February, 2026);

        Handle<YieldTermStructure> ois_curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(d1, 0.04, act360, Continuous, Annual)));
        ext::shared_ptr<OvernightIndex> eonia(new Eonia(ois_curve));

        OvernightIndexedCoupon oic(d2, 100000.0, d1, d2, eonia);
        // Set the pricer (needed for FloatingRateCoupon::rate())
        ext::shared_ptr<FloatingRateCouponPricer> pricer(
            new CompoundingOvernightIndexedCouponPricer());
        oic.setPricer(pricer);

        std::cout << "  \"overnight_coupon\": {\n";
        std::cout << "    \"start_serial\": " << d1.serialNumber() << ",\n";
        std::cout << "    \"end_serial\": " << d2.serialNumber() << ",\n";
        std::cout << "    \"nominal\": 100000,\n";
        std::cout << "    \"accrual_period\": " << oic.accrualPeriod() << ",\n";
        std::cout << "    \"rate\": " << oic.rate() << ",\n";
        std::cout << "    \"amount\": " << oic.amount() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // IborCoupon: 3M Euribor at flat 3.5%, Jan1->Apr1, Act/360
    // (eval date set well before the fixing date so the index
    //  forecasts the fixing instead of looking it up)
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(1, December, 2025);

        Date d1(1, January, 2026);
        Date d2(1, April, 2026);
        Date payment = d2;

        Handle<YieldTermStructure> ibor_curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(Date(1, December, 2025), 0.035, act360, Simple, Annual)));
        ext::shared_ptr<IborIndex> idx(new Euribor3M(ibor_curve));

        IborCoupon ic(payment, 100000.0, d1, d2,
                       idx->fixingDays(), idx,
                       /*gearing*/1.0, /*spread*/0.0);
        ext::shared_ptr<IborCouponPricer> pricer(new BlackIborCouponPricer());
        ic.setPricer(pricer);

        std::cout << "  \"ibor_coupon\": {\n";
        std::cout << "    \"start_serial\": " << d1.serialNumber() << ",\n";
        std::cout << "    \"end_serial\": " << d2.serialNumber() << ",\n";
        std::cout << "    \"accrual_period\": " << ic.accrualPeriod() << ",\n";
        std::cout << "    \"index_fixing\": " << ic.indexFixing() << ",\n";
        std::cout << "    \"rate\": " << ic.rate() << ",\n";
        std::cout << "    \"amount\": " << ic.amount() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
