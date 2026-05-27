// L3-B mega-probe: Bond family + DiscountingBondEngine + BondForward
//
// Emits reference values for FixedRateBond, ZeroCouponBond, FloatingRateBond,
// AmortizingFixedRateBond clean/dirty/yield/accrued + DiscountingBondEngine
// NPV + BondForward forwardPrice/cleanForwardPrice/spotIncome.
//
// All bond schedules and curves use simple deterministic settings so the
// Python port can reproduce them with the existing L1/L2 ports.

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/bond.hpp>
#include <ql/instruments/bondforward.hpp>
#include <ql/instruments/bonds/amortizingfixedratebond.hpp>
#include <ql/instruments/bonds/fixedratebond.hpp>
#include <ql/instruments/bonds/floatingratebond.hpp>
#include <ql/instruments/bonds/zerocouponbond.hpp>
#include <ql/pricingengines/bond/bondfunctions.hpp>
#include <ql/pricingengines/bond/discountingbondengine.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    Actual360 act360;
    Actual365Fixed act365;
    Thirty360 th360(Thirty360::BondBasis);
    NullCalendar nullCal;
    TARGET target;

    // ============================================================
    // FixedRateBond: 5-year, 5% Semiannual, 30/360 BondBasis,
    // face 100. Issued Jan 15, 2025; matures Jan 15, 2030.
    // Discount curve: FlatForward(5%, Compounded, Annual, Act/365).
    // Settlement = Jan 15, 2025 + 2 BD = Jan 17, 2025 (TARGET).
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2025);
        Date issue(15, January, 2025);
        Date maturity(15, January, 2030);

        Schedule sched(issue, maturity, Period(Semiannual), target,
                       Unadjusted, Unadjusted,
                       DateGeneration::Backward, false);

        FixedRateBond frb(2, 100.0, sched,
                          std::vector<Rate>(1, 0.05),
                          th360, Following, 100.0, issue);

        Handle<YieldTermStructure> curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(issue, 0.05, act365, Compounded, Annual)));
        ext::shared_ptr<DiscountingBondEngine> engine(
            new DiscountingBondEngine(curve));
        frb.setPricingEngine(engine);

        Date settle = frb.settlementDate();
        double clean = frb.cleanPrice();
        double dirty = frb.dirtyPrice();
        double accrued = frb.accruedAmount(settle);
        double settlement_value = frb.settlementValue();
        double npv_engine = frb.NPV();
        double notional = frb.notional(settle);

        // Yield round-trip (LOOSE tier — iterative).
        double y = frb.yield(th360, Compounded, Semiannual,
                             1.0e-10, 100, 0.05, Bond::Price::Clean);
        // Re-price from yield to verify roundtrip
        double clean_from_y = frb.cleanPrice(y, th360, Compounded, Semiannual);

        std::cout << "  \"fixed_rate_bond\": {\n";
        std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
        std::cout << "    \"issue_serial\": " << issue.serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << maturity.serialNumber() << ",\n";
        std::cout << "    \"clean_price\": " << clean << ",\n";
        std::cout << "    \"dirty_price\": " << dirty << ",\n";
        std::cout << "    \"accrued_amount\": " << accrued << ",\n";
        std::cout << "    \"settlement_value\": " << settlement_value << ",\n";
        std::cout << "    \"npv\": " << npv_engine << ",\n";
        std::cout << "    \"notional\": " << notional << ",\n";
        std::cout << "    \"yield\": " << y << ",\n";
        std::cout << "    \"clean_price_from_yield\": " << clean_from_y << ",\n";
        // Cashflow count (10 coupons + 1 redemption = 11)
        std::cout << "    \"n_cashflows\": " << frb.cashflows().size() << ",\n";
        std::cout << "    \"start_date_serial\": " << frb.startDate().serialNumber() << ",\n";
        std::cout << "    \"maturity_date_serial\": " << frb.maturityDate().serialNumber() << ",\n";
        std::cout << "    \"is_tradable\": " << (frb.isTradable() ? 1 : 0) << ",\n";
        std::cout << "    \"is_expired\": " << (frb.isExpired() ? 1 : 0) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // FixedRateBond — accrued at mid-period.
    // Probe settlement mid first coupon: Apr 15, 2025
    // (~90/180 days into the first 6-month period).
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2025);
        Date issue(15, January, 2025);
        Date maturity(15, January, 2030);

        Schedule sched(issue, maturity, Period(Semiannual), target,
                       Unadjusted, Unadjusted,
                       DateGeneration::Backward, false);

        FixedRateBond frb(2, 100.0, sched, std::vector<Rate>(1, 0.05),
                          th360, Following, 100.0, issue);

        Date mid(15, April, 2025);
        double accrued_mid = frb.accruedAmount(mid);
        double next_cf = frb.nextCashFlowDate(mid).serialNumber();
        double prev_cf = frb.previousCashFlowDate(mid).serialNumber();
        double next_coupon_rate = frb.nextCouponRate(mid);

        std::cout << "  \"accrued_mid_period\": {\n";
        std::cout << "    \"settle_serial\": " << mid.serialNumber() << ",\n";
        std::cout << "    \"accrued_amount\": " << accrued_mid << ",\n";
        std::cout << "    \"next_cf_serial\": " << next_cf << ",\n";
        std::cout << "    \"prev_cf_serial\": " << prev_cf << ",\n";
        std::cout << "    \"next_coupon_rate\": " << next_coupon_rate << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // ZeroCouponBond: 5-year, face 100, redemption 100.
    // Discount curve: FlatForward(5%, Compounded, Annual, Act/365).
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2025);
        Date issue(15, January, 2025);
        Date maturity(15, January, 2030);

        ZeroCouponBond zcb(2, target, 100.0, maturity, Following, 100.0, issue);

        Handle<YieldTermStructure> curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(issue, 0.05, act365, Compounded, Annual)));
        ext::shared_ptr<DiscountingBondEngine> engine(
            new DiscountingBondEngine(curve));
        zcb.setPricingEngine(engine);

        Date settle = zcb.settlementDate();
        double clean = zcb.cleanPrice();
        double dirty = zcb.dirtyPrice();
        double npv_engine = zcb.NPV();
        // Direct discount factor: bond pays 100 at adjusted maturity.
        Date redemption_date = target.adjust(maturity, Following);
        double df_at_redemption = (*curve)->discount(redemption_date);
        double expected_settlement_value = 100.0 * df_at_redemption;

        std::cout << "  \"zero_coupon_bond\": {\n";
        std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
        std::cout << "    \"redemption_serial\": " << redemption_date.serialNumber() << ",\n";
        std::cout << "    \"clean_price\": " << clean << ",\n";
        std::cout << "    \"dirty_price\": " << dirty << ",\n";
        std::cout << "    \"npv\": " << npv_engine << ",\n";
        std::cout << "    \"df_at_redemption\": " << df_at_redemption << ",\n";
        std::cout << "    \"expected_settlement_value\": " << expected_settlement_value << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // FloatingRateBond: 5-year, Euribor 6M, Semiannual,
    // face 100, gearing=1, spread=0, flat forecast curve at 3.5%.
    // Eval date Dec 1, 2024 so the first fixing (~Jan 13, 2025) is in
    // the future and gets forecast from the curve (no historical fixing
    // required).
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(1, December, 2024);
        Date issue(15, January, 2025);
        Date maturity(15, January, 2030);

        Handle<YieldTermStructure> forecast_curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(Date(1, December, 2024), 0.035, act360, Simple, Annual)));
        ext::shared_ptr<IborIndex> idx(new Euribor6M(forecast_curve));

        Schedule sched(issue, maturity, Period(Semiannual), target,
                       Unadjusted, Unadjusted,
                       DateGeneration::Backward, false);

        FloatingRateBond floater(2, 100.0, sched, idx, act360,
                                 Following, 2, /*gearings*/{1.0},
                                 /*spreads*/{0.0}, /*caps*/{}, /*floors*/{},
                                 /*inArrears*/false, 100.0, issue);

        Handle<YieldTermStructure> disc_curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(Date(1, December, 2024), 0.04, act365, Compounded, Annual)));
        ext::shared_ptr<DiscountingBondEngine> engine(
            new DiscountingBondEngine(disc_curve));
        floater.setPricingEngine(engine);

        // Set pricer for IborCoupon
        ext::shared_ptr<IborCouponPricer> pricer(new BlackIborCouponPricer());
        setCouponPricer(floater.cashflows(), pricer);

        double npv = floater.NPV();
        double clean = floater.cleanPrice();
        double dirty = floater.dirtyPrice();
        Date settle = floater.settlementDate();
        double accrued = floater.accruedAmount(settle);
        double notional = floater.notional(settle);

        std::cout << "  \"floating_rate_bond\": {\n";
        std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
        std::cout << "    \"npv\": " << npv << ",\n";
        std::cout << "    \"clean_price\": " << clean << ",\n";
        std::cout << "    \"dirty_price\": " << dirty << ",\n";
        std::cout << "    \"accrued_amount\": " << accrued << ",\n";
        std::cout << "    \"notional\": " << notional << ",\n";
        std::cout << "    \"n_cashflows\": " << floater.cashflows().size() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // AmortizingFixedRateBond: 4-year, 5% Annual, 30/360.
    // Notionals 100 -> 75 -> 50 -> 25 -> 0 (linear straight-line).
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2025);
        Date issue(15, January, 2025);
        Date maturity(15, January, 2029);

        Schedule sched(issue, maturity, Period(Annual), target,
                       Unadjusted, Unadjusted,
                       DateGeneration::Backward, false);

        // 4 coupon periods, notional steps down each period.
        std::vector<Real> notionals = {100.0, 75.0, 50.0, 25.0};
        std::vector<Rate> coupons(1, 0.05);

        AmortizingFixedRateBond afrb(2, notionals, sched, coupons,
                                     th360, Following, issue);

        Handle<YieldTermStructure> curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(issue, 0.05, act365, Compounded, Annual)));
        ext::shared_ptr<DiscountingBondEngine> engine(
            new DiscountingBondEngine(curve));
        afrb.setPricingEngine(engine);

        Date settle = afrb.settlementDate();
        double npv = afrb.NPV();
        double clean = afrb.cleanPrice();
        double dirty = afrb.dirtyPrice();
        double notional_settle = afrb.notional(settle);
        double notional_y1 = afrb.notional(Date(15, January, 2026));
        double notional_y2 = afrb.notional(Date(15, January, 2027));
        double notional_y3 = afrb.notional(Date(15, January, 2028));

        std::cout << "  \"amortizing_fixed_rate_bond\": {\n";
        std::cout << "    \"settle_serial\": " << settle.serialNumber() << ",\n";
        std::cout << "    \"npv\": " << npv << ",\n";
        std::cout << "    \"clean_price\": " << clean << ",\n";
        std::cout << "    \"dirty_price\": " << dirty << ",\n";
        std::cout << "    \"notional_at_settle\": " << notional_settle << ",\n";
        std::cout << "    \"notional_y1\": " << notional_y1 << ",\n";
        std::cout << "    \"notional_y2\": " << notional_y2 << ",\n";
        std::cout << "    \"notional_y3\": " << notional_y3 << ",\n";
        std::cout << "    \"n_cashflows\": " << afrb.cashflows().size() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // BondForward: forward contract on a 5y 5% FRB.
    // valueDate (settlement) = Jan 17, 2025 (T+2),
    // deliveryDate = Apr 15, 2025.
    // Discount + incomeDiscount = FlatForward(5%, Continuous, Annual, Act/365)
    // ============================================================
    {
        Settings::instance().evaluationDate() = Date(15, January, 2025);
        Date issue(15, January, 2025);
        Date bond_maturity(15, January, 2030);

        Schedule sched(issue, bond_maturity, Period(Semiannual), target,
                       Unadjusted, Unadjusted,
                       DateGeneration::Backward, false);

        ext::shared_ptr<Bond> bond(new FixedRateBond(
            2, 100.0, sched, std::vector<Rate>(1, 0.05),
            th360, Following, 100.0, issue));

        Handle<YieldTermStructure> curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(issue, 0.05, act365, Continuous, Annual)));
        Handle<YieldTermStructure> income_curve(ext::shared_ptr<YieldTermStructure>(
            new FlatForward(issue, 0.05, act365, Continuous, Annual)));

        ext::shared_ptr<DiscountingBondEngine> bond_engine(
            new DiscountingBondEngine(curve));
        bond->setPricingEngine(bond_engine);

        Date value_date(17, January, 2025);
        Date delivery_date(15, April, 2025);

        BondForward bf(value_date, delivery_date, Position::Long, /*strike*/100.0,
                       /*settlementDays*/2, act365, target, Following, bond,
                       curve, income_curve);

        double spot_value = bf.spotValue();
        double spot_income = bf.spotIncome(income_curve);
        double forward_price = bf.forwardPrice();
        double clean_forward = bf.cleanForwardPrice();

        std::cout << "  \"bond_forward\": {\n";
        std::cout << "    \"value_date_serial\": " << value_date.serialNumber() << ",\n";
        std::cout << "    \"delivery_date_serial\": " << delivery_date.serialNumber() << ",\n";
        std::cout << "    \"spot_value\": " << spot_value << ",\n";
        std::cout << "    \"spot_income\": " << spot_income << ",\n";
        std::cout << "    \"forward_price\": " << forward_price << ",\n";
        std::cout << "    \"clean_forward_price\": " << clean_forward << ",\n";
        // No bond coupon in [valueDate, deliveryDate] → spotIncome should be 0.
        std::cout << "    \"strike\": 100\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
