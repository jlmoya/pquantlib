// migration-harness/cpp/probes/v143_cf_irr/probe.cpp
//
// Reference values for CashFlows::IrrFinder — the objective class nested in
// CashFlows that drives CashFlows::yield.
//
// IrrFinder is private, so it cannot be instantiated from a probe. What it
// does is observable through yield() in three ways, and all three are pinned
// here because PQuantLib got all three subtly wrong:
//
//   1. the root itself. PQuantLib solved with Brent over a sign-flipped
//      objective (target - NPV) and a fixed 1e-4 step; C++ solves with
//      NewtonSafe over (NPV - target) using IrrFinder::derivative, stepping
//      guess/10. Same root in easy cases, different iterate path — and a
//      different answer whenever the bracket search matters. Several guesses
//      per leg are probed for exactly that reason.
//
//   2. IrrFinder::derivative, which is -modifiedDuration * P. It is not
//      directly callable, but modifiedDuration and npv are, so the product is
//      emitted at the solved yield: a port whose derivative is wrong (sign,
//      or simple vs modified duration) cannot reproduce it.
//
//   3. IrrFinder::checkSign, run from the constructor. A (leg, target price)
//      pair whose flows never change sign relative to -price is rejected with
//      "the given cash flows cannot result in the given market price due to
//      their sign". PQuantLib had no such guard at all, so this is pinned as
//      an explicit {"raises": true} case.
//
// Legs are built inline from literal tables (no market fixture, no index) so a
// mismatch localises to the yield machinery.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/cf/irr.json.

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/cashflows/cashflows.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

const Date kSettlement(15, June, 2026);

// A plain amortising-free fixed-rate bond leg plus its redemption, built by
// hand so the probe depends on nothing but Schedule + FixedRateCoupon.
Leg buildLeg(Rate couponRate, Real notional, Size years, Frequency freq) {
    Schedule schedule = MakeSchedule()
                            .from(kSettlement)
                            .to(kSettlement + Period(static_cast<Integer>(years), Years))
                            .withFrequency(freq)
                            .withCalendar(TARGET())
                            .withConvention(Unadjusted)
                            .backwards();
    // Unadjusted payment dates keep payment == accrual end, so the redemption
    // pushed at the schedule's last date cannot land before the final coupon's
    // payment date (which would make the leg non-monotonic in date and trip
    // getStepwiseDiscountTime's negative-time guard).
    Leg leg = FixedRateLeg(schedule)
                  .withNotionals(notional)
                  .withCouponRates(couponRate, Thirty360(Thirty360::BondBasis))
                  .withPaymentAdjustment(Unadjusted);
    leg.push_back(ext::make_shared<SimpleCashFlow>(notional, schedule.dates().back()));
    return leg;
}

void emitCashflows(const Leg& leg) {
    std::cout << "    \"cashflows\": [";
    for (Size i = 0; i < leg.size(); ++i) {
        if (i != 0)
            std::cout << ",";
        std::cout << "\n      {\"date_serial\": " << leg[i]->date().serialNumber()
                  << ", \"amount\": " << leg[i]->amount() << "}";
    }
    std::cout << "\n    ]";
}

void emitYieldCase(const char* key,
                   const Leg& leg,
                   Real targetNpv,
                   const DayCounter& dc,
                   Compounding comp,
                   Frequency freq,
                   Rate guess,
                   bool trailingComma) {
    std::cout << "  \"" << key << "\": {\n"
              << "    \"target_npv\": " << targetNpv << ",\n"
              << "    \"compounding\": " << static_cast<int>(comp) << ",\n"
              << "    \"frequency\": " << static_cast<int>(freq) << ",\n"
              << "    \"guess\": " << guess << ",\n"
              << "    \"settlement_serial\": " << kSettlement.serialNumber() << ",\n";
    emitCashflows(leg);
    std::cout << ",\n";
    try {
        // Same defaults as CashFlows::yield's non-template overload:
        // accuracy 1e-10, maxIterations 100.
        Rate y = CashFlows::yield(leg, targetNpv, dc, comp, freq, false, kSettlement,
                                  kSettlement, 1.0e-10, 100, guess);
        InterestRate rate(y, dc, comp, freq);
        Real npvAtRoot = CashFlows::npv(leg, rate, false, kSettlement, kSettlement);
        Real modDur =
            CashFlows::duration(leg, rate, Duration::Modified, false, kSettlement, kSettlement);
        std::cout << "    \"raises\": false,\n"
                  << "    \"yield\": " << y << ",\n"
                  // IrrFinder::operator()(root) — must be ~0 by construction.
                  << "    \"objective_at_root\": " << (npvAtRoot - targetNpv) << ",\n"
                  << "    \"npv_at_root\": " << npvAtRoot << ",\n"
                  << "    \"modified_duration_at_root\": " << modDur << ",\n"
                  // IrrFinder::derivative(root).
                  << "    \"derivative_at_root\": " << (-modDur * npvAtRoot) << "\n";
    } catch (const std::exception& e) {
        // The message is pinned too: checkSign has a distinctive one, and a
        // port that raises for the wrong reason is not a port that agrees.
        std::cout << "    \"raises\": true,\n"
                  << "    \"error\": \"" << e.what() << "\"\n";
    }
    std::cout << "  }" << (trailingComma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const Actual365Fixed a365;
    const Thirty360 t360(Thirty360::BondBasis);

    const Leg fiveYear = buildLeg(0.04, 100.0, 5, Semiannual);
    const Leg tenYear = buildLeg(0.055, 100.0, 10, Annual);

    // --- ordinary roots, several guesses: NewtonSafe's bracket search
    //     starts at [guess - guess/10, guess + guess/10] and expands, so the
    //     guess is not cosmetic.
    emitYieldCase("par_5y_semi_guess_005", fiveYear, 100.0, t360, Compounded, Semiannual, 0.05,
                  true);
    emitYieldCase("par_5y_semi_guess_001", fiveYear, 100.0, t360, Compounded, Semiannual, 0.01,
                  true);
    emitYieldCase("par_5y_semi_guess_020", fiveYear, 100.0, t360, Compounded, Semiannual, 0.20,
                  true);
    emitYieldCase("premium_5y_semi", fiveYear, 112.5, t360, Compounded, Semiannual, 0.05, true);
    emitYieldCase("discount_5y_semi", fiveYear, 87.25, t360, Compounded, Semiannual, 0.05, true);

    // --- the other compounding conventions reach different branches of
    //     modifiedDuration, hence a different IrrFinder::derivative.
    emitYieldCase("par_10y_annual_compounded", tenYear, 100.0, t360, Compounded, Annual, 0.05,
                  true);
    emitYieldCase("par_10y_annual_continuous", tenYear, 100.0, a365, Continuous, Annual, 0.05,
                  true);
    emitYieldCase("par_10y_annual_simple", tenYear, 100.0, a365, Simple, Annual, 0.05, true);
    emitYieldCase("par_10y_simple_then_comp", tenYear, 100.0, a365, SimpleThenCompounded, Annual,
                  0.05, true);
    emitYieldCase("par_10y_comp_then_simple", tenYear, 100.0, a365, CompoundedThenSimple, Annual,
                  0.05, true);

    // --- checkSign rejections. All flows are positive, so a NEGATIVE target
    //     price gives sign(-npv) = +1 and no sign change: nonsensical IRR.
    emitYieldCase("check_sign_negative_target", fiveYear, -100.0, t360, Compounded, Semiannual,
                  0.05, true);

    // A leg of one single positive flow against a zero target: sign(-0) == 0,
    // so lastSign starts at 0 and never multiplies to a negative product.
    Leg singleFlow;
    singleFlow.push_back(ext::make_shared<SimpleCashFlow>(100.0, kSettlement + Period(1, Years)));
    emitYieldCase("check_sign_single_flow_zero_target", singleFlow, 0.0, a365, Compounded, Annual,
                  0.05, false);

    std::cout << "}\n";
    return 0;
}
