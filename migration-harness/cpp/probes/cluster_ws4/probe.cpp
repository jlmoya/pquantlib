// W-S4 cluster (sibling-packages): ConvertibleBond subsystem into pquantlib
// CORE. PRIMARY (same-method, same-N) cross-validation reference.
//
// BinomialConvertibleEngine<CoxRossRubinstein> (Tsiveriotis-Fernandes
// credit-adjusted lattice) IS in v1.42.1, so the Python port cross-validates
// against the C++ SAME engine at the SAME number of time steps -> TIGHT
// (tree-vs-identical-tree, no model gap; there is no analytic convertible
// value, so the C++ same-engine value IS the reference).
//
// Scenario is based on convertiblebonds.cpp CommonVars, hardened into three
// emitted convertible NPVs:
//   * conv_fixed_eu        — ConvertibleFixedCouponBond, EuropeanExercise,
//                            no callability, no dividends.
//   * conv_fixed_am        — same, AmericanExercise (issueDate..maturity).
//   * conv_fixed_am_callput — AmericanExercise + a CallabilitySchedule mixing
//                            a SoftCallability (Call w/ trigger) and a plain
//                            Put, + a FixedDividend schedule.
//
// All three priced with N = TIME_STEPS recorded below. The Python test must
// use the identical step count.
//
// C++ parity: v1.42.1 (099987f0).

#include <ql/instruments/bonds/convertiblebonds.hpp>
#include <ql/instruments/callabilityschedule.hpp>
#include <ql/instruments/dividendschedule.hpp>
#include <ql/cashflows/dividend.hpp>
#include <ql/pricingengines/bond/binomialconvertibleengine.hpp>
#include <ql/methods/lattices/binomialtree.hpp>
#include <ql/exercise.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/schedule.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace QuantLib;

namespace {

constexpr Size TIME_STEPS = 801;

// Flat yield term structure helper (mirrors test-suite flatRate).
ext::shared_ptr<YieldTermStructure> flatRate(const Date& today,
                                             Rate forward,
                                             const DayCounter& dc) {
    return ext::make_shared<FlatForward>(today, forward, dc);
}

ext::shared_ptr<BlackVolTermStructure> flatVol(const Date& today,
                                               Volatility vol,
                                               const DayCounter& dc) {
    return ext::make_shared<BlackConstantVol>(today, TARGET(), vol, dc);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);

    // ------------------------------------------------------------------
    // CommonVars-style setup, with a FIXED evaluation date (the test-suite
    // uses Settings::instance().evaluationDate() which defaults to "today";
    // we pin it so the reference is reproducible).
    // ------------------------------------------------------------------
    Date today(2, January, 2020);
    Settings::instance().evaluationDate() = today;

    Calendar calendar = TARGET();
    DayCounter dayCounter = Actual360();
    Frequency frequency = Annual;
    Natural settlementDays = 3;

    Natural fixingDays = 2;
    Date issueDate = calendar.advance(today, fixingDays, Days);
    Date maturityDate = calendar.advance(issueDate, 10, Years);
    issueDate = calendar.advance(maturityDate, -10, Years);

    Handle<Quote> underlying(ext::make_shared<SimpleQuote>(50.0));
    Handle<YieldTermStructure> dividendYield(flatRate(today, 0.02, dayCounter));
    Handle<YieldTermStructure> riskFreeRate(flatRate(today, 0.05, dayCounter));
    Handle<BlackVolTermStructure> volatility(flatVol(today, 0.15, dayCounter));

    auto process = ext::make_shared<BlackScholesMertonProcess>(
        underlying, dividendYield, riskFreeRate, volatility);

    Handle<Quote> creditSpread(ext::make_shared<SimpleQuote>(0.005));

    Real redemption = 100.0;
    Real conversionRatio = redemption / underlying->value(); // 2.0

    std::vector<Rate> coupons(1, 0.05);

    Schedule schedule = MakeSchedule().from(issueDate)
                                      .to(maturityDate)
                                      .withFrequency(frequency)
                                      .withCalendar(calendar)
                                      .backwards();

    ext::shared_ptr<Exercise> euExercise =
        ext::make_shared<EuropeanExercise>(maturityDate);
    ext::shared_ptr<Exercise> amExercise =
        ext::make_shared<AmericanExercise>(issueDate, maturityDate);

    CallabilitySchedule no_callability;
    DividendSchedule no_dividends;

    auto engine = ext::make_shared<BinomialConvertibleEngine<CoxRossRubinstein>>(
        process, TIME_STEPS, creditSpread, no_dividends);

    std::cout << "{\n";
    std::cout << "  \"time_steps\": " << TIME_STEPS << ",\n";
    std::cout << "  \"today_serial\": " << today.serialNumber() << ",\n";
    std::cout << "  \"issue_serial\": " << issueDate.serialNumber() << ",\n";
    std::cout << "  \"maturity_serial\": " << maturityDate.serialNumber() << ",\n";
    std::cout << "  \"conversion_ratio\": " << conversionRatio << ",\n";

    // --- European, no callability, no dividends ---
    {
        ConvertibleFixedCouponBond bond(euExercise, conversionRatio,
                                        no_callability, issueDate,
                                        settlementDays, coupons, dayCounter,
                                        schedule, redemption);
        bond.setPricingEngine(engine);
        std::cout << "  \"conv_fixed_eu\": " << bond.NPV() << ",\n";
    }

    // --- American, no callability, no dividends ---
    {
        ConvertibleFixedCouponBond bond(amExercise, conversionRatio,
                                        no_callability, issueDate,
                                        settlementDays, coupons, dayCounter,
                                        schedule, redemption);
        bond.setPricingEngine(engine);
        std::cout << "  \"conv_fixed_am\": " << bond.NPV() << ",\n";
    }

    // --- American + call/put schedule + dividends ---
    {
        // SoftCallability (Call, trigger 1.10) at year 5, plain Put at year 7.
        CallabilitySchedule callability;
        Date callDate = calendar.advance(issueDate, 5, Years);
        Date putDate = calendar.advance(issueDate, 7, Years);
        callability.push_back(ext::make_shared<SoftCallability>(
            Bond::Price(108.0, Bond::Price::Clean), callDate, 1.10));
        callability.push_back(ext::make_shared<Callability>(
            Bond::Price(101.0, Bond::Price::Clean), Callability::Put, putDate));

        // Two fixed dividends.
        DividendSchedule dividends;
        Date div1 = calendar.advance(issueDate, 1, Years);
        Date div2 = calendar.advance(issueDate, 3, Years);
        dividends.push_back(ext::make_shared<FixedDividend>(1.0, div1));
        dividends.push_back(ext::make_shared<FixedDividend>(1.5, div2));

        auto divEngine =
            ext::make_shared<BinomialConvertibleEngine<CoxRossRubinstein>>(
                process, TIME_STEPS, creditSpread, dividends);

        std::cout << "    \"call_serial\": " << callDate.serialNumber() << ",\n";
        std::cout << "    \"put_serial\": " << putDate.serialNumber() << ",\n";
        std::cout << "    \"div1_serial\": " << div1.serialNumber() << ",\n";
        std::cout << "    \"div2_serial\": " << div2.serialNumber() << ",\n";

        ConvertibleFixedCouponBond bond(amExercise, conversionRatio,
                                        callability, issueDate,
                                        settlementDays, coupons, dayCounter,
                                        schedule, redemption);
        bond.setPricingEngine(divEngine);
        std::cout << "  \"conv_fixed_am_callput\": " << bond.NPV() << "\n";
    }

    std::cout << "}\n";
    return 0;
}
