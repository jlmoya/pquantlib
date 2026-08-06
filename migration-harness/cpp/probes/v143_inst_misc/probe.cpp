// migration-harness/cpp/probes/v143_inst_misc/probe.cpp
//
// Reference values for four small instruments-subsystem classes PQuantLib
// lacked:
//
//   * Stock                    (ql/instruments/stock.hpp)
//   * CompositeInstrument      (ql/instruments/compositeinstrument.hpp)
//   * FaceValueAccrualClaim    (ql/instruments/claim.hpp)
//   * ImpliedVolatilityHelper  (ql/instruments/impliedvolatility.hpp)
//
// Small does not mean trivial to get right:
//
//   - CompositeInstrument's multiplier is signed and subtract() is defined as
//     add(-multiplier); a port that forgets the negation still returns a
//     number. So the composite is probed with asymmetric multipliers and with
//     mixed add/subtract, and each component's own NPV is emitted alongside
//     the total so a cancelling pair of errors cannot hide.
//   - isExpired() on a composite is an AND over components, not an OR — an
//     easy inversion. It is probed with all-expired, none-expired and mixed.
//   - FaceValueAccrualClaim divides accrued by notional(d), so it depends on
//     the reference bond's amortisation, not just on its coupon. It is probed
//     against an amortising bond at several dates, including a coupon date.
//   - ImpliedVolatilityHelper::calculate is a Brent solve over a cloned
//     process whose vol is driven by a quote; the clone must keep the state
//     variable, dividend and risk-free curves and only replace the vol. The
//     probe therefore emits, for each case, both the recovered vol AND the
//     price the original engine gives at that vol.
//
// Everything is built inline from literal tables.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/misc.json.

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include <ql/exercise.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/instruments/compositeinstrument.hpp>
#include <ql/instruments/impliedvolatility.hpp>
#include <ql/instruments/stock.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/instruments/bond.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

const Date kToday(15, June, 2026);
const Actual365Fixed kA365;

ext::shared_ptr<GeneralizedBlackScholesProcess> makeProcess(Real spot,
                                                            Rate q,
                                                            Rate r,
                                                            Volatility vol) {
    Handle<Quote> s(ext::make_shared<SimpleQuote>(spot));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(kToday, q, kA365));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(kToday, r, kA365));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(kToday, TARGET(), vol, kA365));
    return ext::make_shared<GeneralizedBlackScholesProcess>(s, qTS, rTS, volTS);
}

ext::shared_ptr<VanillaOption> makeOption(Option::Type type, Real strike, const Date& expiry) {
    return ext::make_shared<VanillaOption>(
        ext::make_shared<PlainVanillaPayoff>(type, strike),
        ext::make_shared<EuropeanExercise>(expiry));
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kToday;
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"today_serial\": " << kToday.serialNumber() << ",\n";

    // =====================================================================
    // Stock
    // =====================================================================
    {
        auto quote = ext::make_shared<SimpleQuote>(123.75);
        Stock stock{Handle<Quote>(quote)};
        std::cout << "  \"stock\": {\n"
                  << "    \"quote\": 123.75,\n"
                  << "    \"npv\": " << stock.NPV() << ",\n"
                  << "    \"is_expired\": " << (stock.isExpired() ? "true" : "false") << ",\n";
        // The quote is observed: bumping it must invalidate the cache.
        quote->setValue(150.25);
        std::cout << "    \"npv_after_quote_bump\": " << stock.NPV() << ",\n";
        Stock nullQuoteStock{Handle<Quote>()};
        try {
            (void)nullQuoteStock.NPV();
            std::cout << "    \"null_quote_raises\": false\n";
        } catch (const std::exception& e) {
            std::cout << "    \"null_quote_raises\": true,\n"
                      << "    \"null_quote_error\": \"" << e.what() << "\"\n";
        }
        std::cout << "  },\n";
    }

    // =====================================================================
    // CompositeInstrument
    // =====================================================================
    {
        auto process = makeProcess(100.0, 0.03, 0.05, 0.20);
        auto engine = ext::make_shared<AnalyticEuropeanEngine>(process);

        const Date expiry = kToday + Period(1, Years);
        auto call90 = makeOption(Option::Call, 90.0, expiry);
        auto call110 = makeOption(Option::Call, 110.0, expiry);
        auto put100 = makeOption(Option::Put, 100.0, expiry);
        call90->setPricingEngine(engine);
        call110->setPricingEngine(engine);
        put100->setPricingEngine(engine);

        std::cout << "  \"composite\": {\n"
                  << "    \"call90_npv\": " << call90->NPV() << ",\n"
                  << "    \"call110_npv\": " << call110->NPV() << ",\n"
                  << "    \"put100_npv\": " << put100->NPV() << ",\n";

        // Asymmetric multipliers, and a subtract() so the sign convention is
        // exercised: 2.5*call90 - 1.75*call110 + 0.5*put100.
        CompositeInstrument composite;
        composite.add(call90, 2.5);
        composite.subtract(call110, 1.75);
        composite.add(put100, 0.5);
        std::cout << "    \"npv\": " << composite.NPV() << ",\n"
                  << "    \"is_expired\": " << (composite.isExpired() ? "true" : "false")
                  << ",\n";

        // Default multiplier is 1.0.
        CompositeInstrument defaults;
        defaults.add(call90);
        defaults.subtract(call110);
        std::cout << "    \"npv_default_multipliers\": " << defaults.NPV() << ",\n";

        // An empty composite is expired (vacuous AND) and worth zero.
        CompositeInstrument empty;
        std::cout << "    \"empty_is_expired\": " << (empty.isExpired() ? "true" : "false")
                  << ",\n"
                  << "    \"empty_npv\": " << empty.NPV() << ",\n";

        // isExpired is an AND: one live component keeps the composite live.
        auto expired = makeOption(Option::Call, 90.0, kToday - Period(1, Days));
        expired->setPricingEngine(engine);
        CompositeInstrument mixed;
        mixed.add(expired, 1.0);
        mixed.add(call90, 1.0);
        CompositeInstrument allExpired;
        allExpired.add(expired, 1.0);
        std::cout << "    \"expired_component_npv\": " << expired->NPV() << ",\n"
                  << "    \"mixed_is_expired\": " << (mixed.isExpired() ? "true" : "false")
                  << ",\n"
                  << "    \"mixed_npv\": " << mixed.NPV() << ",\n"
                  << "    \"all_expired_is_expired\": "
                  << (allExpired.isExpired() ? "true" : "false") << ",\n"
                  << "    \"all_expired_npv\": " << allExpired.NPV() << "\n"
                  << "  },\n";
    }

    // =====================================================================
    // FaceValueAccrualClaim (and FaceValueClaim for contrast)
    // =====================================================================
    {
        Schedule schedule = MakeSchedule()
                                .from(Date(15, June, 2025))
                                .to(Date(15, June, 2030))
                                .withFrequency(Annual)
                                .withCalendar(TARGET())
                                .withConvention(Unadjusted)
                                .backwards();
        // Amortising notionals: notional(d) changes over life, so a claim
        // that divides accrued by the WRONG notional is visible. Built as a
        // generic Bond over an explicit leg rather than via a concrete bond
        // class, so this probe does not depend on the amortising-bond port.
        const std::vector<Real> notionals = {100.0, 90.0, 80.0, 70.0, 60.0};
        const Thirty360 bondDc(Thirty360::BondBasis);
        Leg coupons;
        for (Size i = 0; i + 1 < schedule.size(); ++i) {
            coupons.push_back(ext::make_shared<FixedRateCoupon>(
                schedule.date(i + 1), notionals[i], 0.04, bondDc, schedule.date(i),
                schedule.date(i + 1)));
            const Real amortised =
                notionals[i] - (i + 1 < notionals.size() ? notionals[i + 1] : 0.0);
            coupons.push_back(
                ext::make_shared<AmortizingPayment>(amortised, schedule.date(i + 1)));
        }
        auto bond = ext::make_shared<Bond>(2, TARGET(), Date(15, June, 2025), coupons);

        FaceValueClaim faceValue;
        FaceValueAccrualClaim accrualClaim(bond);

        const std::vector<Date> dates = {Date(15, June, 2026), Date(15, September, 2026),
                                         Date(15, December, 2026), Date(15, March, 2027),
                                         Date(15, June, 2028)};
        std::cout << "  \"claim\": {\n    \"cases\": [";
        for (Size i = 0; i < dates.size(); ++i) {
            if (i != 0)
                std::cout << ",";
            const Date& d = dates[i];
            std::cout << "\n      {\"date_serial\": " << d.serialNumber()
                      << ", \"bond_notional\": " << bond->notional(d)
                      << ", \"bond_accrued\": " << bond->accruedAmount(d)
                      << ", \"face_value_claim\": " << faceValue.amount(d, 1000.0, 0.4)
                      << ", \"face_value_accrual_claim\": "
                      << accrualClaim.amount(d, 1000.0, 0.4)
                      << ", \"face_value_accrual_claim_rr0\": "
                      << accrualClaim.amount(d, 1000.0, 0.0) << "}";
        }
        std::cout << "\n    ]\n  },\n";
    }

    // =====================================================================
    // ImpliedVolatilityHelper
    // =====================================================================
    {
        std::cout << "  \"implied_vol\": {\n    \"cases\": [";
        struct Case {
            Option::Type type;
            Real strike;
            Volatility trueVol;
            Rate q;
            Rate r;
            Real spot;
        };
        const std::vector<Case> cases = {
            {Option::Call, 100.0, 0.20, 0.03, 0.05, 100.0},
            {Option::Put, 100.0, 0.20, 0.03, 0.05, 100.0},
            {Option::Call, 120.0, 0.35, 0.00, 0.02, 100.0},
            {Option::Put, 80.0, 0.15, 0.06, 0.01, 100.0},
            {Option::Call, 100.0, 0.45, 0.03, 0.05, 130.0},
        };
        const Date expiry = kToday + Period(1, Years);
        for (Size i = 0; i < cases.size(); ++i) {
            if (i != 0)
                std::cout << ",";
            const Case& c = cases[i];
            // Price at the true vol with the ORIGINAL process...
            auto process = makeProcess(c.spot, c.q, c.r, c.trueVol);
            auto option = makeOption(c.type, c.strike, expiry);
            option->setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));
            const Real target = option->NPV();

            // ... then recover it through the helper, on a cloned process.
            auto volQuote = ext::make_shared<SimpleQuote>(0.0);
            auto cloned = detail::ImpliedVolatilityHelper::clone(process, volQuote);
            auto ivEngine = ext::make_shared<AnalyticEuropeanEngine>(cloned);
            const Volatility recovered = detail::ImpliedVolatilityHelper::calculate(
                *option, *ivEngine, *volQuote, target, 1.0e-8, 200, 1.0e-7, 4.0);

            std::cout << "\n      {\"option_type\": " << static_cast<int>(c.type)
                      << ", \"strike\": " << c.strike << ", \"spot\": " << c.spot
                      << ", \"dividend_yield\": " << c.q << ", \"risk_free\": " << c.r
                      << ", \"true_vol\": " << c.trueVol
                      << ", \"expiry_serial\": " << expiry.serialNumber()
                      << ", \"target_value\": " << target
                      << ", \"implied_vol\": " << recovered
                      // The clone must preserve spot / q / r — if it did not,
                      // the recovered vol would not reproduce the target.
                      << ", \"cloned_spot\": " << cloned->stateVariable()->value()
                      << ", \"cloned_dividend_discount\": "
                      << cloned->dividendYield()->discount(expiry)
                      << ", \"cloned_riskfree_discount\": "
                      << cloned->riskFreeRate()->discount(expiry)
                      << ", \"cloned_vol_at_recovered\": "
                      << cloned->blackVolatility()->blackVol(expiry, c.strike) << "}";
        }
        std::cout << "\n    ]\n  }\n";
    }

    std::cout << "}\n";
    return 0;
}
