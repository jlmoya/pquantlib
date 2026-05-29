// Phase 11 W3-C cluster probe: hazard-rate specialty curves + CDO/Basket cluster.
//
// Captures reference values for:
//   * FactorSpreadedHazardRateCurve: multiplicative-spread on hazard rate.
//   * SpreadedHazardRateCurve: additive-spread on hazard rate.
//   * Basket: size, basketNotional, attachment/detachment amounts, trancheNotional.
//   * SyntheticCDO + IntegralCDOEngine + GaussianLHPLossModel: full NPV, premiumValue,
//     protectionValue on a small 5-issuer basket.
//   * SyntheticCDO + MidPointCDOEngine on the same inputs (convergence check).
//
// C++ parity:
//   ql/experimental/credit/factorspreadedhazardratecurve.hpp
//   ql/experimental/credit/spreadedhazardratecurve.hpp
//   ql/experimental/credit/basket.{hpp,cpp}
//   ql/experimental/credit/syntheticcdo.{hpp,cpp}
//   ql/experimental/credit/integralcdoengine.{hpp,cpp}
//   ql/experimental/credit/midpointcdoengine.{hpp,cpp}
//   ql/experimental/credit/gaussianlhplossmodel.hpp
//   @ v1.42.1 (099987f0).

#include <ql/currencies/america.hpp>
#include <ql/default.hpp>
#include <ql/experimental/credit/basket.hpp>
#include <ql/experimental/credit/defaultprobabilitykey.hpp>
#include <ql/experimental/credit/defaulttype.hpp>
#include <ql/experimental/credit/factorspreadedhazardratecurve.hpp>
#include <ql/experimental/credit/gaussianlhplossmodel.hpp>
#include <ql/experimental/credit/integralcdoengine.hpp>
#include <ql/experimental/credit/issuer.hpp>
#include <ql/experimental/credit/midpointcdoengine.hpp>
#include <ql/experimental/credit/pool.hpp>
#include <ql/experimental/credit/spreadedhazardratecurve.hpp>
#include <ql/experimental/credit/syntheticcdo.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/claim.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/credit/flathazardrate.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/schedule.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ============================================================
    // Common reference: today = 15-Jan-2024, 2% flat hazard, ACT/365F.
    // ============================================================
    Date today(15, January, 2024);
    Settings::instance().evaluationDate() = today;
    DayCounter dc = Actual365Fixed();
    ext::shared_ptr<SimpleQuote> baseQuote(new SimpleQuote(0.02));
    ext::shared_ptr<DefaultProbabilityTermStructure> base(
        new FlatHazardRate(today, Handle<Quote>(baseQuote), dc));

    // 1) FactorSpreadedHazardRateCurve: hazardRate(t) = base * (1 + factor)
    {
        ext::shared_ptr<SimpleQuote> factor(new SimpleQuote(1.0)); // 100% = doubles
        Handle<DefaultProbabilityTermStructure> baseH(base);
        Handle<Quote> factorH(factor);
        FactorSpreadedHazardRateCurve curveA(baseH, factorH);
        std::cout << "  \"factor_spreaded_hazard_rate_curve\": {\n";
        std::cout << "    \"h_t1\": " << curveA.hazardRate(1.0, true) << ",\n";
        std::cout << "    \"h_t5\": " << curveA.hazardRate(5.0, true) << ",\n";
        std::cout << "    \"survival_t1\": " << curveA.survivalProbability(1.0, true) << ",\n";
        std::cout << "    \"survival_t5\": " << curveA.survivalProbability(5.0, true) << "\n";
        std::cout << "  },\n";
    }

    // 2) SpreadedHazardRateCurve: hazardRate(t) = base + spread
    {
        ext::shared_ptr<SimpleQuote> spread(new SimpleQuote(0.01));
        Handle<DefaultProbabilityTermStructure> baseH(base);
        Handle<Quote> spreadH(spread);
        SpreadedHazardRateCurve curveB(baseH, spreadH);
        std::cout << "  \"spreaded_hazard_rate_curve\": {\n";
        std::cout << "    \"h_t1\": " << curveB.hazardRate(1.0, true) << ",\n";
        std::cout << "    \"h_t5\": " << curveB.hazardRate(5.0, true) << ",\n";
        std::cout << "    \"survival_t1\": " << curveB.survivalProbability(1.0, true) << ",\n";
        std::cout << "    \"survival_t5\": " << curveB.survivalProbability(5.0, true) << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) Basket: 5 issuers @ 1MM notional each, attach=0.0, detach=0.1.
    // ============================================================
    USDCurrency usd;
    DefaultType dt(AtomicDefault::Bankruptcy, Restructuring::NoRestructuring);
    DefaultProbKey contractKey(
        std::vector<ext::shared_ptr<DefaultType>>{ext::make_shared<DefaultType>(dt)},
        usd,
        SeniorUnSec);

    std::vector<std::string> names = {"N0", "N1", "N2", "N3", "N4"};
    std::vector<Real> notionals(5, 1.0e6);

    ext::shared_ptr<Pool> pool(new Pool());
    {
        Handle<DefaultProbabilityTermStructure> baseH(base);
        std::vector<Issuer::key_curve_pair> probs{{contractKey, baseH}};
        Issuer issuer(probs);
        for (const auto& n : names) {
            pool->add(n, issuer, contractKey);
        }
    }

    Date inceptionDate = today;
    ext::shared_ptr<Basket> basket(new Basket(
        inceptionDate, names, notionals, pool, 0.0, 0.1,
        ext::shared_ptr<Claim>(new FaceValueClaim())));

    std::cout << "  \"basket\": {\n";
    std::cout << "    \"size\": " << basket->size() << ",\n";
    std::cout << "    \"basket_notional\": " << basket->basketNotional() << ",\n";
    std::cout << "    \"attachment_amount\": " << basket->attachmentAmount() << ",\n";
    std::cout << "    \"detachment_amount\": " << basket->detachmentAmount() << ",\n";
    std::cout << "    \"tranche_notional\": " << basket->trancheNotional() << ",\n";
    std::cout << "    \"attachment_ratio\": " << basket->attachmentRatio() << ",\n";
    std::cout << "    \"detachment_ratio\": " << basket->detachmentRatio() << "\n";
    std::cout << "  },\n";

    // ============================================================
    // 4) SyntheticCDO + IntegralCDOEngine + GaussianLHPLossModel.
    //    5-year schedule, quarterly payments. correlation=0.3, recovery=0.4.
    // ============================================================
    Calendar calendar = TARGET();
    Date maturity = today + 5 * Years;
    Schedule schedule(
        today, maturity, Period(Quarterly),
        calendar, Following, Following,
        DateGeneration::Forward, false);

    ext::shared_ptr<SyntheticCDO> cdo(new SyntheticCDO(
        basket, Protection::Seller, schedule,
        /* upfront= */ 0.0,
        /* running= */ 0.05,
        dc, Following));

    // Flat 3% discount curve.
    ext::shared_ptr<YieldTermStructure> discountCurve(
        new FlatForward(today, 0.03, dc));

    // Gaussian LHP loss model with 30% correlation, 40% recovery on all names.
    std::vector<Real> recoveries(5, 0.4);
    ext::shared_ptr<DefaultLossModel> lossModel(
        new GaussianLHPLossModel(0.3, recoveries));
    basket->setLossModel(lossModel);

    {
        ext::shared_ptr<PricingEngine> engine(
            new IntegralCDOEngine(
                Handle<YieldTermStructure>(discountCurve), 3 * Months));
        cdo->setPricingEngine(engine);

        std::cout << "  \"synthetic_cdo_integral\": {\n";
        std::cout << "    \"npv\": " << cdo->NPV() << ",\n";
        std::cout << "    \"premium_value\": " << cdo->premiumValue() << ",\n";
        std::cout << "    \"protection_value\": " << cdo->protectionValue() << ",\n";
        std::cout << "    \"fair_premium\": " << cdo->fairPremium() << ",\n";
        std::cout << "    \"remaining_notional\": " << cdo->remainingNotional() << ",\n";
        std::cout << "    \"leverage_factor\": " << cdo->leverageFactor() << "\n";
        std::cout << "  },\n";
    }

    {
        ext::shared_ptr<PricingEngine> engine(
            new MidPointCDOEngine(Handle<YieldTermStructure>(discountCurve)));
        cdo->setPricingEngine(engine);

        std::cout << "  \"synthetic_cdo_midpoint\": {\n";
        std::cout << "    \"npv\": " << cdo->NPV() << ",\n";
        std::cout << "    \"premium_value\": " << cdo->premiumValue() << ",\n";
        std::cout << "    \"protection_value\": " << cdo->protectionValue() << ",\n";
        std::cout << "    \"fair_premium\": " << cdo->fairPremium() << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
