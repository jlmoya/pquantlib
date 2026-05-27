// L3-E mega-probe: forwards + FRAs.
//
// Captures reference values for the L3-E layer:
//
//   * ForwardTypePayoff: Long/Short evaluation at several prices.
//   * FxForward: settlement date + isExpired check + DiscountingFxForwardEngine
//     fairForwardRate + NPVs (source + target ccy terms).
//   * ForwardRateAgreement: forwardRate + amount + NPV for both
//     useIndexedCoupon=true and useIndexedCoupon=false branches.
//   * FraRateHelper (useIndexedCoupon=true): implied_quote roundtrip
//     vs the L2-C carry-over branch (LOOSE tier: index.fixing returns
//     forecast on a forwarding curve).
//
// C++ parity:
//   ql/instruments/forward.{hpp,cpp},
//   ql/instruments/fxforward.{hpp,cpp},
//   ql/instruments/forwardrateagreement.{hpp,cpp},
//   ql/pricingengines/forward/discountingfxforwardengine.{hpp,cpp},
//   ql/termstructures/yield/ratehelpers.{hpp,cpp} (FraRateHelper)
//   @ v1.42.1 (099987f0).

#include <ql/instruments/forward.hpp>
#include <ql/instruments/fxforward.hpp>
#include <ql/instruments/forwardrateagreement.hpp>
#include <ql/pricingengines/forward/discountingfxforwardengine.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/businessdayconvention.hpp>
#include <ql/settings.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/currencies/america.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    Date evalDate(17, January, 2024);  // Wed; avoids weekends + MLK day
    Settings::instance().evaluationDate() = evalDate;

    std::cout << "{\n";

    // -----------------------------------------------------------------
    // ForwardTypePayoff
    // -----------------------------------------------------------------
    {
        ForwardTypePayoff longPayoff(Position::Long, 100.0);
        ForwardTypePayoff shortPayoff(Position::Short, 100.0);
        std::cout << "  \"forward_type_payoff\": {\n";
        std::cout << "    \"name\": \"" << longPayoff.name() << "\",\n";
        std::cout << "    \"long_at_120\": " << longPayoff(120.0) << ",\n";
        std::cout << "    \"long_at_100\": " << longPayoff(100.0) << ",\n";
        std::cout << "    \"long_at_80\": " << longPayoff(80.0) << ",\n";
        std::cout << "    \"short_at_120\": " << shortPayoff(120.0) << ",\n";
        std::cout << "    \"short_at_100\": " << shortPayoff(100.0) << ",\n";
        std::cout << "    \"short_at_80\": " << shortPayoff(80.0) << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // FxForward + DiscountingFxForwardEngine
    //
    // 1M EUR (source) → 1.10 strike EUR/USD → 1.10M USD (target).
    //
    // EUR discount curve = FlatForward(3%)
    // USD discount curve = FlatForward(5%)
    // Spot FX = 1.10 (target/source = USD/EUR)
    // Maturity = eval + 365 days
    //
    // Fair rate per engine:  F = spotFx * df_target(T) / df_source(T)
    //   with df_source(t) = df_EUR(t)/df_EUR(settlement)
    //        df_target(t) = df_USD(t)/df_USD(settlement)
    // NPV in source ccy: paySource ⇒ -N_src*df_src + (N_tgt*df_tgt)/spotFx
    // -----------------------------------------------------------------
    {
        Date maturity = evalDate + 365;
        Real sourceNominal = 1'000'000.0;
        Real strikeRate = 1.10;
        Real targetNominal = sourceNominal * strikeRate;
        Real spotFx = 1.10;

        // Discount curves
        Handle<Quote> qEur(ext::make_shared<SimpleQuote>(0.03));
        Handle<Quote> qUsd(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> tsEur(
            ext::make_shared<FlatForward>(evalDate, qEur, Actual360()));
        Handle<YieldTermStructure> tsUsd(
            ext::make_shared<FlatForward>(evalDate, qUsd, Actual360()));

        Handle<Quote> hSpot(ext::make_shared<SimpleQuote>(spotFx));
        // Instrument: paySource=true (pay EUR, receive USD)
        FxForward fxFwd(sourceNominal, EURCurrency(),
                        targetNominal, USDCurrency(),
                        maturity, /*paySourceCurrency=*/true,
                        /*settlementDays=*/2, TARGET());
        auto engine = ext::make_shared<DiscountingFxForwardEngine>(
            tsEur, tsUsd, hSpot);
        fxFwd.setPricingEngine(engine);

        Date settlementDate = fxFwd.settlementDate();

        std::cout << "  \"fx_forward\": {\n";
        std::cout << "    \"eval_date_serial\": " << evalDate.serialNumber() << ",\n";
        std::cout << "    \"settlement_serial\": " << settlementDate.serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << maturity.serialNumber() << ",\n";
        std::cout << "    \"source_nominal\": " << sourceNominal << ",\n";
        std::cout << "    \"target_nominal\": " << targetNominal << ",\n";
        std::cout << "    \"strike_rate\": " << strikeRate << ",\n";
        std::cout << "    \"spot_fx\": " << spotFx << ",\n";
        std::cout << "    \"npv\": " << fxFwd.NPV() << ",\n";
        std::cout << "    \"fair_forward_rate\": " << fxFwd.fairForwardRate() << ",\n";
        std::cout << "    \"npv_source_ccy\": " << fxFwd.npvSourceCurrency() << ",\n";
        std::cout << "    \"npv_target_ccy\": " << fxFwd.npvTargetCurrency() << "\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // ForwardRateAgreement (both branches).
    //
    // 3M Euribor, 3M into the future, strike 5%, notional 1M EUR.
    //
    // forwardingTermStructure is FlatForward(5%) under Actual360
    // (continuous compounding by default).
    //   forward_rate = (d(valueDate) / d(maturityDate) - 1) / yearFraction
    // (par-coupon branch); indexed-coupon branch calls index.fixing.
    // -----------------------------------------------------------------
    {
        Handle<Quote> q(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, q, Actual360(), Continuous, Annual));
        auto index = ext::make_shared<Euribor3M>(ts);

        // Value date = today + 3M, advanced on TARGET with index conventions.
        Date valueDate = TARGET().advance(
            evalDate, 3 * Months, index->businessDayConvention(),
            index->endOfMonth());
        Date maturityDate = index->maturityDate(valueDate);

        Position::Type type = Position::Long;
        Rate strike = 0.05;
        Real notional = 1'000'000.0;

        // Indexed-coupon branch (constructor 1)
        ForwardRateAgreement fra_indexed(index, valueDate, type, strike,
                                          notional);
        Rate forwardRate_indexed = fra_indexed.forwardRate().rate();
        Real amount_indexed = fra_indexed.amount();
        Real npv_indexed = fra_indexed.NPV();

        // Par-coupon branch (constructor 2: explicit valueDate + maturityDate)
        ForwardRateAgreement fra_par(index, valueDate, maturityDate, type,
                                      strike, notional);
        Rate forwardRate_par = fra_par.forwardRate().rate();
        Real amount_par = fra_par.amount();
        Real npv_par = fra_par.NPV();

        std::cout << "  \"forward_rate_agreement\": {\n";
        std::cout << "    \"value_date_serial\": " << valueDate.serialNumber() << ",\n";
        std::cout << "    \"maturity_date_serial\": " << maturityDate.serialNumber() << ",\n";
        std::cout << "    \"strike\": " << strike << ",\n";
        std::cout << "    \"notional\": " << notional << ",\n";
        std::cout << "    \"indexed\": {\n";
        std::cout << "      \"forward_rate\": " << forwardRate_indexed << ",\n";
        std::cout << "      \"amount\": " << amount_indexed << ",\n";
        std::cout << "      \"npv\": " << npv_indexed << "\n";
        std::cout << "    },\n";
        std::cout << "    \"par\": {\n";
        std::cout << "      \"forward_rate\": " << forwardRate_par << ",\n";
        std::cout << "      \"amount\": " << amount_par << ",\n";
        std::cout << "      \"npv\": " << npv_par << "\n";
        std::cout << "    }\n";
        std::cout << "  },\n";
    }

    // -----------------------------------------------------------------
    // FraRateHelper.impliedQuote with useIndexedCoupon=TRUE.
    //
    // L2-C carry-over: discount-factor branch was already ported; here
    // we cross-check the indexed-coupon branch via
    // index.fixing(fixingDate, /*forecastTodaysFixing=*/true).
    // -----------------------------------------------------------------
    {
        Handle<Quote> q(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, q, Actual360(), Continuous, Annual));
        // 3M into 3M FRA. useIndexedCoupon=TRUE.
        auto helper = ext::make_shared<FraRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.05)),
            3, 6, 2, TARGET(), ModifiedFollowing, true, Actual360(),
            Pillar::LastRelevantDate, Date(), /*useIndexedCoupon=*/true);
        helper->setTermStructure(ts.currentLink().get());
        Real implied = helper->impliedQuote();
        std::cout << "  \"fra_rate_helper_indexed\": {\n";
        std::cout << "    \"earliest_serial\": "
                  << helper->earliestDate().serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": "
                  << helper->maturityDate().serialNumber() << ",\n";
        std::cout << "    \"latest_relevant_serial\": "
                  << helper->latestRelevantDate().serialNumber() << ",\n";
        std::cout << "    \"implied_quote\": " << implied << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
