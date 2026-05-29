// Phase 11 W4-B cluster probe: compound + chooser + extensible
// options + Kirk spread engine + AnalyticPdfHeston + continuous-Asian
// Levy/Vecer.
//
// Captures reference values for:
//
//   * CompoundOption + AnalyticCompoundOptionEngine — Wystup 2002
//     closed-form (uses Brent solver internally + bivariate-normal).
//
//   * SimpleChooserOption + AnalyticSimpleChooserEngine — Rubinstein
//     1991 closed-form.
//
//   * HolderExtensibleOption + AnalyticHolderExtensibleOptionEngine —
//     Haug 2007 closed-form (uses Newton-Raphson + bivariate-normal).
//
//   * WriterExtensibleOption + AnalyticWriterExtensibleOptionEngine —
//     Haug 2007 closed-form (bivariate-normal).
//
//   * Spread option via SpreadBlackScholesVanillaEngine + KirkEngine
//     under BasketOption(SpreadBasketPayoff(...)). Kirk 1995 closed-form
//     for spread options on two correlated futures.
//
//   * AnalyticPDFHestonEngine — Dragulescu-Yakovenko 2002 PDF-based
//     pricing for European vanilla under Heston. Cross-checked against
//     the AnalyticHestonEngine on the same setup (LOOSE: quadrature
//     noise).
//
//   * ContinuousArithmeticAsianLevyEngine — Levy 1992 closed-form
//     approximation.
//
//   * ContinuousArithmeticAsianVecerEngine — Vecer 2001 PDE.
//
// C++ parity:
//   ql/instruments/compoundoption.hpp
//   ql/pricingengines/exotic/analyticcompoundoptionengine.hpp
//   ql/instruments/simplechooseroption.hpp
//   ql/pricingengines/exotic/analyticsimplechooserengine.hpp
//   ql/instruments/holderextensibleoption.hpp
//   ql/pricingengines/exotic/analyticholderextensibleoptionengine.hpp
//   ql/instruments/writerextensibleoption.hpp
//   ql/pricingengines/exotic/analyticwriterextensibleoptionengine.hpp
//   ql/pricingengines/basket/kirkengine.hpp
//   ql/pricingengines/basket/spreadblackscholesvanillaengine.hpp
//   ql/pricingengines/vanilla/analyticpdfhestonengine.hpp
//   ql/pricingengines/asian/continuousarithmeticasianlevyengine.hpp
//   ql/experimental/exoticoptions/continuousarithmeticasianvecerengine.hpp
//   @ v1.42.1 (099987f0).

#include <ql/exercise.hpp>
#include <ql/experimental/exoticoptions/continuousarithmeticasianvecerengine.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/asianoption.hpp>
#include <ql/instruments/basketoption.hpp>
#include <ql/instruments/compoundoption.hpp>
#include <ql/instruments/holderextensibleoption.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/instruments/simplechooseroption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/instruments/writerextensibleoption.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/pricingengines/asian/continuousarithmeticasianlevyengine.hpp>
#include <ql/pricingengines/basket/kirkengine.hpp>
#include <ql/pricingengines/exotic/analyticcompoundoptionengine.hpp>
#include <ql/pricingengines/exotic/analyticholderextensibleoptionengine.hpp>
#include <ql/pricingengines/exotic/analyticsimplechooserengine.hpp>
#include <ql/pricingengines/exotic/analyticwriterextensibleoptionengine.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/pricingengines/vanilla/analyticpdfhestonengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    DayCounter dc = Actual365Fixed();
    Calendar cal = NullCalendar();
    Date ref(15, June, 2026);
    Settings::instance().evaluationDate() = ref;

    // ============================================================
    // 1) CompoundOption + AnalyticCompoundOptionEngine.
    //    Mother: 6-month call on a 1-year daughter call.
    //    Setup chosen so all closed-form branches exercise.
    // ============================================================
    {
        Real spot = 100.0;
        Real rf_rate = 0.05;
        Real q_rate = 0.02;
        Real vol = 0.30;
        Real mother_strike = 5.0;   // strike on the daughter option's value
        Real daughter_strike = 95.0;

        Date mother_exp = ref + 182;   // ~0.5y
        Date daughter_exp = ref + 365; // ~1.0y

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div = ext::make_shared<FlatForward>(ref, q_rate, dc);
        auto vol_ts = ext::make_shared<BlackConstantVol>(ref, cal, vol, dc);
        auto spot_q = ext::make_shared<SimpleQuote>(spot);

        auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(spot_q),
            Handle<YieldTermStructure>(div),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts));

        // Call-on-call.
        auto daughter_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, daughter_strike);
        auto mother_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, mother_strike);
        auto daughter_ex = ext::make_shared<EuropeanExercise>(daughter_exp);
        auto mother_ex = ext::make_shared<EuropeanExercise>(mother_exp);

        CompoundOption call_on_call(
            mother_payoff, mother_ex, daughter_payoff, daughter_ex);
        call_on_call.setPricingEngine(
            ext::make_shared<AnalyticCompoundOptionEngine>(process));

        // Put-on-call.
        auto mother_put_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, mother_strike);
        CompoundOption put_on_call(
            mother_put_payoff, mother_ex, daughter_payoff, daughter_ex);
        put_on_call.setPricingEngine(
            ext::make_shared<AnalyticCompoundOptionEngine>(process));

        // Call-on-put.
        auto daughter_put_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, daughter_strike);
        CompoundOption call_on_put(
            mother_payoff, mother_ex, daughter_put_payoff, daughter_ex);
        call_on_put.setPricingEngine(
            ext::make_shared<AnalyticCompoundOptionEngine>(process));

        // Put-on-put.
        CompoundOption put_on_put(
            mother_put_payoff, mother_ex, daughter_put_payoff, daughter_ex);
        put_on_put.setPricingEngine(
            ext::make_shared<AnalyticCompoundOptionEngine>(process));

        std::cout << "  \"compound\": {\n";
        std::cout << "    \"call_on_call\": " << call_on_call.NPV() << ",\n";
        std::cout << "    \"put_on_call\": " << put_on_call.NPV() << ",\n";
        std::cout << "    \"call_on_put\": " << call_on_put.NPV() << ",\n";
        std::cout << "    \"put_on_put\": " << put_on_put.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 2) SimpleChooserOption + AnalyticSimpleChooserEngine.
    //    Rubinstein 1991 — caller can choose call vs put at choosing
    //    date prior to a single expiry.
    // ============================================================
    {
        Real spot = 50.0;
        Real rf_rate = 0.08;
        Real q_rate = 0.05;
        Real vol = 0.25;
        Real strike = 50.0;

        Date choosing = ref + 91;  // ~0.25y
        Date expiry = ref + 182;   // ~0.5y

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div = ext::make_shared<FlatForward>(ref, q_rate, dc);
        auto vol_ts = ext::make_shared<BlackConstantVol>(ref, cal, vol, dc);
        auto spot_q = ext::make_shared<SimpleQuote>(spot);

        auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(spot_q),
            Handle<YieldTermStructure>(div),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts));

        SimpleChooserOption chooser(
            choosing, strike, ext::make_shared<EuropeanExercise>(expiry));
        chooser.setPricingEngine(
            ext::make_shared<AnalyticSimpleChooserEngine>(process));

        std::cout << "  \"simple_chooser\": {\n";
        std::cout << "    \"npv\": " << chooser.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 3) HolderExtensibleOption + AnalyticHolderExtensibleOptionEngine.
    //    Haug 2007 — option holder pays premium A to extend maturity
    //    from t1 to T2 with new strike X2.
    // ============================================================
    {
        Real spot = 100.0;
        Real rf_rate = 0.08;
        Real q_rate = 0.0;
        Real vol = 0.25;
        Real X1 = 100.0;        // original strike
        Real X2 = 105.0;        // extended strike
        Real premium = 1.0;     // premium paid to extend

        Date t1 = ref + 182;    // ~0.5y first expiry
        Date T2 = ref + 273;    // ~0.75y extended expiry

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div = ext::make_shared<FlatForward>(ref, q_rate, dc);
        auto vol_ts = ext::make_shared<BlackConstantVol>(ref, cal, vol, dc);
        auto spot_q = ext::make_shared<SimpleQuote>(spot);

        auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(spot_q),
            Handle<YieldTermStructure>(div),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts));

        // Call holder-extensible.
        auto call_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, X1);
        HolderExtensibleOption holder_call(
            Option::Call, premium, T2, X2,
            call_payoff, ext::make_shared<EuropeanExercise>(t1));
        holder_call.setPricingEngine(
            ext::make_shared<AnalyticHolderExtensibleOptionEngine>(process));

        // Put holder-extensible.
        auto put_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, X1);
        HolderExtensibleOption holder_put(
            Option::Put, premium, T2, X2,
            put_payoff, ext::make_shared<EuropeanExercise>(t1));
        holder_put.setPricingEngine(
            ext::make_shared<AnalyticHolderExtensibleOptionEngine>(process));

        std::cout << "  \"holder_extensible\": {\n";
        std::cout << "    \"call_npv\": " << holder_call.NPV() << ",\n";
        std::cout << "    \"put_npv\": " << holder_put.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 4) WriterExtensibleOption + AnalyticWriterExtensibleOptionEngine.
    //    Haug 2007 — if option is OTM at t1, writer extends to T2 with
    //    new strike X2.
    // ============================================================
    {
        Real spot = 80.0;
        Real rf_rate = 0.08;
        Real q_rate = 0.0;
        Real vol = 0.30;
        Real X1 = 90.0;
        Real X2 = 82.0;

        Date t1 = ref + 182;
        Date t2 = ref + 365;

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div = ext::make_shared<FlatForward>(ref, q_rate, dc);
        auto vol_ts = ext::make_shared<BlackConstantVol>(ref, cal, vol, dc);
        auto spot_q = ext::make_shared<SimpleQuote>(spot);

        auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(spot_q),
            Handle<YieldTermStructure>(div),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts));

        // Call writer-extensible.
        auto call_payoff1 = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, X1);
        auto call_payoff2 = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, X2);
        WriterExtensibleOption writer_call(
            call_payoff1, ext::make_shared<EuropeanExercise>(t1),
            call_payoff2, ext::make_shared<EuropeanExercise>(t2));
        writer_call.setPricingEngine(
            ext::make_shared<AnalyticWriterExtensibleOptionEngine>(process));

        // Put writer-extensible.
        auto put_payoff1 = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, X1);
        auto put_payoff2 = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, X2);
        WriterExtensibleOption writer_put(
            put_payoff1, ext::make_shared<EuropeanExercise>(t1),
            put_payoff2, ext::make_shared<EuropeanExercise>(t2));
        writer_put.setPricingEngine(
            ext::make_shared<AnalyticWriterExtensibleOptionEngine>(process));

        std::cout << "  \"writer_extensible\": {\n";
        std::cout << "    \"call_npv\": " << writer_call.NPV() << ",\n";
        std::cout << "    \"put_npv\": " << writer_put.NPV() << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 5) Kirk spread engine via BasketOption + SpreadBasketPayoff.
    //    Kirk 1995 — energy-market spread approximation.
    // ============================================================
    {
        Real S1 = 100.0;
        Real S2 = 90.0;
        Real rf_rate = 0.05;
        Real q1 = 0.0;
        Real q2 = 0.0;
        Real vol1 = 0.20;
        Real vol2 = 0.25;
        Real strike = 5.0;
        Real correlation = 0.5;

        Date expiry = ref + 365;

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div1 = ext::make_shared<FlatForward>(ref, q1, dc);
        auto div2 = ext::make_shared<FlatForward>(ref, q2, dc);
        auto vol_ts1 = ext::make_shared<BlackConstantVol>(ref, cal, vol1, dc);
        auto vol_ts2 = ext::make_shared<BlackConstantVol>(ref, cal, vol2, dc);

        auto process1 = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(S1)),
            Handle<YieldTermStructure>(div1),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts1));
        auto process2 = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(S2)),
            Handle<YieldTermStructure>(div2),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts2));

        auto base_call = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, strike);
        auto spread_payoff = ext::make_shared<SpreadBasketPayoff>(base_call);

        BasketOption spread_opt(
            spread_payoff, ext::make_shared<EuropeanExercise>(expiry));
        spread_opt.setPricingEngine(
            ext::make_shared<KirkEngine>(process1, process2, correlation));
        Real call_npv = spread_opt.NPV();

        // Put.
        auto base_put = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, strike);
        auto spread_payoff_put = ext::make_shared<SpreadBasketPayoff>(base_put);
        BasketOption spread_opt_put(
            spread_payoff_put,
            ext::make_shared<EuropeanExercise>(expiry));
        spread_opt_put.setPricingEngine(
            ext::make_shared<KirkEngine>(process1, process2, correlation));
        Real put_npv = spread_opt_put.NPV();

        std::cout << "  \"kirk_spread\": {\n";
        std::cout << "    \"call_npv\": " << call_npv << ",\n";
        std::cout << "    \"put_npv\": " << put_npv << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 6) AnalyticPDFHestonEngine vs AnalyticHestonEngine.
    //    PDF engine should agree with Gauss-Laguerre engine within
    //    quadrature noise.
    // ============================================================
    {
        Real spot = 100.0;
        Real rf_rate = 0.05;
        Real q_rate = 0.02;
        Real v0 = 0.04;
        Real kappa = 1.0;
        Real theta = 0.04;
        Real sigma = 0.3;
        Real rho = -0.5;

        Date expiry = ref + 365;

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div = ext::make_shared<FlatForward>(ref, q_rate, dc);

        auto process = ext::make_shared<HestonProcess>(
            Handle<YieldTermStructure>(rf),
            Handle<YieldTermStructure>(div),
            Handle<Quote>(ext::make_shared<SimpleQuote>(spot)),
            v0, kappa, theta, sigma, rho);

        auto model = ext::make_shared<HestonModel>(process);

        // ATM call.
        auto payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, 100.0);
        VanillaOption opt(payoff, ext::make_shared<EuropeanExercise>(expiry));

        opt.setPricingEngine(
            ext::make_shared<AnalyticPDFHestonEngine>(model, 1e-8, 100000));
        Real pdf_npv = opt.NPV();

        opt.setPricingEngine(
            ext::make_shared<AnalyticHestonEngine>(model));
        Real classic_npv = opt.NPV();

        std::cout << "  \"analytic_pdf_heston\": {\n";
        std::cout << "    \"pdf_npv\": " << pdf_npv << ",\n";
        std::cout << "    \"classic_heston_npv\": " << classic_npv << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 7) ContinuousArithmeticAsianLevyEngine — Levy 1992 approximation.
    //    Use a seasoned setup with start_date < ref.
    // ============================================================
    {
        Real spot = 100.0;
        Real rf_rate = 0.05;
        Real q_rate = 0.02;
        Real vol = 0.20;
        Real strike = 100.0;
        Real current_average = 100.0;

        Date start = ref - 91;   // ~0.25y in the past
        Date expiry = ref + 274; // ~0.75y to maturity

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div = ext::make_shared<FlatForward>(ref, q_rate, dc);
        auto vol_ts = ext::make_shared<BlackConstantVol>(ref, cal, vol, dc);

        auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(spot)),
            Handle<YieldTermStructure>(div),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts));

        Handle<Quote> avg_quote(ext::make_shared<SimpleQuote>(current_average));

        // Use the start-date constructor (deprecated but still works).
        QL_DEPRECATED_DISABLE_WARNING
        auto levy_engine =
            ext::make_shared<ContinuousArithmeticAsianLevyEngine>(
                process, avg_quote, start);
        QL_DEPRECATED_ENABLE_WARNING

        auto call_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, strike);
        ContinuousAveragingAsianOption levy_call(
            Average::Arithmetic, call_payoff,
            ext::make_shared<EuropeanExercise>(expiry));
        levy_call.setPricingEngine(levy_engine);
        Real call_npv = levy_call.NPV();

        auto put_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, strike);
        ContinuousAveragingAsianOption levy_put(
            Average::Arithmetic, put_payoff,
            ext::make_shared<EuropeanExercise>(expiry));
        levy_put.setPricingEngine(levy_engine);
        Real put_npv = levy_put.NPV();

        std::cout << "  \"asian_levy\": {\n";
        std::cout << "    \"call_npv\": " << call_npv << ",\n";
        std::cout << "    \"put_npv\": " << put_npv << "\n";
        std::cout << "  },\n";
    }

    // ============================================================
    // 8) ContinuousArithmeticAsianVecerEngine — Vecer 2001 PDE.
    //    Unseasoned setup (start_date = ref).
    // ============================================================
    {
        Real spot = 100.0;
        Real rf_rate = 0.05;
        Real q_rate = 0.02;
        Real vol = 0.20;
        Real strike = 100.0;

        Date start = ref;
        Date expiry = ref + 365;

        auto rf = ext::make_shared<FlatForward>(ref, rf_rate, dc);
        auto div = ext::make_shared<FlatForward>(ref, q_rate, dc);
        auto vol_ts = ext::make_shared<BlackConstantVol>(ref, cal, vol, dc);

        auto process = ext::make_shared<GeneralizedBlackScholesProcess>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(spot)),
            Handle<YieldTermStructure>(div),
            Handle<YieldTermStructure>(rf),
            Handle<BlackVolTermStructure>(vol_ts));

        Handle<Quote> avg_quote(ext::make_shared<SimpleQuote>(100.0));

        // Vecer PDE — call.
        auto call_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Call, strike);
        ContinuousAveragingAsianOption vecer_call(
            Average::Arithmetic, call_payoff,
            ext::make_shared<EuropeanExercise>(expiry));
        vecer_call.setPricingEngine(
            ext::make_shared<ContinuousArithmeticAsianVecerEngine>(
                process, avg_quote, start, 100, 100, -1.0, 1.0));
        Real call_npv = vecer_call.NPV();

        // Vecer PDE — put.
        auto put_payoff = ext::make_shared<PlainVanillaPayoff>(
            Option::Put, strike);
        ContinuousAveragingAsianOption vecer_put(
            Average::Arithmetic, put_payoff,
            ext::make_shared<EuropeanExercise>(expiry));
        vecer_put.setPricingEngine(
            ext::make_shared<ContinuousArithmeticAsianVecerEngine>(
                process, avg_quote, start, 100, 100, -1.0, 1.0));
        Real put_npv = vecer_put.NPV();

        std::cout << "  \"asian_vecer\": {\n";
        std::cout << "    \"call_npv\": " << call_npv << ",\n";
        std::cout << "    \"put_npv\": " << put_npv << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
