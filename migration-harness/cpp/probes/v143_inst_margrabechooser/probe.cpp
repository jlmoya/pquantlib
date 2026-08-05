// migration-harness/cpp/probes/v143_inst_margrabechooser/probe.cpp
//
// Reference values for the "Margrabe + complex chooser + swing forward payoff"
// cluster of C++ QuantLib v1.43:
//
//   * MargrabeOption            (ql/instruments/margrabeoption.{hpp,cpp})
//     + AnalyticEuropeanMargrabeEngine
//     + AnalyticAmericanMargrabeEngine
//   * ComplexChooserOption      (ql/instruments/complexchooseroption.{hpp,cpp})
//     + AnalyticComplexChooserEngine
//   * VanillaForwardPayoff      (ql/instruments/vanillaswingoption.{hpp,cpp})
//
// WHAT IS PINNED AND WHY
//
// 1. MargrabeOption. Every constructor / engine argument must be shown to move
//    an output, otherwise a dropped argument is invisible (the failure mode this
//    port has already been bitten by). So the grid varies, one at a time from a
//    common base case: Q1, Q2, the two dividend yields, the two volatilities,
//    the correlation (including rho = 0 and rho < 0), the two spots and the
//    maturity. For every case we pin NPV plus the *full* result surface the
//    instrument exposes -- delta1/delta2/gamma1/gamma2 from MargrabeOption and
//    delta/gamma/theta/vega/rho/dividendRho from MultiAssetOption.
//
//    Several of those are deliberately NOT filled by the engines:
//      - AnalyticEuropeanMargrabeEngine fills value, delta1, delta2, gamma1,
//        gamma2, theta, rho -- and leaves delta, gamma, vega, dividendRho unset,
//        so those accessors throw.
//      - AnalyticAmericanMargrabeEngine fills value ONLY, so every greek throws.
//    Those throwing accessors are pinned as {"raises": true}: they are part of
//    the observable contract, and a port that helpfully returns 0.0 instead of
//    raising is wrong.
//
// 2. ComplexChooserOption. Call strike, put strike, choosing date, call expiry
//    and put expiry are varied INDEPENDENTLY, so a port that swaps the two
//    strikes, swaps the two expiries, or reads the choosing date from the wrong
//    field fails at least one case. The engine fills value only; the greeks are
//    pinned as raising for the same reason as above.
//
// 3. VanillaForwardPayoff. name(), description() and operator()(price) across a
//    price range for both Call and Put -- the payoff is linear and unclamped
//    (price - strike / strike - price), which is exactly the detail a port
//    copied from PlainVanillaPayoff would get wrong by clamping at zero.
//
// 4. The QL_REQUIRE failure branches of MargrabeOption::arguments::validate,
//    ComplexChooserOption::arguments::validate and both Margrabe engines'
//    exercise-type checks, pinned as {"raises": true}.
//
// All values are analytic closed forms. The complex chooser routes through
// BivariateCumulativeNormalDistributionDr78 (Drezner 1978, ~6 decimal places),
// which the Python port does not reproduce bit-for-bit -- see the test file for
// the tolerance derivation.
//
// C++ parity:
//   ql/instruments/margrabeoption.{hpp,cpp}
//   ql/instruments/complexchooseroption.{hpp,cpp}
//   ql/instruments/vanillaswingoption.{hpp,cpp}
//   ql/pricingengines/exotic/analyticeuropeanmargrabeengine.{hpp,cpp}
//   ql/pricingengines/exotic/analyticamericanmargrabeengine.{hpp,cpp}
//   ql/pricingengines/exotic/analyticcomplexchooserengine.{hpp,cpp}
//   @ v1.43.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/inst/margrabechooser.json.

#include <ql/exercise.hpp>
#include <ql/handle.hpp>
#include <ql/instruments/complexchooseroption.hpp>
#include <ql/instruments/margrabeoption.hpp>
#include <ql/instruments/vanillaswingoption.hpp>
#include <ql/pricingengines/exotic/analyticamericanmargrabeengine.hpp>
#include <ql/pricingengines/exotic/analyticcomplexchooserengine.hpp>
#include <ql/pricingengines/exotic/analyticeuropeanmargrabeengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

const DayCounter kDc = Actual365Fixed();
const Date kToday(15, January, 2024);

ext::shared_ptr<GeneralizedBlackScholesProcess>
makeProcess(Real spot, Rate r, Rate q, Volatility sigma) {
    Handle<Quote> spotH(ext::make_shared<SimpleQuote>(spot));
    Handle<YieldTermStructure> rH(ext::make_shared<FlatForward>(kToday, r, kDc));
    Handle<YieldTermStructure> qH(ext::make_shared<FlatForward>(kToday, q, kDc));
    Handle<BlackVolTermStructure> vH(
        ext::make_shared<BlackConstantVol>(kToday, NullCalendar(), sigma, kDc));
    return ext::make_shared<GeneralizedBlackScholesProcess>(spotH, qH, rH, vH);
}

// Emit `"<key>": <double>` or `"<key>": {"raises": true}` when the accessor
// throws. Both are meaningful contract, so both are recorded.
void emitMaybe(const std::string& indent,
               const char* key,
               const std::function<Real()>& f,
               bool comma) {
    std::cout << indent << "\"" << key << "\": ";
    try {
        std::cout << f();
    } catch (const std::exception&) {
        std::cout << "{\"raises\": true}";
    }
    std::cout << (comma ? "," : "") << "\n";
}

void emitRaises(const std::string& indent,
                const char* key,
                const std::function<void()>& f,
                bool comma) {
    bool raised = false;
    try {
        f();
    } catch (const std::exception&) {
        raised = true;
    }
    std::cout << indent << "\"" << key << "\": {\"raises\": " << (raised ? "true" : "false")
              << "}" << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------------- Margrabe --

struct MargrabeCase {
    const char* key;
    Integer q1;      // quantity of asset 1
    Integer q2;      // quantity of asset 2
    Real s1, s2;     // spots
    Rate div1, div2; // dividend yields
    Volatility v1, v2;
    Real rho;
    Integer months; // maturity in months from kToday
};

void emitMargrabeResults(const std::string& indent, const MargrabeOption& opt) {
    emitMaybe(indent, "value", [&] { return opt.NPV(); }, true);
    emitMaybe(indent, "delta1", [&] { return opt.delta1(); }, true);
    emitMaybe(indent, "delta2", [&] { return opt.delta2(); }, true);
    emitMaybe(indent, "gamma1", [&] { return opt.gamma1(); }, true);
    emitMaybe(indent, "gamma2", [&] { return opt.gamma2(); }, true);
    emitMaybe(indent, "delta", [&] { return opt.delta(); }, true);
    emitMaybe(indent, "gamma", [&] { return opt.gamma(); }, true);
    emitMaybe(indent, "theta", [&] { return opt.theta(); }, true);
    emitMaybe(indent, "vega", [&] { return opt.vega(); }, true);
    emitMaybe(indent, "rho", [&] { return opt.rho(); }, true);
    emitMaybe(indent, "dividend_rho", [&] { return opt.dividendRho(); }, false);
}

void emitMargrabeCase(const MargrabeCase& c, bool comma) {
    const Date maturity = kToday + Period(c.months, Months);
    auto p1 = makeProcess(c.s1, 0.05, c.div1, c.v1);
    auto p2 = makeProcess(c.s2, 0.05, c.div2, c.v2);

    std::cout << "    \"" << c.key << "\": {\n";
    std::cout << "      \"inputs\": {\n"
              << "        \"Q1\": " << c.q1 << ",\n"
              << "        \"Q2\": " << c.q2 << ",\n"
              << "        \"s1\": " << c.s1 << ",\n"
              << "        \"s2\": " << c.s2 << ",\n"
              << "        \"r\": " << 0.05 << ",\n"
              << "        \"q1\": " << c.div1 << ",\n"
              << "        \"q2\": " << c.div2 << ",\n"
              << "        \"v1\": " << c.v1 << ",\n"
              << "        \"v2\": " << c.v2 << ",\n"
              << "        \"rho\": " << c.rho << ",\n"
              << "        \"maturity_serial\": " << maturity.serialNumber() << "\n"
              << "      },\n";

    {
        MargrabeOption opt(c.q1, c.q2, ext::make_shared<EuropeanExercise>(maturity));
        opt.setPricingEngine(ext::make_shared<AnalyticEuropeanMargrabeEngine>(p1, p2, c.rho));
        std::cout << "      \"european\": {\n";
        emitMargrabeResults("        ", opt);
        std::cout << "      },\n";
    }
    {
        MargrabeOption opt(c.q1, c.q2,
                           ext::make_shared<AmericanExercise>(kToday, maturity));
        opt.setPricingEngine(ext::make_shared<AnalyticAmericanMargrabeEngine>(p1, p2, c.rho));
        std::cout << "      \"american\": {\n";
        emitMargrabeResults("        ", opt);
        std::cout << "      }\n";
    }
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

// ---------------------------------------------------------- Complex chooser --

struct ChooserCase {
    const char* key;
    Real spot;
    Rate r, q;
    Volatility vol;
    Real strikeCall, strikePut;
    Integer chooseMonths, callMonths, putMonths;
};

void emitChooserCase(const ChooserCase& c, bool comma) {
    const Date choosing = kToday + Period(c.chooseMonths, Months);
    const Date callExp = kToday + Period(c.callMonths, Months);
    const Date putExp = kToday + Period(c.putMonths, Months);

    auto process = makeProcess(c.spot, c.r, c.q, c.vol);
    ComplexChooserOption opt(choosing, c.strikeCall, c.strikePut,
                             ext::make_shared<EuropeanExercise>(callExp),
                             ext::make_shared<EuropeanExercise>(putExp));
    opt.setPricingEngine(ext::make_shared<AnalyticComplexChooserEngine>(process));

    std::cout << "    \"" << c.key << "\": {\n"
              << "      \"inputs\": {\n"
              << "        \"spot\": " << c.spot << ",\n"
              << "        \"r\": " << c.r << ",\n"
              << "        \"q\": " << c.q << ",\n"
              << "        \"vol\": " << c.vol << ",\n"
              << "        \"strike_call\": " << c.strikeCall << ",\n"
              << "        \"strike_put\": " << c.strikePut << ",\n"
              << "        \"choosing_serial\": " << choosing.serialNumber() << ",\n"
              << "        \"call_expiry_serial\": " << callExp.serialNumber() << ",\n"
              << "        \"put_expiry_serial\": " << putExp.serialNumber() << "\n"
              << "      },\n";
    emitMaybe("      ", "value", [&] { return opt.NPV(); }, true);
    emitMaybe("      ", "delta", [&] { return opt.delta(); }, true);
    emitMaybe("      ", "gamma", [&] { return opt.gamma(); }, true);
    emitMaybe("      ", "theta", [&] { return opt.theta(); }, true);
    emitMaybe("      ", "vega", [&] { return opt.vega(); }, true);
    emitMaybe("      ", "rho", [&] { return opt.rho(); }, true);
    emitMaybe("      ", "dividend_rho", [&] { return opt.dividendRho(); }, false);
    std::cout << "    }" << (comma ? "," : "") << "\n";
}

// -------------------------------------------------------- Forward payoff -----

void emitForwardPayoff(const char* key, Option::Type type, Real strike,
                       const std::vector<Real>& prices, bool comma) {
    VanillaForwardPayoff payoff(type, strike);
    std::cout << "    \"" << key << "\": {\n"
              << "      \"strike\": " << strike << ",\n"
              << "      \"name\": \"" << payoff.name() << "\",\n"
              << "      \"description\": \"" << payoff.description() << "\",\n"
              << "      \"values\": [";
    for (Size i = 0; i < prices.size(); ++i) {
        std::cout << (i != 0U ? ", " : "") << payoff(prices[i]);
    }
    std::cout << "]\n    }" << (comma ? "," : "") << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    Settings::instance().evaluationDate() = kToday;

    std::cout << "{\n";

    std::cout << "  \"meta\": {\n"
              << "    \"today_serial\": " << kToday.serialNumber() << ",\n"
              << "    \"day_counter\": \"" << kDc.name() << "\"\n"
              << "  },\n";

    // -- Margrabe grid: base case, then one perturbation per argument. --------
    const std::vector<MargrabeCase> margrabe = {
        //  key              Q1  Q2   s1     s2    q1     q2     v1    v2   rho   m
        {"base", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.30, 12},
        {"quantity1_2", 2, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.30, 12},
        {"quantity2_3", 1, 3, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.30, 12},
        {"quantity_ratio_5_4", 5, 4, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.30, 12},
        {"div1_high", 1, 1, 100.0, 105.0, 0.09, 0.03, 0.20, 0.25, 0.30, 12},
        {"div2_zero", 1, 1, 100.0, 105.0, 0.02, 0.00, 0.20, 0.25, 0.30, 12},
        {"vol1_high", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.45, 0.25, 0.30, 12},
        {"vol2_low", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.10, 0.30, 12},
        {"rho_zero", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.00, 12},
        {"rho_negative", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, -0.55, 12},
        {"rho_high", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.85, 12},
        {"spot1_low", 1, 1, 80.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.30, 12},
        {"spot2_low", 1, 1, 100.0, 85.0, 0.02, 0.03, 0.20, 0.25, 0.30, 12},
        {"maturity_2y", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.30, 24},
        {"maturity_3m", 1, 1, 100.0, 105.0, 0.02, 0.03, 0.20, 0.25, 0.30, 3},
        // div1 <= 0 and div1 <= div2 sends the American engine down the
        // European branch of the Bjerksund-Stensland approximation.
        {"american_european_branch", 1, 1, 100.0, 105.0, 0.00, 0.03, 0.20, 0.25, 0.30, 12},
        // Deep in the money with a large asset-1 yield: early exercise is
        // optimal, so the American engine hits its immediate-exercise branch.
        {"american_immediate_branch", 1, 1, 200.0, 60.0, 0.30, 0.01, 0.15, 0.15, 0.90, 12},
    };
    std::cout << "  \"margrabe\": {\n";
    for (Size i = 0; i < margrabe.size(); ++i) {
        emitMargrabeCase(margrabe[i], i + 1 != margrabe.size());
    }
    std::cout << "  },\n";

    // -- Margrabe failure branches -------------------------------------------
    {
        const Date maturity = kToday + Period(12, Months);
        auto p1 = makeProcess(100.0, 0.05, 0.02, 0.20);
        auto p2 = makeProcess(105.0, 0.05, 0.03, 0.25);
        std::cout << "  \"margrabe_errors\": {\n";
        emitRaises("    ", "q1_zero", [&] {
            MargrabeOption opt(0, 1, ext::make_shared<EuropeanExercise>(maturity));
            opt.setPricingEngine(ext::make_shared<AnalyticEuropeanMargrabeEngine>(p1, p2, 0.3));
            opt.NPV();
        }, true);
        emitRaises("    ", "q2_negative", [&] {
            MargrabeOption opt(1, -2, ext::make_shared<EuropeanExercise>(maturity));
            opt.setPricingEngine(ext::make_shared<AnalyticEuropeanMargrabeEngine>(p1, p2, 0.3));
            opt.NPV();
        }, true);
        emitRaises("    ", "european_engine_american_exercise", [&] {
            MargrabeOption opt(1, 1, ext::make_shared<AmericanExercise>(kToday, maturity));
            opt.setPricingEngine(ext::make_shared<AnalyticEuropeanMargrabeEngine>(p1, p2, 0.3));
            opt.NPV();
        }, true);
        emitRaises("    ", "american_engine_european_exercise", [&] {
            MargrabeOption opt(1, 1, ext::make_shared<EuropeanExercise>(maturity));
            opt.setPricingEngine(ext::make_shared<AnalyticAmericanMargrabeEngine>(p1, p2, 0.3));
            opt.NPV();
        }, false);
        std::cout << "  },\n";
    }

    // -- Complex chooser grid: one perturbation per argument. ----------------
    // NOTE on the date grid: AnalyticComplexChooserEngine::bsCalculator prices
    // its inner vanillas at t = callMaturity - 2*T (resp. putMaturity - 2*T),
    // so a maturity of exactly twice the choosing time degenerates to t = 0 and
    // the engine throws. Every case below keeps both maturities strictly beyond
    // twice the choosing date.
    const std::vector<ChooserCase> chooser = {
        //  key                 spot    r     q     vol   Xc    Xp    tc  tCall tPut
        {"base", 50.0, 0.10, 0.05, 0.35, 55.0, 48.0, 3, 9, 10},
        {"strike_call_high", 50.0, 0.10, 0.05, 0.35, 62.0, 48.0, 3, 9, 10},
        {"strike_put_high", 50.0, 0.10, 0.05, 0.35, 55.0, 53.0, 3, 9, 10},
        {"choosing_later", 50.0, 0.10, 0.05, 0.35, 55.0, 48.0, 4, 9, 10},
        {"call_expiry_later", 50.0, 0.10, 0.05, 0.35, 55.0, 48.0, 3, 12, 10},
        {"put_expiry_later", 50.0, 0.10, 0.05, 0.35, 55.0, 48.0, 3, 9, 15},
        // Put expiry BEFORE call expiry: a port that swaps the two exercises
        // gets a different number here than in "put_expiry_later".
        {"put_expiry_before_call", 50.0, 0.10, 0.05, 0.35, 55.0, 48.0, 2, 11, 7},
        {"vol_low", 50.0, 0.10, 0.05, 0.20, 55.0, 48.0, 3, 9, 10},
        {"dividend_zero", 50.0, 0.10, 0.00, 0.35, 55.0, 48.0, 3, 9, 10},
        {"rate_low", 50.0, 0.02, 0.05, 0.35, 55.0, 48.0, 3, 9, 10},
        {"spot_high", 62.0, 0.10, 0.05, 0.35, 55.0, 48.0, 3, 9, 10},
        // Strikes swapped relative to base: pins that Xc and Xp are not
        // interchangeable.
        {"strikes_swapped", 50.0, 0.10, 0.05, 0.35, 48.0, 55.0, 3, 9, 10},
    };
    std::cout << "  \"complex_chooser\": {\n";
    for (Size i = 0; i < chooser.size(); ++i) {
        emitChooserCase(chooser[i], i + 1 != chooser.size());
    }
    std::cout << "  },\n";

    // -- Complex chooser failure branches ------------------------------------
    {
        auto process = makeProcess(50.0, 0.10, 0.05, 0.35);
        const Date callExp = kToday + Period(9, Months);
        const Date putExp = kToday + Period(10, Months);
        std::cout << "  \"complex_chooser_errors\": {\n";
        emitRaises("    ", "null_choosing_date", [&] {
            ComplexChooserOption opt(Date(), 55.0, 48.0,
                                     ext::make_shared<EuropeanExercise>(callExp),
                                     ext::make_shared<EuropeanExercise>(putExp));
            opt.setPricingEngine(ext::make_shared<AnalyticComplexChooserEngine>(process));
            opt.NPV();
        }, true);
        emitRaises("    ", "choosing_after_call_expiry", [&] {
            ComplexChooserOption opt(kToday + Period(12, Months), 55.0, 48.0,
                                     ext::make_shared<EuropeanExercise>(callExp),
                                     ext::make_shared<EuropeanExercise>(putExp));
            opt.setPricingEngine(ext::make_shared<AnalyticComplexChooserEngine>(process));
            opt.NPV();
        }, true);
        // Choosing date sits between the put expiry and the call expiry, so
        // only the put check fires.
        emitRaises("    ", "choosing_after_put_expiry", [&] {
            ComplexChooserOption opt(kToday + Period(8, Months), 55.0, 48.0,
                                     ext::make_shared<EuropeanExercise>(callExp),
                                     ext::make_shared<EuropeanExercise>(kToday + Period(7, Months)));
            opt.setPricingEngine(ext::make_shared<AnalyticComplexChooserEngine>(process));
            opt.NPV();
        }, false);
        std::cout << "  },\n";
    }

    // -- VanillaForwardPayoff -------------------------------------------------
    const std::vector<Real> prices = {0.0,  20.0,  50.0,  87.5,   99.0,
                                      100.0, 101.0, 120.0, 175.0, 250.0};
    std::cout << "  \"vanilla_forward_payoff\": {\n";
    std::cout << "    \"prices\": [";
    for (Size i = 0; i < prices.size(); ++i) {
        std::cout << (i != 0U ? ", " : "") << prices[i];
    }
    std::cout << "],\n";
    // strike 87.5 renders identically under C++ ostream defaults and Python
    // repr, so description() can be compared verbatim.
    emitForwardPayoff("call_87_5", Option::Call, 87.5, prices, true);
    emitForwardPayoff("put_87_5", Option::Put, 87.5, prices, true);
    emitForwardPayoff("call_100", Option::Call, 100.0, prices, true);
    emitForwardPayoff("put_100", Option::Put, 100.0, prices, false);
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
