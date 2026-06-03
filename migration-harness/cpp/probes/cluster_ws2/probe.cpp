// W-S2 cluster probe: SECONDARY economic-sanity reference for the
// retired-API dividend-option compat layer (hosted in pquantlib-helpers).
//
// These three JQuantLib classes (DividendVanillaOption,
// BinomialDividendVanillaEngine, BlackScholesDividendLattice) do NOT exist in
// C++ QuantLib v1.42.1 — v1.42.1 prices discrete-dividend Europeans with
// AnalyticDividendEuropeanEngine(process, DividendSchedule) on a plain
// VanillaOption. This probe emits that closed-form escrowed-dividend value as
// a coarse economic-sanity reference for the European compat-engine NPV
// (LOOSE ~1e-3, BY DESIGN: CRR tree discretization vs analytic, plus the
// JQuantLib lattice's known escrowed-amount approximation).
//
// It also emits the no-dividend AnalyticEuropeanEngine value for context.
//
// Scenario (mirrors CRRDividendOptionTest / FDDividendOptionTest):
//   type=Put, S=36, K=40, r=0.06, q=0.00, vol=0.20,
//   today=15-May-1998, settlement=17-May-1998, maturity=17-May-1999,
//   dayCounter=Actual365Fixed, calendar=Target,
//   3 dividends of 2.06 at today + i*3 months + 15 days (i=1,2,3).
//
// C++ parity: v1.42.1 (099987f0).

#include <ql/cashflows/dividend.hpp>
#include <ql/exercise.hpp>
#include <ql/instruments/dividendschedule.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/analyticdividendeuropeanengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);

    const Calendar calendar = TARGET();
    const DayCounter dc = Actual365Fixed();
    const Date today(15, May, 1998);
    const Date settlementDate(17, May, 1998);
    const Date maturityDate(17, May, 1999);

    Settings::instance().evaluationDate() = today;

    const Option::Type type = Option::Put;
    const Real strike = 40.0;
    const Real underlying = 36.0;
    const Rate riskFreeRate = 0.06;
    const Volatility volatility = 0.20;
    const Rate dividendYield = 0.00;

    // Discrete dividends: today + i*3 months + 15 days, i = 1, 2, 3.
    std::vector<Date> dividendDates;
    std::vector<Real> dividends;
    for (int i = 1; i <= 3; ++i) {
        Date d = today + Period(i * 3, Months) + Period(15, Days);
        dividendDates.push_back(d);
        dividends.push_back(2.06);
    }

    // Term structures anchored at the settlement date (matches the
    // JQuantLib CRRDividendOptionHelper referenceDate).
    auto spot = ext::make_shared<SimpleQuote>(underlying);
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(settlementDate, riskFreeRate, dc));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(settlementDate, dividendYield, dc));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(settlementDate, calendar, volatility, dc));

    auto process = ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(spot), qTS, rTS, volTS);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, strike);
    auto exercise = ext::make_shared<EuropeanExercise>(maturityDate);

    // Analytic discrete-dividend European (the v1.42.1 replacement for the
    // retired DividendVanillaOption + AnalyticDividendEuropeanEngine path).
    VanillaOption divOption(payoff, exercise);
    divOption.setPricingEngine(ext::make_shared<AnalyticDividendEuropeanEngine>(
        process, DividendVector(dividendDates, dividends)));

    // No-dividend analytic European, for context.
    VanillaOption plainOption(payoff, exercise);
    plainOption.setPricingEngine(
        ext::make_shared<AnalyticEuropeanEngine>(process));

    std::cout << "{\n";
    std::cout << "  \"scenario\": {\n";
    std::cout << "    \"type\": \"Put\",\n";
    std::cout << "    \"underlying\": " << underlying << ",\n";
    std::cout << "    \"strike\": " << strike << ",\n";
    std::cout << "    \"risk_free_rate\": " << riskFreeRate << ",\n";
    std::cout << "    \"dividend_yield\": " << dividendYield << ",\n";
    std::cout << "    \"volatility\": " << volatility << ",\n";
    std::cout << "    \"today_serial\": " << today.serialNumber() << ",\n";
    std::cout << "    \"settlement_serial\": " << settlementDate.serialNumber() << ",\n";
    std::cout << "    \"maturity_serial\": " << maturityDate.serialNumber() << ",\n";
    std::cout << "    \"dividend_amount\": 2.06,\n";
    std::cout << "    \"dividend_date_serials\": [";
    for (std::size_t i = 0; i < dividendDates.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << dividendDates[i].serialNumber();
    }
    std::cout << "]\n";
    std::cout << "  },\n";

    std::cout << "  \"analytic_dividend_european\": {\n";
    std::cout << "    \"npv\": " << divOption.NPV() << ",\n";
    std::cout << "    \"delta\": " << divOption.delta() << ",\n";
    std::cout << "    \"gamma\": " << divOption.gamma() << "\n";
    std::cout << "  },\n";

    std::cout << "  \"analytic_european_no_dividend\": {\n";
    std::cout << "    \"npv\": " << plainOption.NPV() << ",\n";
    std::cout << "    \"delta\": " << plainOption.delta() << ",\n";
    std::cout << "    \"gamma\": " << plainOption.gamma() << "\n";
    std::cout << "  }\n";

    std::cout << "}\n";
    return 0;
}
