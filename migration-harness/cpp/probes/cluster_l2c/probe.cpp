// L2-C mega-probe: 8 IBOR concretes (default-market params + sample forecasts),
// 2 swap indexes, 7 rate helpers (implied_quote for known FlatForward curve).
//
// Goal: emit JSON reference values cross-checkable by Python ports.

#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/usdlibor.hpp>
#include <ql/indexes/ibor/gbplibor.hpp>
#include <ql/indexes/ibor/eonia.hpp>
#include <ql/indexes/ibor/sofr.hpp>
#include <ql/indexes/ibor/sonia.hpp>
#include <ql/indexes/ibor/fedfunds.hpp>
#include <ql/indexes/ibor/estr.hpp>
#include <ql/indexes/swap/euriborswap.hpp>
#include <ql/indexes/swap/usdliborswap.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/termstructures/yield/oisratehelper.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/calendars/jointcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/businessdayconvention.hpp>
#include <ql/settings.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

using namespace QuantLib;

namespace {
    std::string bdc_name(BusinessDayConvention c) {
        switch (c) {
            case Following: return "Following";
            case ModifiedFollowing: return "ModifiedFollowing";
            case Preceding: return "Preceding";
            case ModifiedPreceding: return "ModifiedPreceding";
            case Unadjusted: return "Unadjusted";
            case HalfMonthModifiedFollowing: return "HalfMonthModifiedFollowing";
            case Nearest: return "Nearest";
        }
        return "Unknown";
    }

    template <class IDX>
    void emit_ibor(const char* key, const IDX& idx) {
        std::cout << "  \"" << key << "\": {\n";
        std::cout << "    \"name\": \"" << idx.name() << "\",\n";
        std::cout << "    \"familyName\": \"" << idx.familyName() << "\",\n";
        std::cout << "    \"tenor_length\": " << idx.tenor().length() << ",\n";
        std::cout << "    \"tenor_units\": " << static_cast<int>(idx.tenor().units()) << ",\n";
        std::cout << "    \"fixingDays\": " << idx.fixingDays() << ",\n";
        std::cout << "    \"currencyCode\": \"" << idx.currency().code() << "\",\n";
        std::cout << "    \"fixingCalendarName\": \"" << idx.fixingCalendar().name() << "\",\n";
        std::cout << "    \"dayCounterName\": \"" << idx.dayCounter().name() << "\",\n";
        std::cout << "    \"businessDayConvention\": \"" << bdc_name(idx.businessDayConvention()) << "\",\n";
        std::cout << "    \"endOfMonth\": " << (idx.endOfMonth() ? "true" : "false") << "\n";
        std::cout << "  }";
    }
}

int main() {
    std::cout << std::setprecision(17);
    Date evalDate(17, January, 2024); // Wed; avoids MLK + weekends
    Settings::instance().evaluationDate() = evalDate;

    std::cout << "{\n";

    // === IBOR concretes — default-market parameters ===
    emit_ibor("euribor_3m", Euribor3M());
    std::cout << ",\n";
    emit_ibor("euribor_6m", Euribor6M());
    std::cout << ",\n";
    emit_ibor("euribor_1y", Euribor1Y());
    std::cout << ",\n";
    emit_ibor("euribor_1w", Euribor1W());
    std::cout << ",\n";
    emit_ibor("usd_libor_3m", USDLibor(3 * Months));
    std::cout << ",\n";
    emit_ibor("gbp_libor_6m", GBPLibor(6 * Months));
    std::cout << ",\n";
    emit_ibor("eonia", Eonia());
    std::cout << ",\n";
    emit_ibor("sofr", Sofr());
    std::cout << ",\n";
    emit_ibor("sonia", Sonia());
    std::cout << ",\n";
    emit_ibor("fed_funds", FedFunds());
    std::cout << ",\n";
    emit_ibor("estr", Estr());
    std::cout << ",\n";

    // === Euribor3M fixing with a known flat-forward curve ===
    {
        Handle<Quote> rate(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, rate, Actual360(), Continuous, Annual));
        Euribor3M idx(ts);
        Date fix(17, January, 2024);
        Date valueDate = idx.valueDate(fix);
        Date maturity = idx.maturityDate(valueDate);
        Time t = idx.dayCounter().yearFraction(valueDate, maturity);
        Real fixing = idx.fixing(fix, true);
        std::cout << "  \"euribor_3m_fixing\": {\n";
        std::cout << "    \"fix_serial\": " << fix.serialNumber() << ",\n";
        std::cout << "    \"value_serial\": " << valueDate.serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << maturity.serialNumber() << ",\n";
        std::cout << "    \"yearFraction\": " << t << ",\n";
        std::cout << "    \"fixing\": " << fixing << "\n";
        std::cout << "  },\n";
    }

    // === USDLibor3M fixing with FlatForward ===
    {
        Handle<Quote> rate(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, rate, Actual360(), Continuous, Annual));
        USDLibor idx(3 * Months, ts);
        Date fix(17, January, 2024);
        Date valueDate = idx.valueDate(fix);
        Date maturity = idx.maturityDate(valueDate);
        Time t = idx.dayCounter().yearFraction(valueDate, maturity);
        Real fixing = idx.fixing(fix, true);
        std::cout << "  \"usd_libor_3m_fixing\": {\n";
        std::cout << "    \"fix_serial\": " << fix.serialNumber() << ",\n";
        std::cout << "    \"value_serial\": " << valueDate.serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << maturity.serialNumber() << ",\n";
        std::cout << "    \"yearFraction\": " << t << ",\n";
        std::cout << "    \"fixing\": " << fixing << "\n";
        std::cout << "  },\n";
    }

    // === Sofr fixing (overnight) with FlatForward ===
    {
        Handle<Quote> rate(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, rate, Actual360(), Continuous, Annual));
        Sofr idx(ts);
        Date fix(17, January, 2024);
        Date valueDate = idx.valueDate(fix);
        Date maturity = idx.maturityDate(valueDate);
        Time t = idx.dayCounter().yearFraction(valueDate, maturity);
        Real fixing = idx.fixing(fix, true);
        std::cout << "  \"sofr_fixing\": {\n";
        std::cout << "    \"fix_serial\": " << fix.serialNumber() << ",\n";
        std::cout << "    \"value_serial\": " << valueDate.serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << maturity.serialNumber() << ",\n";
        std::cout << "    \"yearFraction\": " << t << ",\n";
        std::cout << "    \"fixing\": " << fixing << "\n";
        std::cout << "  },\n";
    }

    // === DepositRateHelper.impliedQuote — should ~roundtrip rate ===
    {
        Handle<Quote> q(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, q, Actual360(), Continuous, Annual));
        auto helper = ext::make_shared<DepositRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.05)),
            3 * Months, 2, TARGET(), ModifiedFollowing, true, Actual360());
        helper->setTermStructure(ts.currentLink().get());
        Real implied = helper->impliedQuote();
        std::cout << "  \"deposit_rate_helper\": {\n";
        std::cout << "    \"earliest_serial\": " << helper->earliestDate().serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << helper->maturityDate().serialNumber() << ",\n";
        std::cout << "    \"implied_quote\": " << implied << "\n";
        std::cout << "  },\n";
    }

    // === FraRateHelper.impliedQuote ===
    {
        Handle<Quote> q(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, q, Actual360(), Continuous, Annual));
        // 3M-into-3M FRA. useIndexedCoupon=false uses discount-factor formula.
        auto helper = ext::make_shared<FraRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.05)),
            3, 6, 2, TARGET(), ModifiedFollowing, true, Actual360(),
            Pillar::LastRelevantDate, Date(), false);
        helper->setTermStructure(ts.currentLink().get());
        Real implied = helper->impliedQuote();
        std::cout << "  \"fra_rate_helper\": {\n";
        std::cout << "    \"earliest_serial\": " << helper->earliestDate().serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << helper->maturityDate().serialNumber() << ",\n";
        std::cout << "    \"implied_quote\": " << implied << "\n";
        std::cout << "  },\n";
    }

    // === FuturesRateHelper.impliedQuote ===
    {
        Handle<Quote> q(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, q, Actual360(), Continuous, Annual));
        Date startDate(20, March, 2024); // IMM date
        auto helper = ext::make_shared<FuturesRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(95.0)),
            startDate, 3, TARGET(), ModifiedFollowing, true, Actual360(),
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.0)));
        helper->setTermStructure(ts.currentLink().get());
        Real implied = helper->impliedQuote();
        std::cout << "  \"futures_rate_helper\": {\n";
        std::cout << "    \"earliest_serial\": " << helper->earliestDate().serialNumber() << ",\n";
        std::cout << "    \"maturity_serial\": " << helper->maturityDate().serialNumber() << ",\n";
        std::cout << "    \"implied_quote\": " << implied << "\n";
        std::cout << "  },\n";
    }

    // === FxSwapRateHelper.impliedQuote ===
    {
        Handle<Quote> q(ext::make_shared<SimpleQuote>(0.05));
        Handle<YieldTermStructure> coll(
            ext::make_shared<FlatForward>(
                evalDate, Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)),
                Actual365Fixed(), Continuous, Annual));
        Handle<YieldTermStructure> ts(
            ext::make_shared<FlatForward>(
                evalDate, q, Actual365Fixed(), Continuous, Annual));
        Handle<Quote> spotFx(ext::make_shared<SimpleQuote>(1.10));
        auto helper = ext::make_shared<FxSwapRateHelper>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(0.01)),
            spotFx, 3 * Months, 2, TARGET(),
            ModifiedFollowing, false, true, coll);
        helper->setTermStructure(ts.currentLink().get());
        Real implied = helper->impliedQuote();
        std::cout << "  \"fx_swap_rate_helper\": {\n";
        std::cout << "    \"earliest_serial\": " << helper->earliestDate().serialNumber() << ",\n";
        std::cout << "    \"latest_serial\": " << helper->latestDate().serialNumber() << ",\n";
        std::cout << "    \"implied_quote\": " << implied << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
