// migration-harness/cpp/probes/v143_indexes_ibor/probe.cpp
//
// Reference values for the ql/indexes/ibor/* families that PQuantLib was
// missing against C++ QuantLib v1.43.
//
// Almost every one of these indexes is a thin constructor wrapper, so what
// actually needs pinning is the exact wiring: name, fixing days, currency,
// fixing calendar, day counter, business-day convention and end-of-month flag.
// Those are cheap to get subtly wrong in a port and stay invisible until a
// fixing date or accrual is off, so they are captured explicitly — together
// with valueDate/maturityDate for a sample mid-week fixing, which is where a
// wrong calendar or convention actually bites.
//
// Several families carry per-tenor overrides (Following/EOM=false on Days and
// Weeks vs ModifiedFollowing/EOM=true on Months and Years; settlement days
// that collapse to 0 on a 1-day tenor; the whole DailyTenor* family whose
// *fixing* calendar differs from the tenor-based sibling), so each tenor that
// C++ exposes as its own subclass is probed separately rather than inferred.
//
// Cdi overrides forecastFixing with a compounded (rather than simple) forward,
// and CustomIborIndex threads separate value/maturity calendars through
// fixingDate/valueDate/maturityDate — both are probed against a flat curve /
// an explicit calendar triple rather than by wiring alone.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/indexes/ibor.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/indexes/ibor/aonia.hpp>
#include <ql/indexes/ibor/audlibor.hpp>
#include <ql/indexes/ibor/bbsw.hpp>
#include <ql/indexes/ibor/bibor.hpp>
#include <ql/indexes/ibor/bkbm.hpp>
#include <ql/indexes/ibor/cadlibor.hpp>
#include <ql/indexes/ibor/cdi.hpp>
#include <ql/indexes/ibor/cdor.hpp>
#include <ql/indexes/ibor/chflibor.hpp>
#include <ql/indexes/ibor/corra.hpp>
#include <ql/indexes/ibor/custom.hpp>
#include <ql/indexes/ibor/destr.hpp>
#include <ql/indexes/ibor/dkklibor.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/eurlibor.hpp>
#include <ql/indexes/ibor/jibar.hpp>
#include <ql/indexes/ibor/jpylibor.hpp>
#include <ql/indexes/ibor/kofr.hpp>
#include <ql/indexes/ibor/mosprime.hpp>
#include <ql/indexes/ibor/nzdlibor.hpp>
#include <ql/indexes/ibor/nzocr.hpp>
#include <ql/indexes/ibor/pribor.hpp>
#include <ql/indexes/ibor/robor.hpp>
#include <ql/indexes/ibor/saron.hpp>
#include <ql/indexes/ibor/seklibor.hpp>
#include <ql/indexes/ibor/shibor.hpp>
#include <ql/indexes/ibor/swestr.hpp>
#include <ql/indexes/ibor/thbfix.hpp>
#include <ql/indexes/ibor/tibor.hpp>
#include <ql/indexes/ibor/tonar.hpp>
#include <ql/indexes/ibor/trlibor.hpp>
#include <ql/indexes/ibor/wibor.hpp>
#include <ql/indexes/ibor/zibor.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/unitedkingdom.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/period.hpp>

using namespace QuantLib;

namespace {

// Mid-week (a Monday) so the value/maturity roll is driven by the index's own
// calendar and convention rather than by a weekend. Each index adjusts it
// forward on its own fixing calendar first, and the adjusted serial is emitted
// alongside the results, so a national holiday cannot silently invalidate a
// fixing.
const Date kSeed(15, June, 2026);

bool first = true;

void emitIndex(const std::string& key, const InterestRateIndex& idx) {
    const Date fixing = idx.fixingCalendar().adjust(kSeed);
    const Date value = idx.valueDate(fixing);
    if (!first)
        std::cout << ",\n";
    first = false;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"name\": \"" << idx.name() << "\",\n"
              << "    \"fixing_days\": " << idx.fixingDays() << ",\n"
              << "    \"currency_code\": \"" << idx.currency().code() << "\",\n"
              << "    \"fixing_calendar\": \"" << idx.fixingCalendar().name() << "\",\n"
              << "    \"day_counter\": \"" << idx.dayCounter().name() << "\",\n"
              << "    \"tenor\": \"" << idx.tenor() << "\",\n"
              << "    \"fixing_serial\": " << fixing.serialNumber() << ",\n"
              << "    \"value_date_serial\": " << value.serialNumber() << ",\n"
              << "    \"fixing_date_serial\": " << idx.fixingDate(value).serialNumber() << ",\n"
              << "    \"maturity_date_serial\": " << idx.maturityDate(value).serialNumber();
}

void emitIbor(const std::string& key, const IborIndex& idx) {
    emitIndex(key, idx);
    std::cout << ",\n"
              << "    \"business_day_convention\": " << static_cast<int>(idx.businessDayConvention())
              << ",\n"
              << "    \"end_of_month\": " << (idx.endOfMonth() ? "true" : "false") << "\n"
              << "  }";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- Oceania -----------------------------------------------------------
    emitIbor("aonia", Aonia());
    emitIbor("nzocr", Nzocr());
    emitIbor("audlibor_3m", AUDLibor(Period(3, Months)));
    emitIbor("audlibor_1w", AUDLibor(Period(1, Weeks)));
    emitIbor("nzdlibor_3m", NZDLibor(Period(3, Months)));
    emitIbor("bbsw_1m", Bbsw1M());
    emitIbor("bbsw_2m", Bbsw2M());
    emitIbor("bbsw_3m", Bbsw3M());
    emitIbor("bbsw_4m", Bbsw4M());
    emitIbor("bbsw_5m", Bbsw5M());
    emitIbor("bbsw_6m", Bbsw6M());
    emitIbor("bkbm_1m", Bkbm1M());
    emitIbor("bkbm_2m", Bkbm2M());
    emitIbor("bkbm_3m", Bkbm3M());
    emitIbor("bkbm_4m", Bkbm4M());
    emitIbor("bkbm_5m", Bkbm5M());
    emitIbor("bkbm_6m", Bkbm6M());

    // --- Asia --------------------------------------------------------------
    emitIbor("bibor_sw", BiborSW());
    emitIbor("bibor_1m", Bibor1M());
    emitIbor("bibor_2m", Bibor2M());
    emitIbor("bibor_3m", Bibor3M());
    emitIbor("bibor_6m", Bibor6M());
    emitIbor("bibor_1y", Bibor1Y());
    emitIbor("thbfix_6m", THBFIX(Period(6, Months)));
    emitIbor("tibor_3m", Tibor(Period(3, Months)));
    emitIbor("tonar", Tonar());
    emitIbor("jpylibor_3m", JPYLibor(Period(3, Months)));
    emitIbor("daily_tenor_jpylibor_0", DailyTenorJPYLibor(0));
    emitIbor("daily_tenor_jpylibor_2", DailyTenorJPYLibor(2));
    emitIbor("kofr", Kofr());
    emitIbor("shibor_on", Shibor(Period(1, Days)));
    emitIbor("shibor_1w", Shibor(Period(1, Weeks)));
    emitIbor("shibor_3m", Shibor(Period(3, Months)));

    // --- Americas ----------------------------------------------------------
    emitIbor("cadlibor_3m", CADLibor(Period(3, Months)));
    emitIbor("cadlibor_on", CADLiborON());
    emitIbor("cdor_3m", Cdor(Period(3, Months)));
    emitIbor("corra", Corra());
    emitIbor("cdi", Cdi());

    // --- Europe ------------------------------------------------------------
    emitIbor("chflibor_3m", CHFLibor(Period(3, Months)));
    emitIbor("daily_tenor_chflibor_0", DailyTenorCHFLibor(0));
    emitIbor("daily_tenor_chflibor_2", DailyTenorCHFLibor(2));
    emitIbor("saron", Saron());
    emitIbor("zibor_3m", Zibor(Period(3, Months)));
    emitIbor("dkklibor_3m", DKKLibor(Period(3, Months)));
    emitIbor("destr", Destr());
    emitIbor("seklibor_3m", SEKLibor(Period(3, Months)));
    emitIbor("swestr", Swestr());
    emitIbor("eurlibor_on", EURLiborON());
    emitIbor("daily_tenor_eurlibor_2", DailyTenorEURLibor(2));
    emitIbor("eurlibor_1m", EURLibor1M());
    emitIbor("eurlibor_3m", EURLibor3M());
    emitIbor("eurlibor_6m", EURLibor6M());
    emitIbor("eurlibor_1y", EURLibor1Y());
    emitIbor("eurlibor_1w", EURLibor(Period(1, Weeks)));
    emitIbor("euribor_1w", Euribor1W());
    emitIbor("euribor_1m", Euribor1M());
    emitIbor("euribor_3m", Euribor3M());
    emitIbor("euribor_6m", Euribor6M());
    emitIbor("euribor_1y", Euribor1Y());
    emitIbor("euribor365_3m", Euribor365(Period(3, Months)));
    emitIbor("euribor365_1w", Euribor365(Period(1, Weeks)));
    emitIbor("mosprime_on", Mosprime(Period(1, Days)));
    emitIbor("mosprime_3m", Mosprime(Period(3, Months)));
    emitIbor("pribor_on", Pribor(Period(1, Days)));
    emitIbor("pribor_3m", Pribor(Period(3, Months)));
    emitIbor("robor_on", Robor(Period(1, Days)));
    emitIbor("robor_3m", Robor(Period(3, Months)));
    emitIbor("wibor_on", Wibor(Period(1, Days)));
    emitIbor("wibor_3m", Wibor(Period(3, Months)));
    emitIbor("trlibor_3m", TRLibor(Period(3, Months)));

    // --- Africa ------------------------------------------------------------
    emitIbor("jibar_3m", Jibar(Period(3, Months)));

    // --- CustomIborIndex: distinct fixing / value / maturity calendars ------
    // The three calendars are deliberately different so a port that collapses
    // them into one cannot pass: fixing rolls on the UK exchange calendar,
    // value dates advance and adjust on TARGET, and maturity advances on the
    // US government-bond calendar.
    emitIbor("custom",
             CustomIborIndex("CustomIndex", Period(3, Months), 2, EURCurrency(),
                             UnitedKingdom(UnitedKingdom::Exchange), TARGET(),
                             UnitedStates(UnitedStates::GovernmentBond),
                             ModifiedFollowing, true, Actual365Fixed()));

    // --- Cdi::forecastFixing — compounded, not simple -----------------------
    // Cdi is the one index in this batch with real behaviour rather than pure
    // wiring: it overrides forecastFixing with (D1/D2)^(1/yf) - 1 instead of
    // the IborIndex simple (D1/D2 - 1)/yf. Both are emitted off the same flat
    // 5% curve so the port cannot pass by inheriting the base implementation.
    Settings::instance().evaluationDate() = Date(15, June, 2026);
    const Handle<YieldTermStructure> flat(ext::make_shared<FlatForward>(
        Date(15, June, 2026), Handle<Quote>(ext::make_shared<SimpleQuote>(0.05)),
        Actual365Fixed()));
    const Cdi cdi(flat);
    const Date cdiFixing = cdi.fixingCalendar().adjust(Date(17, June, 2026));
    const OvernightIndex plainBrl("CDI-plain", 0, BRLCurrency(),
                                  Brazil(Brazil::Settlement), Business252(), flat);
    std::cout << ",\n"
              << "  \"cdi_forecast\": {\n"
              << "    \"fixing_serial\": " << cdiFixing.serialNumber() << ",\n"
              << "    \"compounded\": " << cdi.forecastFixing(cdiFixing) << ",\n"
              << "    \"simple\": " << plainBrl.forecastFixing(cdiFixing) << "\n"
              << "  }";

    std::cout << "\n}\n";
    return 0;
}
