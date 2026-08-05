// migration-harness/cpp/probes/v143_indexes_equity/probe.cpp
//
// Reference values for ql/indexes/equityindex.hpp and for the batch-fixing
// semantics of ql/index.hpp + ql/indexes/indexmanager.hpp at C++ v1.43.
//
// EquityIndex is the only index in this batch with a real forecast: a forward
// carried off the spot quote and two curves,
//   forward = spot * D_dividend(t) / D_interest(t),
// with the dividend leg dropped when no dividend curve is supplied and the
// spot falling back to the last stored historical fixing when the quote is
// absent. All three paths are emitted, plus the past/spot/forecast branch
// selection in fixing().
//
// Index::addFixings is probed for its ordering guarantee rather than its happy
// path: C++ stores every acceptable fixing and only then raises for the
// rejected ones, so a batch containing one invalid date still commits the
// valid entries. That is easy to get backwards in a port and only shows up as
// silently missing history.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/indexes/equity.json.

#include <iomanip>
#include <iostream>
#include <vector>

#include <ql/currencies/america.hpp>
#include <ql/indexes/equityindex.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/indexmanager.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);

    const Date today(15, June, 2026);
    Settings::instance().evaluationDate() = today;

    const Calendar cal = UnitedStates(UnitedStates::GovernmentBond);
    const Handle<YieldTermStructure> interest(ext::make_shared<FlatForward>(
        today, Handle<Quote>(ext::make_shared<SimpleQuote>(0.045)), Actual365Fixed()));
    const Handle<YieldTermStructure> dividend(ext::make_shared<FlatForward>(
        today, Handle<Quote>(ext::make_shared<SimpleQuote>(0.015)), Actual365Fixed()));
    const Handle<Quote> spot(ext::make_shared<SimpleQuote>(4321.0));

    const Date future(15, December, 2026);
    const Date past(1, June, 2026);

    const EquityIndex withDividend("eqIndexDiv", cal, USDCurrency(), interest, dividend, spot);
    const EquityIndex noDividend("eqIndexNoDiv", cal, USDCurrency(), interest, {}, spot);

    // Spot-less index: forecasting falls back to the last stored fixing, taken
    // at today rolled Preceding on the index calendar.
    EquityIndex noSpot("eqIndexNoSpot", cal, USDCurrency(), interest, dividend, {});
    noSpot.addFixing(cal.adjust(today, Preceding), 4000.0);

    // fixing() branch selection on an index that has both history and spot.
    EquityIndex branches("eqIndexBranches", cal, USDCurrency(), interest, dividend, spot);
    branches.addFixing(past, 4100.0);

    std::cout << "{\n"
              << "  \"today_serial\": " << today.serialNumber() << ",\n"
              << "  \"future_serial\": " << future.serialNumber() << ",\n"
              << "  \"past_serial\": " << past.serialNumber() << ",\n"
              << "  \"spot\": " << spot->value() << ",\n"
              << "  \"name_with_dividend\": \"" << withDividend.name() << "\",\n"
              << "  \"fixing_calendar\": \"" << withDividend.fixingCalendar().name() << "\",\n"
              << "  \"currency_code\": \"" << withDividend.currency().code() << "\",\n"
              << "  \"forecast_with_dividend\": " << withDividend.forecastFixing(future) << ",\n"
              << "  \"forecast_without_dividend\": " << noDividend.forecastFixing(future) << ",\n"
              << "  \"forecast_spotless_last_fixing\": " << 4000.0 << ",\n"
              << "  \"forecast_spotless\": " << noSpot.forecastFixing(future) << ",\n"
              << "  \"fixing_future\": " << branches.fixing(future) << ",\n"
              << "  \"fixing_past\": " << branches.fixing(past) << ",\n"
              << "  \"fixing_today_falls_back_to_spot\": " << branches.fixing(today) << ",\n"
              << "  \"fixing_today_forecast\": " << branches.fixing(today, true) << ",\n";

    // --- Index::addFixings: partial commit, then raise ----------------------
    // Two valid business days bracketing one weekend date. C++ stores the two
    // valid entries and then throws for the invalid one.
    const Date d0(1, June, 2026);   // Monday
    const Date bad(6, June, 2026);  // Saturday — not a valid fixing date
    const Date d1(8, June, 2026);   // Monday
    std::vector<Date> dates{d0, bad, d1};
    std::vector<Real> values{10.0, 99.0, 12.0};

    Euribor3M batch;
    bool threw = false;
    try {
        batch.addFixings(dates.begin(), dates.end(), values.begin());
    } catch (const std::exception&) {
        threw = true;
    }
    std::cout << "  \"batch_threw\": " << (threw ? "true" : "false") << ",\n"
              << "  \"batch_d0_serial\": " << d0.serialNumber() << ",\n"
              << "  \"batch_bad_serial\": " << bad.serialNumber() << ",\n"
              << "  \"batch_d1_serial\": " << d1.serialNumber() << ",\n"
              << "  \"batch_stored_d0\": " << batch.timeSeries()[d0] << ",\n"
              << "  \"batch_stored_d1\": " << batch.timeSeries()[d1] << ",\n"
              << "  \"batch_bad_is_valid_fixing_date\": "
              << (batch.isValidFixingDate(bad) ? "true" : "false") << ",\n";

    // Re-adding an identical value is accepted silently; a different one is a
    // duplicate error.
    bool sameThrew = false;
    try {
        batch.addFixing(d0, 10.0);
    } catch (const std::exception&) {
        sameThrew = true;
    }
    bool conflictThrew = false;
    try {
        batch.addFixing(d0, 11.0);
    } catch (const std::exception&) {
        conflictThrew = true;
    }
    std::cout << "  \"readding_same_value_throws\": " << (sameThrew ? "true" : "false") << ",\n"
              << "  \"readding_different_value_throws\": "
              << (conflictThrew ? "true" : "false") << ",\n"
              << "  \"force_overwrite_wins\": ";
    batch.addFixing(d0, 11.0, true);
    std::cout << batch.timeSeries()[d0] << "\n";

    std::cout << "}\n";
    return 0;
}
