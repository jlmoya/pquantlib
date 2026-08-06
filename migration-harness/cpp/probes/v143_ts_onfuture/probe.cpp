// migration-harness/cpp/probes/v143_ts_onfuture/probe.cpp
//
// Pins OvernightIndexFuture and the two rate helpers built on it.
//
// The instrument prices itself: NPV = 100 * (1 - (convexityAdjustment + rate))
// where rate is either the SIMPLE average or the COMPOUNDED average of the
// overnight fixings over [valueDate, maturityDate). Those two differ by the
// cross terms, so probing only one would leave half the class unverified —
// both are emitted for the same period.
//
// The valuation date matters as much as the period: three regimes are probed
// for each contract, because the code branches on them.
//   * today <= valueDate: everything is forecast off the curve; the compounded
//     rate collapses to a single telescopic discount ratio.
//   * today strictly inside the reference period: past fixings must come from
//     the index history, today's fixing is used if published, and the
//     telescopic part starts from the day after. Both "today's fixing is
//     published" and "it isn't" are probed, since they take different
//     branches and pick a different forwardDiscountStart.
//   * a period whose maturity falls on a weekend, so the daily roll can
//     overshoot maturity — the averaged branch caps d2 at maturity and the
//     compounded branch does not.
//
// SofrFutureRateHelper is probed for both a quarterly contract (third
// Wednesday to third Wednesday, compounded) and a monthly one (first of month
// to first of next month, simple average), plus a case with a non-zero
// convexity adjustment.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/onfuture.json.

#include <iomanip>
#include <iostream>
#include <string>

#include <ql/indexes/ibor/sofr.hpp>
#include <ql/instruments/overnightindexfuture.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/overnightindexfutureratehelper.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace QuantLib;

namespace {

const Date kCurveRef(2, January, 2024);

RelinkableHandle<YieldTermStructure> curveHandle;

ext::shared_ptr<YieldTermStructure> flatCurve() {
    auto c = ext::make_shared<FlatForward>(kCurveRef, 0.0525, Actual365Fixed(), Continuous,
                                           Annual);
    c->enableExtrapolation();
    return c;
}

bool firstEmitted = false;

void emitFuture(const std::string& key,
                const Date& valueDate,
                const Date& maturityDate,
                RateAveraging::Type averaging,
                const Date& evaluationDate,
                Real convexity = Null<Real>()) {
    Settings::instance().evaluationDate() = evaluationDate;
    auto index = ext::make_shared<Sofr>(curveHandle);
    Handle<Quote> adj;
    if (convexity != Null<Real>())
        adj = Handle<Quote>(ext::make_shared<SimpleQuote>(convexity));
    OvernightIndexFuture f(index, valueDate, maturityDate, adj, averaging);

    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"evaluation_date\": " << evaluationDate.serialNumber() << ",\n"
              << "    \"value_date\": " << valueDate.serialNumber() << ",\n"
              << "    \"maturity_date\": " << maturityDate.serialNumber() << ",\n"
              << "    \"convexity_adjustment\": " << f.convexityAdjustment() << ",\n"
              << "    \"is_expired\": " << (f.isExpired() ? "true" : "false") << ",\n"
              << "    \"npv\": " << f.NPV() << "\n"
              << "  }";
}

void emitHelper(const std::string& key,
                const ext::shared_ptr<OvernightIndexFutureRateHelper>& h,
                const Date& evaluationDate,
                const ext::shared_ptr<YieldTermStructure>& curve) {
    Settings::instance().evaluationDate() = evaluationDate;
    h->setTermStructure(curve.get());

    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"evaluation_date\": " << evaluationDate.serialNumber() << ",\n"
              << "    \"earliest_date\": " << h->earliestDate().serialNumber() << ",\n"
              << "    \"latest_date\": " << h->latestDate().serialNumber() << ",\n"
              << "    \"pillar_date\": " << h->pillarDate().serialNumber() << ",\n"
              << "    \"convexity_adjustment\": " << h->convexityAdjustment() << ",\n"
              << "    \"implied_quote\": " << h->impliedQuote() << "\n"
              << "  }";
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    auto curve = flatCurve();
    curveHandle.linkTo(curve);

    // --- the instrument, valued before the reference period starts ---------
    const Date q1Start(20, March, 2024);   // third Wednesday of March 2024
    const Date q1End(19, June, 2024);      // third Wednesday of June 2024
    emitFuture("compounded_before_start", q1Start, q1End, RateAveraging::Compound,
               Date(2, January, 2024));
    emitFuture("averaged_before_start", q1Start, q1End, RateAveraging::Simple,
               Date(2, January, 2024));
    emitFuture("compounded_with_convexity", q1Start, q1End, RateAveraging::Compound,
               Date(2, January, 2024), 0.0012);

    // A monthly contract: 1 April to 1 May, so maturity is a Wednesday but the
    // start is a Monday; the daily roll crosses several weekends.
    emitFuture("monthly_averaged_before_start", Date(1, April, 2024), Date(1, May, 2024),
               RateAveraging::Simple, Date(2, January, 2024));
    emitFuture("monthly_compounded_before_start", Date(1, April, 2024),
               Date(1, May, 2024), RateAveraging::Compound, Date(2, January, 2024));

    // Maturity on a Saturday: the daily advance overshoots it, which the
    // averaged branch caps and the compounded branch does not.
    emitFuture("weekend_maturity_averaged", Date(1, April, 2024), Date(4, May, 2024),
               RateAveraging::Simple, Date(2, January, 2024));
    emitFuture("weekend_maturity_compounded", Date(1, April, 2024), Date(4, May, 2024),
               RateAveraging::Compound, Date(2, January, 2024));

    // --- valued INSIDE the reference period, so history is consulted -------
    {
        // Populate SOFR fixings for every business day from the period start
        // up to and including 5 April 2024.
        auto index = ext::make_shared<Sofr>(curveHandle);
        Calendar cal = index->fixingCalendar();
        Date d = Date(20, March, 2024);
        Real r = 0.0530;
        while (d <= Date(5, April, 2024)) {
            if (cal.isBusinessDay(d))
                index->addFixing(d, r, true);
            r += 0.00001;
            d += 1;
        }
    }
    // 5 April is a Friday and its fixing IS published, so forwardDiscountStart
    // moves to the next business day.
    emitFuture("compounded_inside_fixing_published", q1Start, q1End,
               RateAveraging::Compound, Date(5, April, 2024));
    emitFuture("averaged_inside_fixing_published", q1Start, q1End,
               RateAveraging::Simple, Date(5, April, 2024));
    // 8 April is a Monday with no fixing, so forwardDiscountStart stays at
    // today and the loop consumes 5 April's fixing on the way.
    emitFuture("compounded_inside_fixing_missing", q1Start, q1End,
               RateAveraging::Compound, Date(8, April, 2024));

    // --- the helpers --------------------------------------------------------
    emitHelper("helper_quarterly",
               ext::make_shared<SofrFutureRateHelper>(94.75, March, 2024, Quarterly),
               Date(2, January, 2024), curve);
    emitHelper("helper_quarterly_convexity",
               ext::make_shared<SofrFutureRateHelper>(94.75, March, 2024, Quarterly,
                                                      0.0012),
               Date(2, January, 2024), curve);
    emitHelper("helper_monthly",
               ext::make_shared<SofrFutureRateHelper>(94.70, April, 2024, Monthly),
               Date(2, January, 2024), curve);
    emitHelper("helper_september_quarterly",
               ext::make_shared<SofrFutureRateHelper>(94.60, September, 2024, Quarterly),
               Date(2, January, 2024), curve);
    emitHelper("helper_december_monthly",
               ext::make_shared<SofrFutureRateHelper>(94.55, December, 2024, Monthly),
               Date(2, January, 2024), curve);
    emitHelper("helper_generic_compound",
               ext::make_shared<OvernightIndexFutureRateHelper>(
                   Handle<Quote>(ext::make_shared<SimpleQuote>(94.80)),
                   Date(1, April, 2024), Date(1, July, 2024),
                   ext::make_shared<Sofr>(), Handle<Quote>(), RateAveraging::Compound,
                   Pillar::MaturityDate),
               Date(2, January, 2024), curve);
    emitHelper("helper_generic_custom_pillar",
               ext::make_shared<OvernightIndexFutureRateHelper>(
                   Handle<Quote>(ext::make_shared<SimpleQuote>(94.80)),
                   Date(1, April, 2024), Date(1, July, 2024),
                   ext::make_shared<Sofr>(), Handle<Quote>(), RateAveraging::Simple,
                   Pillar::CustomDate, Date(3, June, 2024)),
               Date(2, January, 2024), curve);

    std::cout << "\n}\n";
    return 0;
}
