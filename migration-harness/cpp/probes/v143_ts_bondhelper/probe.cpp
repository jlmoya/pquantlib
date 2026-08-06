// migration-harness/cpp/probes/v143_ts_bondhelper/probe.cpp
//
// Pins BondHelper / FixedRateBondHelper: the two dates and the implied quote.
//
// BondHelper::impliedQuote prices the wrapped bond off the curve being
// bootstrapped and returns its clean or dirty price (bondhelpers.cpp:53-69).
// The dates are NOT the bond's own maturity and start:
//
//     latestDate_   = bond->cashflows().back()->date()
//     earliestDate_ = bond->nextCashFlowDate()
//
// The first is the last CASHFLOW date, which lands after the maturity date
// whenever the payment convention rolls the redemption forward — the schedule
// below is deliberately built so that happens (a maturity on a Saturday, so
// Following pushes the redemption payment into the next month). The second is
// the next cashflow after settlement, not the issue date, so it moves with the
// evaluation date.
//
// Clean and dirty are both emitted because they differ by the accrued amount,
// and a helper that reports the wrong one is otherwise silent: at an accrual
// boundary the two coincide.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/v143/ts/bondhelper.json.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/instruments/bonds/zerocouponbond.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/bondhelpers.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>

using namespace QuantLib;

namespace {

const Date kEval(17, January, 2024);

ext::shared_ptr<YieldTermStructure> flatCurve() {
    auto c = ext::make_shared<FlatForward>(kEval, 0.035, Actual365Fixed(), Continuous,
                                           Annual);
    c->enableExtrapolation();
    return c;
}

bool firstEmitted = false;

void emit(const std::string& key, const ext::shared_ptr<BondHelper>& h) {
    auto curve = flatCurve();
    h->setTermStructure(curve.get());
    const auto& bond = h->bond();

    if (firstEmitted)
        std::cout << ",\n";
    firstEmitted = true;
    std::cout << "  \"" << key << "\": {\n"
              << "    \"earliest_date\": " << h->earliestDate().serialNumber() << ",\n"
              << "    \"latest_date\": " << h->latestDate().serialNumber() << ",\n"
              << "    \"pillar_date\": " << h->pillarDate().serialNumber() << ",\n"
              << "    \"bond_maturity_date\": " << bond->maturityDate().serialNumber() << ",\n"
              << "    \"bond_settlement_date\": " << bond->settlementDate().serialNumber() << ",\n"
              << "    \"n_cashflows\": " << bond->cashflows().size() << ",\n"
              << "    \"implied_quote\": " << h->impliedQuote() << ",\n"
              << "    \"quote_error\": " << h->quoteError() << "\n"
              << "  }";
}

} // namespace

int main() {
    Settings::instance().evaluationDate() = kEval;
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    const Calendar cal = TARGET();
    const DayCounter thirty360 = Thirty360(Thirty360::BondBasis);

    // Maturity 15 June 2030 is a Saturday, so a Following payment convention
    // pushes the final coupon and the redemption to Monday 17 June — making
    // latestDate() strictly later than the bond's maturity date.
    Schedule schedule(Date(15, June, 2020), Date(15, June, 2030), Period(Annual), cal,
                      Unadjusted, Unadjusted, DateGeneration::Backward, false);

    emit("fixed_rate_clean",
         ext::make_shared<FixedRateBondHelper>(
             Handle<Quote>(ext::make_shared<SimpleQuote>(101.5)), 3, 100.0, schedule,
             std::vector<Rate>{0.04}, thirty360, Following, 100.0, Date(15, June, 2020),
             cal, Period(), Calendar(), Unadjusted, false, Bond::Price::Clean));

    emit("fixed_rate_dirty",
         ext::make_shared<FixedRateBondHelper>(
             Handle<Quote>(ext::make_shared<SimpleQuote>(101.5)), 3, 100.0, schedule,
             std::vector<Rate>{0.04}, thirty360, Following, 100.0, Date(15, June, 2020),
             cal, Period(), Calendar(), Unadjusted, false, Bond::Price::Dirty));

    // Zero settlement days: settlement is the evaluation date itself, so
    // earliestDate moves.
    emit("fixed_rate_settle0",
         ext::make_shared<FixedRateBondHelper>(
             Handle<Quote>(ext::make_shared<SimpleQuote>(101.5)), 0, 100.0, schedule,
             std::vector<Rate>{0.04}, thirty360, Following, 100.0, Date(15, June, 2020),
             cal, Period(), Calendar(), Unadjusted, false, Bond::Price::Clean));

    // Semiannual coupons + a different day counter and face amount, to catch a
    // port that hard-codes any of them.
    Schedule semi(Date(1, March, 2021), Date(1, March, 2029), Period(Semiannual), cal,
                  ModifiedFollowing, ModifiedFollowing, DateGeneration::Backward, false);
    emit("fixed_rate_semi_act365",
         ext::make_shared<FixedRateBondHelper>(
             Handle<Quote>(ext::make_shared<SimpleQuote>(98.25)), 2, 1000.0, semi,
             std::vector<Rate>{0.025}, Actual365Fixed(), ModifiedFollowing, 100.0,
             Date(1, March, 2021), cal, Period(), Calendar(), Unadjusted, false,
             Bond::Price::Clean));

    // Plain BondHelper over a zero-coupon bond: exercises the base class with a
    // bond that has a single cashflow.
    emit("zero_coupon_clean",
         ext::make_shared<BondHelper>(
             Handle<Quote>(ext::make_shared<SimpleQuote>(82.0)),
             ext::make_shared<ZeroCouponBond>(2, cal, 100.0, Date(15, June, 2030),
                                              Following, 100.0, Date(15, June, 2020)),
             Bond::Price::Clean));

    std::cout << "\n}\n";
    return 0;
}
