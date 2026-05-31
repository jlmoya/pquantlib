// Phase 11 W7-C cluster probe: commodity index/curve/cashflow + energy
// instruments.
//
// Captures reference values for:
//
//   * CommodityCurve::price(d) — forward-flat interpolated price at a node and
//     between nodes (the C++ ForwardFlat interpolator).
//   * CommodityIndex::fixing(d) — past fixing from the index history.
//   * CommodityIndex::forwardPrice(d) — forward price routed through the
//     forward curve (with UOM conversion factor == 1 here).
//   * CommodityIndex::lastQuoteDate / empty / forwardCurveEmpty.
//   * CommodityCashFlow::amount() — discounted Money amount round-trip.
//   * EnergyFuture::NPV — quantity x (indexPrice - tradePrice) x lot, signed by
//     buy/sell.
//   * EnergyVanillaSwap::NPV — floating-vs-fixed leg difference summed over the
//     pricing period (the canonical energy-swap test value), plus the daily
//     position count.
//
// C++ parity:
//   ql/experimental/commodities/commoditycurve.hpp
//   ql/experimental/commodities/commodityindex.hpp
//   ql/experimental/commodities/commoditycashflow.hpp
//   ql/experimental/commodities/energyfuture.hpp
//   ql/experimental/commodities/energyvanillaswap.hpp
//   @ v1.42.1 (099987f0).

#include <ql/experimental/commodities/commoditycurve.hpp>
#include <ql/experimental/commodities/commodityindex.hpp>
#include <ql/experimental/commodities/commoditycashflow.hpp>
#include <ql/experimental/commodities/commoditysettings.hpp>
#include <ql/experimental/commodities/energyfuture.hpp>
#include <ql/experimental/commodities/energyvanillaswap.hpp>
#include <ql/experimental/commodities/petroleumunitsofmeasure.hpp>
#include <ql/experimental/commodities/commoditytype.hpp>
#include <ql/experimental/commodities/pricingperiod.hpp>
#include <ql/experimental/commodities/quantity.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/currencies/america.hpp>
#include <ql/settings.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {
    std::string jr(const std::string& key, Real v) {
        std::ostringstream os;
        os << "  \"" << key << "\": " << std::setprecision(17) << v;
        return os.str();
    }
}

int main() {
    std::vector<std::string> out;

    Date today(15, March, 2020);
    Settings::instance().evaluationDate() = today;

    NullCommodityType nct;
    BarrelUnitOfMeasure bbl;
    NullCalendar cal;
    Actual365Fixed dc;
    USDCurrency usd;

    // ---- CommodityCurve: forward price curve ----
    std::vector<Date> dates = {
        Date(15, March, 2020),
        Date(15, June, 2020),
        Date(15, September, 2020),
        Date(15, December, 2020)
    };
    std::vector<Real> prices = {30.0, 35.0, 40.0, 45.0};

    auto fwdCurve = ext::make_shared<CommodityCurve>(
        "WTI_FWD", nct, usd, bbl, cal, dates, prices, dc);

    // price at a node (15-Jun-2020 -> 35) and between nodes.
    out.push_back(jr("curve_price_node1", fwdCurve->price(
        Date(15, June, 2020), ext::shared_ptr<ExchangeContracts>(), 0)));
    // between node 0 and 1 (forward-flat -> takes the *right* endpoint value).
    out.push_back(jr("curve_price_mid01", fwdCurve->price(
        Date(15, April, 2020), ext::shared_ptr<ExchangeContracts>(), 0)));
    out.push_back(jr("curve_price_node3", fwdCurve->price(
        Date(15, December, 2020), ext::shared_ptr<ExchangeContracts>(), 0)));
    out.push_back(jr("curve_max_date_serial",
                     (Real)fwdCurve->maxDate().serialNumber()));

    // ---- CommodityIndex: history + forward curve ----
    auto index = ext::make_shared<CommodityIndex>(
        "WTI", nct, usd, bbl, cal, 1000.0, fwdCurve,
        ext::shared_ptr<ExchangeContracts>(), 0);

    // empty before any fixing.
    out.push_back(jr("index_empty_before", index->empty() ? 1.0 : 0.0));
    out.push_back(jr("index_fwd_curve_empty", index->forwardCurveEmpty() ? 1.0 : 0.0));

    // add a couple of historical fixings.
    index->addFixing(Date(13, March, 2020), 29.5);
    index->addFixing(Date(15, March, 2020), 31.0);

    out.push_back(jr("index_fixing_hist", index->fixing(Date(15, March, 2020))));
    out.push_back(jr("index_empty_after", index->empty() ? 1.0 : 0.0));
    out.push_back(jr("index_last_quote_serial",
                     (Real)index->lastQuoteDate().serialNumber()));
    // forward price from the curve at a future date (node -> 40).
    out.push_back(jr("index_forward_price",
                     index->forwardPrice(Date(15, September, 2020))));

    // ---- CommodityCashFlow: amount() == discounted Money value ----
    Money disc(usd, 12345.67);
    Money undisc(usd, 12500.00);
    CommodityCashFlow ccf(Date(20, December, 2020), disc, undisc, disc, undisc,
                          0.987, 0.987, false);
    out.push_back(jr("cashflow_amount", ccf.amount()));
    out.push_back(jr("cashflow_disc_factor", ccf.discountFactor()));

    // ---- EnergyFuture: NPV ----
    {
        CommoditySettings::instance().currency() = usd;
        CommoditySettings::instance().unitOfMeasure() = bbl;

        Quantity q(nct, bbl, 1000.0);
        CommodityUnitCost tradePrice(Money(usd, 28.0), bbl);
        auto futIndex = ext::make_shared<CommodityIndex>(
            "WTI_FUT", nct, usd, bbl, cal, 1.0, fwdCurve,
            ext::shared_ptr<ExchangeContracts>(), 0);
        futIndex->addFixing(Date(15, March, 2020), 31.0);

        EnergyFuture fut(1, q, tradePrice, futIndex, nct,
                         ext::shared_ptr<SecondaryCosts>());
        out.push_back(jr("future_npv", fut.NPV()));
    }

    // ---- EnergyVanillaSwap: NPV over a single pricing period ----
    {
        CommoditySettings::instance().currency() = usd;
        CommoditySettings::instance().unitOfMeasure() = bbl;

        // index quotes daily across the period via fixings.
        auto swapIndex = ext::make_shared<CommodityIndex>(
            "WTI_SWAP", nct, usd, bbl, cal, 1.0, fwdCurve,
            ext::shared_ptr<ExchangeContracts>(), 0);

        Date periodStart(16, March, 2020);
        Date periodEnd(20, March, 2020);
        Date payDate(31, March, 2020);
        // fixings for every day in [start, end] (NullCalendar: all are bd).
        Real q = 30.0;
        for (Date d = periodStart; d <= periodEnd; d = cal.advance(d, 1*Days)) {
            swapIndex->addFixing(d, q);
            q += 1.0;  // 30, 31, 32, 33, 34
        }

        Quantity quantity(nct, bbl, 1000.0);
        std::vector<ext::shared_ptr<PricingPeriod> > periods;
        periods.push_back(ext::make_shared<PricingPeriod>(
            periodStart, periodEnd, payDate, quantity));

        Handle<YieldTermStructure> flatTs(
            ext::make_shared<FlatForward>(today, 0.01, dc));

        // fixed price 30; floating avg (30..34) = 32 -> non-trivial NPV.
        EnergyVanillaSwap swap(
            true, cal, Money(usd, 30.0), bbl, swapIndex, usd, usd,
            periods, nct, ext::shared_ptr<SecondaryCosts>(),
            flatTs, flatTs, flatTs);

        out.push_back(jr("swap_npv", swap.NPV()));
        out.push_back(jr("swap_daily_positions",
                         (Real)swap.dailyPositions().size()));
        out.push_back(jr("swap_payment_cashflows",
                         (Real)swap.paymentCashFlows().size()));
    }

    std::cout << "{\n";
    for (Size i = 0; i < out.size(); ++i)
        std::cout << out[i] << (i + 1 < out.size() ? ",\n" : "\n");
    std::cout << "}\n";
    return 0;
}
