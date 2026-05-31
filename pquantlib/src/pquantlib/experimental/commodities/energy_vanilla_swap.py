"""EnergyVanillaSwap — fixed-vs-floating energy swap.

# C++ parity: ql/experimental/commodities/energyvanillaswap.hpp +
#             energyvanillaswap.cpp (v1.42.1).

A self-pricing :class:`EnergySwap` that exchanges a fixed price for the
floating index price over each pricing period. For each business day in a
period the floating (index) and fixed leg prices are recorded as an
:class:`~pquantlib.experimental.commodities.energy_commodity.EnergyDailyPosition`;
the period's quantity is spread evenly across its days, leg values are
summed (discounted per the supplied term structures), and the discounted
net (``dDelta``) is accumulated into the NPV and emitted as a
:class:`~pquantlib.experimental.commodities.commodity_cash_flow.CommodityCashFlow`.

# C++ parity notes:
# - ``payer`` True => ``payReceive_ == 1`` (pay leg = fixed, receive leg =
#   floating); False => receive leg = fixed.
# - The C++ ``Handle<YieldTermStructure>`` arguments map to plain
#   ``YieldTermStructure`` objects (no Handle smart-pointer in this port).
# - Discounting only applies when ``paymentDate >= evalDate + 2`` (the C++
#   2-settlement-day rule); otherwise all discount factors are 1.
# - FX conversion factors are all 1 in the same-currency case (the
#   cross-currency ExchangeRateManager path is deferred — see
#   CommodityPricingHelper).
# - ``Settings::instance().evaluationDate()`` -> ``ObservableSettings()
#   .evaluation_date_or_today()``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.money import Money
from pquantlib.experimental.commodities.commodity import (
    PricingErrorLevel,
    SecondaryCosts,
)
from pquantlib.experimental.commodities.commodity_cash_flow import CommodityCashFlow
from pquantlib.experimental.commodities.commodity_index import CommodityIndex
from pquantlib.experimental.commodities.commodity_settings import CommoditySettings
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.energy_commodity import EnergyDailyPosition
from pquantlib.experimental.commodities.energy_swap import EnergySwap
from pquantlib.experimental.commodities.pricing_period import PricingPeriods
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit


class EnergyVanillaSwap(EnergySwap):
    """Fixed-vs-floating energy swap (self-pricing over pricing periods)."""

    def __init__(
        self,
        payer: bool,
        calendar: Calendar,
        fixed_price: Money,
        fixed_price_unit_of_measure: UnitOfMeasure,
        index: CommodityIndex,
        pay_currency: Currency,
        receive_currency: Currency,
        pricing_periods: PricingPeriods,
        commodity_type: CommodityType,
        secondary_costs: SecondaryCosts | None,
        pay_leg_term_structure: YieldTermStructure,
        receive_leg_term_structure: YieldTermStructure,
        discount_term_structure: YieldTermStructure,
    ) -> None:
        super().__init__(
            calendar,
            pay_currency,
            receive_currency,
            pricing_periods,
            commodity_type,
            secondary_costs,
        )
        self._pay_receive = 1 if payer else 0
        self._fixed_price = fixed_price
        self._fixed_price_unit_of_measure = fixed_price_unit_of_measure
        self._index = index
        self._pay_leg_term_structure = pay_leg_term_structure
        self._receive_leg_term_structure = receive_leg_term_structure
        self._discount_term_structure = discount_term_structure
        qassert.require(len(pricing_periods) > 0, "no pricing periods")

    # ---- inspectors ----

    @property
    def pay_receive(self) -> int:
        return self._pay_receive

    @property
    def fixed_price(self) -> Money:
        return self._fixed_price

    @property
    def fixed_price_unit_of_measure(self) -> UnitOfMeasure:
        return self._fixed_price_unit_of_measure

    @property
    def index(self) -> CommodityIndex:
        return self._index

    def is_expired(self) -> bool:
        last_end = self._pricing_periods[-1].end_date
        evaluation_date = ObservableSettings().evaluation_date_or_today()
        return last_end < evaluation_date

    # ---- pricing ----

    def _perform_calculations(self) -> None:  # noqa: PLR0915 — faithful port of a long C++ method
        if self._index.empty() and not self._index.forward_curve_empty():
            self.add_pricing_error(
                PricingErrorLevel.WARNING,
                f"index [{self._index.name()}] does not have any quotes; "
                f"using forward prices",
            )
        elif self._index.empty():
            qassert.fail(f"index [{self._index.name()}] does not have any quotes")

        self._npv = 0.0
        self._additional_results = {}
        self._daily_positions = {}
        self._payment_cash_flows = {}

        evaluation_date = ObservableSettings().evaluation_date_or_today()
        base_currency = CommoditySettings.instance().currency
        base_uom = CommoditySettings.instance().unit_of_measure

        period0_qty = self._pricing_periods[0].quantity
        quantity_uom_factor = self.calculate_uom_conversion_factor(
            period0_qty.commodity_type, base_uom, period0_qty.unit_of_measure
        )
        fixed_price_uom_factor = self.calculate_uom_conversion_factor(
            period0_qty.commodity_type, self._fixed_price_unit_of_measure, base_uom
        )
        index_uom_factor = self.calculate_uom_conversion_factor(
            self._index.commodity_type, self._index.unit_of_measure, base_uom
        )

        fixed_price_fx_factor = self.calculate_fx_conversion_factor(
            self._fixed_price.currency, base_currency, evaluation_date
        )
        index_price_fx_factor = self.calculate_fx_conversion_factor(
            self._index.currency, base_currency, evaluation_date
        )
        pay_leg_fx_factor = self.calculate_fx_conversion_factor(
            base_currency,
            self._pay_currency if self._pay_receive > 0 else self._receive_currency,
            evaluation_date,
        )
        receive_leg_fx_factor = self.calculate_fx_conversion_factor(
            base_currency,
            self._receive_currency if self._pay_receive > 0 else self._pay_currency,
            evaluation_date,
        )

        last_quote_date = self._index.last_quote_date()
        if last_quote_date < evaluation_date - 1:
            self.add_pricing_error(
                PricingErrorLevel.WARNING,
                f"index [{self._index.name()}] has last quote date of "
                f"{last_quote_date}",
            )

        total_quantity_amount = 0.0

        for pricing_period in self._pricing_periods:
            qassert.require(
                pricing_period.quantity.amount != 0, "quantity is zero"
            )

            period_day_count = 0
            period_start_date = self._calendar.adjust(pricing_period.start_date)
            period_dates: list[Date] = []

            step_date = period_start_date
            while step_date <= pricing_period.end_date:
                unrealized = step_date > evaluation_date
                if step_date <= last_quote_date:
                    quote_value = self._index.fixing(step_date)
                else:
                    quote_value = self._index.forward_price(step_date)

                if quote_value == 0:
                    self.add_pricing_error(
                        PricingErrorLevel.WARNING,
                        f"pay quote value for curve [{self._index.name()}] "
                        f"is 0 for date {step_date}",
                    )

                fixed_leg_price_value = (
                    self._fixed_price.value
                    * fixed_price_uom_factor
                    * fixed_price_fx_factor
                )
                floating_leg_price_value = (
                    quote_value * index_uom_factor * index_price_fx_factor
                )
                pay_leg_price_value = (
                    fixed_leg_price_value
                    if self._pay_receive > 0
                    else floating_leg_price_value
                )
                receive_leg_price_value = (
                    floating_leg_price_value
                    if self._pay_receive > 0
                    else fixed_leg_price_value
                )

                self._daily_positions[step_date] = EnergyDailyPosition(
                    date=step_date,
                    pay_leg_price=pay_leg_price_value,
                    receive_leg_price=receive_leg_price_value,
                    unrealized=unrealized,
                )
                period_dates.append(step_date)
                period_day_count += 1
                step_date = self._calendar.advance(step_date, 1, TimeUnit.Days)

            period_quantity_amount = (
                pricing_period.quantity.amount * quantity_uom_factor
            )
            total_quantity_amount += period_quantity_amount

            avg_daily_quantity_amount = (
                0.0
                if period_day_count == 0
                else period_quantity_amount / period_day_count
            )

            pay_leg_value = 0.0
            receive_leg_value = 0.0
            for d in period_dates:
                daily_position = self._daily_positions[d]
                daily_position.quantity_amount = avg_daily_quantity_amount
                daily_position.risk_delta = (
                    -daily_position.pay_leg_price + daily_position.receive_leg_price
                ) * avg_daily_quantity_amount
                pay_leg_value += (
                    -daily_position.pay_leg_price * avg_daily_quantity_amount
                )
                receive_leg_value += (
                    daily_position.receive_leg_price * avg_daily_quantity_amount
                )

            discount_factor = 1.0
            pay_leg_discount_factor = 1.0
            receive_leg_discount_factor = 1.0
            if pricing_period.payment_date >= evaluation_date + 2:
                discount_factor = self._discount_term_structure.discount(
                    pricing_period.payment_date
                )
                pay_leg_discount_factor = self._pay_leg_term_structure.discount(
                    pricing_period.payment_date
                )
                receive_leg_discount_factor = (
                    self._receive_leg_term_structure.discount(
                        pricing_period.payment_date
                    )
                )

            u_delta = receive_leg_value + pay_leg_value
            d_delta = (receive_leg_value * receive_leg_discount_factor) + (
                pay_leg_value * pay_leg_discount_factor
            )
            pmt_fx_factor = (
                pay_leg_fx_factor
                if (d_delta * self._pay_receive) > 0
                else receive_leg_fx_factor
            )
            pmt_currency = (
                self._receive_currency
                if (d_delta * self._pay_receive) > 0
                else self._pay_currency
            )
            pmt_discount_factor = (
                receive_leg_discount_factor if d_delta > 0 else pay_leg_discount_factor
            )

            self._payment_cash_flows[pricing_period.payment_date] = CommodityCashFlow(
                pricing_period.payment_date,
                Money(base_currency, u_delta * discount_factor),
                Money(base_currency, u_delta),
                Money(pmt_currency, d_delta * pmt_fx_factor),
                Money(pmt_currency, u_delta * pmt_fx_factor),
                discount_factor,
                pmt_discount_factor,
                pricing_period.payment_date <= evaluation_date,
            )

            self.calculate_secondary_cost_amounts(
                period0_qty.commodity_type, total_quantity_amount, evaluation_date
            )

            self._npv += d_delta

        qassert.require(len(self._payment_cash_flows) > 0, "no cashflows")

        for amount in self._secondary_cost_amounts.values():
            self._npv -= amount.value

        self._additional_results["dailyPositions"] = self._daily_positions
