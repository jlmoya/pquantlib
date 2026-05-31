"""EnergyBasisSwap — basis swap between two energy indices.

# C++ parity: ql/experimental/commodities/energybasisswap.hpp +
#             energybasisswap.cpp (v1.42.1).

A self-pricing :class:`EnergySwap` exchanging one index's price (pay leg)
for another index's price (receive leg), with a fixed *basis* spread added
to whichever leg ``spread_to_pay_leg`` selects. Daily positions are built
exactly as in :class:`EnergyVanillaSwap`, but both legs are floating
(driven by ``pay_index`` / ``receive_index``); the quote source switches to
forward prices once past the earlier of the two indices' last quote dates.

# C++ parity notes:
# - The C++ ``Handle<YieldTermStructure>`` arguments map to plain
#   ``YieldTermStructure`` objects (no Handle smart-pointer in this port).
# - ``spreadIndex`` is stored for parity but, like the C++ code, is not used
#   in ``performCalculations`` (the basis comes from the ``basis`` unit cost).
# - Discounting only applies when ``paymentDate >= evalDate + 2``.
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
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.energy_commodity import EnergyDailyPosition
from pquantlib.experimental.commodities.energy_swap import EnergySwap
from pquantlib.experimental.commodities.pricing_period import PricingPeriods
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit


class EnergyBasisSwap(EnergySwap):
    """Basis swap between two energy indices (self-pricing over periods)."""

    def __init__(
        self,
        calendar: Calendar,
        spread_index: CommodityIndex,
        pay_index: CommodityIndex,
        receive_index: CommodityIndex,
        spread_to_pay_leg: bool,
        pay_currency: Currency,
        receive_currency: Currency,
        pricing_periods: PricingPeriods,
        basis: CommodityUnitCost,
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
        self._spread_index = spread_index
        self._pay_index = pay_index
        self._receive_index = receive_index
        self._spread_to_pay_leg = spread_to_pay_leg
        self._basis = basis
        self._pay_leg_term_structure = pay_leg_term_structure
        self._receive_leg_term_structure = receive_leg_term_structure
        self._discount_term_structure = discount_term_structure
        qassert.require(len(pricing_periods) > 0, "no payment dates")

    # ---- inspectors ----

    @property
    def pay_index(self) -> CommodityIndex:
        return self._pay_index

    @property
    def receive_index(self) -> CommodityIndex:
        return self._receive_index

    @property
    def basis(self) -> CommodityUnitCost:
        return self._basis

    # ---- pricing ----

    def _perform_calculations(self) -> None:  # noqa: PLR0915 — faithful port of a long C++ method
        for idx in (self._pay_index, self._receive_index):
            if idx.empty():
                if idx.forward_curve_empty():
                    qassert.fail(
                        f"index [{idx.name()}] does not have any quotes "
                        f"or forward prices"
                    )
                else:
                    self.add_pricing_error(
                        PricingErrorLevel.WARNING,
                        f"index [{idx.name()}] does not have any quotes; "
                        f"using forward prices",
                    )

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
        pay_index_uom_factor = self.calculate_uom_conversion_factor(
            self._pay_index.commodity_type, self._pay_index.unit_of_measure, base_uom
        )
        receive_index_uom_factor = self.calculate_uom_conversion_factor(
            self._receive_index.commodity_type,
            self._receive_index.unit_of_measure,
            base_uom,
        )

        pay_index_fx_factor = self.calculate_fx_conversion_factor(
            self._pay_index.currency, base_currency, evaluation_date
        )
        receive_index_fx_factor = self.calculate_fx_conversion_factor(
            self._receive_index.currency, base_currency, evaluation_date
        )
        pay_leg_fx_factor = self.calculate_fx_conversion_factor(
            base_currency, self._pay_currency, evaluation_date
        )
        receive_leg_fx_factor = self.calculate_fx_conversion_factor(
            base_currency, self._receive_currency, evaluation_date
        )

        basis_uom_factor = self.calculate_uom_conversion_factor(
            period0_qty.commodity_type, self._basis.unit_of_measure, base_uom
        )
        basis_fx_factor = self.calculate_fx_conversion_factor(
            base_currency, self._basis.amount.currency, evaluation_date
        )
        basis_value = self._basis.amount.value * basis_uom_factor * basis_fx_factor

        last_pay_quote_date = self._pay_index.last_quote_date()
        last_receive_quote_date = self._receive_index.last_quote_date()
        if last_pay_quote_date < evaluation_date - 1:
            self.add_pricing_error(
                PricingErrorLevel.WARNING,
                f"index [{self._pay_index.name()}] has last quote date of "
                f"{last_pay_quote_date}",
            )
        if last_receive_quote_date < evaluation_date - 1:
            self.add_pricing_error(
                PricingErrorLevel.WARNING,
                f"index [{self._receive_index.name()}] has last quote date of "
                f"{last_receive_quote_date}",
            )
        last_quote_date = min(last_pay_quote_date, last_receive_quote_date)

        total_quantity_amount = 0.0

        for pricing_period in self._pricing_periods:
            period_day_count = 0
            period_start_date = self._calendar.adjust(pricing_period.start_date)
            period_dates: list[Date] = []

            step_date = period_start_date
            while step_date <= pricing_period.end_date:
                unrealized = step_date > evaluation_date
                if step_date <= last_quote_date:
                    pay_quote_value = self._pay_index.fixing(step_date)
                    receive_quote_value = self._receive_index.fixing(step_date)
                else:
                    pay_quote_value = self._pay_index.forward_price(step_date)
                    receive_quote_value = self._receive_index.forward_price(step_date)

                pay_leg_price_value = (
                    pay_quote_value * pay_index_uom_factor * pay_index_fx_factor
                )
                receive_leg_price_value = (
                    receive_quote_value
                    * receive_index_uom_factor
                    * receive_index_fx_factor
                )

                if self._spread_to_pay_leg:
                    pay_leg_price_value += basis_value
                else:
                    receive_leg_price_value += basis_value

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
                pay_leg_fx_factor if d_delta > 0 else receive_leg_fx_factor
            )
            pmt_currency = (
                self._receive_currency if d_delta > 0 else self._pay_currency
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
