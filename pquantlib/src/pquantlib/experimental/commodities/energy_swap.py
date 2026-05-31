"""EnergySwap — abstract base for energy swaps (pricing periods).

# C++ parity: ql/experimental/commodities/energyswap.hpp +
#             energyswap.cpp (v1.42.1).

An :class:`EnergyCommodity` priced over a list of
:class:`~pquantlib.experimental.commodities.pricing_period.PricingPeriod`
objects, holding the per-day positions and payment cash flows produced by
the concrete swap's ``performCalculations``. The aggregate
:meth:`quantity` sums the periods' quantities.

# C++ parity note: ``isExpired()`` uses ``detail::simple_event(paymentDate)
# .hasOccurred()`` which reads ``Settings::instance().evaluationDate()``.
# PQuantLib compares the last payment date against
# ``ObservableSettings().evaluation_date_or_today()`` (event has occurred
# when its date is strictly before the evaluation date).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.experimental.commodities.commodity import SecondaryCosts
from pquantlib.experimental.commodities.commodity_cash_flow import CommodityCashFlows
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.energy_commodity import (
    EnergyCommodity,
    EnergyDailyPositions,
)
from pquantlib.experimental.commodities.pricing_period import PricingPeriods
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.calendar import Calendar


class EnergySwap(EnergyCommodity):
    """Abstract energy swap (priced over a list of pricing periods)."""

    def __init__(
        self,
        calendar: Calendar,
        pay_currency: Currency,
        receive_currency: Currency,
        pricing_periods: PricingPeriods,
        commodity_type: CommodityType,
        secondary_costs: SecondaryCosts | None = None,
    ) -> None:
        super().__init__(commodity_type, secondary_costs)
        self._calendar = calendar
        self._pay_currency = pay_currency
        self._receive_currency = receive_currency
        self._pricing_periods = pricing_periods
        self._daily_positions: EnergyDailyPositions = {}
        self._payment_cash_flows: CommodityCashFlows = {}

    # ---- inspectors ----

    @property
    def calendar(self) -> Calendar:
        return self._calendar

    @property
    def pay_currency(self) -> Currency:
        return self._pay_currency

    @property
    def receive_currency(self) -> Currency:
        return self._receive_currency

    @property
    def pricing_periods(self) -> PricingPeriods:
        return self._pricing_periods

    @property
    def daily_positions(self) -> EnergyDailyPositions:
        return self._daily_positions

    @property
    def payment_cash_flows(self) -> CommodityCashFlows:
        return self._payment_cash_flows

    @property
    def commodity_type(self) -> CommodityType:
        qassert.require(len(self._pricing_periods) > 0, "no pricing periods")
        return self._pricing_periods[0].quantity.commodity_type

    def quantity(self) -> Quantity:
        total = 0.0
        for pricing_period in self._pricing_periods:
            total += pricing_period.quantity.amount
        return Quantity(
            self._pricing_periods[0].quantity.commodity_type,
            self._pricing_periods[0].quantity.unit_of_measure,
            total,
        )

    def is_expired(self) -> bool:
        if not self._pricing_periods:
            return True
        last_payment = self._pricing_periods[-1].payment_date
        evaluation_date = ObservableSettings().evaluation_date_or_today()
        return last_payment < evaluation_date
