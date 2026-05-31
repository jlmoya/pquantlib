"""EnergyCommodity — abstract base for energy commodity instruments.

# C++ parity: ql/experimental/commodities/energycommodity.hpp +
#             energycommodity.cpp (v1.42.1).

Extends the :class:`~pquantlib.experimental.commodities.commodity.Commodity`
base with energy-specific scheduling enums, the per-day position bookkeeping
struct (:class:`EnergyDailyPosition`), and the static UOM/FX conversion +
secondary-cost helpers used by the concrete energy instruments (future /
vanilla swap / basis swap).

# C++ parity notes:
# - The static ``calculateUomConversionFactor`` / ``calculateFxConversionFactor``
#   delegate to the same arithmetic as ``CommodityPricingHelper`` (the C++
#   bodies are duplicated between the two classes). We forward to the helper
#   so the cross-currency deferral (ExchangeRateManager not yet ported) is
#   shared in one place.
# - ``setupArguments`` / ``fetchResults`` are part of the C++ engine plumbing
#   but the concrete energy instruments self-price in ``performCalculations``
#   (no separate engine is attached). They are omitted here.
# - ``EnergyDailyPosition`` is a plain mutable record (C++ ``struct``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from pquantlib.currencies.currency import Currency
from pquantlib.currencies.money import Money
from pquantlib.experimental.commodities.commodity import Commodity, SecondaryCosts
from pquantlib.experimental.commodities.commodity_pricing_helpers import (
    CommodityPricingHelper,
)
from pquantlib.experimental.commodities.commodity_settings import CommoditySettings
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.time.date import Date


class DeliverySchedule(IntEnum):
    """Energy delivery schedule (parity with C++ ``DeliverySchedule``)."""

    CONSTANT = 0
    WINDOW = 1
    HOURLY = 2
    DAILY = 3
    WEEKLY = 4
    MONTHLY = 5
    QUARTERLY = 6
    YEARLY = 7


class QuantityPeriodicity(IntEnum):
    """Quantity periodicity (parity with C++ ``QuantityPeriodicity``)."""

    ABSOLUTE = 0
    PER_HOUR = 1
    PER_DAY = 2
    PER_WEEK = 3
    PER_MONTH = 4
    PER_QUARTER = 5
    PER_YEAR = 6


class PaymentSchedule(IntEnum):
    """Payment schedule (parity with C++ ``PaymentSchedule``)."""

    WINDOW_SETTLEMENT = 0
    MONTHLY_SETTLEMENT = 1
    QUARTERLY_SETTLEMENT = 2
    YEARLY_SETTLEMENT = 3


@dataclass
class EnergyDailyPosition:
    """Per-day position record in an energy swap (parity with the C++ struct)."""

    date: Date = field(default_factory=Date)
    quantity_amount: float = 0.0
    pay_leg_price: float = 0.0
    receive_leg_price: float = 0.0
    risk_delta: float = 0.0
    unrealized: bool = False


# C++ parity: ``typedef std::map<Date, EnergyDailyPosition> EnergyDailyPositions;``
EnergyDailyPositions = dict[Date, EnergyDailyPosition]


class EnergyCommodity(Commodity):
    """Abstract energy commodity instrument (scheduling enums + cost helpers)."""

    # Nested enum aliases for the C++ idiom ``EnergyCommodity.Daily`` etc.
    DeliverySchedule = DeliverySchedule
    QuantityPeriodicity = QuantityPeriodicity
    PaymentSchedule = PaymentSchedule

    def __init__(
        self,
        commodity_type: CommodityType,
        secondary_costs: SecondaryCosts | None = None,
    ) -> None:
        super().__init__(secondary_costs)
        self._commodity_type = commodity_type

    def quantity(self) -> Quantity:
        """Total priced quantity (subclass-specific)."""
        raise NotImplementedError

    @property
    def commodity_type(self) -> CommodityType:
        return self._commodity_type

    # ---- static conversion helpers (delegate to CommodityPricingHelper) ----

    @staticmethod
    def calculate_uom_conversion_factor(
        commodity_type: CommodityType,
        from_unit_of_measure: UnitOfMeasure,
        to_unit_of_measure: UnitOfMeasure,
    ) -> float:
        return CommodityPricingHelper.calculate_uom_conversion_factor(
            commodity_type, from_unit_of_measure, to_unit_of_measure
        )

    @staticmethod
    def calculate_fx_conversion_factor(
        from_currency: Currency,
        to_currency: Currency,
        evaluation_date: Date,
    ) -> float:
        return CommodityPricingHelper.calculate_fx_conversion_factor(
            from_currency, to_currency, evaluation_date
        )

    def calculate_unit_cost(
        self,
        commodity_type: CommodityType,
        unit_cost: CommodityUnitCost,
        evaluation_date: Date,
    ) -> float:
        """Unit cost re-expressed in the base currency + base UOM."""
        return CommodityPricingHelper.calculate_unit_cost(
            commodity_type,
            unit_cost,
            CommoditySettings.instance().currency,
            CommoditySettings.instance().unit_of_measure,
            evaluation_date,
        )

    def calculate_secondary_cost_amounts(
        self,
        commodity_type: CommodityType,
        total_quantity_value: float,
        evaluation_date: Date,
    ) -> None:
        """Populate ``secondary_cost_amounts`` from the configured costs."""
        self._secondary_cost_amounts.clear()
        if not self._secondary_costs:
            return
        base_currency = CommoditySettings.instance().currency
        for key, cost in self._secondary_costs.items():
            if isinstance(cost, CommodityUnitCost):
                value = (
                    self.calculate_unit_cost(commodity_type, cost, evaluation_date)
                    * total_quantity_value
                )
                self._secondary_cost_amounts[key] = Money(base_currency, value)
            elif isinstance(cost, Money):
                fx = self.calculate_fx_conversion_factor(
                    cost.currency, base_currency, evaluation_date
                )
                self._secondary_cost_amounts[key] = Money(
                    base_currency, cost.value * fx
                )
