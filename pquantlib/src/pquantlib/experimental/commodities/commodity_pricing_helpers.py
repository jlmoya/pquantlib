"""CommodityPricingHelper — static helpers for commodity pricing engines.

# C++ parity: ql/experimental/commodities/commoditypricinghelpers.hpp +
#             commoditypricinghelpers.cpp (v1.43).

Self-contained date/UOM/FX arithmetic helpers used by commodity engines.

Deferred to W7-C (where ``EnergyCommodity`` lands):
- ``createPricingPeriods`` — depends on ``EnergyCommodity.DeliverySchedule`` /
  ``EnergyCommodity.QuantityPeriodicity`` enums.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.currencies.exchange_rate import ExchangeRateType
from pquantlib.currencies.exchange_rate_manager import ExchangeRateManager
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.experimental.commodities.unit_of_measure_conversion_manager import (
    UnitOfMeasureConversionManager,
)
from pquantlib.time.date import Date


class CommodityPricingHelper:
    """Static commodity-pricing helper methods (parity with the C++ class)."""

    @staticmethod
    def calculate_uom_conversion_factor(
        commodity_type: CommodityType,
        from_unit_of_measure: UnitOfMeasure,
        to_unit_of_measure: UnitOfMeasure,
    ) -> float:
        """Factor to convert ``from`` UOM into ``to`` UOM (1 if identical)."""
        if to_unit_of_measure != from_unit_of_measure:
            conv = UnitOfMeasureConversionManager.instance().lookup(
                commodity_type, from_unit_of_measure, to_unit_of_measure
            )
            return conv.conversion_factor
        return 1.0

    @staticmethod
    def calculate_fx_conversion_factor(
        from_currency: Currency,
        to_currency: Currency,
        evaluation_date: Date,
    ) -> float:
        """FX factor to convert ``from`` currency into ``to`` currency.

        Asks ``ExchangeRateManager`` for a *direct* rate and inverts it when
        the stored rate runs the other way round.
        """
        if from_currency != to_currency:
            rate = ExchangeRateManager.instance().lookup(
                from_currency, to_currency, evaluation_date, ExchangeRateType.Direct
            )
            assert rate.rate is not None
            if from_currency != rate.source:
                return 1.0 / rate.rate
            return rate.rate
        return 1.0

    @staticmethod
    def calculate_unit_cost(
        commodity_type: CommodityType,
        unit_cost: CommodityUnitCost,
        base_currency: Currency,
        base_unit_of_measure: UnitOfMeasure,
        evaluation_date: Date,
    ) -> float:
        """Unit cost re-expressed in the base currency + base UOM."""
        if unit_cost.amount.value != 0:
            uom_factor = CommodityPricingHelper.calculate_uom_conversion_factor(
                commodity_type,
                unit_cost.unit_of_measure,
                base_unit_of_measure,
            )
            fx_factor = CommodityPricingHelper.calculate_fx_conversion_factor(
                unit_cost.amount.currency,
                base_currency,
                evaluation_date,
            )
            return unit_cost.amount.value * uom_factor * fx_factor
        return 0.0
