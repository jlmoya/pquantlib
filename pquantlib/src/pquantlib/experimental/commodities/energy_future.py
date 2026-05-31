"""EnergyFuture — energy futures contract.

# C++ parity: ql/experimental/commodities/energyfuture.hpp +
#             energyfuture.cpp (v1.42.1).

A self-pricing :class:`EnergyCommodity` whose NPV is

    ((quotePrice - tradePrice) * quantityAmount) * lotQuantity * buySell

where ``quotePrice`` is the index fixing (if a recent quote exists) or the
index forward price, all re-expressed in the base currency + base UOM via
the inherited conversion helpers, less any secondary costs.

# C++ parity notes:
# - The C++ ``performCalculations`` reads ``Settings::instance().evaluationDate()``.
#   PQuantLib uses ``ObservableSettings().evaluation_date_or_today()``.
# - When the index's last quote date is older than ``evalDate - 1`` the
#   forward price is used and a ``PricingError.Warning`` is recorded.
"""

from __future__ import annotations

from pquantlib.experimental.commodities.commodity import (
    PricingErrorLevel,
    SecondaryCosts,
)
from pquantlib.experimental.commodities.commodity_index import CommodityIndex
from pquantlib.experimental.commodities.commodity_settings import CommoditySettings
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.commodity_unit_cost import CommodityUnitCost
from pquantlib.experimental.commodities.energy_commodity import EnergyCommodity
from pquantlib.experimental.commodities.quantity import Quantity
from pquantlib.patterns.observable_settings import ObservableSettings


class EnergyFuture(EnergyCommodity):
    """Energy future (self-pricing: NPV = signed quantity x price delta)."""

    def __init__(
        self,
        buy_sell: int,
        quantity: Quantity,
        trade_price: CommodityUnitCost,
        index: CommodityIndex,
        commodity_type: CommodityType,
        secondary_costs: SecondaryCosts | None = None,
    ) -> None:
        super().__init__(commodity_type, secondary_costs)
        self._buy_sell = buy_sell
        self._quantity = quantity
        self._trade_price = trade_price
        self._index = index

    def is_expired(self) -> bool:
        return False

    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def trade_price(self) -> CommodityUnitCost:
        return self._trade_price

    @property
    def index(self) -> CommodityIndex:
        return self._index

    def _perform_calculations(self) -> None:
        self._npv = 0.0
        self._additional_results = {}

        evaluation_date = ObservableSettings().evaluation_date_or_today()
        base_currency = CommoditySettings.instance().currency
        base_uom = CommoditySettings.instance().unit_of_measure

        quantity_uom_factor = (
            self.calculate_uom_conversion_factor(
                self._quantity.commodity_type, base_uom, self._quantity.unit_of_measure
            )
            * self._index.lot_quantity
        )
        index_uom_factor = self.calculate_uom_conversion_factor(
            self._index.commodity_type, self._index.unit_of_measure, base_uom
        )
        trade_price_uom_factor = self.calculate_uom_conversion_factor(
            self._quantity.commodity_type, self._trade_price.unit_of_measure, base_uom
        )

        trade_price_fx_factor = self.calculate_fx_conversion_factor(
            self._trade_price.amount.currency, base_currency, evaluation_date
        )
        index_price_fx_factor = self.calculate_fx_conversion_factor(
            self._index.currency, base_currency, evaluation_date
        )

        last_quote_date = self._index.last_quote_date()
        if last_quote_date >= evaluation_date - 1:
            quote_value = self._index.fixing(evaluation_date)
        else:
            quote_value = self._index.forward_price(evaluation_date)
            self.add_pricing_error(
                PricingErrorLevel.WARNING,
                f"curve [{self._index.name()}] has last quote date of "
                f"{last_quote_date} using forward price",
            )

        # C++ parity: a ``quoteValue != Null<Real>()`` check guards a missing
        # quote; Python has no Null<Real> sentinel (fixing()/forward_price()
        # raise instead of returning a sentinel), so the guard is vestigial.

        trade_price_value = (
            self._trade_price.amount.value
            * trade_price_uom_factor
            * trade_price_fx_factor
        )
        quote_price_value = quote_value * index_uom_factor * index_price_fx_factor

        quantity_amount = self._quantity.amount * quantity_uom_factor

        delta = (
            ((quote_price_value - trade_price_value) * quantity_amount)
            * self._index.lot_quantity
        ) * self._buy_sell

        self._npv = delta

        self.calculate_secondary_cost_amounts(
            self._quantity.commodity_type, self._quantity.amount, evaluation_date
        )
        for amount in self._secondary_cost_amounts.values():
            self._npv -= amount.value
