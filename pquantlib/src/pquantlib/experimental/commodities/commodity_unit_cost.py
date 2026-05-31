"""CommodityUnitCost — a cost per unit of measure.

# C++ parity: ql/experimental/commodities/commodityunitcost.hpp (v1.42.1).

A simple value type pairing a :class:`~pquantlib.currencies.money.Money`
amount with the :class:`~pquantlib.experimental.commodities.unit_of_measure.UnitOfMeasure`
that the amount is "per".
"""

from __future__ import annotations

from pquantlib.currencies.money import Money
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure


class CommodityUnitCost:
    """Cost of one unit of a commodity (a Money amount per unit of measure)."""

    def __init__(
        self,
        amount: Money | None = None,
        unit_of_measure: UnitOfMeasure | None = None,
    ) -> None:
        self._amount: Money = amount if amount is not None else Money()
        self._unit_of_measure: UnitOfMeasure = (
            unit_of_measure if unit_of_measure is not None else UnitOfMeasure()
        )

    @property
    def amount(self) -> Money:
        return self._amount

    @property
    def unit_of_measure(self) -> UnitOfMeasure:
        return self._unit_of_measure

    def __str__(self) -> str:
        return f"{self._amount} per {self._unit_of_measure.code}"

    def __repr__(self) -> str:
        return f"CommodityUnitCost({self.__str__()!r})"
