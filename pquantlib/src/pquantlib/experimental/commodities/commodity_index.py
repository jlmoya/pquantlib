"""CommodityIndex — observable commodity spot/forward index.

# C++ parity: ql/experimental/commodities/commodityindex.hpp +
#             commodityindex.cpp (v1.42.1).

An :class:`~pquantlib.indexes.index.Index` whose past fixings are held in
the shared index history (``IndexManager``) and whose *forward* price is
read off an attached :class:`CommodityCurve`. ``fixing(date)`` returns the
recorded past fixing; ``forward_price(date)`` reads the forward curve
(scaled by a UOM conversion factor computed at construction).

# C++ parity notes:
# - The C++ ctor registers with ``Settings::instance().evaluationDate()``
#   and the index notifier, then computes a UOM conversion factor between
#   the forward curve's UOM and the index UOM (via CommodityPricingHelper).
#   PQuantLib omits the Settings registration (no global eval-date
#   observable is wired into Index here); the UOM factor is reproduced.
# - ``fixing(date, forecastTodaysFixing)`` ignores the second flag and
#   returns ``pastFixing(date)`` — identical to C++.
# - ``isValidFixingDate`` delegates to the index calendar's business-day
#   test (parity with C++).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.experimental.commodities.commodity_curve import CommodityCurve
from pquantlib.experimental.commodities.commodity_pricing_helpers import (
    CommodityPricingHelper,
)
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.exchange_contract import ExchangeContracts
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.indexes.index import Index
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class CommodityIndex(Index):
    """Commodity index (history-backed fixings + forward-curve forward prices)."""

    def __init__(
        self,
        name: str,
        commodity_type: CommodityType,
        currency: Currency,
        unit_of_measure: UnitOfMeasure,
        calendar: Calendar,
        lot_quantity: float,
        forward_curve: CommodityCurve | None = None,
        exchange_contracts: ExchangeContracts | None = None,
        nearby_offset: int = 0,
    ) -> None:
        super().__init__()
        self._name = name
        self._commodity_type = commodity_type
        self._unit_of_measure = unit_of_measure
        self._currency = currency
        self._calendar = calendar
        self._lot_quantity = lot_quantity
        self._forward_curve = forward_curve
        self._exchange_contracts = exchange_contracts
        self._nearby_offset = nearby_offset
        self._forward_curve_uom_conversion_factor: float = 1.0

        if forward_curve is not None:
            self._forward_curve_uom_conversion_factor = (
                CommodityPricingHelper.calculate_uom_conversion_factor(
                    commodity_type,
                    forward_curve.unit_of_measure,
                    unit_of_measure,
                )
            )

    # ---- Index interface ----

    def name(self) -> str:
        return self._name

    def fixing_calendar(self) -> Calendar:
        return self._calendar

    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        return self.fixing_calendar().is_business_day(fixing_date)

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        # C++ parity: ignores forecast_todays_fixing; returns past fixing.
        del forecast_todays_fixing
        return self.past_fixing(fixing_date)

    # ---- inspectors ----

    @property
    def commodity_type(self) -> CommodityType:
        return self._commodity_type

    @property
    def currency(self) -> Currency:
        return self._currency

    @property
    def unit_of_measure(self) -> UnitOfMeasure:
        return self._unit_of_measure

    @property
    def forward_curve(self) -> CommodityCurve | None:
        return self._forward_curve

    @property
    def lot_quantity(self) -> float:
        return self._lot_quantity

    def forward_price(self, date: Date) -> float:
        """Forward price at ``date`` (forward curve price x UOM factor)."""
        qassert.require(
            self._forward_curve is not None,
            f"index [{self._name}] has no forward curve",
        )
        assert self._forward_curve is not None
        try:
            fwd = self._forward_curve.price(
                date, self._exchange_contracts, self._nearby_offset
            )
        except Exception as e:
            qassert.fail(f"error fetching forward price for index {self._name}: {e}")
        return fwd * self._forward_curve_uom_conversion_factor

    def last_quote_date(self) -> Date:
        return self.time_series().last_date()

    def empty(self) -> bool:
        return self.time_series().empty()

    def forward_curve_empty(self) -> bool:
        if self._forward_curve is not None:
            return self._forward_curve.empty()
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommodityIndex):
            return NotImplemented
        return self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"CommodityIndex([{self._name}] {self._currency.code}/{self._unit_of_measure.code})"
