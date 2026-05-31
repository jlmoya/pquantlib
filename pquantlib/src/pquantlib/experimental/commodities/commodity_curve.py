"""CommodityCurve — forward price curve (term structure) for a commodity.

# C++ parity: ql/experimental/commodities/commoditycurve.hpp +
#             commoditycurve.cpp (v1.42.1).

A :class:`~pquantlib.termstructures.term_structure.TermStructure` whose
"value" is a forward *price* (not a discount factor): a set of
``(date, price)`` nodes interpolated with a forward-flat interpolator
(``price(t)`` holds the left-hand knot value going forward). Supports an
optional chained *basis* curve whose prices are added on top (the
``basisOfCurve_`` mechanism, with a UOM conversion factor).

# C++ parity notes:
# - The C++ ctor anchors the term structure at ``dates[0]`` (fixed
#   reference date) and uses ``Actual365Fixed`` by default. We mirror both.
# - ``price(d, exchangeContracts, nearbyOffset)`` rolls the pricing date on
#   the underlying contract expiry when ``nearbyOffset > 0`` (nearby curves);
#   for ``nearbyOffset == 0`` it prices directly at ``d``. The interpolation
#   is always evaluated with extrapolation enabled (C++ ``interpolation_(t,
#   true)``).
# - The two C++ ctors (with / without nodes) collapse into one Python ctor
#   where ``dates``/``prices`` are optional (matching the W7-B convention).
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.commodities.commodity_pricing_helpers import (
    CommodityPricingHelper,
)
from pquantlib.experimental.commodities.commodity_type import CommodityType
from pquantlib.experimental.commodities.exchange_contract import ExchangeContracts
from pquantlib.experimental.commodities.unit_of_measure import UnitOfMeasure
from pquantlib.math.array import Array
from pquantlib.math.interpolations.forward_flat import ForwardFlatInterpolation
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class CommodityCurve(TermStructure):
    """Commodity forward-price term structure (forward-flat interpolated)."""

    def __init__(
        self,
        name: str,
        commodity_type: CommodityType,
        currency: Currency,
        unit_of_measure: UnitOfMeasure,
        calendar: Calendar,
        dates: list[Date] | None = None,
        prices: list[float] | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()

        self._name = name
        self._commodity_type = commodity_type
        self._unit_of_measure = unit_of_measure
        self._currency = currency
        self._basis_of_curve: CommodityCurve | None = None
        self._basis_of_curve_uom_conversion_factor: float = 1.0
        self._dates: list[Date] = []
        self._times: list[float] = []
        self._data: list[float] = []
        self._interpolation: ForwardFlatInterpolation | None = None

        if dates is not None and prices is not None:
            # Node-bearing ctor: anchor at dates[0] (fixed reference date).
            TermStructure.__init__(
                self, reference_date=dates[0], calendar=calendar, day_counter=dc
            )
            qassert.require(len(dates) > 1, "too few dates")
            qassert.require(
                len(prices) == len(dates), "dates/prices count mismatch"
            )
            self._dates = list(dates)
            self._data = list(prices)
            self._rebuild_interpolation()
        else:
            # Empty ctor: moving reference date (settlement_days=0).
            TermStructure.__init__(
                self, settlement_days=0, calendar=calendar, day_counter=dc
            )

    # ---- interpolation rebuild (shared by ctor / set_prices) ----

    def _rebuild_interpolation(self) -> None:
        dc = self.day_counter()
        n = len(self._dates)
        self._times = [0.0] * n
        for i in range(1, n):
            qassert.require(
                self._dates[i] > self._dates[i - 1],
                f"invalid date ({self._dates[i]}, vs {self._dates[i - 1]})",
            )
            self._times[i] = dc.year_fraction(self._dates[0], self._dates[i])
        xs: Array = np.array(self._times, dtype=np.float64)
        ys: Array = np.array(self._data, dtype=np.float64)
        self._interpolation = ForwardFlatInterpolation(xs, ys)

    # ---- inspectors ----

    @property
    def name(self) -> str:
        return self._name

    @property
    def commodity_type(self) -> CommodityType:
        return self._commodity_type

    @property
    def unit_of_measure(self) -> UnitOfMeasure:
        return self._unit_of_measure

    @property
    def currency(self) -> Currency:
        return self._currency

    def max_date(self) -> Date:
        return self._dates[-1]

    def times(self) -> list[float]:
        return self._times

    def dates(self) -> list[Date]:
        return self._dates

    def prices(self) -> list[float]:
        return self._data

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))

    def empty(self) -> bool:
        return len(self._dates) == 0

    @property
    def basis_of_curve(self) -> CommodityCurve | None:
        return self._basis_of_curve

    # ---- mutators ----

    def set_prices(self, prices: dict[Date, float]) -> None:
        """Replace the node set from a ``{date: price}`` map (parity)."""
        qassert.require(len(prices) > 1, "too few prices")
        # C++ iterates a std::map -> keys are sorted ascending.
        self._dates = sorted(prices.keys())
        self._data = [prices[d] for d in self._dates]
        self._rebuild_interpolation()

    def set_basis_of_curve(self, basis_of_curve: CommodityCurve) -> None:
        """Chain a basis curve whose prices add on top (with UOM factor)."""
        self._basis_of_curve = basis_of_curve
        self._basis_of_curve_uom_conversion_factor = (
            CommodityPricingHelper.calculate_uom_conversion_factor(
                self._commodity_type,
                basis_of_curve._unit_of_measure,
                self._unit_of_measure,
            )
        )

    # ---- pricing ----

    def _price_impl(self, t: float) -> float:
        qassert.require(
            self._interpolation is not None,
            f"curve [{self._name}] has no price data",
        )
        assert self._interpolation is not None
        return self._interpolation(t, allow_extrapolation=True)

    def _basis_of_price_impl(self, t: float) -> float:
        if self._basis_of_curve is not None:
            try:
                basis_value = (
                    self._basis_of_curve._price_impl(t)
                    * self._basis_of_curve_uom_conversion_factor
                )
            except Exception as e:
                qassert.fail(f"error retrieving price for curve [{self._name}]: {e}")
            return basis_value + self._basis_of_curve._basis_of_price_impl(t)
        return 0.0

    def basis_of_price(self, d: Date) -> float:
        return self._basis_of_price_impl(self.time_from_reference(d))

    def underlying_price_date(
        self,
        date: Date,
        exchange_contracts: ExchangeContracts | None,
        nearby_offset: int,
    ) -> Date:
        """Roll the pricing date onto the underlying contract expiry (nearby)."""
        qassert.require(nearby_offset > 0, "nearby offset must be > 0")
        qassert.require(
            exchange_contracts is not None,
            "exchange contracts required for nearby offset",
        )
        assert exchange_contracts is not None
        # C++ uses std::map::lower_bound(date): first key >= date.
        keys = sorted(k for k in exchange_contracts if k >= date)
        if keys:
            # advance (nearbyOffset - 1) further entries.
            idx = nearby_offset - 1
            qassert.require(
                idx < len(keys),
                f"not enough nearby contracts available for curve "
                f"[{self._name}] for date [{date}].",
            )
            return exchange_contracts[keys[idx]].underlying_start_date
        return date

    def price(
        self,
        d: Date,
        exchange_contracts: ExchangeContracts | None = None,
        nearby_offset: int = 0,
    ) -> float:
        """Forward price at ``d`` (interpolated + any chained basis curves)."""
        date = (
            self.underlying_price_date(d, exchange_contracts, nearby_offset)
            if nearby_offset > 0
            else d
        )
        t = self.time_from_reference(date)
        try:
            price_value = self._price_impl(t)
        except Exception as e:
            qassert.fail(f"error retrieving price for curve [{self._name}]: {e}")
        return price_value + self._basis_of_price_impl(t)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommodityCurve):
            return NotImplemented
        return self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"CommodityCurve([{self._name}] {self._currency.code}/{self._unit_of_measure.code})"
