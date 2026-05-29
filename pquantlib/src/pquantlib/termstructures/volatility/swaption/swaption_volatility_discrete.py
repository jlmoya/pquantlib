"""SwaptionVolatilityDiscrete — abstract base for grid-of-points swaption vol surfaces.

# C++ parity: ql/termstructures/volatility/swaption/swaptionvoldiscrete.{hpp,cpp}
# (v1.42.1).

Provides the pillar bookkeeping (option_tenors / option_dates /
option_times / swap_tenors / swap_lengths) shared by the
``SwaptionVolatilityMatrix`` family and the deferred SABR-cube
variants. Subclasses fill the interpolation surface in their own
``__init__`` and override ``_volatility_impl``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class SwaptionVolatilityDiscrete(SwaptionVolatilityStructure):
    """Discrete-pillar swaption-volatility structure base."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention,
        option_tenors: Sequence[Period],
        swap_tenors: Sequence[Period],
        calendar: Calendar,
        day_counter: DayCounter,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        qassert.require(len(option_tenors) > 0, "empty option-tenor vector")
        qassert.require(len(swap_tenors) > 0, "empty swap-tenor vector")
        for i in range(1, len(option_tenors)):
            qassert.require(
                option_tenors[i - 1] < option_tenors[i],
                f"non-increasing option tenors at index {i}",
            )
        for j in range(1, len(swap_tenors)):
            qassert.require(
                swap_tenors[j - 1] < swap_tenors[j],
                f"non-increasing swap tenors at index {j}",
            )
        self._option_tenors: list[Period] = list(option_tenors)
        self._swap_tenors: list[Period] = list(swap_tenors)

        ref = self.reference_date()
        self._option_dates: list[Date] = [
            self.option_date_from_tenor(p) for p in self._option_tenors
        ]
        self._option_times: list[float] = [
            day_counter.year_fraction(ref, d) for d in self._option_dates
        ]
        self._swap_lengths: list[float] = [
            self.swap_length(p) for p in self._swap_tenors
        ]

    # --- inspectors ------------------------------------------------------

    def option_tenors(self) -> list[Period]:
        return list(self._option_tenors)

    def option_dates(self) -> list[Date]:
        return list(self._option_dates)

    def option_times(self) -> list[float]:
        return list(self._option_times)

    def swap_tenors(self) -> list[Period]:
        return list(self._swap_tenors)

    def swap_lengths(self) -> list[float]:
        return list(self._swap_lengths)

    # --- TermStructure interface (subclass still must override) ----------

    @abstractmethod
    def max_date(self) -> Date:
        ...

    @abstractmethod
    def _volatility_impl(
        self, option_time: float, swap_length: float, strike: float
    ) -> float:
        ...
