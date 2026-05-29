"""StrippedOptionlet — concrete container for exogenous caplet vols.

# C++ parity: ql/termstructures/volatility/optionlet/strippedoptionlet.{hpp,cpp}
# (v1.42.1).

Wraps a matrix of caplet volatilities (rows = option dates, columns =
strikes) plus their fixing schedule. Strikes can be either a single
shared vector or a per-date vector — the constructor accepts the
shared-vector form (the most common; the per-date variant is in the
carve-outs).
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_base import (
    StrippedOptionletBase,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class StrippedOptionlet(StrippedOptionletBase):
    """Concrete container for exogenous (precomputed) caplet vols."""

    def __init__(
        self,
        *,
        settlement_days: int,
        calendar: Calendar,
        business_day_convention: BusinessDayConvention,
        ibor_index: object,
        optionlet_dates: Sequence[Date],
        strikes: Sequence[float],
        optionlet_volatilities: Sequence[Sequence[float]],
        day_counter: DayCounter,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
    ) -> None:
        qassert.require(len(optionlet_dates) > 0, "no optionlet dates given")
        qassert.require(len(strikes) > 0, "no strikes given")
        qassert.require(
            len(optionlet_volatilities) == len(optionlet_dates),
            f"vol rows ({len(optionlet_volatilities)}) mismatch dates "
            f"({len(optionlet_dates)})",
        )
        for i, row in enumerate(optionlet_volatilities):
            qassert.require(
                len(row) == len(strikes),
                f"vol row {i} length ({len(row)}) mismatch strikes "
                f"({len(strikes)})",
            )

        self._settlement_days: int = settlement_days
        self._calendar: Calendar = calendar
        self._business_day_convention: BusinessDayConvention = business_day_convention
        self._ibor_index: object = ibor_index
        self._optionlet_dates: list[Date] = list(optionlet_dates)
        self._strikes: list[float] = [float(s) for s in strikes]
        self._optionlet_volatilities: list[list[float]] = [
            [float(v) for v in row] for row in optionlet_volatilities
        ]
        self._day_counter: DayCounter = day_counter
        self._volatility_type: VolatilityType = volatility_type
        self._displacement: float = displacement

        # Optionlet fixing times — derived against the first optionlet
        # date as a proxy reference. The C++ class chains times off
        # the term-vol surface's reference; PQuantLib uses the
        # explicit-fixing-date form. Real consumers
        # (``OptionletStripper1``, the adapter) override
        # ``optionlet_fixing_times`` indirectly via their own
        # reference-date wiring, so this proxy is only used by
        # standalone ``StrippedOptionlet`` instances built from already-
        # stripped vols.
        self._optionlet_times: list[float] = [
            day_counter.year_fraction(optionlet_dates[0], d) for d in optionlet_dates
        ]
        # ATM rates default to NaN; a stripper subclass can override.
        self._atm_rates: list[float] = [float("nan")] * len(optionlet_dates)

    # --- StrippedOptionletBase interface --------------------------------

    def optionlet_strikes(self, i: int) -> list[float]:
        _ = i  # strikes shared across dates
        return list(self._strikes)

    def optionlet_volatilities(self, i: int) -> list[float]:
        return list(self._optionlet_volatilities[i])

    def optionlet_fixing_dates(self) -> list[Date]:
        return list(self._optionlet_dates)

    def optionlet_fixing_times(self) -> list[float]:
        return list(self._optionlet_times)

    def optionlet_maturities(self) -> int:
        return len(self._optionlet_dates)

    def atm_optionlet_rates(self) -> list[float]:
        return list(self._atm_rates)

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def calendar(self) -> Calendar:
        return self._calendar

    def settlement_days(self) -> int:
        return self._settlement_days

    def business_day_convention(self) -> BusinessDayConvention:
        return self._business_day_convention

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def displacement(self) -> float:
        return self._displacement
