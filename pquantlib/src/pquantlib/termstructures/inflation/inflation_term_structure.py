"""InflationTermStructure — abstract base for inflation curves.

# C++ parity: ql/termstructures/inflationtermstructure.{hpp,cpp} (v1.42.1).

C++ supports three constructor modes (delegated, fixed, moving) inherited
from ``TermStructure``. We keep the same three modes via the existing
TermStructure base. C++ also carries a base-rate, a frequency, a base
date (set in the constructor), and an optional seasonality.

L7-A spec divergences (documented):

- C++ ``observationLag()`` is ``[[deprecated]]`` in v1.42.1. The L7-A
  spec re-instates ``observation_lag`` as an explicit, non-deprecated
  constructor argument because:
  * older C++ versions kept the field active,
  * downstream YoY cap/floor engines (L7-D) and inflation cashflows
    (L7-C) still need a lag for `index.fixing(d - observation_lag)`-style
    plumbing.
- C++ v1.42.1 dropped the ``nominal_term_structure`` accessor. The L7-A
  spec re-adds it as a typed Protocol slot because Black-style YoY
  cap/floor engines must discount on a nominal curve that is decoupled
  from the inflation curve itself; storing it on the abstract is the
  cleanest Pythonic seam.
- We treat both as opt-in: if a caller doesn't supply them, the accessor
  returns ``None`` (Pythonic optional). Concrete L7-D engines may
  ``qassert.require`` they are present before pricing.

The C++ ``hasExplicitBaseDate()`` is also ``[[deprecated]]`` — modern
curves always carry one, so we don't expose it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period

if TYPE_CHECKING:
    from pquantlib.termstructures.inflation.seasonality import Seasonality
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class InflationTermStructure(TermStructure):
    """Abstract base for inflation term structures.

    # C++ parity: ``InflationTermStructure`` in inflationtermstructure.hpp.
    # See module docstring for divergences from v1.42.1.
    """

    def __init__(
        self,
        *,
        base_date: Date,
        frequency: Frequency,
        day_counter: DayCounter,
        observation_lag: Period | None = None,
        nominal_term_structure: YieldTermStructureProtocol | None = None,
        base_rate: float | None = None,
        seasonality: Seasonality | None = None,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._base_date: Date = base_date
        self._frequency: Frequency = frequency
        self._observation_lag: Period | None = observation_lag
        self._nominal_term_structure: YieldTermStructureProtocol | None = (
            nominal_term_structure
        )
        self._base_rate: float | None = base_rate
        self._seasonality: Seasonality | None = None
        if seasonality is not None:
            self.set_seasonality(seasonality)

    # ---- inflation interface -----------------------------------------

    def frequency(self) -> Frequency:
        return self._frequency

    def base_date(self) -> Date:
        return self._base_date

    def observation_lag(self) -> Period | None:
        return self._observation_lag

    def nominal_term_structure(self) -> YieldTermStructureProtocol | None:
        return self._nominal_term_structure

    def base_rate(self) -> float:
        """Return the base rate.

        # C++ parity: ``baseRate()`` raises if no base rate is set; we
        # mirror that — clients that need an optional getter should
        # branch on ``has_base_rate()`` first.
        """
        qassert.require(self._base_rate is not None, "base rate not available")
        assert self._base_rate is not None
        return self._base_rate

    def has_base_rate(self) -> bool:
        return self._base_rate is not None

    # ---- seasonality --------------------------------------------------

    def seasonality(self) -> Seasonality | None:
        return self._seasonality

    def has_seasonality(self) -> bool:
        return self._seasonality is not None

    def set_seasonality(self, seasonality: Seasonality | None) -> None:
        """Install (or clear) the seasonality and re-validate consistency.

        # C++ parity: ``InflationTermStructure::setSeasonality`` always
        # resets and re-checks ``isConsistent``. We propagate ``update()``
        # through the Observable base.
        """
        self._seasonality = seasonality
        if seasonality is not None:
            qassert.require(
                seasonality.is_consistent(self),
                "Seasonality inconsistent with inflation term structure",
            )
        self.update()

    # ---- range checks -------------------------------------------------

    def check_range(self, d: Date, extrapolate: bool) -> None:
        """C++ parity: ``InflationTermStructure::checkRange(Date, bool)``.

        Overrides ``TermStructure.check_range`` which checks against
        ``reference_date()``; inflation curves check against
        ``base_date()`` (which may sit earlier than the reference date).
        """
        qassert.require(
            d >= self.base_date(),
            f"date ({d}) is before base date ({self.base_date()})",
        )
        qassert.require(
            extrapolate or self.allows_extrapolation() or d <= self.max_date(),
            f"date ({d}) is past max curve date ({self.max_date()})",
        )

    def check_time_range(self, t: float, extrapolate: bool) -> None:
        """C++ parity: ``InflationTermStructure::checkRange(Time, bool)``."""
        base_t = self.day_counter().year_fraction(self.reference_date(), self.base_date())
        qassert.require(t >= base_t, f"time ({t}) is before base date")
        qassert.require(
            extrapolate or self.allows_extrapolation() or t <= self.max_time(),
            f"time ({t}) is past max curve time ({self.max_time()})",
        )
