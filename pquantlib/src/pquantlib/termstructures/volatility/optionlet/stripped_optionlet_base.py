"""StrippedOptionletBase — abstract container for stripped caplet vols.

# C++ parity: ql/termstructures/volatility/optionlet/strippedoptionletbase.hpp
# (v1.42.1).

Defines the interface for a vector of caplet vols indexed by
(fixing_date, strike). Subclasses implement either the static
container (``StrippedOptionlet``) or a stripper that derives the
matrix from cap term vols (``OptionletStripper``).

C++ inherits from ``LazyObject``; PQuantLib's port skips the lazy
caching here (subclasses can opt in via their own ``calculate`` if
they need it). The minimal interface that downstream consumers
(``StrippedOptionletAdapter``) require is faithfully ported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class StrippedOptionletBase(ABC):
    """Abstract base interface for stripped optionlets."""

    @abstractmethod
    def optionlet_strikes(self, i: int) -> list[float]:
        """Return the strike vector at the i-th optionlet date."""

    @abstractmethod
    def optionlet_volatilities(self, i: int) -> list[float]:
        """Return the vol vector at the i-th optionlet date."""

    @abstractmethod
    def optionlet_fixing_dates(self) -> list[Date]:
        """Return the optionlet fixing dates."""

    @abstractmethod
    def optionlet_fixing_times(self) -> list[float]:
        """Return the optionlet fixing times."""

    @abstractmethod
    def optionlet_maturities(self) -> int:
        """Number of optionlet maturities held by this base."""

    @abstractmethod
    def atm_optionlet_rates(self) -> list[float]:
        """ATM forward rates per optionlet (live or cached)."""

    @abstractmethod
    def day_counter(self) -> DayCounter:
        """Day counter."""

    @abstractmethod
    def calendar(self) -> Calendar:
        """Calendar."""

    @abstractmethod
    def settlement_days(self) -> int:
        """Settlement days."""

    @abstractmethod
    def business_day_convention(self) -> BusinessDayConvention:
        """Business day convention."""

    @abstractmethod
    def volatility_type(self) -> VolatilityType:
        """Vol type."""

    @abstractmethod
    def displacement(self) -> float:
        """Displacement for shifted-lognormal vols."""
