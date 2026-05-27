"""LocalVolTermStructure — abstract base for local volatility surfaces.

# C++ parity: ql/termstructures/volatility/equityfx/localvoltermstructure.hpp +
#             localvoltermstructure.cpp (v1.42.1).

Local volatility (Dupire) — ``sigma_L(t, S)`` is the diffusion coefficient
of a one-factor local-vol diffusion ``dS = mu dt + sigma_L(t,S) S dW``.
Subclasses implement ``_local_vol_impl(t, underlying_level)``; the
``local_vol`` public method performs range + strike checks first and
then delegates.

Defaults the business-day convention to ``Following``, mirroring C++.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility_term_structure import VolatilityTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class LocalVolTermStructure(VolatilityTermStructure):
    """Abstract local volatility term structure."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.Following,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )

    # --- subclass-implemented hook ----------------------------------------

    @abstractmethod
    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        """Subclass: local vol at (``t``, ``underlying_level``).

        Range and strike-range checks have already been performed; treat
        the call as if extrapolation is required.
        """

    # --- public API --------------------------------------------------------

    def local_vol(
        self, d: Date, underlying_level: float, extrapolate: bool = False
    ) -> float:
        self.check_range(d, extrapolate)
        self.check_strike(underlying_level, extrapolate)
        t = self.time_from_reference(d)
        return self._local_vol_impl(t, underlying_level)

    def local_vol_at_time(
        self, t: float, underlying_level: float, extrapolate: bool = False
    ) -> float:
        """Time-anchored variant."""
        self.check_time_range(t, extrapolate)
        self.check_strike(underlying_level, extrapolate)
        return self._local_vol_impl(t, underlying_level)
