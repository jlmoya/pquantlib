"""LocalVolTermStructure view of an Andreasen-Huge interpolation.

# C++ parity: ql/termstructures/volatility/equityfx/andreasenhugelocalvoladapter.hpp +
#             andreasenhugelocalvoladapter.cpp (v1.43).

The thin half of the pair: :class:`AndreasenHugeVolatilityInterpl` already
produces local volatility, so this class only has to present it through the
``LocalVolTermStructure`` interface. Two things it does add:

* the strike is **clamped** into the interpolation's own
  ``[min_strike, max_strike]`` before the lookup (cpp:46-48), so a query
  outside the calibration grid returns the edge value rather than an
  extrapolated one;
* its own ``min_strike`` / ``max_strike`` are ``0`` / ``QL_MAX_REAL``, i.e.
  deliberately *wider* than the interpolation's, which is what makes the
  clamp above the operative bound.

Metadata is delegated to the interpolation's risk-free curve, so — as with
:class:`AndreasenHugeVolatilityAdapter` — ``calendar()`` and
``settlement_days()`` throw when that curve was built on an explicit
reference date. That is the pass-through, not a gap.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.termstructures.volatility.equity_fx.andreasen_huge_volatility_interpl import (
    AndreasenHugeVolatilityInterpl,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class AndreasenHugeLocalVolAdapter(LocalVolTermStructure):
    """``LocalVolTermStructure`` over an Andreasen-Huge interpolation.

    # C++ parity: ``class AndreasenHugeLocalVolAdapter``
    # (andreasenhugelocalvoladapter.hpp:34).
    """

    def __init__(self, local_vol: AndreasenHugeVolatilityInterpl) -> None:
        # C++ default-constructs its base (delegated mode); every accessor is
        # overridden below to read the risk-free curve.
        super().__init__(business_day_convention=BusinessDayConvention.Following)
        self._local_vol: AndreasenHugeVolatilityInterpl = local_vol

    # --- TermStructure interface, all delegated ----------------------------

    def max_date(self) -> Date:
        """# C++ parity: andreasenhugelocalvoladapter.cpp:32-34."""
        return self._local_vol.max_date()

    def min_strike(self) -> float:
        """# C++ parity: cpp:36-38 — 0, not the interpolation's own bound."""
        return 0.0

    def max_strike(self) -> float:
        """# C++ parity: cpp:40-42 — ``QL_MAX_REAL``."""
        return QL_MAX_REAL

    def calendar(self) -> Calendar:
        """# C++ parity: cpp:51-53. Throws when the curve has no calendar."""
        return self._local_vol.risk_free_rate().calendar()

    def day_counter(self) -> DayCounter:
        """# C++ parity: cpp:54-56."""
        return self._local_vol.risk_free_rate().day_counter()

    def settlement_days(self) -> int:
        """# C++ parity: cpp:60-62. Throws when the curve has none set."""
        return self._local_vol.risk_free_rate().settlement_days()

    def reference_date(self) -> Date:
        """# C++ parity: cpp:57-59."""
        return self._local_vol.risk_free_rate().reference_date()

    # --- LocalVolTermStructure hook ----------------------------------------

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        """# C++ parity: ``localVolImpl`` (cpp:44-49).

        The strike is clamped into the interpolation's calibrated range first.
        """
        return self._local_vol.local_vol(
            t,
            min(self._local_vol.max_strike(), max(self._local_vol.min_strike(), underlying_level)),
        )


__all__ = ["AndreasenHugeLocalVolAdapter"]
