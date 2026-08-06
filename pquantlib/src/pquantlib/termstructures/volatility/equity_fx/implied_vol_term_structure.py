"""ImpliedVolTermStructure — a Black vol surface seen from a future date.

# C++ parity: ql/termstructures/volatility/equityfx/impliedvoltermstructure.hpp
# (v1.43, header-only).

Wraps an existing :class:`BlackVolTermStructure` and re-anchors it at a
future ``reference_date``. The variance quoted at time ``t`` from the new
anchor is the *forward* variance of the original structure between the
shift and the shifted horizon::

    time_shift = day_counter.year_fraction(original.reference_date(),
                                           reference_date)
    variance(t, K) = original.black_forward_variance(time_shift,
                                                     time_shift + t, K,
                                                     extrapolate=True)

``time_shift`` is recomputed on every call rather than cached: the wrapped
structure may move underneath (this class stays *linked* to it, and
observes it), so a cached shift could go stale.

Warning, carried over from C++: it does not make financial sense to build
an implied vol term structure over an asset-dependent surface. Use it with
time-dependent-only structures.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
    BlackVolTermStructure,
)
from pquantlib.time.date import Date


class ImpliedVolTermStructure(BlackVarianceTermStructure):
    """Black vol term structure implied at a future reference date."""

    def __init__(
        self,
        *,
        original_ts: BlackVolTermStructure,
        reference_date: Date,
    ) -> None:
        # C++ parity: ``BlackVarianceTermStructure(referenceDate)`` — no
        # calendar and no day counter of its own; ``day_counter()`` is
        # overridden below to delegate, and ``calendar()`` is left
        # unavailable exactly as the default-constructed C++ Calendar is.
        super().__init__(reference_date=reference_date)
        self._original_ts: BlackVolTermStructure = original_ts
        self._original_ts.register_with(self)

    # --- TermStructure interface -------------------------------------------

    def day_counter(self) -> DayCounter:
        return self._original_ts.day_counter()

    def max_date(self) -> Date:
        return self._original_ts.max_date()

    # --- VolatilityTermStructure interface ---------------------------------

    def min_strike(self) -> float:
        return self._original_ts.min_strike()

    def max_strike(self) -> float:
        return self._original_ts.max_strike()

    # --- BlackVarianceTermStructure hook -----------------------------------

    def _black_variance_impl(self, t: float, strike: float) -> float:
        # The time shift (and hence the variance at the evaluation date)
        # cannot be cached: the original curve may change between calls.
        time_shift = self.day_counter().year_fraction(
            self._original_ts.reference_date(), self.reference_date()
        )
        # ``t`` is relative to this structure's reference date; convert it
        # to the original curve's clock before asking for forward variance.
        return self._original_ts.black_forward_variance_at_time(
            time_shift, time_shift + t, strike, True
        )


__all__ = ["ImpliedVolTermStructure"]
