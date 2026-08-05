"""ImpliedVolTermStructure — a Black vol surface seen from a future date.

# C++ parity: ql/termstructures/volatility/equityfx/impliedvoltermstructure.hpp
# (v1.43).

Given an existing Black vol surface and a future date, this structure
exposes the vol surface *as it will be seen from that date*: variance
measured from the new reference date is the original surface's forward
variance between the two dates (``blackVarianceImpl``,
impliedvoltermstructure.hpp:96-110).

Used by :class:`~pquantlib.pricingengines.forward.forward_vanilla_engine.ForwardVanillaEngine`
to build the process a forward-starting option sees at its reset date.

The structure stays linked to the original surface.

C++ warning, carried over: it makes no financial sense to imply an
asset-dependent vol surface forward — use this only with structures that
are time-dependent.

Not ported: the ``accept(AcyclicVisitor&)`` override; pquantlib has no
acyclic-visitor infrastructure.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
    BlackVolTermStructure,
)
from pquantlib.time.date import Date


class ImpliedVolTermStructure(BlackVarianceTermStructure):
    """Black vol surface implied at a given future date.

    # C++ parity: ``class ImpliedVolTermStructure : public
    # BlackVarianceTermStructure``.

    Args:
        original_ts: the surface to imply forward.
        reference_date: the implied reference date.
    """

    def __init__(self, original_ts: BlackVolTermStructure, reference_date: Date) -> None:
        super().__init__(reference_date=reference_date)
        self._original_ts: BlackVolTermStructure = original_ts
        original_ts.register_with(self)

    # --- TermStructure interface ---------------------------------------------

    def day_counter(self) -> DayCounter:
        return self._original_ts.day_counter()

    def max_date(self) -> Date:
        return self._original_ts.max_date()

    # --- VolatilityTermStructure interface -----------------------------------

    def min_strike(self) -> float:
        return self._original_ts.min_strike()

    def max_strike(self) -> float:
        return self._original_ts.max_strike()

    # --- BlackVarianceTermStructure implementation ----------------------------

    def _black_variance_impl(self, t: float, strike: float) -> float:
        """Forward variance of the original surface over the implied window.

        # C++ parity: ``ImpliedVolTermStructure::blackVarianceImpl``
        # (impliedvoltermstructure.hpp:96-110). The shift is recomputed on
        # every call rather than cached, because the original curve may
        # move between invocations.
        """
        time_shift = self.day_counter().year_fraction(
            self._original_ts.reference_date(), self.reference_date()
        )
        # ``t`` is relative to this structure's reference date; convert it to
        # the original curve's time axis before asking for forward variance.
        return self._original_ts.black_forward_variance_at_time(time_shift, time_shift + t, strike, True)


__all__ = ["ImpliedVolTermStructure"]
