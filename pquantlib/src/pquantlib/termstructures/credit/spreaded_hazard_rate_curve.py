"""SpreadedHazardRateCurve — additive hazard-rate spread.

# C++ parity: ql/experimental/credit/spreadedhazardratecurve.hpp
   (v1.42.1).

Given a base ``DefaultProbabilityTermStructure`` and an additive
``Quote`` ``spread``, the resulting hazard rate is::

    h(t) = h_base(t) + spread

The survival probability is derived by the
:class:`pquantlib.termstructures.credit.hazard_rate_structure.HazardRateStructure`
adapter via quadrature on ``h(t)``.

Remains linked to the base curve and the quote — any update propagates
through observer notification.
"""

from __future__ import annotations

from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.credit.hazard_rate_structure import HazardRateStructure


class SpreadedHazardRateCurve(HazardRateStructure):
    """Additive-spread default-probability term structure.

    Constructed from a base curve and an additive ``spread`` Quote.

    # C++ parity divergence: ctor unwraps the C++ ``Handle<...>`` to plain
    # refs; observability is wired through ``register_with``.
    """

    def __init__(
        self,
        base: DefaultProbabilityTermStructure,
        spread: Quote,
    ) -> None:
        # Mirror the base curve's day-counter so the inherited
        # TermStructure state is consistent (eval-date mode).
        super().__init__(
            reference_date=None,
            calendar=None,
            day_counter=base.day_counter(),
            jumps=None,
            jump_dates=None,
        )
        self._base: DefaultProbabilityTermStructure = base
        self._spread: Quote = spread
        base.register_with(self)
        spread.register_with(self)

    # ---- TermStructure interface ------------------------------------------

    def day_counter(self):  # type: ignore[no-untyped-def]
        # C++ parity: ``originalCurve_->dayCounter()``.
        return self._base.day_counter()

    def calendar(self):  # type: ignore[no-untyped-def]
        # C++ parity: ``originalCurve_->calendar()``.
        return self._base.calendar()

    def reference_date(self):  # type: ignore[no-untyped-def]
        # C++ parity: ``originalCurve_->referenceDate()``.
        return self._base.reference_date()

    def max_date(self):  # type: ignore[no-untyped-def]
        # C++ parity: ``originalCurve_->maxDate()``.
        return self._base.max_date()

    def max_time(self) -> float:
        # C++ parity: ``originalCurve_->maxTime()``.
        return self._base.max_time()

    # ---- HazardRateStructure override -------------------------------------

    def _hazard_rate_impl(self, t: float) -> float:
        """h(t) = h_base(t) + spread."""
        # C++ parity: spreadedhazardratecurve.hpp:92-94.
        return self._base.hazard_rate(t, True) + self._spread.value()


__all__ = ["SpreadedHazardRateCurve"]
