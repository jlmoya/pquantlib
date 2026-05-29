"""FactorSpreadedHazardRateCurve — multiplicative hazard-rate spread.

# C++ parity: ql/experimental/credit/factorspreadedhazardratecurve.hpp
   (v1.42.1).

Given a base ``DefaultProbabilityTermStructure`` and a multiplicative
``Quote`` ``factor``, the resulting hazard rate is::

    h(t) = h_base(t) * (1 + factor)

The survival probability is derived by the
:class:`pquantlib.termstructures.credit.hazard_rate_structure.HazardRateStructure`
adapter via quadrature on ``h(t)``.

The class remains linked to the base curve and the quote — any update
propagates through observer notification.
"""

from __future__ import annotations

from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.credit.hazard_rate_structure import HazardRateStructure


class FactorSpreadedHazardRateCurve(HazardRateStructure):
    """Multiplicative-spread default-probability term structure.

    Constructed from a base curve and a multiplicative ``factor`` Quote.

    # C++ parity divergence: the C++ ctor takes a ``Handle<...>`` on both
    # the base curve and the spread quote. The Python port unwraps to
    # plain references; observability is wired through ``register_with``.
    """

    def __init__(
        self,
        base: DefaultProbabilityTermStructure,
        factor: Quote,
    ) -> None:
        # The base curve carries the day-counter, calendar, and reference
        # date; we mirror its construction so the inherited TermStructure
        # state is consistent (eval-date mode).
        super().__init__(
            reference_date=None,
            calendar=None,
            day_counter=base.day_counter(),
            jumps=None,
            jump_dates=None,
        )
        self._base: DefaultProbabilityTermStructure = base
        self._factor: Quote = factor
        base.register_with(self)
        factor.register_with(self)

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
        """h(t) = h_base(t) * (1 + factor)."""
        # C++ parity: factorspreadedhazardratecurve.hpp:90-92.
        return self._base.hazard_rate(t, True) * (1.0 + self._factor.value())


__all__ = ["FactorSpreadedHazardRateCurve"]
