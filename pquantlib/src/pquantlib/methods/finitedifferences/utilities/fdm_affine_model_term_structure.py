"""FdmAffineModelTermStructure — a curve implied by an affine model state.

# C++ parity: ql/methods/finitedifferences/utilities/fdmaffinemodeltermstructure.{hpp,cpp}
# @ v1.43 (6b57206e0).

Wraps an :class:`~pquantlib.models.model.AffineModel` and a state vector
``r`` into a ``YieldTermStructure`` anchored at ``reference_date``::

    discount(T) = model.discount_bond(t, T + t, r)

where ``t = dayCounter.yearFraction(modelReferenceDate, referenceDate)`` is
the model time of the anchor. ``set_variable`` swaps the state in place and
notifies observers, so the FD sweep can reuse one curve object across grid
nodes (this is exactly how ``FdmAffineModelSwapInnerValue`` drives it).
"""

from __future__ import annotations

from typing import Protocol, final, runtime_checkable

import numpy as np

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.patterns.observer import Observer
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


@runtime_checkable
class AffineModelLike(Protocol):
    """The slice of C++'s ``AffineModel`` this curve actually uses.

    # C++ parity: the constructor takes ``ext::shared_ptr<AffineModel>``.

    PQuantLib spells that ABC ``pquantlib.models.model.AffineModel``, but
    its ``G2`` — unlike C++'s ``class G2 : public TwoFactorModel, public
    AffineModel, public TermStructureConsistentModel`` — does not declare
    ``AffineModel`` as a base even though it implements the whole surface.
    Typing the parameter nominally would therefore reject ``G2``, which is
    one of only two models C++ instantiates this curve with. A structural
    type keeps both ``HullWhite`` (a real ``AffineModel``) and ``G2``
    admissible without weakening the contract.

    ``discount_bond`` is declared positional-only because the concretes
    disagree on the third parameter's name (``factors`` vs ``x_or_factors``).
    """

    def discount_bond(self, now: float, maturity: float, factors: Array, /) -> float: ...

    def register_with(self, observer: Observer, /) -> None: ...


@final
class FdmAffineModelTermStructure(YieldTermStructure):
    """Yield curve implied by an affine short-rate model at a fixed state.

    # C++ parity: ``class FdmAffineModelTermStructure : public
    # YieldTermStructure``.
    """

    def __init__(
        self,
        r: Array,
        cal: Calendar | None,
        day_counter: DayCounter,
        reference_date: Date,
        model_reference_date: Date,
        model: AffineModelLike,
    ) -> None:
        # ``cal`` is ``Calendar | None`` rather than ``Calendar`` because the
        # C++ callers pass ``discount->calendar()``, which is a *default
        # constructed* (empty) ``Calendar`` for curves such as ``FlatForward``.
        # PQuantLib models the empty calendar as ``None`` (``TermStructure``'s
        # "none-calendar mode"), so the null case has to be representable
        # here too. The calendar is never read by this curve — the reference
        # date is fixed at construction.
        super().__init__(
            reference_date=reference_date, calendar=cal, day_counter=day_counter
        )
        self._r: Array = np.asarray(r, dtype=np.float64)
        self._t: float = day_counter.year_fraction(model_reference_date, reference_date)
        self._model: AffineModelLike = model
        # C++ parity: ``registerWith(model_)`` — in PQuantLib the observable
        # owns the registration call, so the direction of the call is
        # inverted relative to C++'s Observer::registerWith.
        model.register_with(self)

    def max_date(self) -> Date:
        """# C++ parity: ``Date::maxDate()``."""
        return Date.max_date()

    def set_variable(self, r: Array) -> None:
        """Replace the model state and notify observers.

        # C++ parity: ``void setVariable(const Array& r)``.
        """
        self._r = np.asarray(r, dtype=np.float64)
        self.notify_observers()

    def _discount_impl(self, t: float) -> float:
        """# C++ parity: ``discountImpl(Time T)`` — ``model->discountBond(t_, T+t_, r_)``."""
        return self._model.discount_bond(self._t, t + self._t, self._r)


__all__ = ["AffineModelLike", "FdmAffineModelTermStructure"]
