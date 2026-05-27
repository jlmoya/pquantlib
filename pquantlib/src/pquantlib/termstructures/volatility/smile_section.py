"""SmileSection — abstract volatility smile at one expiry.

# C++ parity: ql/termstructures/volatility/smilesection.hpp + smilesection.cpp (v1.42.1).

The C++ base supports both *date-anchored* construction (an exercise
``Date`` plus an optional reference ``Date``; if the reference is left
default, the section *floats* by registering with
``Settings::instance().evaluationDate()``) and *time-anchored*
construction (a raw ``Time`` exercise).

PQuantLib L2-E ports both modes but keeps the floating variant in an
**explicit** form: callers supply a reference date when constructing
a date-anchored section. Mode "let the global evaluation date drift
the reference" is deferred until ObservableSettings.evaluation_date
becomes observable — the same deferral as TermStructure mode 3.

Notes:

- ``variance(strike)`` is provided by the base as
  ``volatility(strike)^2 * exercise_time``. Subclasses override
  ``_volatility_impl`` only; ``_variance_impl`` is overridable for the
  rare cases where the variance is computed directly.
- ``atm_level()`` is abstract — concretes either store it (as
  ``FlatSmileSection`` does) or derive it from a referenced curve.
- The expensive option-pricing methods (``option_price``,
  ``digital_option_price``, ``vega``, ``density``, the implied-vol
  conversion ``volatility(strike, type, shift)``) require a ``BlackFormula``
  port that isn't in L2-E. They will land alongside vanilla pricing in
  Phase 3 (L3 instruments + pricingengines).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.patterns.observer import Observable
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class SmileSection(Observable, ABC):
    """Abstract volatility smile at a single expiry.

    Subclasses MUST override ``min_strike()``, ``max_strike()``,
    ``atm_level()``, and ``_volatility_impl(strike)``. They MAY override
    ``_variance_impl(strike)`` if the variance is computed directly
    rather than as ``vol^2 * t``.
    """

    def __init__(
        self,
        *,
        exercise_date: Date | None = None,
        exercise_time: float | None = None,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
    ) -> None:
        """Construct a smile section.

        Two construction modes:

        - **Date-anchored**: pass ``exercise_date`` AND ``day_counter``
          AND ``reference_date``. ``exercise_time`` is then derived as
          ``day_counter.year_fraction(reference_date, exercise_date)``.
          ``exercise_date >= reference_date`` is enforced.
        - **Time-anchored**: pass ``exercise_time`` (a non-negative
          float). ``day_counter`` is optional in this mode (it falls
          back to ``DayCounter()`` semantically; PQuantLib stores
          ``None`` in that case). ``exercise_date`` and
          ``reference_date`` remain ``None``.
        """
        Observable.__init__(self)

        # Determine which constructor mode was used.
        if exercise_date is not None:
            qassert.require(
                reference_date is not None,
                "SmileSection date-anchored mode requires a reference_date "
                "(floating-via-global-evaluation-date is deferred)",
            )
            qassert.require(
                day_counter is not None,
                "SmileSection date-anchored mode requires a day_counter",
            )
            assert reference_date is not None
            assert day_counter is not None
            qassert.require(
                exercise_date >= reference_date,
                f"expiry date ({exercise_date}) must be greater than or equal "
                f"to reference date ({reference_date})",
            )
            self._exercise_date: Date | None = exercise_date
            self._reference_date: Date | None = reference_date
            self._day_counter: DayCounter | None = day_counter
            self._exercise_time: float = day_counter.year_fraction(reference_date, exercise_date)
        else:
            qassert.require(
                exercise_time is not None,
                "SmileSection time-anchored mode requires an exercise_time",
            )
            assert exercise_time is not None
            qassert.require(
                exercise_time >= 0.0,
                f"expiry time must be non-negative: {exercise_time} not allowed",
            )
            self._exercise_date = None
            self._reference_date = None
            self._day_counter = day_counter
            self._exercise_time = exercise_time

        self._volatility_type: VolatilityType = volatility_type
        self._shift: float = shift

    # --- subclass-implemented hooks ----------------------------------------

    @abstractmethod
    def min_strike(self) -> float:
        """Smallest strike for which the smile returns volatilities."""

    @abstractmethod
    def max_strike(self) -> float:
        """Largest strike for which the smile returns volatilities."""

    @abstractmethod
    def atm_level(self) -> float:
        """At-the-money level. May raise if the section can't supply one."""

    @abstractmethod
    def _volatility_impl(self, strike: float) -> float:
        """Subclass: return the volatility at ``strike``."""

    def _variance_impl(self, strike: float) -> float:
        """Default variance impl: ``vol(strike)^2 * exercise_time``.

        # C++ parity: SmileSection::varianceImpl
        """
        v = self._volatility_impl(strike)
        return v * v * self._exercise_time

    # --- public API --------------------------------------------------------

    def volatility(self, strike: float) -> float:
        return self._volatility_impl(strike)

    def variance(self, strike: float) -> float:
        return self._variance_impl(strike)

    def shift(self) -> float:
        return self._shift

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def exercise_time(self) -> float:
        return self._exercise_time

    def exercise_date(self) -> Date:
        qassert.require(
            self._exercise_date is not None,
            "exercise date not available for this time-anchored SmileSection",
        )
        assert self._exercise_date is not None
        return self._exercise_date

    def reference_date(self) -> Date:
        qassert.require(
            self._reference_date is not None,
            "referenceDate not available for this instance",
        )
        assert self._reference_date is not None
        return self._reference_date

    def day_counter(self) -> DayCounter:
        qassert.require(
            self._day_counter is not None,
            "day counter not available for this SmileSection",
        )
        assert self._day_counter is not None
        return self._day_counter

    def update(self) -> None:
        """Observer.update — propagates to own observers.

        Floating-via-global-evaluation-date mode is deferred (see class
        docstring); this implementation therefore just notifies.
        """
        self.notify_observers()
