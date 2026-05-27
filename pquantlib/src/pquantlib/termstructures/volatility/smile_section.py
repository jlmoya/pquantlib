"""SmileSection — abstract volatility smile at one expiry.

# C++ parity: ql/termstructures/volatility/smilesection.hpp + smilesection.cpp (v1.42.1).

The C++ base supports both *date-anchored* construction (an exercise
``Date`` plus an optional reference ``Date``; if the reference is left
default, the section *floats* by registering with
``Settings::instance().evaluationDate()``) and *time-anchored*
construction (a raw ``Time`` exercise).

PQuantLib L2-E ported both modes, but kept the floating variant
disabled pending L3-A's Settings.evaluation_date observable wiring.
Floating mode now works: construct with ``exercise_date`` AND
``day_counter`` but leave ``reference_date=None``; the section
registers with ``ObservableSettings()`` so its reference date and
exercise time re-snap on every eval-date change.

Notes:

- ``variance(strike)`` is provided by the base as
  ``volatility(strike)^2 * exercise_time``. Subclasses override
  ``_volatility_impl`` only; ``_variance_impl`` is overridable for the
  rare cases where the variance is computed directly.
- ``atm_level()`` is abstract — concretes either store it (as
  ``FlatSmileSection`` does) or derive it from a referenced curve.
- The expensive option-pricing methods (``option_price``,
  ``digital_option_price``, ``vega``, ``density``, the implied-vol
  conversion ``volatility(strike, type, shift)``) require ``BlackFormula``
  and will be wired during Phase 3 L3-D (vanilla option pricing).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.patterns.observable_settings import ObservableSettings
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

        Three construction modes:

        - **Date-anchored, fixed reference**: pass ``exercise_date`` AND
          ``day_counter`` AND ``reference_date``. ``exercise_time`` is
          derived as
          ``day_counter.year_fraction(reference_date, exercise_date)``.
          ``exercise_date >= reference_date`` is enforced.
        - **Date-anchored, floating reference**: pass ``exercise_date``
          AND ``day_counter``, leave ``reference_date=None``. The section
          registers with ``ObservableSettings()`` so the reference date
          re-snaps to ``evaluation_date_or_today()`` whenever the global
          eval date moves; the cached ``exercise_time`` is recomputed
          lazily on each read.
        - **Time-anchored**: pass ``exercise_time`` (a non-negative
          float). ``day_counter`` is optional in this mode.
          ``exercise_date`` and ``reference_date`` remain ``None``.
        """
        Observable.__init__(self)

        self._floating: bool = False
        if exercise_date is not None:
            qassert.require(
                day_counter is not None,
                "SmileSection date-anchored mode requires a day_counter",
            )
            assert day_counter is not None
            if reference_date is None:
                # Floating mode — register with Settings to invalidate
                # the cached exercise_time on eval-date changes.
                self._floating = True
                ObservableSettings().register_with(self)
                self._exercise_date: Date | None = exercise_date
                self._reference_date: Date | None = None
                self._day_counter: DayCounter | None = day_counter
                self._exercise_time: float = 0.0  # filled by _refresh_floating
                self._refresh_floating()
            else:
                qassert.require(
                    exercise_date >= reference_date,
                    f"expiry date ({exercise_date}) must be greater than or equal "
                    f"to reference date ({reference_date})",
                )
                self._exercise_date = exercise_date
                self._reference_date = reference_date
                self._day_counter = day_counter
                self._exercise_time = day_counter.year_fraction(reference_date, exercise_date)
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

    # --- floating-mode helper ----------------------------------------------

    def _refresh_floating(self) -> None:
        """Re-snap the floating reference date + exercise time."""
        assert self._exercise_date is not None
        assert self._day_counter is not None
        ref = ObservableSettings().evaluation_date_or_today()
        qassert.require(
            self._exercise_date >= ref,
            f"expiry date ({self._exercise_date}) must be greater than or equal "
            f"to floating reference date ({ref})",
        )
        self._reference_date = ref
        self._exercise_time = self._day_counter.year_fraction(ref, self._exercise_date)

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
        """Observer.update — refresh floating cache + propagate.

        In floating mode, the reference date is the global evaluation
        date, so the cached exercise time must be recomputed.
        """
        if self._floating:
            self._refresh_floating()
        self.notify_observers()
