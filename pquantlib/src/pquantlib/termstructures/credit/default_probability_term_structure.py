"""DefaultProbabilityTermStructure — abstract default-probability TS.

# C++ parity: ql/termstructures/defaulttermstructure.{hpp,cpp} (v1.42.1)

The C++ abstract class extends ``TermStructure`` with three quantities tied
together by integration / differentiation:

- ``survival_probability(t)`` = S(t)
- ``default_probability(t)`` = 1 - S(t); also a two-date variant returning
  ``S(t1) - S(t2)``
- ``default_density(t)`` = p(t) = -S'(t)
- ``hazard_rate(t)`` = h(t) = p(t) / S(t)

Concrete subclasses (FlatHazardRate, InterpolatedSurvivalProbabilityCurve,
PiecewiseDefaultCurve) override one of ``survival_probability_impl``,
``hazard_rate_impl`` or ``default_density_impl``, plus inherit the cross-
computation defaults via the intermediate adapters
(SurvivalProbabilityStructure / HazardRateStructure / DefaultDensityStructure).

Jump support mirrors C++ ``YieldTermStructure``: ``survival_probability(t)``
multiplies the base ``survival_probability_impl(t)`` by every jump quote
whose ``jump_times[i] < t`` (jumps must be positive and ≤ 1).
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class DefaultProbabilityTermStructure(TermStructure):
    """Abstract base for default-probability term structures.

    Subclasses implement at least one of:

    - ``survival_probability_impl(t)``
    - ``hazard_rate_impl(t)``
    - ``default_density_impl(t)``

    via the intermediate adapter classes
    (:class:`SurvivalProbabilityStructure` / :class:`HazardRateStructure`
    / :class:`DefaultDensityStructure`) which fill in the others by
    numerical integration / differentiation.
    """

    def __init__(
        self,
        *,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        settlement_days: int | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._jumps: list[Quote] = list(jumps) if jumps else []
        self._jump_dates: list[Date] = list(jump_dates) if jump_dates else []
        self._n_jumps: int = len(self._jumps)
        self._jump_times: list[float] = [0.0] * len(self._jump_dates)
        self._latest_reference: Date | None = None
        # In fixed mode the reference_date is already available; in
        # delegated/moving mode it must be resolved via reference_date().
        if reference_date is not None:
            self._set_jumps(reference_date)
        for q in self._jumps:
            q.register_with(self)

    # ---- jumps -------------------------------------------------------------

    def _set_jumps(self, reference_date: Date) -> None:
        """C++ parity: ``DefaultProbabilityTermStructure::setJumps``."""
        if not self._jump_dates and self._jumps:
            # Turn-of-year dates: 31 December of each subsequent year.
            self._jump_dates = [
                Date.from_ymd(31, Month.December, reference_date.year() + i)
                for i in range(self._n_jumps)
            ]
            self._jump_times = [0.0] * self._n_jumps
        else:
            qassert.require(
                len(self._jump_dates) == self._n_jumps,
                f"mismatch between number of jumps ({self._n_jumps}) "
                f"and jump dates ({len(self._jump_dates)})",
            )
        for i in range(self._n_jumps):
            self._jump_times[i] = self.time_from_reference(self._jump_dates[i])
        self._latest_reference = reference_date

    def jump_dates(self) -> list[Date]:
        return list(self._jump_dates)

    def jump_times(self) -> list[float]:
        return list(self._jump_times)

    # ---- abstract subclass hooks ------------------------------------------

    @abstractmethod
    def _survival_probability_impl(self, t: float) -> float:
        """Survival probability at time ``t`` (range-check pre-done)."""

    @abstractmethod
    def _default_density_impl(self, t: float) -> float:
        """Default density at time ``t`` (range-check pre-done)."""

    def _hazard_rate_impl(self, t: float) -> float:
        """Hazard rate at time ``t``.

        # C++ parity: ``DefaultProbabilityTermStructure::hazardRateImpl``
        # is inline in defaulttermstructure.hpp:
        #     S = survivalProbability(t, true);
        #     return S == 0.0 ? 0.0 : defaultDensity(t, true)/S;
        """
        s = self.survival_probability(t, extrapolate=True)
        if s == 0.0:
            return 0.0
        return self.default_density(t, extrapolate=True) / s

    # ---- survival_probability overloads -----------------------------------

    def survival_probability(self, t: float | Date, extrapolate: bool = False) -> float:
        """Survival probability at a time or date.

        # C++ parity: ``DefaultProbabilityTermStructure::survivalProbability``.
        """
        if isinstance(t, Date):
            return self.survival_probability(self.time_from_reference(t), extrapolate)
        self.check_time_range(t, extrapolate)
        if not self._jumps:
            return self._survival_probability_impl(t)
        jump_effect = 1.0
        for i in range(self._n_jumps):
            jt = self._jump_times[i]
            if jt >= t:
                break
            qassert.require(self._jumps[i].is_valid(), f"invalid {i + 1}-th jump quote")
            this_jump = self._jumps[i].value()
            qassert.require(
                this_jump > 0.0 and this_jump <= 1.0,
                f"invalid {i + 1}-th jump value: {this_jump}",
            )
            jump_effect *= this_jump
        return jump_effect * self._survival_probability_impl(t)

    # ---- default_probability overloads -----------------------------------

    def default_probability(
        self,
        t: float | Date,
        t2: float | Date | bool | None = None,
        extrapolate: bool = False,
    ) -> float:
        """Default probability at a time / date, or between two times / dates.

        # C++ parity: ``DefaultProbabilityTermStructure::defaultProbability``
        # has four overloads: (Time, bool), (Date, bool), (Time, Time, bool),
        # (Date, Date, bool). We collapse them into a single Python method
        # by polymorphism on ``t2``.

        Variants:

        - ``default_probability(t)`` → 1 - S(t).
        - ``default_probability(t1, t2)`` → S(t1) - S(t2) (single-period
          default probability between two times / dates).
        - The two-time variant also accepts a bool as the third positional
          ``extrapolate``.
        """
        # Two-arg case: t2 is the second time/date.
        if isinstance(t2, (Date, float, int)) and not isinstance(t2, bool):
            t1 = t
            t2_val = t2
            if isinstance(t1, Date) and isinstance(t2_val, Date):
                qassert.require(
                    t1 <= t2_val,
                    f"initial date ({t1}) later than final date ({t2_val})",
                )
                p1 = 0.0 if t1 < self.reference_date() else self.default_probability(
                    t1, extrapolate=extrapolate,
                )
                p2 = self.default_probability(t2_val, extrapolate=extrapolate)
                return p2 - p1
            # Numeric two-arg.
            assert not isinstance(t1, Date)
            assert not isinstance(t2_val, Date)
            qassert.require(
                t1 <= t2_val,
                f"initial time ({t1}) later than final time ({t2_val})",
            )
            p1 = 0.0 if t1 < 0.0 else self.default_probability(t1, extrapolate=extrapolate)
            p2 = self.default_probability(t2_val, extrapolate=extrapolate)
            return p2 - p1
        # Single-arg case. ``t2`` may be a bool meaning "extrapolate".
        if isinstance(t2, bool):
            extrapolate = t2
        return 1.0 - self.survival_probability(t, extrapolate)

    # ---- default_density overloads ---------------------------------------

    def default_density(self, t: float | Date, extrapolate: bool = False) -> float:
        if isinstance(t, Date):
            return self.default_density(self.time_from_reference(t), extrapolate)
        self.check_time_range(t, extrapolate)
        return self._default_density_impl(t)

    # ---- hazard_rate overloads -------------------------------------------

    def hazard_rate(self, t: float | Date, extrapolate: bool = False) -> float:
        if isinstance(t, Date):
            return self.hazard_rate(self.time_from_reference(t), extrapolate)
        self.check_time_range(t, extrapolate)
        return self._hazard_rate_impl(t)

    # ---- Observer interface ----------------------------------------------

    def update(self) -> None:
        super().update()
        try:
            new_reference = self.reference_date()
        except Exception:
            return
        if new_reference != self._latest_reference:
            self._set_jumps(new_reference)


__all__ = ["DefaultProbabilityTermStructure"]
