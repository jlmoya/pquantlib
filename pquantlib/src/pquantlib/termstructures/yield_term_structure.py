"""YieldTermStructure — abstract base for interest-rate curves.

# C++ parity: ql/termstructures/yieldtermstructure.hpp +
#             ql/termstructures/yieldtermstructure.cpp (v1.42.1)

The C++ abstract class extends ``TermStructure`` with three pieces of
core functionality:

1. ``discount(t)`` / ``discount(d)``: discount factor at a time/date.
2. ``zero_rate(t|d, comp, freq)`` and ``forward_rate(t1|d1, t2|d2,
   comp, freq)`` returning ``InterestRate`` objects.
3. Jump-date / jump-quote support: between ``setJumps`` updates the
   ``discount`` accessor multiplies the bare ``discountImpl(t)``
   result by every active jump quote whose ``jumpTimes_[i] < t``.

Subclasses implement ``_discount_impl(t)``. Three modes (delegated /
fixed / moving) are inherited from ``TermStructure``; the moving mode
remains deferred until Settings.evaluation_date is wired (see
``pquantlib.termstructures.term_structure``).

Divergences from C++:

- ``forward_rate`` overloaded variants — the C++ ``forwardRate(date,
  Period, ...)`` helper that wraps ``d + p`` is omitted; callers
  construct ``d2 = d + p`` themselves. Period arithmetic is already
  available on ``pquantlib.time.date.Date``.
- The Protocol surface (``pquantlib.termstructures.protocols``) exposes
  simplified ``zero_rate(t)`` / ``forward_rate(t1, t2)`` that return
  plain floats; this class returns ``InterestRate`` for parity with
  the C++ richer API. Both surfaces coexist.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Final

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

# C++ parity: yieldtermstructure.cpp anonymous namespace: ``const Time dt = 0.0001;``.
_DT: Final[float] = 0.0001


class YieldTermStructure(TermStructure):
    """Interest-rate term structure with discount / zero / forward methods."""

    def __init__(
        self,
        *,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> None:
        # C++ uses Handle<Quote> for jumps; Python uses Quote directly
        # (Quote is itself Observable and equivalent to the Handle's
        # tracked underlying for these purposes).
        super().__init__(reference_date=reference_date, calendar=calendar, day_counter=day_counter)
        self._jumps: list[Quote] = list(jumps) if jumps else []
        self._jump_dates: list[Date] = list(jump_dates) if jump_dates else []
        self._jump_times: list[float] = [0.0] * len(self._jump_dates)
        self._n_jumps: int = len(self._jumps)
        self._latest_reference: Date | None = None
        # Set up jumps if reference date is set. In delegated mode this
        # happens at first call (via update()).
        if reference_date is not None:
            self._set_jumps(reference_date)
        for q in self._jumps:
            q.register_with(self)

    def _set_jumps(self, reference_date: Date) -> None:
        """C++ parity: ``YieldTermStructure::setJumps``."""
        if not self._jump_dates and self._jumps:
            # Turn-of-year dates: 31 December of each subsequent year.
            self._jump_dates = [Date.from_ymd(31, Month.December, reference_date.year() + i) for i in range(self._n_jumps)]
            self._jump_times = [0.0] * self._n_jumps
        else:
            qassert.require(
                len(self._jump_dates) == self._n_jumps,
                f"mismatch between number of jumps ({self._n_jumps}) and jump dates ({len(self._jump_dates)})",
            )
        for i in range(self._n_jumps):
            self._jump_times[i] = self.time_from_reference(self._jump_dates[i])
        self._latest_reference = reference_date

    @abstractmethod
    def _discount_impl(self, t: float) -> float:
        """Discount factor at time ``t`` — subclasses implement.

        # C++ parity: ``YieldTermStructure::discountImpl(Time)``.

        When called, range-check has already been performed (the caller
        is the public ``discount``); subclasses may assume extrapolation
        is required.
        """

    # ---- discount overloads ------------------------------------------------

    def discount(self, t: float | Date, extrapolate: bool = False) -> float:
        """Discount factor at a time or date.

        # C++ parity: ``YieldTermStructure::discount(Time|Date, bool)``.

        The date-overload converts via ``time_from_reference``; the
        time-overload range-checks via ``check_time_range`` (or the
        date-flavored ``check_range`` if jumps are present).
        """
        if isinstance(t, Date):
            return self.discount(self.time_from_reference(t), extrapolate)
        self.check_time_range(t, extrapolate)
        if not self._jumps:
            return self._discount_impl(t)
        jump_effect = 1.0
        for i in range(self._n_jumps):
            jt = self._jump_times[i]
            if jt > 0 and jt < t:
                qassert.require(self._jumps[i].is_valid(), f"invalid {i + 1}-th jump quote")
                this_jump = self._jumps[i].value()
                qassert.require(this_jump > 0.0, f"invalid {i + 1}-th jump value: {this_jump}")
                jump_effect *= this_jump
        return jump_effect * self._discount_impl(t)

    # ---- zero_rate overloads -----------------------------------------------

    def zero_rate(
        self,
        t: float | Date,
        compounding: Compounding,
        frequency: Frequency = Frequency.Annual,
        extrapolate: bool = False,
        result_day_counter: DayCounter | None = None,
    ) -> InterestRate:
        """Implied zero yield rate for a given time or date.

        # C++ parity: ``YieldTermStructure::zeroRate(Time|Date, ...)``.

        - The Date overload optionally takes ``result_day_counter``
          (C++ second positional after ``Date``); if omitted, falls
          back to ``self.day_counter()``.
        - The Time overload uses ``self.day_counter()`` directly.
        """
        if isinstance(t, Date):
            dc = result_day_counter if result_day_counter is not None else self.day_counter()
            d = t
            t_num = self.time_from_reference(d)
            if t_num == 0.0:
                compound = 1.0 / self.discount(_DT, extrapolate)
                # t was computed with a possibly different daycounter,
                # but the difference shouldn't matter for very small times.
                return InterestRate.implied_rate(compound, dc, compounding, frequency, _DT)
            compound = 1.0 / self.discount(t_num, extrapolate)
            return InterestRate.implied_rate_dates(
                compound, dc, compounding, frequency, self.reference_date(), d
            )
        # Time overload
        if t == 0.0:
            t = _DT
        compound = 1.0 / self.discount(t, extrapolate)
        return InterestRate.implied_rate(compound, self.day_counter(), compounding, frequency, t)

    # ---- forward_rate overloads --------------------------------------------

    def forward_rate(
        self,
        t1: float | Date,
        t2: float | Date,
        compounding: Compounding,
        frequency: Frequency = Frequency.Annual,
        extrapolate: bool = False,
        result_day_counter: DayCounter | None = None,
    ) -> InterestRate:
        """Forward interest rate between two times or two dates.

        # C++ parity: ``YieldTermStructure::forwardRate(...)``.

        If both args are equal the instantaneous forward rate is returned
        (computed by finite-difference around ``t``).
        """
        if isinstance(t1, Date) and isinstance(t2, Date):
            dc = result_day_counter if result_day_counter is not None else self.day_counter()
            d1, d2 = t1, t2
            if d1 == d2:
                self.check_range(d1, extrapolate)
                t1n = max(self.time_from_reference(d1) - _DT / 2.0, 0.0)
                t2n = t1n + _DT
                compound = self.discount(t1n, True) / self.discount(t2n, True)
                return InterestRate.implied_rate(compound, dc, compounding, frequency, _DT)
            qassert.require(d1 < d2, f"{d1} later than {d2}")
            compound = self.discount(d1, extrapolate) / self.discount(d2, extrapolate)
            return InterestRate.implied_rate_dates(compound, dc, compounding, frequency, d1, d2)
        # Time overload
        assert not isinstance(t1, Date), "forward_rate: mixed Date/float args not allowed"
        assert not isinstance(t2, Date), "forward_rate: mixed Date/float args not allowed"
        # C++ parity: yieldtermstructure.cpp:154-172 mutates ``t1`` and
        # ``t2`` directly in the t2==t1 branch, so that the subsequent
        # ``impliedRate(..., t2-t1)`` call uses the shifted interval
        # ``_DT`` rather than zero. The original Python port computed
        # ``t1_adj`` / ``t2_adj`` locally but then used the *unmodified*
        # ``t1, t2`` for ``implied_rate`` — which triggers a
        # ``positive time required`` failure at ``t1 == t2``. Mirror
        # C++ exactly by rebinding ``t1`` / ``t2``.
        if t2 == t1:
            self.check_time_range(t1, extrapolate)
            t1 = max(t1 - _DT / 2.0, 0.0)
            t2 = t1 + _DT
            compound = self.discount(t1, True) / self.discount(t2, True)
        else:
            qassert.require(t2 > t1, f"t2 ({t2}) < t1 ({t1})")
            compound = self.discount(t1, extrapolate) / self.discount(t2, extrapolate)
        return InterestRate.implied_rate(
            compound, self.day_counter(), compounding, frequency, t2 - t1
        )

    # ---- jump inspectors ---------------------------------------------------

    def jump_dates(self) -> list[Date]:
        return list(self._jump_dates)

    def jump_times(self) -> list[float]:
        return list(self._jump_times)

    # ---- Observer interface ------------------------------------------------

    def update(self) -> None:
        super().update()
        new_reference: Date | None = None
        try:
            new_reference = self.reference_date()
            if new_reference != self._latest_reference:
                self._set_jumps(new_reference)
        except Exception:
            if new_reference is None:
                # Couldn't calculate reference date — usually because an
                # underlying handle isn't set yet. Absorb and continue;
                # jumps will be set when a valid underlying appears.
                return
            raise
