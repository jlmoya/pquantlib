"""Concrete interest-rate class with compounding/equivalence algebra.

# C++ parity: ql/interestrate.hpp + ql/interestrate.cpp (v1.42.1)

The C++ class encapsulates:

- a rate ``r``, a ``DayCounter`` ``dc``, a ``Compounding`` ``comp``,
  and (when meaningful) a ``Frequency`` ``freq``.
- ``compound_factor(t)`` and ``discount_factor(t)`` (inverses).
- ``InterestRate.implied_rate(...)`` factory that recovers a rate from
  a compound factor under a chosen compounding/frequency.
- ``equivalent_rate(...)`` re-expression in another compounding.

The Python port keeps the same public surface; differences from C++:

- C++ uses ``operator Rate()`` for implicit conversion to float; Python
  exposes ``rate()`` (the C++ inspector) and clients call it explicitly.
- C++ ``freqMakesSense_`` is a private bool toggled in the ctor. Python
  uses the same logic but exposes via ``frequency()`` returning
  ``Frequency.NoFrequency`` when the underlying ``Compounding`` is
  Simple or Continuous (mirrors C++ behavior).
- C++ default ctor produces a "null" rate (``Null<Real>()``). Python's
  factory ``InterestRate.null()`` returns the same sentinel state;
  callers check via ``is_null()`` rather than comparing against a
  poison-double.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

_NULL_RATE: Final[float] = float("nan")


class InterestRate:
    """Concrete interest-rate carrying day-counter, compounding, frequency."""

    __slots__ = ("_compounding", "_day_counter", "_freq", "_freq_makes_sense", "_rate")

    def __init__(
        self,
        rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
    ) -> None:
        """Construct a fully-specified interest rate.

        # C++ parity: InterestRate::InterestRate(r, dc, comp, freq).

        Raises if ``compounding`` is Compounded / SimpleThenCompounded /
        CompoundedThenSimple and ``frequency`` is NoFrequency or Once.
        """
        self._rate: float = rate
        self._day_counter: DayCounter = day_counter
        self._compounding: Compounding = compounding
        self._freq_makes_sense: bool = False
        # C++ stores frequency as Real for arithmetic on power exponents.
        self._freq: float = float(int(frequency))

        if compounding in (
            Compounding.Compounded,
            Compounding.SimpleThenCompounded,
            Compounding.CompoundedThenSimple,
        ):
            self._freq_makes_sense = True
            qassert.require(
                frequency not in (Frequency.NoFrequency, Frequency.Once),
                "frequency not allowed for this interest rate",
            )

    @classmethod
    def null(cls) -> InterestRate:
        """Return a null/invalid rate (C++ ``InterestRate()`` default ctor)."""
        # We use NaN as the sentinel because Python lacks Null<Real>() poison
        # values. ``is_null()`` checks via ``math.isnan``.
        # The other fields are set to dummies — null rates are never asked
        # to compound. Day counter uses a placeholder one-day-counter
        # equivalent; callers are expected to check is_null() first.
        # We need *some* DayCounter; use a minimal placeholder via the
        # ``OneDayCounter`` import locally to avoid an import cycle.
        # C++ parity: only ``r_`` is set to Null; other fields are
        # default-constructed (empty DayCounter, etc.). Python's
        # OneDayCounter is the closest analogue available.
        from pquantlib.daycounters.one_day_counter import OneDayCounter  # noqa: PLC0415
        return cls(_NULL_RATE, OneDayCounter(), Compounding.Simple, Frequency.NoFrequency)

    def is_null(self) -> bool:
        """Return True iff this rate was produced by ``null()``."""
        return math.isnan(self._rate)

    def rate(self) -> float:
        return self._rate

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def compounding(self) -> Compounding:
        return self._compounding

    def frequency(self) -> Frequency:
        # C++: ``freq_makes_sense_ ? Frequency(Integer(freq_)) : NoFrequency``.
        if self._freq_makes_sense:
            return Frequency(int(self._freq))
        return Frequency.NoFrequency

    def compound_factor(self, t: float) -> float:
        """Compound factor implied by this rate over time ``t``.

        # C++ parity: InterestRate::compoundFactor(Time t).
        """
        qassert.require(t >= 0.0, f"negative time ({t}) not allowed")
        qassert.require(not self.is_null(), "null interest rate")
        r = self._rate
        comp = self._compounding
        if comp == Compounding.Simple:
            return 1.0 + r * t
        if comp == Compounding.Compounded:
            return math.pow(1.0 + r / self._freq, self._freq * t)
        if comp == Compounding.Continuous:
            return math.exp(r * t)
        if comp == Compounding.SimpleThenCompounded:
            if t <= 1.0 / self._freq:
                return 1.0 + r * t
            return math.pow(1.0 + r / self._freq, self._freq * t)
        if comp == Compounding.CompoundedThenSimple:
            if t > 1.0 / self._freq:
                return 1.0 + r * t
            return math.pow(1.0 + r / self._freq, self._freq * t)
        qassert.fail("unknown compounding convention")  # pragma: no cover

    def discount_factor(self, t: float) -> float:
        """Discount factor implied by this rate over time ``t``."""
        return 1.0 / self.compound_factor(t)

    def discount_factor_dates(
        self,
        d1: Date,
        d2: Date,
        ref_start: Date | None = None,
        ref_end: Date | None = None,
    ) -> float:
        """Discount factor for the year-fraction between two dates.

        # C++ parity: InterestRate::discountFactor(d1, d2, refStart, refEnd).
        """
        qassert.require(d2 >= d1, f"d1 ({d1}) later than d2 ({d2})")
        t = self._day_counter.year_fraction(d1, d2, ref_start, ref_end)
        return self.discount_factor(t)

    def compound_factor_dates(
        self,
        d1: Date,
        d2: Date,
        ref_start: Date | None = None,
        ref_end: Date | None = None,
    ) -> float:
        """Compound factor for the year-fraction between two dates."""
        qassert.require(d2 >= d1, f"d1 ({d1}) later than d2 ({d2})")
        t = self._day_counter.year_fraction(d1, d2, ref_start, ref_end)
        return self.compound_factor(t)

    @classmethod
    def implied_rate(
        cls,
        compound: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        t: float,
    ) -> InterestRate:
        """Recover an interest rate from a compound factor and time.

        # C++ parity: InterestRate::impliedRate(compound, dc, comp, freq, t).
        """
        qassert.require(compound > 0.0, "positive compound factor required")
        if compound == 1.0:
            qassert.require(t >= 0.0, f"non negative time ({t}) required")
            r = 0.0
        else:
            qassert.require(t > 0.0, f"positive time ({t}) required")
            freq_f = float(int(frequency))
            if compounding == Compounding.Simple:
                r = (compound - 1.0) / t
            elif compounding == Compounding.Compounded:
                r = (math.pow(compound, 1.0 / (freq_f * t)) - 1.0) * freq_f
            elif compounding == Compounding.Continuous:
                r = math.log(compound) / t
            elif compounding == Compounding.SimpleThenCompounded:
                if t <= 1.0 / freq_f:
                    r = (compound - 1.0) / t
                else:
                    r = (math.pow(compound, 1.0 / (freq_f * t)) - 1.0) * freq_f
            elif compounding == Compounding.CompoundedThenSimple:
                if t > 1.0 / freq_f:
                    r = (compound - 1.0) / t
                else:
                    r = (math.pow(compound, 1.0 / (freq_f * t)) - 1.0) * freq_f
            else:
                qassert.fail(f"unknown compounding convention ({int(compounding)})")  # pragma: no cover
        return cls(r, day_counter, compounding, frequency)

    @classmethod
    def implied_rate_dates(
        cls,
        compound: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        d1: Date,
        d2: Date,
        ref_start: Date | None = None,
        ref_end: Date | None = None,
    ) -> InterestRate:
        """Recover an interest rate from a compound factor over dates."""
        qassert.require(d2 >= d1, f"d1 ({d1}) later than d2 ({d2})")
        t = day_counter.year_fraction(d1, d2, ref_start, ref_end)
        return cls.implied_rate(compound, day_counter, compounding, frequency, t)

    def equivalent_rate(
        self,
        compounding: Compounding,
        frequency: Frequency,
        t: float,
    ) -> InterestRate:
        """Re-express this rate under a different compounding for time ``t``."""
        return InterestRate.implied_rate(
            self.compound_factor(t), self._day_counter, compounding, frequency, t
        )

    def equivalent_rate_dates(
        self,
        result_day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        d1: Date,
        d2: Date,
        ref_start: Date | None = None,
        ref_end: Date | None = None,
    ) -> InterestRate:
        """Re-express this rate under a different (DC, compounding) over dates.

        # C++ parity: ``equivalentRate(resultDC, comp, freq, d1, d2, refStart, refEnd)``.

        The C++ version uses two distinct year-fractions: ``t1`` from the
        original day-counter (for ``compoundFactor``) and ``t2`` from
        ``resultDC`` (passed to ``implied_rate``). We mirror that.
        """
        qassert.require(d2 >= d1, f"d1 ({d1}) later than d2 ({d2})")
        t1 = self._day_counter.year_fraction(d1, d2, ref_start, ref_end)
        t2 = result_day_counter.year_fraction(d1, d2, ref_start, ref_end)
        return InterestRate.implied_rate(
            self.compound_factor(t1), result_day_counter, compounding, frequency, t2
        )

    def __repr__(self) -> str:
        if self.is_null():
            return "InterestRate(null)"
        return (
            f"InterestRate(rate={self._rate!r}, dc={self._day_counter.name()!r}, "
            f"comp={self._compounding.name}, freq={self.frequency().name})"
        )
