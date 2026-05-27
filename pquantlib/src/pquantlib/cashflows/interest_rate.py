"""Interest rate algebra.

# C++ parity: ql/interestrate.hpp + ql/interestrate.cpp (v1.42.1).

The C++ class wraps (rate, day_counter, compounding, frequency) and exposes
``compoundFactor(t)``, ``discountFactor(t)``, ``impliedRate(...)``, and
``equivalentRate(...)``.

Python design notes:
- Translated as ``@dataclass(frozen=True, slots=True)`` — InterestRate is
  pure value-object algebra with no observer behaviour.
- ``freq_makes_sense`` (C++ ``freqMakesSense_``) is exposed as a derived
  read-only property to avoid storing it (it is fully determined by
  ``compounding``).
- C++ stores the frequency as ``Real`` to allow non-integer freqs in the
  Compounded branches; we keep ``Frequency`` IntEnum and cast to ``float``
  inside the compute methods. ``NoFrequency`` is the default for
  compoundings that don't need a frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pquantlib import qassert
from pquantlib.cashflows.compounding import Compounding
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


@dataclass(frozen=True, slots=True)
class InterestRate:
    """Rate + day-counter + compounding convention + frequency."""

    rate: float
    day_counter: DayCounter
    compounding: Compounding
    frequency: Frequency = field(default=Frequency.Annual)

    def __post_init__(self) -> None:
        # Validate that Compounded variants require a real frequency.
        if self.compounding in (
            Compounding.Compounded,
            Compounding.SimpleThenCompounded,
            Compounding.CompoundedThenSimple,
        ):
            qassert.require(
                self.frequency not in (Frequency.NoFrequency, Frequency.Once),
                "frequency not allowed for this interest rate",
            )

    # --- inspectors ------------------------------------------------------

    @property
    def freq_makes_sense(self) -> bool:
        """Whether the stored frequency is meaningful for this compounding."""
        return self.compounding in (
            Compounding.Compounded,
            Compounding.SimpleThenCompounded,
            Compounding.CompoundedThenSimple,
        )

    # --- compound / discount factors ------------------------------------

    def compound_factor(self, t: float) -> float:
        """Compound (capitalization) factor implied by the rate at time t.

        C++ parity: ql/interestrate.cpp:44-68 ``Real InterestRate::compoundFactor(Time t)``.
        """
        qassert.require(t >= 0.0, f"negative time ({t}) not allowed")
        r = self.rate
        if self.compounding == Compounding.Simple:
            return 1.0 + r * t
        if self.compounding == Compounding.Compounded:
            f = float(self.frequency)
            return (1.0 + r / f) ** (f * t)
        if self.compounding == Compounding.Continuous:
            return math.exp(r * t)
        if self.compounding == Compounding.SimpleThenCompounded:
            f = float(self.frequency)
            if t <= 1.0 / f:
                return 1.0 + r * t
            return (1.0 + r / f) ** (f * t)
        if self.compounding == Compounding.CompoundedThenSimple:
            f = float(self.frequency)
            if t > 1.0 / f:
                return 1.0 + r * t
            return (1.0 + r / f) ** (f * t)
        qassert.fail("unknown compounding convention")

    def discount_factor(self, t: float) -> float:
        """1.0 / compound_factor(t).

        C++ parity: ql/interestrate.hpp:69-72.
        """
        return 1.0 / self.compound_factor(t)

    def compound_factor_between(
        self,
        d1: Date,
        d2: Date,
        ref_start: Date | None = None,
        ref_end: Date | None = None,
    ) -> float:
        """Compound factor for the period d1->d2 using this rate's day counter.

        C++ parity: ql/interestrate.hpp:98-107.
        """
        qassert.require(d2 >= d1, f"d1 ({d1!r}) later than d2 ({d2!r})")
        t = self.day_counter.year_fraction(d1, d2, ref_start, ref_end)
        return self.compound_factor(t)

    def discount_factor_between(
        self,
        d1: Date,
        d2: Date,
        ref_start: Date | None = None,
        ref_end: Date | None = None,
    ) -> float:
        """1.0 / compound_factor_between."""
        return 1.0 / self.compound_factor_between(d1, d2, ref_start, ref_end)

    # --- implied rate ---------------------------------------------------

    @classmethod
    def implied_rate(
        cls,
        compound: float,
        result_dc: DayCounter,
        comp: Compounding,
        freq: Frequency,
        t: float,
    ) -> InterestRate:
        """Invert ``compound_factor`` to recover the implicit rate.

        C++ parity: ql/interestrate.cpp:70-112.
        """
        qassert.require(compound > 0.0, "positive compound factor required")
        if compound == 1.0:
            qassert.require(t >= 0.0, f"non negative time ({t}) required")
            r = 0.0
        else:
            qassert.require(t > 0.0, f"positive time ({t}) required")
            if comp == Compounding.Simple:
                r = (compound - 1.0) / t
            elif comp == Compounding.Compounded:
                f = float(freq)
                r = (compound ** (1.0 / (f * t)) - 1.0) * f
            elif comp == Compounding.Continuous:
                r = math.log(compound) / t
            elif comp == Compounding.SimpleThenCompounded:
                f = float(freq)
                r = (
                    (compound - 1.0) / t
                    if t <= 1.0 / f
                    else (compound ** (1.0 / (f * t)) - 1.0) * f
                )
            elif comp == Compounding.CompoundedThenSimple:
                f = float(freq)
                r = (
                    (compound - 1.0) / t
                    if t > 1.0 / f
                    else (compound ** (1.0 / (f * t)) - 1.0) * f
                )
            else:
                qassert.fail(f"unknown compounding convention ({int(comp)})")
        return cls(r, result_dc, comp, freq)

    # --- equivalent rate ------------------------------------------------

    def equivalent_rate(
        self,
        comp: Compounding,
        freq: Frequency,
        t: float,
    ) -> InterestRate:
        """Rate equivalent to ``self`` under a different (comp, freq) at time t.

        C++ parity: ql/interestrate.hpp:155-159.
        """
        return InterestRate.implied_rate(
            self.compound_factor(t), self.day_counter, comp, freq, t
        )

    def equivalent_rate_between(
        self,
        result_dc: DayCounter,
        comp: Compounding,
        freq: Frequency,
        d1: Date,
        d2: Date,
        ref_start: Date | None = None,
        ref_end: Date | None = None,
    ) -> InterestRate:
        """Date-pair flavour of equivalent_rate.

        C++ parity: ql/interestrate.hpp:165-178.
        """
        qassert.require(d2 >= d1, f"d1 ({d1!r}) later than d2 ({d2!r})")
        t1 = self.day_counter.year_fraction(d1, d2, ref_start, ref_end)
        t2 = result_dc.year_fraction(d1, d2, ref_start, ref_end)
        return InterestRate.implied_rate(
            self.compound_factor(t1), result_dc, comp, freq, t2
        )
