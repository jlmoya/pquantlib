"""EuropeanOption — vanilla option that *requires* European exercise.

# C++ parity: ql/instruments/europeanoption.{hpp,cpp} (v1.42.1) —
# ``class EuropeanOption : public VanillaOption``.

Trivial subclass that exists to discriminate at the type level
(engines that only handle European exercise can accept this type).
The C++ constructor takes a generic ``Exercise`` but the engine
contract requires it to be European; the Python port enforces this
at construction.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import StrikedTypePayoff


class EuropeanOption(VanillaOption):
    """Vanilla option restricted to European exercise.

    # C++ parity: ``EuropeanOption(payoff, exercise)`` — Python port
    # also asserts the exercise IS a ``EuropeanExercise`` (or has the
    # ``Exercise.Type.European`` discriminant) to fail fast.
    """

    def __init__(self, payoff: StrikedTypePayoff, exercise: Exercise) -> None:
        qassert.require(
            exercise.type() == Exercise.Type.European,
            f"EuropeanOption requires European exercise; got {exercise.type()}",
        )
        super().__init__(payoff, exercise)


__all__ = ["EuropeanOption"]
