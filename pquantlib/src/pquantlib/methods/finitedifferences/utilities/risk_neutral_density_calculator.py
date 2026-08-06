"""RiskNeutralDensityCalculator — the pdf/cdf/invcdf interface, plus InvCDFHelper.

# C++ parity: ql/methods/finitedifferences/utilities/riskneutraldensitycalculator.{hpp,cpp}
# (v1.43).

Three pure virtuals — ``pdf``, ``cdf``, ``invcdf`` — shared by every terminal
risk-neutral density in the FD utilities directory, plus a protected nested
helper that inverts an arbitrary ``cdf`` with Brent.

``InvCDFHelper`` is nested and protected in C++; Python has no protected
access control, so it is a module-level class. It is genuinely behavioural
(it fixes the solver, the accuracy, the evaluation cap and the initial step
size), so it is ported rather than folded away.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.math.solvers1d.brent import Brent


class RiskNeutralDensityCalculator(ABC):
    """Terminal risk-neutral density interface.

    # C++ parity: ``class RiskNeutralDensityCalculator``.
    """

    # The three arguments are positional-only: the C++ headers name them
    # differently per subclass (``x`` / ``f`` / ``v`` for the state, ``q`` or
    # ``p`` for the probability) and the ports keep those names.
    @abstractmethod
    def pdf(self, x: float, t: float, /) -> float:
        """Probability density at ``x`` for maturity ``t``."""

    @abstractmethod
    def cdf(self, x: float, t: float, /) -> float:
        """Cumulative probability at ``x`` for maturity ``t``."""

    @abstractmethod
    def invcdf(self, p: float, t: float, /) -> float:
        """Quantile: the ``x`` with ``cdf(x, t) == p``."""


class InvCDFHelper:
    """Invert a ``RiskNeutralDensityCalculator``'s ``cdf`` with Brent.

    # C++ parity: ``RiskNeutralDensityCalculator::InvCDFHelper`` — a
    # protected nested class in C++, module-level here.
    """

    __slots__ = ("_accuracy", "_calculator", "_guess", "_max_evaluations", "_step_size")

    def __init__(
        self,
        calculator: RiskNeutralDensityCalculator,
        guess: float,
        accuracy: float,
        max_evaluations: int,
        step_size: float = 0.01,
    ) -> None:
        self._calculator: RiskNeutralDensityCalculator = calculator
        self._guess: float = guess
        self._accuracy: float = accuracy
        self._max_evaluations: int = max_evaluations
        self._step_size: float = step_size

    def inverse_cdf(self, p: float, t: float) -> float:
        """# C++ parity: ``InvCDFHelper::inverseCDF`` — unbracketed Brent solve."""
        solver = Brent()
        solver.set_max_evaluations(self._max_evaluations)
        return solver.solve(
            lambda x: self._calculator.cdf(x, t) - p,
            self._accuracy,
            self._guess,
            self._step_size,
        )


__all__ = ["InvCDFHelper", "RiskNeutralDensityCalculator"]
