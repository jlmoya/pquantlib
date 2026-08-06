"""Structural protocol for the FD inner-value calculator.

# C++ parity: ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp
# (v1.43) — ``class FdmInnerValueCalculator``.

C++ step conditions take an ``ext::shared_ptr<FdmInnerValueCalculator>``
and call exactly one of its two virtuals, ``innerValue(iter, t)``.
(``avgInnerValue`` is used by the *solvers*, never by a step condition.)

This module declares the **minimal structural contract** that the
step conditions in this package depend on, as a ``typing.Protocol``.
Rationale:

* Python protocols are structural, so the concrete
  ``FdmInnerValueCalculator`` hierarchy (``FdmLogInnerValue``,
  ``FdmCellAveragingInnerValue``, ``FdmZeroInnerValue``, ...) satisfies
  it without any nominal dependency — no import cycle between
  ``step_conditions`` and ``utilities``.
* Declaring only ``inner_value`` keeps the requirement honest: a step
  condition that never averages must not force callers to supply an
  averaging implementation.

The method is declared **positional-only** so an implementation is free
to name its parameters whatever reads best (``iter``, ``iterator``,
``it``) and still satisfy the protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)


@runtime_checkable
class FdmInnerValueCalculatorLike(Protocol):
    """Anything that can price the immediate-exercise value at a grid node.

    # C++ parity: the ``innerValue`` half of ``FdmInnerValueCalculator``.
    """

    def inner_value(self, iterator: FdmLinearOpIterator, t: float, /) -> float:
        """Inner (immediate-exercise / cashflow) value at ``iterator``, time ``t``.

        # C++ parity: ``FdmInnerValueCalculator::innerValue``.
        """
        ...


__all__ = ["FdmInnerValueCalculatorLike"]
