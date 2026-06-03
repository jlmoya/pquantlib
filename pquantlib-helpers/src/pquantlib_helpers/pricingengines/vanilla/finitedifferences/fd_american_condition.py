"""FDAmericanCondition — adds the American early-exercise step condition.

# Retired-API compat layer — see fd_vanilla_engine docstring.

Java parity: ``org.jquantlib.pricingengines.vanilla.finitedifferences.FDAmericanCondition``.

A thin :class:`FDStepConditionEngine` subclass whose step condition is an
:class:`~pquantlib_helpers.methods.finitedifferences.step_condition.AmericanCondition`
built from the intrinsic-value array (``max(value, intrinsic)`` per node). It is
the base the ``FDDividendAmericanEngine`` adapter parameterises over
(``FDAmericanCondition<FDDividendEngine>`` in C++/Java).
"""

from __future__ import annotations

from pquantlib_helpers.methods.finitedifferences.step_condition import (
    AmericanCondition,
)
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_step_condition_engine import (
    FDStepConditionEngine,
)


class FDAmericanCondition(FDStepConditionEngine):
    """FD step-condition engine with the American early-exercise condition.

    Java parity: ``FDAmericanCondition<T>``.
    """

    def initialize_step_condition(self) -> None:
        """Install ``AmericanCondition`` over the sampled intrinsic values.

        Java parity: ``FDAmericanCondition.initializeStepCondition`` —
        ``stepCondition = new AmericanCondition(intrinsicValues.values())``.
        """
        self.step_condition = AmericanCondition(values=self.intrinsic_values.values())


__all__ = ["FDAmericanCondition"]
