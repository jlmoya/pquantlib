"""FDDividendAmericanEngine — FD engine for American dividend options.

# Retired-API compat layer — see fd_vanilla_engine docstring.

Java parity: ``org.jquantlib.pricingengines.vanilla.finitedifferences.FDDividendAmericanEngine``
(C++ ``typedef FDEngineAdapter<FDAmericanCondition<FDDividendEngine>,
DividendVanillaOption::engine>``).

Drives the control-variate American step-condition engine
(:class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_american_condition.FDAmericanCondition`)
through the :class:`FDEngineAdapter`.

# Faithful-port note: matching C++/Java, the base
# ``FDAmericanCondition extends FDStepConditionEngine extends FDVanillaEngine``,
# NOT ``FDMultiPeriodEngine`` — so the American FD engine does NOT do
# multi-period dividend stepping. Its ``setup_arguments`` reads only the payoff
# + exercise (dividends are ignored on this path), exactly as the Java engine.
# The JQuantLib FDDividendAmericanEngine carries a ``@bug results are not overly
# reliable`` note for this reason; we reproduce it verbatim for cross-validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_american_condition import (
    FDAmericanCondition,
)
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_engine_adapter import (
    FDEngineAdapter,
)

if TYPE_CHECKING:
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )


class FDDividendAmericanEngine(FDEngineAdapter):
    """FD pricing engine for American dividend options.

    Java parity: ``FDDividendAmericanEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int = 100,
        grid_points: int = 100,
        time_dependent: bool = False,
    ) -> None:
        """Build the American-condition base engine + bridge it to GenericEngine."""
        base = FDAmericanCondition(process, time_steps, grid_points, time_dependent)
        super().__init__(base, process)


__all__ = ["FDDividendAmericanEngine"]
