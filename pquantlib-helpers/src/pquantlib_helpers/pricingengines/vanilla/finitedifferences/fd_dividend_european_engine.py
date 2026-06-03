"""FDDividendEuropeanEngine — FD engine for European dividend options.

# Retired-API compat layer — see fd_vanilla_engine docstring.

Java parity: ``org.jquantlib.pricingengines.vanilla.finitedifferences.FDDividendEuropeanEngine``
(C++ ``typedef FDEngineAdapter<FDDividendEngine, DividendVanillaOption::engine>``).

Drives the Merton-73 escrowed-dividend FD base engine
(:class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_dividend_engine.FDDividendEngine`)
through the :class:`FDEngineAdapter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_dividend_engine import (
    FDDividendEngine,
)
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_engine_adapter import (
    FDEngineAdapter,
)

if TYPE_CHECKING:
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )


class FDDividendEuropeanEngine(FDEngineAdapter):
    """FD pricing engine for European dividend options.

    Java parity: ``FDDividendEuropeanEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int = 100,
        grid_points: int = 100,
        time_dependent: bool = False,
    ) -> None:
        """Build the Merton-73 base engine + bridge it to the GenericEngine surface."""
        base = FDDividendEngine(process, time_steps, grid_points, time_dependent)
        super().__init__(base, process)


__all__ = ["FDDividendEuropeanEngine"]
