"""FDEngineAdapter — bridges a base FD engine to the GenericEngine surface.

# Retired-API compat layer — see fd_vanilla_engine docstring.

Java parity: ``org.jquantlib.pricingengines.vanilla.finitedifferences.FDEngineAdapter``.

The Java ``FDEngineAdapter<Base, Engine>`` uses reflection + a p-impl idiom to
host a base FD engine inside a ``OneAssetOption.Engine``. In Python the base FD
engine is just held as a field; :meth:`calculate` copies the option arguments
into the base engine, runs its scheme, and the base writes straight into the
shared results object — so no p-impl indirection is needed. The adapter is a
:class:`~pquantlib.pricingengines.generic_engine.GenericEngine` over the
dividend-option argument/result carriers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib_helpers.instruments.dividend_vanilla_option import (
    DividendVanillaOptionArguments,
    DividendVanillaOptionResults,
)

if TYPE_CHECKING:
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_vanilla_engine import (
        FDVanillaEngine,
    )


class FDEngineAdapter(
    GenericEngine[DividendVanillaOptionArguments, DividendVanillaOptionResults]
):
    """GenericEngine adapter hosting a base FD engine.

    Java parity: ``FDEngineAdapter<Base, Engine>``.
    """

    def __init__(
        self,
        base: FDVanillaEngine,
        process: GeneralizedBlackScholesProcess,
    ) -> None:
        """Bind the base FD engine + register on the process."""
        super().__init__(
            DividendVanillaOptionArguments(), DividendVanillaOptionResults()
        )
        self._base: FDVanillaEngine = base
        process.register_with(self)

    def calculate(self) -> None:
        """Drive the base FD engine.

        Java parity: ``FDEngineAdapter.calculate`` —
        ``base.setupArguments(args); base.calculate(results)``.
        """
        self._base.setup_arguments(self._arguments)
        self._base.calculate(self._results)


__all__ = ["FDEngineAdapter"]
