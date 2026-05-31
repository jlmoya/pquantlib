"""NoArbSabrSwaptionVolatilityCube — no-arb SABR-fitted swaption vol cube.

# C++ parity: ql/experimental/volatility/noarbsabrswaptionvolatilitycube.hpp
# = ``XabrSwaptionVolatilityCube<SwaptionVolCubeNoArbSabrModel>`` (v1.42.1).

Thin wrapper around :class:`XabrSwaptionVolatilityCube` that fixes
``model_kind`` to :attr:`XabrModelKind.NOARB_SABR`. Each grid cell is
fitted with :class:`NoArbSabrInterpolation` and the smile section is a
:class:`NoArbSabrSmileSection` (Doust 2012 no-arbitrage SABR).

Same public surface as :class:`SabrSwaptionVolatilityCube`. Documented
divergences vs C++ are inherited from
:class:`XabrSwaptionVolatilityCube`. Note that, because each no-arb
model evaluation prices + integrates the terminal density, fitting a
NOARB_SABR cube is materially slower than a SABR cube; callers should
prefer pinning beta (and possibly rho) via ``is_parameter_fixed`` to
keep the per-cell fit well-determined and fast.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.swaption.swaption_volatility_cube import (
    AtmSwapIndexProtocol,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.swaption.xabr_swaption_volatility_cube import (
    XabrModelKind,
    XabrSwaptionVolatilityCube,
)
from pquantlib.time.period import Period


class NoArbSabrSwaptionVolatilityCube(XabrSwaptionVolatilityCube):
    """No-arbitrage SABR-fitted swaption volatility cube (Doust 2012).

    Args:
        atm_vol_structure / option_tenors / swap_tenors / strike_spreads
            / vol_spreads / swap_index_base / short_swap_index_base /
            vega_weighted_smile_fit: same as
            :class:`SwaptionVolatilityCube`.
        no_arb_sabr_initial_guess: optional outer list shape
            ``(n_option_tenors x n_swap_tenors) x 4`` of initial
            ``(alpha, beta, nu, rho)`` quadruples. If ``None``, each cell
            uses :class:`NoArbSabrInterpolation`'s default initial guess.
        is_parameter_fixed: 4-element ``(alpha_fixed, beta_fixed,
            nu_fixed, rho_fixed)`` mask shared across grid cells.
    """

    def __init__(
        self,
        *,
        atm_vol_structure: SwaptionVolatilityStructure,
        option_tenors: Sequence[Period],
        swap_tenors: Sequence[Period],
        strike_spreads: Sequence[float],
        vol_spreads: Sequence[Sequence[Quote]],
        swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        short_swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        vega_weighted_smile_fit: bool = False,
        no_arb_sabr_initial_guess: (
            Sequence[Sequence[tuple[float, float, float, float]]] | None
        ) = None,
        is_parameter_fixed: tuple[bool, bool, bool, bool] = (
            False, False, False, False,
        ),
    ) -> None:
        super().__init__(
            model_kind=XabrModelKind.NOARB_SABR,
            atm_vol_structure=atm_vol_structure,
            option_tenors=option_tenors,
            swap_tenors=swap_tenors,
            strike_spreads=strike_spreads,
            vol_spreads=vol_spreads,
            swap_index_base=swap_index_base,
            short_swap_index_base=short_swap_index_base,
            vega_weighted_smile_fit=vega_weighted_smile_fit,
            initial_guess=no_arb_sabr_initial_guess,
            is_parameter_fixed=is_parameter_fixed,
        )

    def no_arb_sabr_parameters(
        self, j: int, k: int
    ) -> tuple[float, float, float, float]:
        """Return the fitted ``(alpha, beta, nu, rho)`` at cell ``(j, k)``.

        Thin typed alias of
        :meth:`XabrSwaptionVolatilityCube.xabr_parameters`.
        """
        params = self.xabr_parameters(j, k)
        return params[0], params[1], params[2], params[3]


__all__ = ["NoArbSabrSwaptionVolatilityCube"]
