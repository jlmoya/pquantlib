"""SabrSwaptionVolatilityCube — SABR-fitted swaption vol cube.

# C++ parity: ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp
# ``typedef XabrSwaptionVolatilityCube<SwaptionVolCubeSabrModel>
#  SabrSwaptionVolatilityCube`` (hpp:1270-1277), the classic Hagan 2002
# SABR instantiation. (v1.43).

Thin wrapper around :class:`XabrSwaptionVolatilityCube` that fixes
``model_kind`` to :attr:`XabrModelKind.SABR`. Originally a standalone
class (Phase 9-C); refactored in Phase 11 W2-A as part of the SABR/ZABR
generalisation so SABR + ZABR share the same infrastructure.

Public API preserved for back-compat:

* :meth:`sabr_parameters` — kept as a thin alias of the generalised
  :meth:`XabrSwaptionVolatilityCube.xabr_parameters`, but typed as
  ``tuple[float, float, float, float]`` (4-tuple).
* :meth:`fitted_forward` / :meth:`recalibrate` / :meth:`smile_section_impl`
  — inherited unchanged.

Documented divergences vs C++ are identical to those carried by
:class:`XabrSwaptionVolatilityCube`.
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


class SabrSwaptionVolatilityCube(XabrSwaptionVolatilityCube):
    """SABR-fitted swaption volatility cube (Hagan 2002).

    Args:
        atm_vol_structure / option_tenors / swap_tenors / strike_spreads
            / vol_spreads / swap_index_base / short_swap_index_base /
            vega_weighted_smile_fit: same as
            :class:`SwaptionVolatilityCube`.
        sabr_initial_guess: optional outer list shape
            ``n_option_tenors x n_swap_tenors``, each cell an
            ``(alpha, beta, nu, rho)`` quadruple of floats or
            :class:`Quote` objects (C++ ``parametersGuess``). If
            ``None``, each cell uses ``SabrInterpolation``'s default
            initial guess.
        is_parameter_fixed: 4-element ``(alpha_fixed, beta_fixed,
            nu_fixed, rho_fixed)`` mask shared across grid cells.
        backward_flat / cutoff_strike: C++ ``backwardFlat`` /
            ``cutoffStrike``; see :class:`XabrSwaptionVolatilityCube`.
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
        sabr_initial_guess: (
            Sequence[Sequence[Sequence[float] | Sequence[Quote]]] | None
        ) = None,
        is_parameter_fixed: tuple[bool, bool, bool, bool] = (
            False, False, False, False,
        ),
        backward_flat: bool = False,
        cutoff_strike: float = 0.0001,
    ) -> None:
        super().__init__(
            model_kind=XabrModelKind.SABR,
            atm_vol_structure=atm_vol_structure,
            option_tenors=option_tenors,
            swap_tenors=swap_tenors,
            strike_spreads=strike_spreads,
            vol_spreads=vol_spreads,
            swap_index_base=swap_index_base,
            short_swap_index_base=short_swap_index_base,
            vega_weighted_smile_fit=vega_weighted_smile_fit,
            initial_guess=sabr_initial_guess,
            is_parameter_fixed=is_parameter_fixed,
            backward_flat=backward_flat,
            cutoff_strike=cutoff_strike,
        )

    # --- typed back-compat accessor ---------------------------------------

    def sabr_parameters(self, j: int, k: int) -> tuple[float, float, float, float]:
        """Return ``(alpha, beta, nu, rho)`` at grid cell ``(j, k)``.

        Thin typed alias of :meth:`XabrSwaptionVolatilityCube.xabr_parameters`
        — preserved for back-compat with L9-C call sites.
        """
        a, b, n, r = self.xabr_parameters(j, k)
        return a, b, n, r
