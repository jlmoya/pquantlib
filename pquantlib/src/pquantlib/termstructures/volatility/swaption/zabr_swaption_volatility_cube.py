"""ZabrSwaptionVolatilityCube — ZABR-fitted swaption vol cube.

# C++ parity: ql/termstructures/volatility/swaption/zabrswaptionvolatilitycube.hpp
#             typedef
#             ``XabrSwaptionVolatilityCube<SwaptionVolCubeZabrModel<>>``.
#             (v1.42.1).

Thin wrapper around :class:`XabrSwaptionVolatilityCube` that fixes
``model_kind`` to :attr:`XabrModelKind.ZABR`. Eager fits a 5-parameter
ZABR slice per ``(option_tenor, swap_tenor)`` cell and routes
``smile_section_impl`` through a :class:`ZabrSmileSection`.

The C++ ``ZabrSwaptionVolatilityCube`` is a typedef of the SABR-mode
template specialised for ZABR via ``SwaptionVolCubeZabrModel``. PQuantLib
collapses both via the IntEnum dispatch on
:class:`XabrSwaptionVolatilityCube`.

Documented divergences vs C++ are identical to those carried by
:class:`XabrSwaptionVolatilityCube`. Additionally:

* The C++ ``ZabrSwaptionVolatilityCube`` rejects non-zero shifts and
  forces normal-vs-lognormal mode through the kernel parameter. The
  PQuantLib port follows the same constraint (``shift`` is forced to
  ``0.0`` in the wrapping :class:`ZabrSmileSection`); shifts on the
  underlying ATM structure are ignored.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
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


class ZabrSwaptionVolatilityCube(XabrSwaptionVolatilityCube):
    """ZABR-fitted swaption volatility cube.

    Args:
        atm_vol_structure / option_tenors / swap_tenors / strike_spreads
            / vol_spreads / swap_index_base / short_swap_index_base /
            vega_weighted_smile_fit: same as
            :class:`SwaptionVolatilityCube`.
        zabr_initial_guess: optional outer list shape
            ``(n_option_tenors x n_swap_tenors) x 5`` of initial
            ``(alpha, beta, nu, rho, gamma)`` quintuples. If ``None``,
            each cell uses ``ZabrInterpolation``'s default initial guess.
        is_parameter_fixed: 5-element ``(alpha_fixed, beta_fixed,
            nu_fixed, rho_fixed, gamma_fixed)`` mask shared across grid
            cells.
        zabr_evaluation: ZABR evaluation mode (default
            :attr:`ZabrEvaluation.ShortMaturityLognormal`). The FD
            modes raise ``LibraryException``.
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
        zabr_initial_guess: (
            Sequence[Sequence[tuple[float, float, float, float, float]]] | None
        ) = None,
        is_parameter_fixed: tuple[bool, bool, bool, bool, bool] = (
            False, False, False, False, False,
        ),
        zabr_evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
    ) -> None:
        super().__init__(
            model_kind=XabrModelKind.ZABR,
            atm_vol_structure=atm_vol_structure,
            option_tenors=option_tenors,
            swap_tenors=swap_tenors,
            strike_spreads=strike_spreads,
            vol_spreads=vol_spreads,
            swap_index_base=swap_index_base,
            short_swap_index_base=short_swap_index_base,
            vega_weighted_smile_fit=vega_weighted_smile_fit,
            initial_guess=zabr_initial_guess,
            is_parameter_fixed=is_parameter_fixed,
            zabr_evaluation=zabr_evaluation,
        )

    # --- typed back-compat accessor ---------------------------------------

    def zabr_parameters(
        self, j: int, k: int
    ) -> tuple[float, float, float, float, float]:
        """Return ``(alpha, beta, nu, rho, gamma)`` at grid cell ``(j, k)``.

        Thin typed alias of :meth:`XabrSwaptionVolatilityCube.xabr_parameters`.
        """
        a, b, n, r, g = self.xabr_parameters(j, k)
        return a, b, n, r, g
