"""ZabrSwaptionVolatilityCube — ZABR-fitted swaption vol cube.

# C++ parity: ql/termstructures/volatility/swaption/zabrswaptionvolatilitycube.hpp
#             typedef
#             ``XabrSwaptionVolatilityCube<SwaptionVolCubeZabrModel<>>``.
#             (v1.43).

Thin wrapper around :class:`XabrSwaptionVolatilityCube` that fixes
``model_kind`` to :attr:`XabrModelKind.ZABR`. Fits a 5-parameter
ZABR slice per ``(option_tenor, swap_tenor)`` cell into the parameter
:class:`~pquantlib.termstructures.volatility.swaption.xabr_swaption_volatility_cube.Cube`
and routes ``smile_section_impl`` through a :class:`ZabrSmileSection`
built from the *interpolated* parameters.

Because ``nParams == 5`` for ZABR, layers 0..4 of the parameter cube are
the model parameters and the forward-rate metadata layer is index 5 —
which, per the C++ ``k <= 4`` threshold (hpp:1011-1017, 1227), means the
forward layer is BILINEAR even when ``backward_flat`` is set, while for a
4-parameter model it is backward-flat. That asymmetry is upstream's and
is reproduced.

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
            ``n_option_tenors x n_swap_tenors``, each cell an
            ``(alpha, beta, nu, rho, gamma)`` quintuple of floats or
            :class:`Quote` objects. If ``None``, each cell uses
            ``ZabrInterpolation``'s default initial guess.
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
            Sequence[Sequence[Sequence[float] | Sequence[Quote]]] | None
        ) = None,
        is_parameter_fixed: tuple[bool, bool, bool, bool, bool] = (
            False, False, False, False, False,
        ),
        backward_flat: bool = False,
        cutoff_strike: float = 0.0001,
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
            backward_flat=backward_flat,
            cutoff_strike=cutoff_strike,
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
