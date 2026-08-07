"""NoArbSabrSwaptionVolatilityCube — no-arb SABR-fitted swaption vol cube.

# C++ parity: ql/experimental/volatility/noarbsabrswaptionvolatilitycube.hpp
# = ``XabrSwaptionVolatilityCube<SwaptionVolCubeNoArbSabrModel>`` (v1.43).

:class:`SwaptionVolCubeNoArbSabrModel` is the model-policy type the C++
cube template is instantiated with; :class:`NoArbSabrSwaptionVolatilityCube`
is that instantiation. The cube is a thin wrapper around
:class:`XabrSwaptionVolatilityCube` that fixes ``model_kind`` to
:attr:`XabrModelKind.NOARB_SABR`. Each grid cell is fitted with
:class:`NoArbSabrInterpolation` and the smile section is a
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
from typing import Any, Final

from pquantlib.experimental.volatility.no_arb_sabr_interpolation import (
    NoArbSabrInterpolation,
)
from pquantlib.experimental.volatility.no_arb_sabr_smile_section import (
    NoArbSabrSmileSection,
)
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
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.period import Period


class SwaptionVolCubeNoArbSabrModel:
    """No-arbitrage SABR model specification for the XABR swaption cube.

    # C++ parity: ``struct SwaptionVolCubeNoArbSabrModel`` +
    # ``struct XabrModelTraits<SwaptionVolCubeNoArbSabrModel>``
    # (noarbsabrswaptionvolatilitycube.hpp:36-80).

    In C++ this is two pieces: a two-typedef tag struct naming the
    interpolation and smile-section types, and an explicit specialisation of
    ``XabrModelTraits`` on that tag carrying the behaviour. Python has no
    template specialisation, so the two collapse into one class: the
    :attr:`Interpolation` / :attr:`SmileSection` attributes are the typedefs
    and the classmethods are the traits members.

    The specialisation exists for one reason, and it is load-bearing:
    ``NoArbSabrInterpolation``'s constructor takes no ``volatilityType``
    argument (the primary ``XabrModelTraits`` template passes one), so the
    cube would not compile against it. :meth:`create_interpolation`
    therefore *drops* ``volatility_type`` — a NOARB_SABR cube built as
    ``Normal`` produces exactly the same interpolation as one built as
    ``ShiftedLognormal``. Cross-validated by ``F2`` of
    ``references/v143/experimental/volatility.json``.
    """

    #: C++ ``typedef NoArbSabrInterpolation Interpolation``.
    Interpolation = NoArbSabrInterpolation
    #: C++ ``typedef NoArbSabrSmileSection SmileSection``.
    SmileSection = NoArbSabrSmileSection

    #: C++ ``static constexpr Size nParams = 4``.
    n_params: Final[int] = 4

    __slots__ = ()

    @classmethod
    def create_interpolation(
        cls,
        strikes: Sequence[float],
        volatilities: Sequence[float],
        t: float,
        forward: float,
        params: Sequence[float],
        param_is_fixed: Sequence[bool],
        vega_weighted: bool,
        end_criteria: Any = None,
        optimization_method: Any = None,
        error_accept: float = 0.0020,
        use_max_error: bool = False,
        max_guesses: int = 50,
        shift: float = 0.0,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
    ) -> NoArbSabrInterpolation:
        """Fit one cube cell.

        # C++ parity: ``XabrModelTraits<SwaptionVolCubeNoArbSabrModel>::
        # createInterpolation`` (noarbsabrswaptionvolatilitycube.hpp:49-66).

        # C++ parity note: ``volatilityType`` is accepted and DISCARDED —
        # the C++ parameter is spelled ``VolatilityType /* volatilityType */``
        # precisely because ``NoArbSabrInterpolation`` has no such argument.
        # ``shift`` IS forwarded, and a non-zero shift is rejected.
        """
        del end_criteria, optimization_method, error_accept, use_max_error
        del volatility_type
        if shift != 0.0:
            raise ValueError(
                "NoArbSabrInterpolation for non zero shift not implemented"
            )
        return NoArbSabrInterpolation(
            strikes,
            volatilities,
            t,
            forward,
            alpha=params[0],
            beta=params[1],
            nu=params[2],
            rho=params[3],
            alpha_is_fixed=param_is_fixed[0],
            beta_is_fixed=param_is_fixed[1],
            nu_is_fixed=param_is_fixed[2],
            rho_is_fixed=param_is_fixed[3],
            vega_weighted=vega_weighted,
            max_guesses=max_guesses,
        )

    @classmethod
    def extract_gamma(cls, interpolation: NoArbSabrInterpolation) -> float:
        """Always 0.0 — no-arb SABR has no gamma parameter.

        # C++ parity: ``extractGamma``
        # (noarbsabrswaptionvolatilitycube.hpp:68-71). With ``nParams == 4``
        # the cube guards the call behind ``if constexpr (nParams >= 5)``, so
        # this is never reached in practice; the traits member still exists.
        """
        del interpolation
        return 0.0

    @classmethod
    def create_smile_section(
        cls,
        option_time: float,
        forward: float,
        params: Sequence[float],
        shift: float = 0.0,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
    ) -> NoArbSabrSmileSection:
        """Build the per-cell smile section.

        # C++ parity: ``createSmileSection``
        # (noarbsabrswaptionvolatilitycube.hpp:73-79). Unlike
        # :meth:`create_interpolation` this one DOES forward
        # ``volatilityType`` — ``NoArbSabrSmileSection`` accepts it.
        """
        alpha, beta, nu, rho = (float(p) for p in params[:4])
        return NoArbSabrSmileSection(
            forward=forward,
            sabr_params=(alpha, beta, nu, rho),
            exercise_time=option_time,
            shift=shift,
            volatility_type=volatility_type,
        )


class NoArbSabrSwaptionVolatilityCube(XabrSwaptionVolatilityCube):
    """No-arbitrage SABR-fitted swaption volatility cube (Doust 2012).

    Args:
        atm_vol_structure / option_tenors / swap_tenors / strike_spreads
            / vol_spreads / swap_index_base / short_swap_index_base /
            vega_weighted_smile_fit: same as
            :class:`SwaptionVolatilityCube`.
        no_arb_sabr_initial_guess: optional outer list shape
            ``n_option_tenors x n_swap_tenors``, each cell an
            ``(alpha, beta, nu, rho)`` quadruple of floats or
            :class:`Quote` objects. If ``None``, each cell uses
            :class:`NoArbSabrInterpolation`'s default initial guess.
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
        no_arb_sabr_initial_guess: (
            Sequence[Sequence[Sequence[float] | Sequence[Quote]]] | None
        ) = None,
        is_parameter_fixed: tuple[bool, bool, bool, bool] = (
            False, False, False, False,
        ),
        backward_flat: bool = False,
        cutoff_strike: float = 0.0001,
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
            backward_flat=backward_flat,
            cutoff_strike=cutoff_strike,
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


__all__ = ["NoArbSabrSwaptionVolatilityCube", "SwaptionVolCubeNoArbSabrModel"]
