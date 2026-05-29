"""XabrSwaptionVolatilityCube — model-parameterised xabr-style swaption cube.

# C++ parity: ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp
#             template ``XabrSwaptionVolatilityCube<Model>`` plus
#             ``ql/termstructures/volatility/swaption/zabrswaptionvolatilitycube.hpp``
#             which is the typedef
#             ``XabrSwaptionVolatilityCube<SwaptionVolCubeZabrModel<>>``.
#             (v1.42.1).

The C++ class is a template parameterised by an ``XabrModelTraits``
specialisation that picks the interpolation class (``SABRInterpolation``,
``ZabrInterpolation<...>``) and the corresponding smile section
(``SabrSmileSection``, ``ZabrSmileSection<...>``).

The PQuantLib generalisation uses an :class:`XabrModelKind` IntEnum
dispatch instead of C++ templates. The eager-fit subset is the same:
at construction (and on any observed Quote update via :meth:`recalibrate`)
the cube fits one parameter vector per ``(option_tenor, swap_tenor)``
cell and stores the result. ``smile_section_impl(t, s)`` resolves the
nearest grid cell and returns the appropriate smile section.

Documented divergences vs C++:

* No ``performCalculations`` LazyObject hook — Python recomputes on
  each ``smile_section_impl`` invocation via the cached parameter cube
  + on Quote-driven re-fit only when explicitly requested via
  :meth:`recalibrate`.
* No bilinear interpolation over the parameter cube between grid
  cells — we return the nearest grid cell's section. This matches the
  C++ ``XabrSwaptionVolatilityCube`` when interrogated at grid pillars
  but diverges in the dense interior.
* No ``backwardFlat`` / ``maxGuesses`` / ``cutoffStrike`` /
  ``errorAccept`` knobs; the SciPy fitter exposes its native
  ``ftol`` / ``xtol`` / ``gtol`` instead.
* SABR mode and ZABR mode use the same constructor: the only
  per-mode difference is the parameter shape (4-tuple for SABR,
  5-tuple for ZABR) and the smile-section class instantiated at
  ``smile_section_impl``. Both modes share the rest of the cube
  infrastructure: grid sanity, ATM-strike dispatch, Quote
  registration, nearest-cell lookup.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.math.interpolations.sabr_interpolation import SabrInterpolation
from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
from pquantlib.math.interpolations.zabr_interpolation import ZabrInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.swaption.swaption_volatility_cube import (
    AtmSwapIndexProtocol,
    SwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection
from pquantlib.time.period import Period


class XabrModelKind(IntEnum):
    """Selector for the xabr model used in :class:`XabrSwaptionVolatilityCube`.

    # C++ parity: indirectly mapped from the template parameter pair
    # ``SwaptionVolCubeSabrModel`` / ``SwaptionVolCubeZabrModel<Kernel>``.
    """

    SABR = 0  # 4-param Hagan 2002 SABR via SabrInterpolation + SabrSmileSection
    ZABR = 1  # 5-param ZABR via ZabrInterpolation + ZabrSmileSection


# Initial guess shape per mode.
_XabrGuessSabr = Sequence[Sequence[tuple[float, float, float, float]]]
_XabrGuessZabr = Sequence[
    Sequence[tuple[float, float, float, float, float]]
]


class XabrSwaptionVolatilityCube(SwaptionVolatilityCube):
    """xabr-style swaption volatility cube, model_kind-dispatched.

    Args:
        model_kind: which xabr model to use (SABR or ZABR).
        atm_vol_structure / option_tenors / swap_tenors / strike_spreads
            / vol_spreads / swap_index_base / short_swap_index_base /
            vega_weighted_smile_fit: same as
            :class:`SwaptionVolatilityCube`.
        initial_guess: optional outer list shape
            ``(n_option_tenors x n_swap_tenors) x n_params`` of initial
            parameter tuples (4-tuples for SABR, 5-tuples for ZABR).
            If ``None``, each cell uses the underlying interpolation's
            default initial guess.
        is_parameter_fixed: per-mode parameter-fix mask
            (4-tuple for SABR, 5-tuple for ZABR) shared across grid cells.
        zabr_evaluation: ZABR evaluation mode (default
            :attr:`ZabrEvaluation.ShortMaturityLognormal`). Ignored
            when ``model_kind == XabrModelKind.SABR``.
    """

    def __init__(
        self,
        *,
        model_kind: XabrModelKind,
        atm_vol_structure: SwaptionVolatilityStructure,
        option_tenors: Sequence[Period],
        swap_tenors: Sequence[Period],
        strike_spreads: Sequence[float],
        vol_spreads: Sequence[Sequence[Quote]],
        swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        short_swap_index_base: SwapIndex | AtmSwapIndexProtocol,
        vega_weighted_smile_fit: bool = False,
        initial_guess: (
            _XabrGuessSabr | _XabrGuessZabr | None
        ) = None,
        is_parameter_fixed: (
            tuple[bool, bool, bool, bool]
            | tuple[bool, bool, bool, bool, bool]
        ) = (False, False, False, False),
        zabr_evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
    ) -> None:
        super().__init__(
            atm_vol_structure=atm_vol_structure,
            option_tenors=option_tenors,
            swap_tenors=swap_tenors,
            strike_spreads=strike_spreads,
            vol_spreads=vol_spreads,
            swap_index_base=swap_index_base,
            short_swap_index_base=short_swap_index_base,
            vega_weighted_smile_fit=vega_weighted_smile_fit,
        )
        self._model_kind: XabrModelKind = model_kind
        n_expected = 5 if model_kind == XabrModelKind.ZABR else 4
        qassert.require(
            len(is_parameter_fixed) == n_expected,
            f"is_parameter_fixed has length {len(is_parameter_fixed)}; "
            f"expected {n_expected} for {model_kind.name}",
        )
        self._is_param_fixed: tuple[bool, ...] = tuple(is_parameter_fixed)
        self._initial_guess: (
            _XabrGuessSabr | _XabrGuessZabr | None
        ) = initial_guess
        self._zabr_evaluation: ZabrEvaluation = zabr_evaluation
        # Fitted parameter tuples per (j, k) cell — populated lazily.
        # Stored as tuples (alpha, beta, nu, rho) for SABR or
        # (alpha, beta, nu, rho, gamma) for ZABR.
        self._fitted_params: list[list[tuple[float, ...]]] | None = None
        self._fitted_forwards: list[list[float]] | None = None

    # --- inspectors --------------------------------------------------------

    def model_kind(self) -> XabrModelKind:
        return self._model_kind

    def zabr_evaluation(self) -> ZabrEvaluation:
        return self._zabr_evaluation

    # --- fit driver --------------------------------------------------------

    def _fit_all(self) -> None:
        """Fit one xabr slice per ``(option_tenor, swap_tenor)`` cell.

        # C++ parity: XabrSwaptionVolatilityCube<Model>::sabrCalibration
        #             (sabrswaptionvolatilitycube.hpp:411-535).
        """
        n_opt = len(self._option_tenors)
        n_swap = len(self._swap_tenors)
        n_strikes = len(self._strike_spreads)
        params: list[list[tuple[float, ...]]] = [
            [()] * n_swap for _ in range(n_opt)
        ]
        forwards: list[list[float]] = [[0.0] * n_swap for _ in range(n_opt)]

        for j in range(n_opt):
            for k in range(n_swap):
                option_date = self._option_dates[j]
                swap_tenor = self._swap_tenors[k]
                option_time = self._option_times[j]
                swap_length = self._swap_lengths[k]
                forward = self.atm_strike(option_date, swap_tenor)
                shift_local = self._atm_vol.shift(
                    option_time, swap_length, extrapolate=True
                )
                atm_vol = self._atm_vol.volatility(
                    option_date, swap_tenor, forward, extrapolate=True
                )
                strikes: list[float] = []
                vols: list[float] = []
                for i in range(n_strikes):
                    strike = forward + self._strike_spreads[i]
                    if strike + shift_local <= 0.0:
                        continue
                    spread = self._vol_spreads[j * n_swap + k][i].value()
                    strikes.append(strike)
                    vols.append(atm_vol + spread)
                params[j][k] = self._fit_one_cell(
                    j, k, strikes, vols, option_time, forward, shift_local,
                )
                forwards[j][k] = forward
        self._fitted_params = params
        self._fitted_forwards = forwards

    def _fit_one_cell(
        self,
        j: int,
        k: int,
        strikes: list[float],
        vols: list[float],
        option_time: float,
        forward: float,
        shift_local: float,
    ) -> tuple[float, ...]:
        """Dispatch to the model-specific fit kernel."""
        if self._model_kind == XabrModelKind.SABR:
            return self._fit_sabr_cell(
                j, k, strikes, vols, option_time, forward, shift_local,
            )
        return self._fit_zabr_cell(
            j, k, strikes, vols, option_time, forward,
        )

    def _fit_sabr_cell(
        self,
        j: int,
        k: int,
        strikes: list[float],
        vols: list[float],
        option_time: float,
        forward: float,
        shift_local: float,
    ) -> tuple[float, float, float, float]:
        if self._initial_guess is not None:
            guess = self._initial_guess[j][k]
            if len(guess) != 4:
                raise LibraryException(
                    f"SABR cube expects 4-tuple guess at cell ({j},{k}); "
                    f"got length {len(guess)}"
                )
            a0, b0, n0, r0 = guess[0], guess[1], guess[2], guess[3]
            interp = SabrInterpolation(
                strikes, vols, option_time, forward,
                alpha=a0, beta=b0, nu=n0, rho=r0,
                alpha_is_fixed=self._is_param_fixed[0],
                beta_is_fixed=self._is_param_fixed[1],
                nu_is_fixed=self._is_param_fixed[2],
                rho_is_fixed=self._is_param_fixed[3],
                vega_weighted=self._vega_weighted_smile_fit,
                shift=shift_local,
                volatility_type=self.volatility_type(),
            )
        else:
            interp = SabrInterpolation(
                strikes, vols, option_time, forward,
                alpha_is_fixed=self._is_param_fixed[0],
                beta_is_fixed=self._is_param_fixed[1],
                nu_is_fixed=self._is_param_fixed[2],
                rho_is_fixed=self._is_param_fixed[3],
                vega_weighted=self._vega_weighted_smile_fit,
                shift=shift_local,
                volatility_type=self.volatility_type(),
            )
        return interp.alpha(), interp.beta(), interp.nu(), interp.rho()

    def _fit_zabr_cell(
        self,
        j: int,
        k: int,
        strikes: list[float],
        vols: list[float],
        option_time: float,
        forward: float,
    ) -> tuple[float, float, float, float, float]:
        if self._initial_guess is not None:
            guess = self._initial_guess[j][k]
            if len(guess) != 5:
                raise LibraryException(
                    f"ZABR cube expects 5-tuple guess at cell ({j},{k}); "
                    f"got length {len(guess)}"
                )
            a0, b0, n0, r0, g0 = (
                guess[0], guess[1], guess[2], guess[3], guess[4],
            )
            interp = ZabrInterpolation(
                strikes=strikes, volatilities=vols,
                expiry_time=option_time, forward=forward,
                alpha=a0, beta=b0, nu=n0, rho=r0, gamma=g0,
                alpha_is_fixed=self._is_param_fixed[0],
                beta_is_fixed=self._is_param_fixed[1],
                nu_is_fixed=self._is_param_fixed[2],
                rho_is_fixed=self._is_param_fixed[3],
                gamma_is_fixed=self._is_param_fixed[4],
                vega_weighted=self._vega_weighted_smile_fit,
                evaluation=self._zabr_evaluation,
            )
        else:
            interp = ZabrInterpolation(
                strikes=strikes, volatilities=vols,
                expiry_time=option_time, forward=forward,
                alpha_is_fixed=self._is_param_fixed[0],
                beta_is_fixed=self._is_param_fixed[1],
                nu_is_fixed=self._is_param_fixed[2],
                rho_is_fixed=self._is_param_fixed[3],
                gamma_is_fixed=self._is_param_fixed[4],
                vega_weighted=self._vega_weighted_smile_fit,
                evaluation=self._zabr_evaluation,
            )
        return (
            interp.alpha(), interp.beta(),
            interp.nu(), interp.rho(), interp.gamma(),
        )

    def _ensure_fitted(self) -> None:
        if self._fitted_params is None:
            self._fit_all()

    # --- generalised inspectors -------------------------------------------

    def xabr_parameters(self, j: int, k: int) -> tuple[float, ...]:
        """Return the fitted parameter tuple at grid cell ``(j, k)``.

        Returns a 4-tuple ``(alpha, beta, nu, rho)`` in SABR mode
        and a 5-tuple ``(alpha, beta, nu, rho, gamma)`` in ZABR mode.
        """
        self._ensure_fitted()
        assert self._fitted_params is not None
        return self._fitted_params[j][k]

    def fitted_forward(self, j: int, k: int) -> float:
        """Return the ATM forward at grid cell ``(j, k)``."""
        self._ensure_fitted()
        assert self._fitted_forwards is not None
        return self._fitted_forwards[j][k]

    # --- smile section -----------------------------------------------------

    def _nearest_cell(self, option_time: float, swap_length: float) -> tuple[int, int]:
        """Index of the grid cell closest to ``(option_time, swap_length)``."""
        j_best = 0
        d_best = abs(self._option_times[0] - option_time)
        for j in range(1, len(self._option_times)):
            d = abs(self._option_times[j] - option_time)
            if d < d_best:
                j_best = j
                d_best = d
        k_best = 0
        d_best = abs(self._swap_lengths[0] - swap_length)
        for k in range(1, len(self._swap_lengths)):
            d = abs(self._swap_lengths[k] - swap_length)
            if d < d_best:
                k_best = k
                d_best = d
        return j_best, k_best

    def smile_section_impl(
        self, option_time: float, swap_length: float
    ) -> SmileSection:
        """Return the smile section at the nearest grid cell.

        # C++ parity: ``XabrSwaptionVolatilityCube::smileSectionImpl`` —
        # diverges on the dense interior (we return nearest-cell instead
        # of interpolating params across cells).
        """
        self._ensure_fitted()
        assert self._fitted_params is not None
        assert self._fitted_forwards is not None
        j, k = self._nearest_cell(option_time, swap_length)
        forward = self._fitted_forwards[j][k]
        shift = self._atm_vol.shift(option_time, swap_length, extrapolate=True)
        if self._model_kind == XabrModelKind.SABR:
            alpha, beta, nu, rho = self._fitted_params[j][k]
            return SabrSmileSection(
                forward=forward,
                sabr_params=(alpha, beta, nu, rho),
                exercise_time=option_time,
                volatility_type=self.volatility_type(),
                shift=shift,
            )
        alpha, beta, nu, rho, gamma = self._fitted_params[j][k]
        return ZabrSmileSection(
            forward=forward,
            zabr_params=(alpha, beta, nu, rho, gamma),
            exercise_time=option_time,
            volatility_type=self.volatility_type(),
            shift=0.0,
            evaluation=self._zabr_evaluation,
        )

    def recalibrate(self) -> None:
        """Force a re-fit at every grid cell.

        # C++ parity: triggered automatically by ``performCalculations``
        # in the lazy-object path. Here it's an explicit method.
        """
        self._fit_all()


__all__ = ["XabrModelKind", "XabrSwaptionVolatilityCube"]
