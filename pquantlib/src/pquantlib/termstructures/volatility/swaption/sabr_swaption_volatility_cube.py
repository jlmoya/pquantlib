"""SabrSwaptionVolatilityCube — SABR-fitted swaption vol cube.

# C++ parity: ql/termstructures/volatility/swaption/sabrswaptionvolatilitycube.hpp
# instantiated with ``SwaptionVolCubeSabrModel`` (Hagan 2002 SABR).
# (v1.42.1).

The C++ class is a template ``XabrSwaptionVolatilityCube<Model>``
specialised for SABR via ``SwaptionVolCubeSabrModel``. The bulk of
the C++ machinery (sparse-cube → dense-cube fill, sparse-smile cache,
spread-vol interpolation) is geared toward the lazy "fit-then-densify"
performance path that supports the ``isAtmCalibrated`` switch.

PQuantLib lands the **eager-fit** subset: at construction (and on any
observed Quote update via ``perform_calculations``) the cube fits a
SABR slice per ``(option_tenor, swap_tenor)`` and stores the 4-parameter
result. ``smile_section_impl(t, s)`` then resolves the nearest
(option_tenor, swap_tenor) grid cell and returns a
:class:`SabrSmileSection` built from those fitted parameters.

Documented divergences vs C++:

* No ``performCalculations`` LazyObject hook — Python recomputes on
  each ``smile_section_impl`` invocation via the cached parameter cube
  + on Quote-driven re-fit only when explicitly requested via
  :meth:`recalibrate`.
* No bilinear interpolation over the parameter cube between grid
  cells — we return the nearest grid cell's section. This matches the
  C++ ``XabrSwaptionVolatilityCube`` when interrogated at grid pillars
  but diverges in the dense interior. The L9-C tests verify pillar
  behaviour.
* No ``backwardFlat`` / ``maxGuesses`` / ``cutoffStrike`` /
  ``errorAccept`` knobs; the SciPy fitter exposes its native
  ``ftol`` / ``xtol`` / ``gtol`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.math.interpolations.sabr_interpolation import SabrInterpolation
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
from pquantlib.time.period import Period


class SabrSwaptionVolatilityCube(SwaptionVolatilityCube):
    """SABR-fitted swaption volatility cube (Hagan 2002).

    Args:
        atm_vol_structure / option_tenors / swap_tenors / strike_spreads
            / vol_spreads / swap_index_base / short_swap_index_base /
            vega_weighted_smile_fit: same as
            :class:`SwaptionVolatilityCube`.
        sabr_initial_guess: optional outer list shape
            ``(n_option_tenors x n_swap_tenors) x 4`` of initial
            ``(alpha, beta, nu, rho)`` quadruples. If ``None``, each
            cell uses ``SabrInterpolation``'s default initial guess.
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
        sabr_initial_guess: Sequence[Sequence[tuple[float, float, float, float]]] | None = None,
        is_parameter_fixed: tuple[bool, bool, bool, bool] = (False, False, False, False),
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
        self._initial_guess: (
            Sequence[Sequence[tuple[float, float, float, float]]] | None
        ) = sabr_initial_guess
        self._is_param_fixed: tuple[bool, bool, bool, bool] = is_parameter_fixed
        # Fitted parameters: ``(alpha, beta, nu, rho)`` per (j, k) grid
        # cell. Filled lazily by :meth:`_fit_all` on first access.
        self._fitted_params: list[list[tuple[float, float, float, float]]] | None = None
        self._fitted_forwards: list[list[float]] | None = None

    # --- fit driver --------------------------------------------------------

    def _fit_all(self) -> None:
        """Fit a SABR slice at each ``(option_tenor, swap_tenor)`` cell."""
        n_opt = len(self._option_tenors)
        n_swap = len(self._swap_tenors)
        n_strikes = len(self._strike_spreads)
        params: list[list[tuple[float, float, float, float]]] = [
            [(0.0, 0.0, 0.0, 0.0)] * n_swap for _ in range(n_opt)
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
                # Default guess: per-cell from constructor, else None
                # → SabrInterpolation uses its built-in defaults.
                if self._initial_guess is not None:
                    a0, b0, n0, r0 = self._initial_guess[j][k]
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
                params[j][k] = (interp.alpha(), interp.beta(), interp.nu(), interp.rho())
                forwards[j][k] = forward
        self._fitted_params = params
        self._fitted_forwards = forwards

    def _ensure_fitted(self) -> None:
        if self._fitted_params is None:
            self._fit_all()

    # --- inspectors --------------------------------------------------------

    def sabr_parameters(self, j: int, k: int) -> tuple[float, float, float, float]:
        """Return ``(alpha, beta, nu, rho)`` at grid cell ``(j, k)``."""
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
        """Return the SabrSmileSection at the nearest grid cell.

        # C++ parity: ``XabrSwaptionVolatilityCube::smileSectionImpl``
        # — diverges on dense interior (we return nearest-cell instead
        # of interpolating SABR params across cells).
        """
        self._ensure_fitted()
        assert self._fitted_params is not None
        assert self._fitted_forwards is not None
        j, k = self._nearest_cell(option_time, swap_length)
        alpha, beta, nu, rho = self._fitted_params[j][k]
        forward = self._fitted_forwards[j][k]
        shift = self._atm_vol.shift(option_time, swap_length, extrapolate=True)
        return SabrSmileSection(
            forward=forward,
            sabr_params=(alpha, beta, nu, rho),
            exercise_time=option_time,
            volatility_type=self.volatility_type(),
            shift=shift,
        )

    def recalibrate(self) -> None:
        """Force a re-fit at every grid cell.

        # C++ parity: triggered automatically by ``performCalculations``
        # in the lazy-object path. Here it's an explicit method.
        """
        self._fit_all()
