"""SabrInterpolatedSmileSection — fit SABR to a strike-vol slice + wrap.

# C++ parity: ql/termstructures/volatility/sabrinterpolatedsmilesection.{hpp,cpp}
# (v1.42.1).

Composition class: fits a :class:`SabrInterpolation` to a market
strike-vol slice and exposes the fitted (alpha, beta, nu, rho) through
a :class:`SabrSmileSection` so all SmileSection methods route through
the closed-form Hagan 2002 evaluation.

PQuantLib divergences:

* **Constructor surface.** C++ exposes two parallel constructors —
  "all market data are quotes" (``Handle<Quote>`` for forward, ATM
  vol, and per-strike vols) and "no quotes" (raw floats). We collapse
  onto the simpler raw-float / no-Quote signature; callers that need
  observability can register an outer observer manually. The
  ``hasFloatingStrikes`` C++ flag (which treats strikes as ATM-relative
  offsets) is deferred — we keep absolute strikes.
* **Optimization method / end-criteria.** C++ accepts ``EndCriteria``
  + ``OptimizationMethod`` overrides. PQuantLib forwards
  ``max_nfev`` + ``max_guesses`` through to
  :class:`SabrInterpolation`; the actual optimizer is fixed at
  ``scipy.optimize.least_squares(method='trf')`` (see the
  ``sabr_interpolation`` module docstring).
* **LazyObject.** C++ is a ``LazyObject`` that defers the SABR fit
  until first ``calculate()``. We fit eagerly in ``__init__`` — the
  fit cost is dominated by the SABR closed-form evaluation per
  strike, which is microseconds.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.sabr_interpolation import SabrInterpolation
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class SabrInterpolatedSmileSection(SmileSection):
    """Smile section fitted from market strike-vol data via SABR.

    Args:
        option_date: option expiry date.
        forward: ATM forward.
        strikes: market strike grid (ascending, length >= 2).
        vols: market vols (same length as strikes).
        alpha, beta, nu, rho: SABR parameter initial values (None to
            use the C++ ``defaultValues`` rule).
        alpha_is_fixed, beta_is_fixed, nu_is_fixed, rho_is_fixed:
            pin parameter to its initial value during the fit.
        vega_weighted: vega-weight residuals during the fit.
        day_counter: day counter to convert ``option_date`` into the
            section's exercise time. Default
            :class:`Actual365Fixed` (matches C++).
        reference_date: reference date for the conversion. Default
            ``None`` selects floating-mode anchoring against the
            global evaluation date.
        volatility_type: SABR vol type (``ShiftedLognormal`` or
            ``Normal``). Default ``ShiftedLognormal``.
        shift: shifted-lognormal shift.
        max_nfev: optimisation budget passed to
            :class:`SabrInterpolation`.
        max_guesses: Halton multi-start guesses (L10-A).
    """

    def __init__(
        self,
        *,
        option_date: Date,
        forward: float,
        strikes: Sequence[float],
        vols: Sequence[float],
        alpha: float | None = None,
        beta: float | None = None,
        nu: float | None = None,
        rho: float | None = None,
        alpha_is_fixed: bool = False,
        beta_is_fixed: bool = False,
        nu_is_fixed: bool = False,
        rho_is_fixed: bool = False,
        vega_weighted: bool = False,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
        max_nfev: int = 1000,
        max_guesses: int = 1,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()
        super().__init__(
            exercise_date=option_date,
            day_counter=dc,
            reference_date=reference_date,
            volatility_type=volatility_type,
            shift=shift,
        )
        self._forward: float = forward
        # Fit SABR.
        self._interp: SabrInterpolation = SabrInterpolation(
            strikes=strikes,
            volatilities=vols,
            expiry_time=self.exercise_time(),
            forward=forward,
            alpha=alpha,
            beta=beta,
            nu=nu,
            rho=rho,
            alpha_is_fixed=alpha_is_fixed,
            beta_is_fixed=beta_is_fixed,
            nu_is_fixed=nu_is_fixed,
            rho_is_fixed=rho_is_fixed,
            vega_weighted=vega_weighted,
            shift=shift,
            volatility_type=volatility_type,
            max_nfev=max_nfev,
            max_guesses=max_guesses,
        )
        # Wrap fitted params in a SabrSmileSection. All SmileSection
        # accessors delegate through this wrapper.
        self._wrapper: SabrSmileSection = SabrSmileSection(
            forward=forward,
            sabr_params=(
                self._interp.alpha(),
                self._interp.beta(),
                self._interp.nu(),
                self._interp.rho(),
            ),
            exercise_time=self.exercise_time(),
            day_counter=dc,
            volatility_type=volatility_type,
            shift=shift,
        )
        # Cache strikes for min/max-strike inspectors.
        self._actual_strikes: list[float] = [float(k) for k in strikes]

    # --- inspectors ---------------------------------------------------

    def alpha(self) -> float:
        return self._interp.alpha()

    def beta(self) -> float:
        return self._interp.beta()

    def nu(self) -> float:
        return self._interp.nu()

    def rho(self) -> float:
        return self._interp.rho()

    def rms_error(self) -> float:
        return self._interp.rms_error()

    def max_error(self) -> float:
        return self._interp.max_error()

    def converged(self) -> bool:
        return self._interp.converged()

    # --- SmileSection ----------------------------------------------

    def min_strike(self) -> float:
        # C++: ``return actualStrikes_.front();`` — we mirror that
        # rather than ``SabrSmileSection``'s ``-shift`` (which would
        # imply allowing extrapolation outside the calibrated grid).
        return self._actual_strikes[0]

    def max_strike(self) -> float:
        return self._actual_strikes[-1]

    def atm_level(self) -> float:
        return self._forward

    def _volatility_impl(self, strike: float) -> float:
        # Route through the wrapping SabrSmileSection so that any
        # SABR-specific clamping (e.g. strike + shift > eps) is honored.
        return self._wrapper.volatility(strike)

    def _variance_impl(self, strike: float) -> float:
        return self._wrapper.variance(strike)


__all__ = ["SabrInterpolatedSmileSection"]
