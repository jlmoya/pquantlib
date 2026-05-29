"""ZabrInterpolatedSmileSection — fit ZABR to a strike-vol slice + wrap.

# C++ parity: ql/termstructures/volatility/zabrinterpolatedsmilesection.hpp
#             (v1.42.1).

Composition class: fits a :class:`ZabrInterpolation` to a market
strike-vol slice and exposes the fitted ``(alpha, beta, nu, rho, gamma)``
through a :class:`ZabrSmileSection` so all SmileSection methods route
through the closed-form ZABR evaluation. Mirrors the L9-C
:class:`SabrInterpolatedSmileSection` port extended with the fifth
ZABR parameter ``gamma`` (closes the L10-C
``ZabrInterpolatedSmileSection`` carve-out).

PQuantLib divergences:

* **Constructor surface.** C++ exposes two parallel constructors —
  "all market data are quotes" (``Handle<Quote>`` for forward, ATM
  vol, and per-strike vols) and "no quotes" (raw floats). We collapse
  onto the simpler raw-float / no-Quote signature; callers that need
  observability can register an outer observer manually. The
  ``hasFloatingStrikes`` C++ flag (which treats strikes as ATM-relative
  offsets) is deferred — we keep absolute strikes. Same shape as
  :class:`SabrInterpolatedSmileSection`.
* **Optimization method / end-criteria.** C++ accepts ``EndCriteria``
  + ``OptimizationMethod`` overrides. PQuantLib forwards ``max_nfev`` +
  ``max_guesses`` through to :class:`ZabrInterpolation`; the actual
  optimizer is fixed at ``scipy.optimize.least_squares(method='trf')``
  (see the ``zabr_interpolation`` module docstring).
* **LazyObject.** C++ is a ``LazyObject`` that defers the ZABR fit
  until first ``calculate()``. We fit eagerly in ``__init__`` — the
  fit cost is dominated by the ZABR closed-form (or RK45) evaluation
  per strike. For ``gamma == 1`` arms the fit is microseconds; for
  ``gamma != 1`` arms the ODE integration makes the constructor
  materially more expensive (still fast enough for unit tests).
* **Shifted / Normal vols.** ZABR does not support shifted-lognormal
  or normal volatility types in its native short-maturity arms; the
  ``shift`` argument is rejected if non-zero and ``volatility_type``
  defaults to ``ShiftedLognormal`` purely so the inherited
  :class:`SmileSection` interface returns a meaningful flag.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
from pquantlib.math.interpolations.zabr_interpolation import ZabrInterpolation
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection
from pquantlib.time.date import Date


class ZabrInterpolatedSmileSection(SmileSection):
    """Smile section fitted from market strike-vol data via ZABR.

    Args:
        option_date: option expiry date.
        forward: ATM forward.
        strikes: market strike grid (ascending, length >= 2).
        vols: market vols (same length as strikes).
        alpha, beta, nu, rho, gamma: ZABR parameter initial values
            (None to use the C++ ``defaultValues`` rule).
        alpha_is_fixed, beta_is_fixed, nu_is_fixed, rho_is_fixed,
            gamma_is_fixed: pin parameter to its initial value during
            the fit.
        vega_weighted: vega-weight residuals during the fit.
        day_counter: day counter to convert ``option_date`` into the
            section's exercise time. Default
            :class:`Actual365Fixed` (matches C++).
        reference_date: reference date for the conversion. Default
            ``None`` selects floating-mode anchoring against the
            global evaluation date.
        evaluation: ZABR evaluation mode used by the underlying
            :class:`ZabrInterpolation` and downstream
            :class:`ZabrSmileSection`. Default
            :attr:`ZabrEvaluation.ShortMaturityLognormal`.
        max_nfev: optimisation budget passed to
            :class:`ZabrInterpolation`.
        max_guesses: Halton multi-start guesses (L10-A pattern).
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
        gamma: float | None = None,
        alpha_is_fixed: bool = False,
        beta_is_fixed: bool = False,
        nu_is_fixed: bool = False,
        rho_is_fixed: bool = False,
        gamma_is_fixed: bool = False,
        vega_weighted: bool = False,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
        max_nfev: int = 1000,
        max_guesses: int = 1,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()
        super().__init__(
            exercise_date=option_date,
            day_counter=dc,
            reference_date=reference_date,
            # ZABR is unshifted; flag is purely informational for
            # downstream SmileSection consumers.
            volatility_type=VolatilityType.ShiftedLognormal,
            shift=0.0,
        )
        self._forward: float = forward
        # Fit ZABR.
        self._interp: ZabrInterpolation = ZabrInterpolation(
            strikes=strikes,
            volatilities=vols,
            expiry_time=self.exercise_time(),
            forward=forward,
            alpha=alpha,
            beta=beta,
            nu=nu,
            rho=rho,
            gamma=gamma,
            alpha_is_fixed=alpha_is_fixed,
            beta_is_fixed=beta_is_fixed,
            nu_is_fixed=nu_is_fixed,
            rho_is_fixed=rho_is_fixed,
            gamma_is_fixed=gamma_is_fixed,
            vega_weighted=vega_weighted,
            evaluation=evaluation,
            max_nfev=max_nfev,
            max_guesses=max_guesses,
        )
        # Wrap fitted params in a ZabrSmileSection. All SmileSection
        # accessors delegate through this wrapper.
        self._wrapper: ZabrSmileSection = ZabrSmileSection(
            forward=forward,
            zabr_params=(
                self._interp.alpha(),
                self._interp.beta(),
                self._interp.nu(),
                self._interp.rho(),
                self._interp.gamma(),
            ),
            exercise_time=self.exercise_time(),
            day_counter=dc,
            evaluation=evaluation,
        )
        # Cache strikes for min/max-strike inspectors (matches C++ —
        # ZabrInterpolatedSmileSection::minStrike returns
        # ``actualStrikes_.front()``).
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

    def gamma(self) -> float:
        return self._interp.gamma()

    def rms_error(self) -> float:
        return self._interp.rms_error()

    def max_error(self) -> float:
        return self._interp.max_error()

    def converged(self) -> bool:
        return self._interp.converged()

    def evaluation(self) -> ZabrEvaluation:
        return self._interp.evaluation()

    # --- SmileSection ----------------------------------------------

    def min_strike(self) -> float:
        # C++: ``return actualStrikes_.front();``
        return self._actual_strikes[0]

    def max_strike(self) -> float:
        return self._actual_strikes[-1]

    def atm_level(self) -> float:
        # C++: ``return forwardValue_;``
        return self._forward

    def _volatility_impl(self, strike: float) -> float:
        # Route through the wrapping ZabrSmileSection so the strike
        # clamp + ZABR mode dispatch is honored.
        return self._wrapper.volatility(strike)

    def _variance_impl(self, strike: float) -> float:
        return self._wrapper.variance(strike)


__all__ = ["ZabrInterpolatedSmileSection"]
