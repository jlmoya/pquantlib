"""NoArbSabrInterpolatedSmileSection — fit no-arb SABR + wrap.

# C++ parity: ql/experimental/volatility/noarbsabrinterpolatedsmilesection.hpp (v1.42.1).

Composition class: fits a :class:`NoArbSabrInterpolation` to a market
strike-vol slice and exposes the fitted ``(alpha, beta, nu, rho)``
through a :class:`NoArbSabrSmileSection`. Mirrors the L9-C
:class:`SabrInterpolatedSmileSection` port.

Divergences match :class:`SviInterpolatedSmileSection`: raw-float
constructor (no Quote handles), eager fit (no LazyObject), the
``hasFloatingStrikes`` mode supported via ``has_floating_strikes``,
and ``EndCriteria`` / ``OptimizationMethod`` replaced by forwarded
``max_nfev`` / ``max_guesses``. Because each no-arb model evaluation is
expensive, ``max_guesses`` defaults to 1.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.volatility.no_arb_sabr_interpolation import (
    NoArbSabrInterpolation,
)
from pquantlib.experimental.volatility.no_arb_sabr_smile_section import (
    NoArbSabrSmileSection,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.time.date import Date


class NoArbSabrInterpolatedSmileSection(SmileSection):
    """Smile section fitted from market strike-vol data via no-arb SABR.

    Args:
        option_date: option expiry date.
        forward: ATM forward.
        strikes: market strike grid (ATM-relative when
            ``has_floating_strikes`` else absolute).
        vols: market vols (ATM-relative spreads when
            ``has_floating_strikes`` and ``atm_volatility`` given).
        atm_volatility: ATM vol, required only when
            ``has_floating_strikes``.
        has_floating_strikes: ATM-relative strike/spread mode.
        alpha, beta, nu, rho: initial values (None → defaultValues).
        alpha_is_fixed .. rho_is_fixed: pin a parameter during the fit.
        vega_weighted: vega-weight residuals during the fit.
        day_counter: day counter (default :class:`Actual365Fixed`).
        reference_date: reference date (None → floating mode).
        max_nfev / max_guesses / multi_start_seed: forwarded to
            :class:`NoArbSabrInterpolation`.
    """

    def __init__(
        self,
        *,
        option_date: Date,
        forward: float,
        strikes: Sequence[float],
        vols: Sequence[float],
        atm_volatility: float | None = None,
        has_floating_strikes: bool = False,
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
        max_nfev: int = 1000,
        max_guesses: int = 1,
        multi_start_seed: int = 42,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()
        super().__init__(
            exercise_date=option_date,
            day_counter=dc,
            reference_date=reference_date,
        )
        self._forward: float = forward

        if has_floating_strikes:
            from pquantlib import qassert  # noqa: PLC0415

            qassert.require(
                atm_volatility is not None,
                "has_floating_strikes requires atm_volatility",
            )
            assert atm_volatility is not None
            actual_strikes = [forward + float(k) for k in strikes]
            actual_vols = [atm_volatility + float(v) for v in vols]
        else:
            actual_strikes = [float(k) for k in strikes]
            actual_vols = [float(v) for v in vols]

        self._actual_strikes: list[float] = actual_strikes
        self._interp: NoArbSabrInterpolation = NoArbSabrInterpolation(
            actual_strikes,
            actual_vols,
            self.exercise_time(),
            forward,
            alpha=alpha,
            beta=beta,
            nu=nu,
            rho=rho,
            alpha_is_fixed=alpha_is_fixed,
            beta_is_fixed=beta_is_fixed,
            nu_is_fixed=nu_is_fixed,
            rho_is_fixed=rho_is_fixed,
            vega_weighted=vega_weighted,
            max_nfev=max_nfev,
            max_guesses=max_guesses,
            multi_start_seed=multi_start_seed,
        )
        self._wrapper: NoArbSabrSmileSection = NoArbSabrSmileSection(
            forward=forward,
            sabr_params=(
                self._interp.alpha(),
                self._interp.beta(),
                self._interp.nu(),
                self._interp.rho(),
            ),
            exercise_time=self.exercise_time(),
            day_counter=dc,
        )

    # --- inspectors -----------------------------------------------------

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

    def model(self) -> NoArbSabrSmileSection:
        return self._wrapper

    # --- SmileSection ---------------------------------------------------

    def min_strike(self) -> float:
        # C++: actualStrikes_.front()
        return self._actual_strikes[0]

    def max_strike(self) -> float:
        # C++: actualStrikes_.back()
        return self._actual_strikes[-1]

    def atm_level(self) -> float:
        # C++: forwardValue_
        return self._forward

    def _volatility_impl(self, strike: float) -> float:
        return self._wrapper.volatility(strike)

    def _variance_impl(self, strike: float) -> float:
        return self._wrapper.variance(strike)


__all__ = ["NoArbSabrInterpolatedSmileSection"]
