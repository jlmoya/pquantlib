"""SviInterpolatedSmileSection — fit SVI to a strike-vol slice + wrap.

# C++ parity: ql/experimental/volatility/sviinterpolatedsmilesection.hpp (v1.42.1).

Composition class: fits a :class:`SviInterpolation` to a market
strike-vol slice and exposes the fitted ``(a, b, sigma, rho, m)``
through a :class:`SviSmileSection` so all SmileSection methods route
through the raw-SVI closed form. Mirrors the L9-C
:class:`SabrInterpolatedSmileSection` / L10-C
:class:`ZabrInterpolatedSmileSection` ports.

PQuantLib divergences (identical in spirit to
:class:`ZabrInterpolatedSmileSection`):

* **Constructor surface.** C++ exposes parallel "all quotes" and "no
  quotes" constructors. We collapse onto the raw-float / no-Quote
  signature; the ``hasFloatingStrikes`` flag (ATM-relative strikes) is
  supported via the ``has_floating_strikes`` kwarg.
* **LazyObject.** C++ defers the fit to first ``calculate()``. We fit
  eagerly in ``__init__`` (the SVI multi-start fit is sub-second).
* **Optimization knobs.** ``EndCriteria`` / ``OptimizationMethod``
  overrides are replaced by the forwarded ``max_nfev`` / ``max_guesses``
  (the optimizer is fixed at ``scipy.optimize.least_squares(trf)``).
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.volatility.svi_interpolation import SviInterpolation
from pquantlib.experimental.volatility.svi_smile_section import SviSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.time.date import Date


class SviInterpolatedSmileSection(SmileSection):
    """Smile section fitted from market strike-vol data via raw SVI.

    Args:
        option_date: option expiry date.
        forward: ATM forward.
        strikes: market strike grid. When ``has_floating_strikes`` is
            ``True`` these are ATM-relative offsets (the actual strikes
            are ``forward + strikes[i]``); otherwise absolute strikes.
        vols: market vols (same length as strikes). When
            ``has_floating_strikes`` is ``True`` and ``atm_volatility``
            is given, ``vols`` are ATM-relative vol spreads (the fitted
            vols are ``atm_volatility + vols[i]``).
        atm_volatility: ATM vol, required only when
            ``has_floating_strikes`` is ``True``.
        has_floating_strikes: treat ``strikes`` / ``vols`` as
            ATM-relative offsets / spreads. Default ``False``.
        a, b, sigma, rho, m: SVI initial values (None → defaultValues).
        a_is_fixed .. m_is_fixed: pin a parameter during the fit.
        vega_weighted: vega-weight residuals during the fit.
        day_counter: day counter for ``option_date`` → exercise time;
            default :class:`Actual365Fixed` (matches C++).
        reference_date: reference date; ``None`` → floating mode.
        max_nfev / max_guesses: forwarded to :class:`SviInterpolation`.
        multi_start_seed: forwarded to :class:`SviInterpolation`.
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
        a: float | None = None,
        b: float | None = None,
        sigma: float | None = None,
        rho: float | None = None,
        m: float | None = None,
        a_is_fixed: bool = False,
        b_is_fixed: bool = False,
        sigma_is_fixed: bool = False,
        rho_is_fixed: bool = False,
        m_is_fixed: bool = False,
        vega_weighted: bool = False,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        max_nfev: int = 1000,
        max_guesses: int = 50,
        multi_start_seed: int = 42,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()
        super().__init__(
            exercise_date=option_date,
            day_counter=dc,
            reference_date=reference_date,
        )
        self._forward: float = forward

        # Resolve actual strikes + vols (C++ performCalculations).
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
        self._interp: SviInterpolation = SviInterpolation(
            actual_strikes,
            actual_vols,
            self.exercise_time(),
            forward,
            a=a,
            b=b,
            sigma=sigma,
            rho=rho,
            m=m,
            a_is_fixed=a_is_fixed,
            b_is_fixed=b_is_fixed,
            sigma_is_fixed=sigma_is_fixed,
            rho_is_fixed=rho_is_fixed,
            m_is_fixed=m_is_fixed,
            vega_weighted=vega_weighted,
            max_nfev=max_nfev,
            max_guesses=max_guesses,
            multi_start_seed=multi_start_seed,
        )
        self._wrapper: SviSmileSection = SviSmileSection(
            forward=forward,
            svi_params=(
                self._interp.a(),
                self._interp.b(),
                self._interp.sigma(),
                self._interp.rho(),
                self._interp.m(),
            ),
            exercise_time=self.exercise_time(),
            day_counter=dc,
        )

    # --- inspectors -----------------------------------------------------

    def a(self) -> float:
        return self._interp.a()

    def b(self) -> float:
        return self._interp.b()

    def sigma(self) -> float:
        return self._interp.sigma()

    def rho(self) -> float:
        return self._interp.rho()

    def m(self) -> float:
        return self._interp.m()

    def rms_error(self) -> float:
        return self._interp.rms_error()

    def max_error(self) -> float:
        return self._interp.max_error()

    def converged(self) -> bool:
        return self._interp.converged()

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


__all__ = ["SviInterpolatedSmileSection"]
