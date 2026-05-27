"""GeneralizedBlackScholesProcess — Black-Scholes process with dividends.

# C++ parity: ql/processes/blackscholesprocess.{hpp,cpp} (v1.42.1).

The C++ process describes the stock S governed by::

    d ln S(t) = (r(t) - q(t) - 0.5 * sigma(t, S)^2) dt + sigma dW_t

(internally computes are in log-space; ``apply`` does ``x0 * exp(dx)``
to convert back).

Python divergences vs C++:

* C++ uses ``Handle<Quote>`` / ``Handle<YieldTermStructure>`` /
  ``Handle<BlackVolTermStructure>``. Python passes the underlying
  objects directly (``Quote`` / ``YieldTermStructure`` /
  ``BlackVolTermStructure``); the relevant ``register_with`` calls
  happen at construction.
* The C++ ``localVolatility()`` accessor dynamically casts the Black
  vol curve to ``BlackConstantVol`` / ``BlackVarianceCurve`` to pick
  the optimal local-vol form. The Python port replicates this with
  ``isinstance`` checks and caches the result in ``_local_vol``.
* C++ also supports an ``externalLocalVolTS_`` constructor; the
  Python port keeps the same overload — pass ``local_vol_ts=...``.
* When the local vol is strike-independent (constant or pure-time)
  AND ``forceDiscretization=False`` (the default), the closed-form
  branches are used for ``expectation`` / ``variance`` / ``evolve``.
  Otherwise the Euler discretization handles the time-stepping.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.stochastic_process_1d import (
    StochasticProcess1D,
    StochasticProcess1DDiscretization,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.volatility.equity_fx.local_constant_vol import (
    LocalConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_curve import (
    LocalVolCurve,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_surface import (
    LocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    pass


class GeneralizedBlackScholesProcess(StochasticProcess1D):
    """Black-Scholes process with deterministic risk-free + dividend yields.

    # C++ parity: ``class GeneralizedBlackScholesProcess : public
    # StochasticProcess1D``.

    Internally calculations are in log-space (``apply`` does
    ``x0 * exp(dx)``).
    """

    def __init__(
        self,
        *,
        x0: Quote,
        dividend_ts: YieldTermStructure,
        risk_free_ts: YieldTermStructure,
        black_vol_ts: BlackVolTermStructure,
        local_vol_ts: LocalVolTermStructure | None = None,
        discretization: StochasticProcess1DDiscretization | None = None,
        force_discretization: bool = False,
    ) -> None:
        if discretization is None:
            discretization = EulerDiscretization()
        super().__init__(discretization)
        self._x0_quote: Quote = x0
        self._risk_free_ts: YieldTermStructure = risk_free_ts
        self._dividend_ts: YieldTermStructure = dividend_ts
        self._black_vol_ts: BlackVolTermStructure = black_vol_ts
        self._has_external_local_vol: bool = local_vol_ts is not None
        self._external_local_vol_ts: LocalVolTermStructure | None = local_vol_ts
        self._force_discretization: bool = force_discretization
        self._updated: bool = False
        self._is_strike_independent: bool = False
        self._local_vol_cached: LocalVolTermStructure | None = None
        # Register so updates propagate.
        x0.register_with(self)
        risk_free_ts.register_with(self)
        dividend_ts.register_with(self)
        black_vol_ts.register_with(self)
        if local_vol_ts is not None:
            local_vol_ts.register_with(self)

    # --- StochasticProcess1D abstract overrides --------------------------

    def x0(self) -> float:
        return self._x0_quote.value()

    def diffusion_1d(self, t: float, x: float) -> float:
        """Diffusion = local volatility at (t, x).

        # C++ parity: ``GeneralizedBlackScholesProcess::diffusion`` —
        # ``localVolatility()->localVol(t, x, true)``.
        """
        return self.local_volatility().local_vol_at_time(t, x, extrapolate=True)

    def drift_1d(self, t: float, x: float) -> float:
        """Drift = (r - q) - 0.5 * sigma^2.

        # C++ parity: ``GeneralizedBlackScholesProcess::drift`` —
        # uses the *forward rate* over a small (1e-4) window for
        # numerical stability vs zero-rate noise at t=0.
        """
        sigma = self.diffusion_1d(t, x)
        t1 = t + 0.0001
        r = self._risk_free_ts.forward_rate(
            t, t1, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        q = self._dividend_ts.forward_rate(
            t, t1, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        return r - q - 0.5 * sigma * sigma

    def apply_1d(self, x0: float, dx: float) -> float:
        """Apply in log-space: ``x0 * exp(dx)``.

        # C++ parity: ``GeneralizedBlackScholesProcess::apply``.
        """
        return x0 * math.exp(dx)

    # --- closed-form overrides when strike-independent -------------------

    def expectation_1d(self, t0: float, x0: float, dt: float) -> float:
        """Strike-independent → exact ``x0 * exp((r-q) dt)``.

        # C++ parity: ``GeneralizedBlackScholesProcess::expectation`` —
        # uses the closed-form on the strike-independent branch and
        # ``QL_FAIL("not implemented")`` otherwise (i.e. the C++ code
        # does NOT expose an Euler-based expectation for the strike-
        # dependent case; only ``evolve`` is supported).
        """
        self.local_volatility()  # trigger isStrikeIndependent_ refresh
        if self._is_strike_independent and not self._force_discretization:
            r = self._risk_free_ts.forward_rate(
                t0, t0 + dt, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            q = self._dividend_ts.forward_rate(
                t0, t0 + dt, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            return x0 * math.exp(dt * (r - q))
        raise LibraryException("not implemented")

    def std_deviation_1d(self, t0: float, x0: float, dt: float) -> float:
        """Closed-form sqrt(variance) when strike-independent.

        # C++ parity: ``GeneralizedBlackScholesProcess::stdDeviation``.
        """
        self.local_volatility()
        if self._is_strike_independent and not self._force_discretization:
            return math.sqrt(self.variance_1d(t0, x0, dt))
        return super().std_deviation_1d(t0, x0, dt)

    def variance_1d(self, t0: float, x0: float, dt: float) -> float:
        """Closed-form variance via Black variance differential.

        # C++ parity: ``GeneralizedBlackScholesProcess::variance`` —
        # ``blackVariance(t0+dt, 0.01) - blackVariance(t0, 0.01)``.
        # The strike value (0.01) is irrelevant for strike-independent
        # vol curves.
        """
        self.local_volatility()
        if self._is_strike_independent and not self._force_discretization:
            return self._black_vol_ts.black_variance_at_time(
                t0 + dt, 0.01, extrapolate=True
            ) - self._black_vol_ts.black_variance_at_time(t0, 0.01, extrapolate=True)
        return super().variance_1d(t0, x0, dt)

    def evolve_1d(self, t0: float, x0: float, dt: float, dw: float) -> float:
        """Strike-independent: ``apply(x0, sqrt(var) * dw + drift)`` with
        drift = (r-q) * dt - 0.5 * var; otherwise raw Euler step.

        # C++ parity: ``GeneralizedBlackScholesProcess::evolve``. The
        # else-branch in C++ explicitly uses the discretization's
        # ``drift`` (not ``expectation``) because ``expectation`` would
        # itself ``QL_FAIL`` in the strike-dependent case.
        """
        self.local_volatility()
        if self._is_strike_independent and not self._force_discretization:
            var = self.variance_1d(t0, x0, dt)
            r = self._risk_free_ts.forward_rate(
                t0, t0 + dt, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            q = self._dividend_ts.forward_rate(
                t0, t0 + dt, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            drift = (r - q) * dt - 0.5 * var
            return self.apply_1d(x0, math.sqrt(var) * dw + drift)
        # Strike-dependent / forced Euler: apply(x0, mu*dt + sigma*sqrt(dt)*dw).
        # We bypass the StochasticProcess1D.evolve_1d default (which would
        # call expectation_1d and raise) and use the discretization directly.
        assert self._discretization_1d is not None
        euler_drift = self._discretization_1d.drift(self, t0, x0, dt)
        return self.apply_1d(x0, euler_drift + self.std_deviation_1d(t0, x0, dt) * dw)

    # --- utilities -------------------------------------------------------

    def time(self, date: Date) -> float:
        """Year fraction via the risk-free curve's day counter.

        # C++ parity: ``GeneralizedBlackScholesProcess::time``.
        """
        return self._risk_free_ts.day_counter().year_fraction(
            self._risk_free_ts.reference_date(), date
        )

    # --- inspectors ------------------------------------------------------

    def state_variable(self) -> Quote:
        return self._x0_quote

    def dividend_yield(self) -> YieldTermStructure:
        return self._dividend_ts

    def risk_free_rate(self) -> YieldTermStructure:
        return self._risk_free_ts

    def black_volatility(self) -> BlackVolTermStructure:
        return self._black_vol_ts

    def local_volatility(self) -> LocalVolTermStructure:
        """Derive (and cache) the local-vol term structure.

        # C++ parity: ``GeneralizedBlackScholesProcess::localVolatility``
        # — picks an optimal subclass based on the Black-vol-curve type.

        Three cases:

        * ``BlackConstantVol`` → ``LocalConstantVol`` (strike-independent).
        * ``BlackVarianceCurve`` → ``LocalVolCurve`` (Dupire on a curve,
          still strike-independent).
        * Otherwise → ``LocalVolSurface`` (Dupire on the surface,
          strike-dependent).
        """
        if self._has_external_local_vol:
            assert self._external_local_vol_ts is not None
            return self._external_local_vol_ts

        if not self._updated:
            self._is_strike_independent = True
            black_ts = self._black_vol_ts

            if isinstance(black_ts, BlackConstantVol):
                # C++ parity: cast → LocalConstantVol with the constant
                # Black vol evaluated at (0.0, x0).
                vol = black_ts.black_vol_at_time(0.0, self._x0_quote.value(), extrapolate=True)
                ref = black_ts.reference_date()
                dc: DayCounter = black_ts.day_counter()
                self._local_vol_cached = LocalConstantVol(
                    reference_date=ref, volatility=vol, day_counter=dc
                )
                self._updated = True
                return self._local_vol_cached

            if isinstance(black_ts, BlackVarianceCurve):
                # C++ parity: cast → LocalVolCurve (Dupire on a variance
                # curve — strike-independent).
                self._local_vol_cached = LocalVolCurve(black_ts)
                self._updated = True
                return self._local_vol_cached

            # Strike-dependent — Dupire on the full surface. Python's
            # ``LocalVolSurface`` accepts only the Black-vol-TS plus an
            # underlying (no rates/div curves — those would lead to
            # additional complexity that's deferred for L3-D).
            self._local_vol_cached = LocalVolSurface(
                black_ts=black_ts,
                underlying=self._x0_quote.value(),
            )
            self._updated = True
            self._is_strike_independent = False
            return self._local_vol_cached

        assert self._local_vol_cached is not None
        return self._local_vol_cached

    # --- Observer interface ---------------------------------------------

    def update(self) -> None:
        """Reset local-vol cache and propagate.

        # C++ parity: ``GeneralizedBlackScholesProcess::update``.
        """
        self._updated = False
        super().update()


__all__ = ["GeneralizedBlackScholesProcess"]
