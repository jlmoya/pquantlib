"""ExtendedBlackScholesMertonProcess — GBSM with a selectable evolve scheme.

# C++ parity: ql/experimental/processes/extendedblackscholesprocess.{hpp,cpp}
# (v1.42.1).

The experimental extended Black-Scholes-Merton process specialises
:class:`GeneralizedBlackScholesProcess` by letting the caller choose the
discretization used inside ``evolve``:

* ``Euler`` — usual ``apply(expectation, stdDeviation * dw)``.
* ``Milstein`` — second-order scheme adding the ``0.5 * sigma^2 *
  (dw^2 - 1) * dt`` correction.
* ``PredictorCorrector`` — equal-weight predictor-corrector step.

The ``drift`` / ``diffusion`` overrides differ from the GBSM base in
that they are *strike-dependent local-vol* evaluations (the base GBSM
overrides only kick in on the strike-independent fast path):

* ``diffusion(t, x) = blackVolatility().blackVol(t, x, true)`` — the
  raw Black vol at (t, x), NOT the Dupire local vol.
* ``drift(t, x) = r_fwd - q_fwd - 0.5 * sigma(t, x)^2`` using the
  forward rate over a 1e-4 window.

Because these overrides feed the Euler discretization's drift/diffusion
(which call ``drift_1d`` / ``diffusion_1d``), the Euler ``evolve`` here
does NOT in general equal the plain GBSM Euler ``evolve`` — the C++
classes likewise diverge (verified by the W7-A probe).
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import final

from pquantlib.exceptions import LibraryException
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import (
    StochasticProcess1DDiscretization,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class Discretization(IntEnum):
    """Evolve-scheme selector.

    # C++ parity: ``ExtendedBlackScholesMertonProcess::Discretization``
    # enum in extendedblackscholesprocess.hpp:39.
    """

    Euler = 0
    Milstein = 1
    PredictorCorrector = 2


@final
class ExtendedBlackScholesMertonProcess(GeneralizedBlackScholesProcess):
    """GBSM process with a built-in selectable evolve discretization.

    # C++ parity: ``class ExtendedBlackScholesMertonProcess : public
    # GeneralizedBlackScholesProcess`` in extendedblackscholesprocess.hpp:36.

    Parameters
    ----------
    x0
        Spot quote.
    dividend_ts
        Dividend-yield curve.
    risk_free_ts
        Risk-free curve.
    black_vol_ts
        Black volatility surface.
    discretization
        The ``StochasticProcess1DDiscretization`` driver (default
        ``EulerDiscretization``) used by ``expectation`` / ``variance``.
    evol_disc
        The :class:`Discretization` scheme used by ``evolve`` (default
        ``Milstein``, matching C++).
    """

    def __init__(
        self,
        *,
        x0: Quote,
        dividend_ts: YieldTermStructure,
        risk_free_ts: YieldTermStructure,
        black_vol_ts: BlackVolTermStructure,
        discretization: StochasticProcess1DDiscretization | None = None,
        evol_disc: Discretization = Discretization.Milstein,
    ) -> None:
        if discretization is None:
            discretization = EulerDiscretization()
        super().__init__(
            x0=x0,
            dividend_ts=dividend_ts,
            risk_free_ts=risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
        )
        self._evol_disc: Discretization = evol_disc

    # --- overrides (strike-dependent raw-Black-vol form) ----------------

    def diffusion_1d(self, t: float, x: float) -> float:
        """Raw Black vol at (t, x).

        # C++ parity: ``ExtendedBlackScholesMertonProcess::diffusion`` —
        # ``blackVolatility()->blackVol(t, x, true)``.
        """
        return self.black_volatility().black_vol_at_time(t, x, extrapolate=True)

    def drift_1d(self, t: float, x: float) -> float:
        """Drift = r_fwd - q_fwd - 0.5 * sigma^2.

        # C++ parity: ``ExtendedBlackScholesMertonProcess::drift`` — uses
        # the forward rate over a (t, t + 1e-4) window.
        """
        sigma = self.diffusion_1d(t, x)
        t1 = t + 0.0001
        r = self.risk_free_rate().forward_rate(
            t, t1, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        q = self.dividend_yield().forward_rate(
            t, t1, Compounding.Continuous, Frequency.NoFrequency, True
        ).rate()
        return r - q - 0.5 * sigma * sigma

    def evolve_1d(self, t0: float, x0: float, dt: float, dw: float) -> float:
        """Evolve via the selected discretization scheme.

        # C++ parity: ``ExtendedBlackScholesMertonProcess::evolve`` —
        # switch over ``discretization_`` (the evol-disc enum).
        """
        if self._evol_disc == Discretization.Milstein:
            # Milstein scheme.
            sigma = self.diffusion_1d(t0, x0)
            return self.apply_1d(
                x0,
                self.drift_1d(t0, x0) * dt
                + 0.5 * (sigma**2) * (dw * dw - 1.0) * dt
                + sigma * math.sqrt(dt) * dw,
            )
        if self._evol_disc == Discretization.Euler:
            # Usual Euler scheme.
            return self.apply_1d(
                self.expectation_1d(t0, x0, dt),
                self.std_deviation_1d(t0, x0, dt) * dw,
            )
        if self._evol_disc == Discretization.PredictorCorrector:
            # Predictor-Corrector scheme with equal weighting.
            predictor = self.apply_1d(
                self.expectation_1d(t0, x0, dt),
                self.std_deviation_1d(t0, x0, dt) * dw,
            )
            t1 = t0 + 0.0001
            sigma0 = self.diffusion_1d(t0, x0)
            sigma1 = self.diffusion_1d(t0 + dt, predictor)
            rate0 = (
                self.risk_free_rate()
                .forward_rate(t0, t1, Compounding.Continuous, Frequency.NoFrequency, True)
                .rate()
                - self.dividend_yield()
                .forward_rate(t0, t1, Compounding.Continuous, Frequency.NoFrequency, True)
                .rate()
                - 0.5 * (sigma0**2)
            )
            rate1 = (
                self.risk_free_rate()
                .forward_rate(
                    t0 + dt, t1 + dt, Compounding.Continuous, Frequency.NoFrequency, True
                )
                .rate()
                - self.dividend_yield()
                .forward_rate(
                    t0 + dt, t1 + dt, Compounding.Continuous, Frequency.NoFrequency, True
                )
                .rate()
                - 0.5 * (sigma1**2)
            )
            drift_term = 0.5 * rate1 + 0.5 * rate0
            diffusion_term = 0.5 * (sigma1 + sigma0)
            return self.apply_1d(
                x0, drift_term * dt + diffusion_term * math.sqrt(dt) * dw
            )
        raise LibraryException("unknown discretization scheme")


__all__ = ["Discretization", "ExtendedBlackScholesMertonProcess"]
