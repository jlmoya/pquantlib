"""Merton76Process — Merton (1976) jump-diffusion process.

# C++ parity: ql/processes/merton76process.{hpp,cpp} (v1.43) —
# ``class Merton76Process : public StochasticProcess1D``.

A thin composite: an inner :class:`BlackScholesMertonProcess` carrying the
diffusive part, plus three quotes describing the compound-Poisson jump part
— intensity ``lambda``, mean log-jump ``mu_J`` and log-jump volatility
``sigma_J``.

It is a ``StochasticProcess1D`` in name only.  ``drift``, ``diffusion`` and
``apply`` all raise: C++ defines them as

    Real drift(Time, Real) const override
        { QL_FAIL("Merton76Process does not implement drift"); }

and likewise for the other two, so the process cannot be simulated — it
exists to carry parameters to
:class:`~pquantlib.pricingengines.vanilla.jump_diffusion_engine.JumpDiffusionEngine`,
which reads the inspectors and rebuilds its own Black-Scholes process per
Poisson term.  ``x0()`` and ``time()`` do delegate to the inner process.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import (
    StochasticProcess1D,
    StochasticProcess1DDiscretization,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.date import Date


class Merton76Process(StochasticProcess1D):
    """Merton-76 jump-diffusion process.

    # C++ parity: ``class Merton76Process``.
    """

    def __init__(
        self,
        *,
        state_variable: Quote,
        dividend_ts: YieldTermStructure,
        risk_free_ts: YieldTermStructure,
        black_vol_ts: BlackVolTermStructure,
        jump_int: Quote,
        log_j_mean: Quote,
        log_j_vol: Quote,
        discretization: StochasticProcess1DDiscretization | None = None,
    ) -> None:
        super().__init__(discretization)
        self._black_process: GeneralizedBlackScholesProcess = BlackScholesMertonProcess(
            x0=state_variable,
            dividend_ts=dividend_ts,
            risk_free_ts=risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
        )
        self._jump_intensity: Quote = jump_int
        self._log_mean_jump: Quote = log_j_mean
        self._log_jump_volatility: Quote = log_j_vol

        self._black_process.register_with(self)
        jump_int.register_with(self)
        log_j_mean.register_with(self)
        log_j_vol.register_with(self)

    # -- StochasticProcess1D interface -------------------------------------

    def x0(self) -> float:
        """Initial value, delegated to the inner Black-Scholes process.

        # C++ parity: ``Merton76Process::x0``.
        """
        return self._black_process.x0()

    def drift_1d(self, t: float, x: float) -> float:
        """Always raises.

        # C++ parity: ``Real drift(Time, Real) const override
        # { QL_FAIL("Merton76Process does not implement drift"); }``.
        """
        del t, x
        qassert.fail("Merton76Process does not implement drift")

    def diffusion_1d(self, t: float, x: float) -> float:
        """Always raises.

        # C++ parity: ``QL_FAIL("Merton76Process does not implement
        # diffusion")``.
        """
        del t, x
        qassert.fail("Merton76Process does not implement diffusion")

    def apply_1d(self, x0: float, dx: float) -> float:
        """Always raises.

        # C++ parity: ``QL_FAIL("Merton76Process does not implement apply")``.
        """
        del x0, dx
        qassert.fail("Merton76Process does not implement apply")

    def time(self, date: Date) -> float:
        """Year fraction to ``date``, delegated to the inner process.

        # C++ parity: ``Merton76Process::time``.
        """
        return self._black_process.time(date)

    # -- inspectors --------------------------------------------------------

    def state_variable(self) -> Quote:
        """# C++ parity: ``Merton76Process::stateVariable``."""
        return self._black_process.state_variable()

    def dividend_yield(self) -> YieldTermStructure:
        """# C++ parity: ``Merton76Process::dividendYield``."""
        return self._black_process.dividend_yield()

    def risk_free_rate(self) -> YieldTermStructure:
        """# C++ parity: ``Merton76Process::riskFreeRate``."""
        return self._black_process.risk_free_rate()

    def black_volatility(self) -> BlackVolTermStructure:
        """# C++ parity: ``Merton76Process::blackVolatility``."""
        return self._black_process.black_volatility()

    def jump_intensity(self) -> Quote:
        """# C++ parity: ``Merton76Process::jumpIntensity``."""
        return self._jump_intensity

    def log_mean_jump(self) -> Quote:
        """# C++ parity: ``Merton76Process::logMeanJump``."""
        return self._log_mean_jump

    def log_jump_volatility(self) -> Quote:
        """# C++ parity: ``Merton76Process::logJumpVolatility``."""
        return self._log_jump_volatility


__all__ = ["Merton76Process"]
