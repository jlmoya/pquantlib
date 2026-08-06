"""Merton76Process — Merton (1976) jump-diffusion process.

# C++ parity: ql/processes/merton76process.{hpp,cpp} (v1.43) —
# ``class Merton76Process : public StochasticProcess1D``
# (merton76process.hpp:36-68, merton76process.cpp:27-80).

The class is a *carrier*, not a simulator: it bundles a
``BlackScholesMertonProcess`` with the three jump parameters
(intensity, log-mean jump, log-jump volatility) that
``JumpDiffusionEngine`` reads off it. Every dynamic method is a
``QL_FAIL``:

    Real drift(Time, Real)     -> QL_FAIL("Merton76Process does not implement drift")
    Real diffusion(Time, Real) -> QL_FAIL("Merton76Process does not implement diffusion")
    Real apply(Real, Real)     -> QL_FAIL("Merton76Process does not implement apply")

That is not an omission in this port: it is what merton76process.hpp:50-52
says. Only ``x0()``, ``time(date)`` and the seven inspectors are live, and
the inherited ``expectation``/``stdDeviation``/``evolve`` all fail because
they route through ``drift`` / ``apply``.

Divergences from C++:

* # C++ parity divergence: C++ takes ``Handle<Quote>`` /
  ``Handle<YieldTermStructure>`` / ``Handle<BlackVolTermStructure>``; this
  port does not implement ``Handle<T>`` and threads the pointed-to objects
  directly.
* The C++ ctor forwards the SAME ``discretization`` instance both to its own
  ``StochasticProcess1D`` base and to the internal
  ``BlackScholesMertonProcess``; this port does the same. Passing ``None``
  makes each of the two build its own ``EulerDiscretization``, matching the
  C++ default argument which constructs one fresh ``EulerDiscretization``
  that is then shared.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.euler_discretization import EulerDiscretization
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

    # C++ parity: ``class Merton76Process : public StochasticProcess1D``
    # (merton76process.hpp:36-68).
    """

    __slots__ = (
        "_black_process",
        "_jump_intensity",
        "_log_jump_volatility",
        "_log_mean_jump",
    )

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
        # C++ parity: merton76process.cpp:27-43. The same discretization
        # instance goes to the base and to the internal BSM process.
        if discretization is None:
            discretization = EulerDiscretization()
        super().__init__(discretization)
        self._black_process: BlackScholesMertonProcess = BlackScholesMertonProcess(
            x0=state_variable,
            dividend_ts=dividend_ts,
            risk_free_ts=risk_free_ts,
            black_vol_ts=black_vol_ts,
            discretization=discretization,
        )
        self._jump_intensity: Quote = jump_int
        self._log_mean_jump: Quote = log_j_mean
        self._log_jump_volatility: Quote = log_j_vol
        # C++ parity: merton76process.cpp:39-42 — registerWith on all four.
        self._black_process.register_with(self)
        jump_int.register_with(self)
        log_j_mean.register_with(self)
        log_j_vol.register_with(self)

    # --- StochasticProcess1D interface ------------------------------------

    def x0(self) -> float:
        # C++ parity: merton76process.cpp:45-47.
        return self._black_process.x0()

    def drift_1d(self, t: float, x: float) -> float:
        """Always fails.

        # C++ parity: merton76process.hpp:50 —
        # ``QL_FAIL("Merton76Process does not implement drift")``.
        """
        del t, x
        qassert.fail("Merton76Process does not implement drift")

    def diffusion_1d(self, t: float, x: float) -> float:
        """Always fails.

        # C++ parity: merton76process.hpp:51 —
        # ``QL_FAIL("Merton76Process does not implement diffusion")``.
        """
        del t, x
        qassert.fail("Merton76Process does not implement diffusion")

    def apply_1d(self, x0: float, dx: float) -> float:
        """Always fails.

        # C++ parity: merton76process.hpp:52 —
        # ``QL_FAIL("Merton76Process does not implement apply")``.
        """
        del x0, dx
        qassert.fail("Merton76Process does not implement apply")

    def time(self, date: Date) -> float:
        # C++ parity: merton76process.cpp:49-51 — delegates to the inner
        # Black-Scholes-Merton process (i.e. to the risk-free curve's
        # day counter).
        return self._black_process.time(date)

    # --- inspectors --------------------------------------------------------

    def state_variable(self) -> Quote:
        # C++ parity: merton76process.cpp:53-55.
        return self._black_process.state_variable()

    def dividend_yield(self) -> YieldTermStructure:
        # C++ parity: merton76process.cpp:57-59.
        return self._black_process.dividend_yield()

    def risk_free_rate(self) -> YieldTermStructure:
        # C++ parity: merton76process.cpp:61-63.
        return self._black_process.risk_free_rate()

    def black_volatility(self) -> BlackVolTermStructure:
        # C++ parity: merton76process.cpp:65-68.
        return self._black_process.black_volatility()

    def jump_intensity(self) -> Quote:
        # C++ parity: merton76process.cpp:70-72.
        return self._jump_intensity

    def log_mean_jump(self) -> Quote:
        # C++ parity: merton76process.cpp:74-76.
        return self._log_mean_jump

    def log_jump_volatility(self) -> Quote:
        # C++ parity: merton76process.cpp:78-80.
        return self._log_jump_volatility


__all__ = ["Merton76Process"]
