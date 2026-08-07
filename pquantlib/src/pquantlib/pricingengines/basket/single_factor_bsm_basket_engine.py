"""SingleFactorBsmBasketEngine — Choi's one-factor basket engine.

# C++ parity:
# ql/pricingengines/basket/singlefactorbsmbasketengine.{hpp,cpp} (v1.43),
# ``QuantLib::SingleFactorBsmBasketEngine`` and
# ``QuantLib::detail::SumExponentialsRootSolver``.

Jaehyuk Choi, *Sum of all Black-Scholes-Merton Models: An efficient Pricing
Method for Spread, Basket and Asian Options*, https://arxiv.org/pdf/1805.03172

Every underlying is driven by the *same* Brownian motion, so the basket value
at maturity is a one-dimensional sum of exponentials

    sum_i  a_i exp(sig_i x),      a_i = w_i F_i exp(-v_i / 2)

and the exercise boundary is the root ``x*`` of that sum minus the strike.
Once ``d = -x*`` is known the price is a plain sum of Black-Scholes terms.

Two classes live here, mirroring the single C++ header:

* :class:`SumExponentialsRootSolver` — the root finder, with its own guards,
  its own linear-approximation start point, four solver strategies and three
  evaluation counters. It is public C++ surface (``detail::`` namespace, but a
  named top-level class with a documented API that the upstream test-suite
  drives directly), not a private helper.
* :class:`SingleFactorBsmBasketEngine` — the engine.

Both are pinned against C++ v1.43 by
``migration-harness/cpp/probes/v143_pe_basket/probe.cpp``. The evaluation
counters are pinned per strategy because they are the only observable that
distinguishes "runs QuantLib's own Brent/Newton/Ridder/Halley" from "delegates
to scipy and happens to land on the same root".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import (
    AverageBasketPayoff,
    BasketOptionResults,
)
from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.math.solvers1d.halley import Halley
from pquantlib.math.solvers1d.newton import Newton
from pquantlib.math.solvers1d.ridder import Ridder
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.vector_bsm_process_extractor import (
    VectorBsmProcessExtractor,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)

_QL_EPSILON: float = float(np.finfo(np.float64).eps)


class SumExponentialsRootSolver:
    """Root of ``sum_i a_i exp(sig_i x) - K``.

    # C++ parity: ``detail::SumExponentialsRootSolver``
    # (singlefactorbsmbasketengine.hpp:57-77,
    #  singlefactorbsmbasketengine.cpp:36-121).

    Args:
        a: the coefficients.
        sig: the exponents. Must have the same length as ``a``.
        k: the level whose crossing is sought.
    """

    class Strategy(IntEnum):
        """Solver choice.

        # C++ parity: ``enum Strategy {Ridder, Newton, Brent, Halley}``
        # (singlefactorbsmbasketengine.hpp:59) — four values, in this
        # declaration order.
        """

        Ridder = 0
        Newton = 1
        Brent = 2
        Halley = 3

    __slots__ = (
        "_a",
        "_f_ctr",
        "_f_double_prime_ctr",
        "_f_prime_ctr",
        "_k",
        "_sig",
    )

    def __init__(
        self, a: Sequence[float] | Array, sig: Sequence[float] | Array, k: float
    ) -> None:
        # C++ parity: singlefactorbsmbasketengine.cpp:36-41.
        self._a: Array = np.asarray(a, dtype=np.float64)
        self._sig: Array = np.asarray(sig, dtype=np.float64)
        qassert.require(
            self._a.shape == self._sig.shape, "Arrays must have the same size"
        )
        self._k: float = k
        self._f_ctr: int = 0
        self._f_prime_ctr: int = 0
        self._f_double_prime_ctr: int = 0

    # --- the function and its derivatives ----------------------------------

    def __call__(self, x: float) -> float:
        """``sum_i a_i exp(sig_i x) - K``.

        # C++ parity: ``operator()`` (singlefactorbsmbasketengine.cpp:43-50).
        """
        self._f_ctr += 1
        s = 0.0
        for i in range(self._a.shape[0]):
            s += float(self._a[i]) * math.exp(float(self._sig[i]) * x)
        return s - self._k

    def derivative(self, x: float) -> float:
        """``sum_i a_i sig_i exp(sig_i x)``.

        # C++ parity: ``derivative`` (singlefactorbsmbasketengine.cpp:52-59).
        """
        self._f_prime_ctr += 1
        s = 0.0
        for i in range(self._a.shape[0]):
            sig_i = float(self._sig[i])
            s += float(self._a[i]) * sig_i * math.exp(sig_i * x)
        return s

    def second_derivative(self, x: float) -> float:
        """``sum_i a_i sig_i**2 exp(sig_i x)``.

        # C++ parity: ``secondDerivative``
        # (singlefactorbsmbasketengine.cpp:61-68).
        """
        self._f_double_prime_ctr += 1
        s = 0.0
        for i in range(self._a.shape[0]):
            sig_i = float(self._sig[i])
            s += float(self._a[i]) * sig_i * sig_i * math.exp(sig_i * x)
        return s

    # --- evaluation counters ------------------------------------------------

    def get_f_ctr(self) -> int:
        """# C++ parity: ``getFCtr`` (singlefactorbsmbasketengine.cpp:70-72)."""
        return self._f_ctr

    def get_derivative_ctr(self) -> int:
        """# C++ parity: ``getDerivativeCtr`` (…cpp:74-76)."""
        return self._f_prime_ctr

    def get_second_derivative_ctr(self) -> int:
        """# C++ parity: ``getSecondDerivativeCtr`` (…cpp:78-80)."""
        return self._f_double_prime_ctr

    # --- the root -----------------------------------------------------------

    def get_root(
        self,
        x_tol: float = 1e6 * _QL_EPSILON,
        strategy: Strategy = Strategy.Brent,
    ) -> float:
        """Solve ``self(x) == 0``.

        # C++ parity: ``getRoot`` (singlefactorbsmbasketengine.cpp:82-121).

        The start point is C++'s linear approximation

            xInit = clamp((K - sum a) / sum(a * sig), -10, 10)

        falling back to ``0.0`` when ``|sum(a * sig)| <= 1000 * QL_EPSILON``;
        every solver is then driven unbracketed with step ``1.0``.

        Raises:
            LibraryException: if any ``a_i * sig_i`` is negative, or if ``K``
                is non-positive while every ``a_i`` is positive (a genuine
                basket, for which no root exists).
        """
        attr = self._a * self._sig
        qassert.require(
            bool(np.all(attr >= 0.0)), "a*sig should not be negative"
        )

        log_prob = bool(np.all(self._a > 0.0))
        qassert.require(
            self._k > 0.0 or not log_prob,
            "non-positive strikes only allowed for spread options",
        )

        denom = float(np.sum(attr))
        if abs(denom) > 1000 * _QL_EPSILON:
            x_init = min(10.0, max(-10.0, (self._k - float(np.sum(self._a))) / denom))
        else:
            x_init = 0.0

        if strategy == SumExponentialsRootSolver.Strategy.Brent:
            return Brent().solve(self, x_tol, x_init, 1.0)
        if strategy == SumExponentialsRootSolver.Strategy.Newton:
            return Newton().solve(self, x_tol, x_init, 1.0)
        if strategy == SumExponentialsRootSolver.Strategy.Ridder:
            return Ridder().solve(self, x_tol, x_init, 1.0)
        if strategy == SumExponentialsRootSolver.Strategy.Halley:
            return Halley().solve(self, x_tol, x_init, 1.0)
        raise AssertionError("unknown strategy type")


class SingleFactorBsmBasketEngine(
    GenericEngine[OptionArguments, BasketOptionResults]
):
    """Basket engine for underlyings driven by one stochastic factor.

    # C++ parity: ``SingleFactorBsmBasketEngine``
    # (singlefactorbsmbasketengine.hpp:42-54,
    #  singlefactorbsmbasketengine.cpp:124-188).

    Args:
        processes: one ``GeneralizedBlackScholesProcess`` per basket leg. They
            must all agree on the risk-free discount factor at maturity.
        x_tol: root-finder accuracy, C++ default ``1e4 * QL_EPSILON``.
    """

    def __init__(
        self,
        processes: Sequence[GeneralizedBlackScholesProcess],
        x_tol: float = 1e4 * _QL_EPSILON,
    ) -> None:
        super().__init__(OptionArguments(), BasketOptionResults())
        self._x_tol: float = x_tol
        self._processes: tuple[GeneralizedBlackScholesProcess, ...] = tuple(processes)
        self._n: int = len(self._processes)
        for p in self._processes:
            p.register_with(self)

    def calculate(self) -> None:
        """Price the basket.

        # C++ parity: ``SingleFactorBsmBasketEngine::calculate``
        # (singlefactorbsmbasketengine.cpp:134-188).
        """
        args = self._arguments
        results = self._results
        results.reset()

        avg_payoff = args.payoff
        qassert.require(
            isinstance(avg_payoff, AverageBasketPayoff),
            "average basket payoff expected",
        )
        assert isinstance(avg_payoff, AverageBasketPayoff)
        payoff = avg_payoff.base_payoff()
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain vanilla payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        strike = payoff.strike()

        weights: Array = avg_payoff.weights()
        qassert.require(
            self._n == int(weights.shape[0]),
            "wrong number of weights arguments in payoff",
        )

        exercise = args.exercise
        qassert.require(exercise is not None, "not an European exercise")
        assert exercise is not None
        qassert.require(
            exercise.type() == Exercise.Type.European, "not an European exercise"
        )
        assert isinstance(exercise, EuropeanExercise)
        maturity_date = exercise.last_date()

        extractor = VectorBsmProcessExtractor(self._processes)
        s = extractor.get_spot()
        dq = extractor.get_dividend_yield_df(maturity_date)
        dr0 = extractor.get_interest_rate_df(maturity_date)

        std_dev = extractor.get_black_std_dev(maturity_date)
        v = std_dev * std_dev

        fwd_basket = weights * s * dq / dr0

        # All vols zero -> the basket is deterministic; return the discounted
        # intrinsic and write NO additionalResults["d"].
        if all(close_enough(float(x), 0.0) for x in std_dev):
            results.value = dr0 * payoff(float(np.sum(fwd_basket)))
            return

        d = -SumExponentialsRootSolver(
            fwd_basket * np.exp(-0.5 * v), std_dev, strike
        ).get_root(self._x_tol, SumExponentialsRootSolver.Strategy.Brent)

        n_dist = CumulativeNormalDistribution()
        cp = 1.0 if payoff.option_type() == OptionType.Call else -1.0

        # C++ std::inner_product with init = -strike*N(cp*d) and the binary op
        # (x, y) -> x*N(cp*(d + y)); the accumulation order is index order.
        acc = -strike * n_dist(cp * d)
        for i in range(self._n):
            acc += float(fwd_basket[i]) * n_dist(cp * (d + float(std_dev[i])))

        results.value = cp * dr0 * acc
        results.additional_results["d"] = d

    def update(self) -> None:
        self.notify_observers()


__all__ = ["SingleFactorBsmBasketEngine", "SumExponentialsRootSolver"]
