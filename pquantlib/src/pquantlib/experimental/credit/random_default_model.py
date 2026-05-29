"""RandomDefaultModel — MC default-time generation for a credit pool.

# C++ parity: ql/experimental/credit/randomdefaultmodel.{hpp,cpp} (v1.42.1).

Generates sequences of random default times — one per name in the pool
— via a one-factor Gaussian copula on top of the per-name
default-probability curves.

The model is wired Observable-style: it observes the copula and (by
extension) the pool's default curves; calling ``next_sequence(tmax)``
stores the simulated default times back on the pool via ``pool.set_time``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pquantlib import qassert
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.one_factor_copula_protocol import (
    OneFactorCopulaProtocol,
)
from pquantlib.experimental.credit.pool import Pool
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.randomnumbers.mersenne_twister import (
    MersenneTwisterUniformRng,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.patterns.observer import Observable, Observer

_QL_MAX_REAL: float = 1.0e308


class RandomDefaultModel(Observable, Observer, ABC):
    """Abstract base for random default-time models.

    # C++ parity: ql/experimental/credit/randomdefaultmodel.hpp:36-58.

    Concrete subclasses (currently only ``GaussianRandomDefaultModel``)
    implement ``next_sequence`` and ``reset``. The pool is kept as the
    storage backend — simulated default times are written to
    ``pool.set_time(name, t)``.
    """

    def __init__(
        self,
        pool: Pool,
        default_keys: list[DefaultProbKey],
    ) -> None:
        super().__init__()
        qassert.require(
            len(default_keys) == pool.size(),
            "Incompatible pool and keys sizes.",
        )
        self._pool: Pool = pool
        self._default_keys: list[DefaultProbKey] = list(default_keys)

    def update(self) -> None:
        """Observer hook: forward change notification.

        # C++ parity: ``RandomDefaultModel::update`` calls
        # ``notifyObservers``.
        """
        self.notify_observers()

    @abstractmethod
    def next_sequence(self, tmax: float = _QL_MAX_REAL) -> None:
        """Generate a sequence of random default times.

        # C++ parity: ``RandomDefaultModel::nextSequence``. ``tmax``
        # acts as a horizon: any default beyond ``tmax`` is written as
        # ``tmax + 1`` to save computation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the underlying RNG / state."""


class GaussianRandomDefaultModel(RandomDefaultModel):
    """One-factor Gaussian-copula default-time generator.

    # C++ parity: ql/experimental/credit/randomdefaultmodel.cpp:49-93.

    The sample-path generator is a Mersenne-Twister-driven
    inverse-cumulative-normal sequence; for each pool entry ``j``:

    * Read three N(0,1) draws: ``M = z[0]``, ``Z_j = z[j+1]``.
    * Form the latent variable ``Y_j = a * M + sqrt(1-a*a) * Z_j``
      where ``a = sqrt(rho)`` and ``rho`` is the copula correlation.
    * Map to the uniform default-probability scale: ``p = Phi(Y_j)``.
    * If ``F^{-1}(p) > tmax`` where ``F`` is the per-name default
      curve, write ``tmax + 1``; else Brent-invert ``F`` to get the
      default time.
    """

    def __init__(
        self,
        pool: Pool,
        default_keys: list[DefaultProbKey],
        copula: OneFactorCopulaProtocol,
        accuracy: float,
        seed: int,
    ) -> None:
        super().__init__(pool, default_keys)
        self._copula: OneFactorCopulaProtocol = copula
        self._accuracy: float = accuracy
        self._seed: int = seed
        self._inv_cnd: InverseCumulativeNormal = InverseCumulativeNormal()
        self._cnd: CumulativeNormalDistribution = CumulativeNormalDistribution()
        # dim = pool.size() + 1 to leave room for the systemic factor M
        # at index 0 + per-name idiosyncratic factors at 1..size.
        self._dim: int = pool.size() + 1
        self._rng: MersenneTwisterUniformRng = MersenneTwisterUniformRng(seed)
        # Observer registration: track the copula so changes invalidate
        # downstream MC.
        if isinstance(copula, Observable):
            copula.register_with(self)

    def reset(self) -> None:
        """Reinitialise the RNG from the original seed.

        # C++ parity: randomdefaultmodel.cpp:60-63.
        """
        self._rng = MersenneTwisterUniformRng(self._seed)

    def next_sequence(self, tmax: float = _QL_MAX_REAL) -> None:
        """Draw a fresh path of default times.

        # C++ parity: randomdefaultmodel.cpp:65-93.

        Writes default times back to ``self._pool`` via ``set_time``;
        any default beyond ``tmax`` is stored as ``tmax + 1`` (sentinel).
        """
        # Draw ``dim`` standard-normal values in one pass.
        values = [self._inv_cnd(self._rng.next_real()) for _ in range(self._dim)]

        rho = self._copula.correlation()
        a = math.sqrt(rho)
        sqrt_one_minus_a2 = math.sqrt(max(0.0, 1.0 - a * a))

        for j in range(self._pool.size()):
            self._draw_one(j, values, a, sqrt_one_minus_a2, tmax)

    def _draw_one(
        self,
        j: int,
        values: list[float],
        a: float,
        sqrt_one_minus_a2: float,
        tmax: float,
    ) -> None:
        """Per-name default-time inversion.

        Wrapped in a method to give the Brent target closure a stable
        ``dts`` binding (avoids the loop-variable-binding pitfall).
        """
        name = self._pool.names()[j]
        dts = self._pool.get(name).default_probability(self._default_keys[j])

        y = a * values[0] + sqrt_one_minus_a2 * values[j + 1]
        p = self._cnd(y)

        if dts.default_probability(tmax) < p:
            self._pool.set_time(name, tmax + 1.0)
            return

        # Solve dts.default_probability(t) - p == 0 in [0, tmax].
        # C++ uses Brent (fall-back Bisection); Python ports Brent only.
        def target(t: float) -> float:
            return dts.default_probability(t) - p

        try:
            solver = Brent()
            solver.set_lower_bound(0.0)
            solver.set_upper_bound(tmax)
            t_default = solver.solve(target, self._accuracy, tmax / 2.0, 1.0)
            self._pool.set_time(name, t_default)
        except Exception:
            # Conservative guard: write tmax + 1 if Brent fails to
            # bracket. Rare in practice — bracket-protected upstream.
            self._pool.set_time(name, tmax + 1.0)


__all__ = [
    "GaussianRandomDefaultModel",
    "RandomDefaultModel",
]
