"""JointStochasticProcess — several processes stacked into one hybrid process.

# C++ parity: ql/processes/jointstochasticprocess.{hpp,cpp} (v1.43) —
# ``class JointStochasticProcess : public StochasticProcess``
# (jointstochasticprocess.hpp:34-97, jointstochasticprocess.cpp:31-309).

Stacks ``N`` arbitrary ``StochasticProcess`` instances into a single
process whose state is their states concatenated. The block-diagonal part
of the covariance comes from the constituents; the off-diagonal (cross-model)
part comes from the abstract :meth:`cross_model_correlation` hook, scaled by
the constituents' own volatilities.

The class is ABSTRACT — five hooks must be supplied by a subclass:

* ``pre_evolve(t0, x0, dt, dv)``  — observation point before the step
* ``post_evolve(t0, x0, dt, dv, y0)`` — final transform of the stepped state
* ``numeraire(t, x)``
* ``correlation_is_state_dependent()`` — controls the correlation cache
* ``cross_model_correlation(t0, x0)``

QuantLib v1.43 ships **no** concrete subclass of this class anywhere in
``ql/`` or its test-suite, so the cross-validation defines one on both sides.

``evolve`` is the substantial method. It:

1. builds the covariance, normalises it to a correlation matrix in place;
2. for each constituent, takes its ``stdDeviation``, row-normalises it (or,
   for a zero-volatility row, fills the row with ``100*i*QL_EPSILON`` "to
   keep the svd happy" — the C++ comment), and inverts it via SVD with a
   ``sqrt(QL_EPSILON)`` singular-value cutoff, writing the result into the
   block-diagonal of a ``(size, modelFactors)`` matrix ``diff``;
3. computes ``rankReducedSqrt(correlation, factors, 1.0, Spectral)``,
   zero-padding it out to ``factors`` columns if fewer were retained;
4. sets ``dv = transpose(diff) * rs * dw`` and caches ``transpose(diff) * rs``
   keyed on ``(t0, dt)`` when the correlation is not state dependent;
5. feeds each constituent its own ``dv`` slice and its own ``x0`` slice, then
   runs the result through ``post_evolve``.

Divergences from C++:

* # C++ parity divergence: C++ holds ``ext::shared_ptr<StochasticProcess>``;
  Python holds the objects directly.
* ``pseudoSqrt(m)`` (used by :meth:`diffusion` and :meth:`std_deviation`)
  has no free-function equivalent in this port yet, so its default
  ``SalvagingAlgorithm::None`` arm is inlined here as ``_pseudo_sqrt``:
  the ``min eigenvalue >= -1e-16`` requirement followed by
  ``CholeskyDecomposition(m, flexible=True)``, exactly as
  pseudosqrt.cpp:154-170 does.
* C++ keys its correlation cache with a ``std::map`` ordered by
  ``CachingKey::operator<``; Python uses a ``dict`` keyed by the same pair.
  :class:`CachingKey` therefore carries both ``__hash__``/``__eq__`` (for the
  dict) and ``__lt__`` (mirroring the C++ comparator).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.matrixutilities.cholesky import cholesky_decomposition
from pquantlib.math.matrixutilities.svd import SVD
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.date import Date


@dataclass(frozen=True, slots=True, order=False)
class CachingKey:
    """Key of ``JointStochasticProcess``'s cross-model correlation cache.

    # C++ parity: nested ``struct JointStochasticProcess::CachingKey``
    # (jointstochasticprocess.hpp:84-94) — a ``(Time t0, Time dt)`` pair used
    # as the key of a ``std::map<CachingKey, Matrix>``.

    C++ makes it usable as a map key by supplying ``operator<``; Python needs
    hashability instead, which the frozen dataclass provides. ``__lt__`` is
    kept as well so the C++ ordering is reproducible (and testable) rather
    than merely implied.
    """

    t0: float
    dt: float

    def __lt__(self, other: object) -> bool:
        """Strict weak ordering: by ``t0`` first, then by ``dt``.

        # C++ parity: jointstochasticprocess.hpp:88-91 —
        # ``t0_ < key.t0_ || (t0_ == key.t0_ && dt_ < key.dt_)``.
        """
        if not isinstance(other, CachingKey):
            return NotImplemented
        return self.t0 < other.t0 or (self.t0 == other.t0 and self.dt < other.dt)


def _pseudo_sqrt(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``pseudoSqrt(m, SalvagingAlgorithm::None)``.

    # C++ parity: ql/math/matrixutilities/pseudosqrt.cpp:154-170 — the
    # ``None`` arm requires the smallest eigenvalue to be ``>= -1e-16`` and
    # then returns ``CholeskyDecomposition(matrix, true)``.
    """
    size = int(m.shape[0])
    qassert.require(
        size == int(m.shape[1]),
        f"non square matrix: {size} rows, {int(m.shape[1])} columns",
    )
    smallest = float(np.min(np.linalg.eigvalsh(m)))
    qassert.require(smallest >= -1e-16, f"negative eigenvalue(s) ({smallest:e})")
    return cholesky_decomposition(m, True)


class JointStochasticProcess(StochasticProcess, ABC):
    """Several stochastic processes stacked into one hybrid process.

    # C++ parity: ``class JointStochasticProcess : public StochasticProcess``.

    Args:
        l: the constituent processes, in order.
        factors: number of Brownian factors the joint process consumes.
            ``None`` reproduces the C++ ``Null<Size>()`` default, which means
            "use the sum of the constituents' factors".
    """

    def __init__(
        self,
        l: Sequence[StochasticProcess],  # noqa: E741 — C++ parameter name
        factors: int | None = None,
    ) -> None:
        # C++ parity: jointstochasticprocess.cpp:31-58.
        super().__init__()
        self._l: list[StochasticProcess] = list(l)
        for proc in self._l:
            proc.register_with(self)

        self._size_: int = 0
        self._model_factors: int = 0
        self._vsize: list[int] = []
        self._vfactors: list[int] = []
        for proc in self._l:
            self._vsize.append(self._size_)
            self._size_ += proc.size()
            self._vfactors.append(self._model_factors)
            self._model_factors += proc.factors()
        self._vsize.append(self._size_)
        self._vfactors.append(self._model_factors)

        if factors is None:
            self._factors: int = self._model_factors
        else:
            qassert.require(factors <= self._size_, "too many factors given")
            self._factors = int(factors)

        self._correlation_cache: dict[CachingKey, npt.NDArray[np.float64]] = {}

    # --- abstract hooks ----------------------------------------------------

    @abstractmethod
    def pre_evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> None:
        """Hook invoked with the CORRELATED increment before the step.

        # C++ parity: ``JointStochasticProcess::preEvolve``
        # (jointstochasticprocess.hpp:53-54). Note the last argument is the
        # internal ``dv``, not the caller's ``dw`` — C++ names the parameter
        # ``dw`` but passes ``dv``.
        """

    @abstractmethod
    def post_evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
        y0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Final transform applied to the stepped state.

        # C++ parity: ``JointStochasticProcess::postEvolve``
        # (jointstochasticprocess.hpp:55-57).
        """

    @abstractmethod
    def numeraire(self, t: float, x: npt.NDArray[np.float64]) -> float:
        """Numeraire of the joint measure at ``(t, x)``.

        # C++ parity: ``JointStochasticProcess::numeraire``
        # (jointstochasticprocess.hpp:59).
        """

    @abstractmethod
    def correlation_is_state_dependent(self) -> bool:
        """Whether :meth:`cross_model_correlation` depends on the state.

        # C++ parity: ``JointStochasticProcess::correlationIsStateDependent``
        # (jointstochasticprocess.hpp:60). When False, ``evolve`` caches the
        # correlation transform keyed on ``(t0, dt)``.
        """

    @abstractmethod
    def cross_model_correlation(
        self, t0: float, x0: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """``size x size`` correlation between the constituents.

        # C++ parity: ``JointStochasticProcess::crossModelCorrelation``
        # (jointstochasticprocess.hpp:61).
        """

    # --- inspectors --------------------------------------------------------

    def constituents(self) -> list[StochasticProcess]:
        """The constituent processes, in order.

        # C++ parity: ``JointStochasticProcess::constituents``
        # (jointstochasticprocess.cpp:293-296).
        """
        return self._l

    # --- StochasticProcess interface --------------------------------------

    def size(self) -> int:
        # C++ parity: jointstochasticprocess.cpp:60-62.
        return self._size_

    def factors(self) -> int:
        # C++ parity: jointstochasticprocess.cpp:64-66.
        return self._factors

    def _slice(self, x: npt.NDArray[np.float64], i: int) -> npt.NDArray[np.float64]:
        """Cut out the ``i``-th constituent's variables.

        # C++ parity: ``JointStochasticProcess::slice``
        # (jointstochasticprocess.cpp:68-75). Protected in C++.
        """
        return np.array(x[self._vsize[i] : self._vsize[i + 1]], dtype=np.float64)

    def initial_values(self) -> npt.NDArray[np.float64]:
        # C++ parity: jointstochasticprocess.cpp:77-88.
        ret = np.empty(self.size(), dtype=np.float64)
        for i, proc in enumerate(self._l):
            ret[self._vsize[i] : self._vsize[i + 1]] = proc.initial_values()
        return ret

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: jointstochasticprocess.cpp:91-104.
        ret = np.empty(self.size(), dtype=np.float64)
        for i, proc in enumerate(self._l):
            ret[self._vsize[i] : self._vsize[i + 1]] = proc.drift(t, self._slice(x, i))
        return ret

    def expectation(
        self, t0: float, x0: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        # C++ parity: jointstochasticprocess.cpp:106-120.
        ret = np.empty(self.size(), dtype=np.float64)
        for i, proc in enumerate(self._l):
            ret[self._vsize[i] : self._vsize[i + 1]] = proc.expectation(
                t0, self._slice(x0, i), dt
            )
        return ret

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Instantaneous diffusion, approximated over a fixed ``dt = 0.001``.

        # C++ parity: jointstochasticprocess.cpp:123-127 — including the
        # verbatim comment "might need some improvement in the future" and the
        # hard-coded step.
        """
        dt = 0.001
        return _pseudo_sqrt(self.covariance(t, x, dt) / dt)

    def covariance(
        self, t0: float, x0: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        """Block-diagonal covariance plus the scaled cross-model correlation.

        # C++ parity: jointstochasticprocess.cpp:130-161.
        """
        n = self.size()
        ret = np.zeros((n, n), dtype=np.float64)
        for j, proc in enumerate(self._l):
            vs = self._vsize[j]
            p_cov = proc.covariance(t0, self._slice(x0, j), dt)
            rows = int(p_cov.shape[0])
            ret[vs : vs + rows, vs : vs + int(p_cov.shape[1])] = p_cov

        # add the cross model covariance matrix
        volatility = np.sqrt(np.diag(ret))
        cross = np.array(self.cross_model_correlation(t0, x0), dtype=np.float64, copy=True)
        cross *= np.outer(volatility, volatility)
        ret += cross
        return ret

    def std_deviation(
        self, t0: float, x0: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        # C++ parity: jointstochasticprocess.cpp:164-168.
        return _pseudo_sqrt(self.covariance(t0, x0, dt))

    def apply(
        self, x0: npt.NDArray[np.float64], dx: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # C++ parity: jointstochasticprocess.cpp:171-183.
        ret = np.empty(self.size(), dtype=np.float64)
        for i, proc in enumerate(self._l):
            ret[self._vsize[i] : self._vsize[i + 1]] = proc.apply(
                self._slice(x0, i), self._slice(dx, i)
            )
        return ret

    def evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Advance the joint state by ``dt`` given independent normals ``dw``.

        # C++ parity: jointstochasticprocess.cpp:185-291.
        """
        state_dependent = self.correlation_is_state_dependent()
        cache_key = CachingKey(t0, dt)
        dv: npt.NDArray[np.float64]

        if state_dependent or cache_key not in self._correlation_cache:
            m = self._correlation_transform(t0, x0, dt)
            if not state_dependent:
                self._correlation_cache[cache_key] = m
            dv = m @ dw
        else:
            # C++ re-tests !correlationIsStateDependent() inside the else
            # branch even though it is necessarily true there; the guard is
            # kept so ``dv`` provenance reads the same on both sides.
            dv = self._correlation_cache[cache_key] @ dw

        self.pre_evolve(t0, x0, dt, dv)

        ret = np.empty(self.size(), dtype=np.float64)
        for i, proc in enumerate(self._l):
            vf = self._vfactors[i]
            dz = np.array(dv[vf : vf + proc.factors()], dtype=np.float64)
            x = np.array(x0[self._vsize[i] : self._vsize[i + 1]], dtype=np.float64)
            ret[self._vsize[i] : self._vsize[i + 1]] = proc.evolve(t0, x, dt, dz)

        return self.post_evolve(t0, x0, dt, dv, ret)

    def _correlation_transform(
        self, t0: float, x0: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        """``transpose(diff) * rs`` — the ``(modelFactors, factors)`` transform.

        # C++ parity: the cached body of ``JointStochasticProcess::evolve``
        # (jointstochasticprocess.cpp:191-258), factored out so the cache-hit
        # and cache-miss arms of ``evolve`` stay legible.
        """
        n = self.size()
        cov = self.covariance(t0, x0, dt)

        # In-place normalisation to a correlation matrix, driven off the upper
        # triangle exactly as C++ does (cov[i][j] = cov[j][i] = ...).
        sqrt_diag = np.sqrt(np.diag(cov))
        for i in range(n):
            for j in range(i, n):
                div = sqrt_diag[i] * sqrt_diag[j]
                value = float(cov[i, j] / div) if div > 0 else 0.0
                cov[i, j] = value
                cov[j, i] = value

        diff = np.zeros((n, self._model_factors), dtype=np.float64)
        for j, proc in enumerate(self._l):
            vs = self._vsize[j]
            vf = self._vfactors[j]

            std_dev = np.array(
                proc.std_deviation(t0, self._slice(x0, j), dt), dtype=np.float64, copy=True
            )
            for i in range(int(std_dev.shape[0])):
                vol = math.sqrt(float(np.dot(std_dev[i, :], std_dev[i, :])))
                if vol > 0.0:
                    std_dev[i, :] /= vol
                else:
                    # C++ comment: "keep the svd happy"
                    std_dev[i, :] = 100.0 * i * QL_EPSILON

            svd = SVD(std_dev)
            s = svd.singular_values()
            w = np.zeros((int(s.size), int(s.size)), dtype=np.float64)
            cutoff = math.sqrt(QL_EPSILON)
            for i in range(int(s.size)):
                if abs(float(s[i])) > cutoff:
                    w[i, i] = 1.0 / float(s[i])

            inv = svd.u() @ w @ svd.v().T
            rows, cols = int(inv.shape[0]), int(inv.shape[1])
            diff[vs : vs + rows, vf : vf + cols] = inv

        rs = rank_reduced_sqrt(cov, self._factors, 1.0, SalvagingAlgorithm.SPECTRAL)
        if int(rs.shape[1]) < self._factors:
            # fewer eigenvalues than expected factors: pad with zeros.
            tmp = np.zeros((n, self._factors), dtype=np.float64)
            tmp[:, : int(rs.shape[1])] = rs
            rs = tmp

        return diff.T @ rs

    # --- utilities ---------------------------------------------------------

    def time(self, date: Date) -> float:
        """Year fraction — delegated to the FIRST constituent.

        # C++ parity: jointstochasticprocess.cpp:298-302. If constituent 0
        # has no date/time conversion of its own the delegation reaches the
        # base ``StochasticProcess::time``, which raises — that is C++
        # behaviour, not a gap in this port.
        """
        qassert.require(len(self._l) > 0, "process list is empty")
        return self._l[0].time(date)

    def update(self) -> None:
        """Clear the correlation cache, then propagate.

        # C++ parity: jointstochasticprocess.cpp:304-309.
        """
        self._correlation_cache.clear()
        super().update()


__all__ = ["CachingKey", "JointStochasticProcess"]
