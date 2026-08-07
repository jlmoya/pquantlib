"""StochasticProcessArray — N correlated 1-D processes as one N-D process.

# C++ parity: ql/processes/stochasticprocessarray.{hpp,cpp} (v1.42.1).

Wraps ``N`` ``StochasticProcess1D`` instances + an ``NxN`` correlation
matrix into a single ``StochasticProcess`` (multi-D).  Each step
takes ``N`` independent Brownian increments ``dw``, premultiplies by
the spectral-square-root of the correlation matrix to inject the
correlation, and feeds component ``dz[i]`` into ``processes[i].evolve``.

The C++ ``stdDeviation`` and ``diffusion`` return ``Matrix`` whose row ``i``
is ``sqrt_corr.row(i) * processes_[i]->...``.  We mirror that by scaling the
``sqrt_corr`` rows in-place.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.matrixutilities.symmetric_schur_decomposition import (
    SymmetricSchurDecomposition,
)
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.date import Date


def _normalize_pseudo_root(
    matrix: npt.NDArray[np.float64], pseudo: npt.NDArray[np.float64]
) -> None:
    """Rescale each row of ``pseudo`` so its norm matches ``matrix``' diagonal.

    # C++ parity: ``normalizePseudoRoot`` (pseudosqrt.cpp, anonymous namespace).
    """
    size = int(matrix.shape[0])
    pseudo_cols = int(pseudo.shape[1])
    for i in range(size):
        norm = 0.0
        for j in range(pseudo_cols):
            norm += float(pseudo[i, j]) * float(pseudo[i, j])
        if norm > 0.0:
            norm_adj = math.sqrt(float(matrix[i, i]) / norm)
            for j in range(pseudo_cols):
                pseudo[i, j] *= norm_adj


def _spectral_sqrt(corr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Spectral square root of a (possibly degenerate) symmetric matrix.

    # C++ parity: ``pseudoSqrt(corr, SalvagingAlgorithm::Spectral)``
    # (ql/math/matrixutilities/pseudosqrt.cpp:377-385):
    #     for i: diagonal[i][i] = sqrt(max(jd.eigenvalues()[i], 0))
    #     result = jd.eigenvectors() * diagonal
    #     normalizePseudoRoot(matrix, result)

    The *value* of this matrix is observable, not merely its Gram product:
    ``StochasticProcessArray.evolve`` uses ``dz = M @ dw`` for a specific
    ``dw``, so any other square root of the same correlation gives a
    statistically equivalent but path-wise different simulation. It must
    therefore be built exactly as C++ builds it —
    :class:`SymmetricSchurDecomposition` (eigenvalues in *decreasing* order,
    with C++'s eigenvector sign convention), not ``numpy.linalg.eigh``, whose
    eigenvalues come out ascending and whose eigenvector signs differ. Pinned
    against C++ by ``migration-harness/references/v143/pe/basket.json``
    (``mc_spectral_pseudo_sqrt_*``).
    """
    qassert.require(
        corr.ndim == 2 and corr.shape[0] == corr.shape[1],
        "correlation matrix must be square",
    )
    qassert.require(
        np.allclose(corr, corr.T, atol=1e-14),
        "correlation matrix must be symmetric",
    )
    size = int(corr.shape[0])
    jd = SymmetricSchurDecomposition(corr)
    diagonal: npt.NDArray[np.float64] = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        diagonal[i, i] = math.sqrt(max(float(jd.eigenvalues()[i]), 0.0))
    result: npt.NDArray[np.float64] = jd.eigenvectors() @ diagonal
    _normalize_pseudo_root(corr, result)
    return result


class StochasticProcessArray(StochasticProcess):
    """N correlated 1-D processes packed into one N-D process.

    # C++ parity: ``class StochasticProcessArray : public StochasticProcess``.

    Args:
        processes: list of ``N`` ``StochasticProcess1D`` instances.
        correlation: ``N x N`` correlation matrix (numpy or list of lists).
    """

    def __init__(
        self,
        processes: Sequence[StochasticProcess1D],
        correlation: npt.NDArray[np.float64] | Sequence[Sequence[float]],
    ) -> None:
        super().__init__()
        qassert.require(len(processes) > 0, "no processes given")
        corr = np.asarray(correlation, dtype=np.float64)
        qassert.require(
            corr.shape[0] == len(processes),
            "mismatch between number of processes and size of correlation matrix",
        )
        for p in processes:
            # C++ does ``registerWith(process)`` on each — Python piggybacks
            # on the Observer Protocol; ``p.register_with(self)`` adds the
            # array as one of ``p``'s observers, so the array notifies its
            # downstream observers when any process notifies its observers.
            p.register_with(self)
        self._processes: list[StochasticProcess1D] = list(processes)
        self._sqrt_correlation: npt.NDArray[np.float64] = _spectral_sqrt(corr)

    # --- StochasticProcess interface -------------------------------------

    def size(self) -> int:
        """N — number of constituent 1-D processes.

        # C++ parity: ``StochasticProcessArray::size``.
        """
        return len(self._processes)

    def factors(self) -> int:
        """N — same as ``size`` (one factor per asset before correlation).

        # C++ parity: defaults from the base class (``size``).
        """
        return self.size()

    def initial_values(self) -> npt.NDArray[np.float64]:
        """``[x0_0, x0_1, ..., x0_{N-1}]``.

        # C++ parity: ``StochasticProcessArray::initialValues``.
        """
        return np.array(
            [p.x0() for p in self._processes],
            dtype=np.float64,
        )

    def drift(
        self,
        t: float,
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Component-wise drift.

        # C++ parity: ``StochasticProcessArray::drift``.
        """
        return np.array(
            [self._processes[i].drift_1d(t, float(x[i])) for i in range(self.size())],
            dtype=np.float64,
        )

    def diffusion(
        self,
        t: float,
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Diffusion matrix at ``(t, x)``.

        # C++ parity: ``StochasticProcessArray::diffusion`` —
        # row ``i`` of ``sqrt_correlation`` scaled by ``sigma_i``.
        """
        tmp = self._sqrt_correlation.copy()
        for i in range(self.size()):
            sigma_i = self._processes[i].diffusion_1d(t, float(x[i]))
            tmp[i, :] *= sigma_i
        return tmp

    def expectation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Component-wise expectation.

        # C++ parity: ``StochasticProcessArray::expectation``.
        """
        return np.array(
            [
                self._processes[i].expectation_1d(t0, float(x0[i]), dt)
                for i in range(self.size())
            ],
            dtype=np.float64,
        )

    def std_deviation(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Standard-deviation matrix.

        # C++ parity: ``StochasticProcessArray::stdDeviation`` — row
        # ``i`` of ``sqrt_correlation`` scaled by ``stdDev_i``.
        """
        tmp = self._sqrt_correlation.copy()
        for i in range(self.size()):
            sigma_i = self._processes[i].std_deviation_1d(t0, float(x0[i]), dt)
            tmp[i, :] *= sigma_i
        return tmp

    def covariance(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Covariance matrix = ``stdDev @ stdDev^T``.

        # C++ parity: ``StochasticProcessArray::covariance`` —
        # ``stdDeviation(t0, x0, dt) * transpose(stdDeviation(...))``.
        """
        s = self.std_deviation(t0, x0, dt)
        return s @ s.T

    def evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evolve given a Brownian increment ``dw``.

        # C++ parity: ``StochasticProcessArray::evolve``. ``dw`` is the
        # vector of independent normals; ``dz = sqrt_correlation @ dw``
        # is the correlated noise, and component ``i`` is forwarded to
        # ``processes[i].evolve(t0, x0[i], dt, dz[i])``.
        """
        dz = self._sqrt_correlation @ dw
        out = np.empty(self.size(), dtype=np.float64)
        for i in range(self.size()):
            out[i] = self._processes[i].evolve_1d(t0, float(x0[i]), dt, float(dz[i]))
        return out

    def apply(
        self,
        x0: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply component-wise via 1-D ``apply``.

        # C++ parity: ``StochasticProcessArray::apply``.
        """
        return np.array(
            [self._processes[i].apply_1d(float(x0[i]), float(dx[i])) for i in range(self.size())],
            dtype=np.float64,
        )

    def time(self, date: Date) -> float:
        """Year fraction for ``date`` — delegated to ``processes[0]``.

        # C++ parity: ``StochasticProcessArray::time`` — ``processes_[0]->time(d)``.
        """
        return self._processes[0].time(date)

    # --- inspectors ------------------------------------------------------

    def process(self, i: int) -> StochasticProcess1D:
        """Return the i-th constituent 1-D process.

        # C++ parity: ``StochasticProcessArray::process``.
        """
        return self._processes[i]

    def correlation(self) -> npt.NDArray[np.float64]:
        """Recover the (possibly salvaged) correlation matrix.

        # C++ parity: ``StochasticProcessArray::correlation`` —
        # ``sqrtCorrelation * transpose(sqrtCorrelation)``.
        """
        return self._sqrt_correlation @ self._sqrt_correlation.T


__all__ = ["StochasticProcessArray"]
