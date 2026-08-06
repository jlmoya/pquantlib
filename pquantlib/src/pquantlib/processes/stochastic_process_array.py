"""StochasticProcessArray — N correlated 1-D processes as one N-D process.

# C++ parity: ql/processes/stochasticprocessarray.{hpp,cpp} (v1.42.1).

Wraps ``N`` ``StochasticProcess1D`` instances + an ``NxN`` correlation
matrix into a single ``StochasticProcess`` (multi-D).  Each step
takes ``N`` independent Brownian increments ``dw``, premultiplies by
the spectral-square-root of the correlation matrix to inject the
correlation, and feeds component ``dz[i]`` into ``processes[i].evolve``.

Python divergences vs C++:

* C++ uses ``Matrix`` and the ``pseudoSqrt(Spectral)`` helper.  This port
  has no free ``pseudo_sqrt``, so ``_spectral_sqrt`` below transcribes it,
  including the ``SymmetricSchurDecomposition`` ordering and sign rules and
  ``normalizePseudoRoot`` — see that function's docstring for why the earlier
  inline ``eigh`` version reproduced ``M M^T`` but not ``M``.
* The C++ ``stdDeviation`` and ``diffusion`` return ``Matrix`` whose
  row ``i`` is ``sqrt_corr.row(i) * processes_[i]->...``.  We mirror
  that by scaling the ``sqrt_corr`` rows in-place.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.date import Date


def _symmetric_schur(
    m: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Eigen-decomposition ordered the way QuantLib's Jacobi solver orders it.

    # C++ parity: ``SymmetricSchurDecomposition``
    # (ql/math/matrixutilities/symmetricschurdecomposition.cpp:115-137) — the
    # post-iteration sort, round-off zeroing and sign convention.

    Three ordering rules, all of them observable in a pseudo-root:

    1. eigenvalues DESCENDING;
    2. on a TIE, ``std::sort(..., std::greater<>())`` on
       ``pair<Real, vector<Real>>`` falls through to comparing the
       EIGENVECTORS lexicographically, also descending. That is what keeps
       ``pseudoSqrt(I)`` equal to ``I`` rather than to a permutation of it;
    3. each eigenvector's sign is pinned so its FIRST component is
       non-negative (applied after the sort, so it does not affect the
       tie-break).

    ``numpy.linalg.eigh`` (LAPACK divide-and-conquer) replaces the C++ cyclic
    Jacobi iteration. Both compute the exact spectral decomposition; for a
    DEGENERATE eigenvalue the two can still pick different bases of the
    eigenspace, and no ordering rule can reconcile that. For the diagonal and
    2x2 correlation matrices this class is used with, they agree.
    """
    eig_vals, eig_vecs = np.linalg.eigh(m)  # ascending
    size = int(m.shape[0])
    order = sorted(
        range(size),
        key=lambda k: (float(eig_vals[k]), tuple(float(v) for v in eig_vecs[:, k])),
        reverse=True,
    )
    values = np.array([float(eig_vals[k]) for k in order], dtype=np.float64)
    vectors = np.array(eig_vecs[:, order], dtype=np.float64)

    max_ev = float(values[0])
    for col in range(size):
        if max_ev != 0.0 and abs(values[col] / max_ev) < 1e-16:
            values[col] = 0.0
        if vectors[0, col] < 0.0:
            vectors[:, col] = -vectors[:, col]
    return values, vectors


def _normalize_pseudo_root(
    matrix: npt.NDArray[np.float64], pseudo: npt.NDArray[np.float64]
) -> None:
    """Rescale each row of ``pseudo`` so its norm matches ``matrix``'s diagonal.

    # C++ parity: ``normalizePseudoRoot`` (pseudosqrt.cpp, in-place).
    """
    for i in range(int(matrix.shape[0])):
        norm = float(np.dot(pseudo[i], pseudo[i]))
        if norm > 0.0:
            pseudo[i, :] *= math.sqrt(float(matrix[i, i]) / norm)


def _spectral_sqrt(corr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Spectral pseudo-square-root of a symmetric matrix.

    # C++ parity: ``pseudoSqrt(corr, SalvagingAlgorithm::Spectral)``
    # in ql/math/matrixutilities/pseudosqrt.cpp:180-187.

    ALIGN(processes): this used to be an inline
    ``numpy.linalg.eigh`` + ``clip`` + ``eigvecs * sqrt(eigvals)``. That
    reproduces ``M @ M.T`` correctly — which is why ``correlation()`` and
    ``covariance()`` matched C++ — but NOT ``M`` itself, and ``M`` is exactly
    what ``diffusion`` / ``std_deviation`` / ``evolve`` hand to the caller:

    * ``eigh`` returns eigenvalues ASCENDING; C++ returns them DESCENDING, so
      the columns came out in reverse order;
    * ``eigh`` picks each eigenvector's sign arbitrarily; C++ pins it so the
      first component is non-negative;
    * ``normalizePseudoRoot`` was missing entirely.

    Concretely, for ``[[1, -0.35], [-0.35, 1]]`` the old code returned row 0 as
    ``[0.570, 0.822]`` where C++ returns ``[0.822, 0.570]``. Every
    ``evolve(dw)`` on a non-identity correlation was mixing the Brownian
    increments differently from C++. Pinned by
    ``migration-harness/references/v143/processes/tail.json`` -> ``spa`` and
    ``end_euler_array``.

    The decomposition is done by :func:`_symmetric_schur` here rather than by
    ``models.marketmodels.models.pseudo_sqrt.rank_reduced_sqrt``, which is the
    same computation for distinct eigenvalues but reverses ``eigh``'s output
    unconditionally and so returns a PERMUTATION matrix for the identity —
    exactly the input this class sees most often.
    """
    qassert.require(corr.ndim == 2 and corr.shape[0] == corr.shape[1], "correlation matrix must be square")
    qassert.require(
        np.allclose(corr, corr.T, atol=1e-14),
        "correlation matrix must be symmetric",
    )
    size = int(corr.shape[0])
    values, vectors = _symmetric_schur(corr)
    diagonal = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        diagonal[i, i] = math.sqrt(max(float(values[i]), 0.0))
    result: npt.NDArray[np.float64] = vectors @ diagonal
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
