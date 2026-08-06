"""LmCorrelationModel — abstract forward-rate correlation model.

# C++ parity: ql/legacy/libormarketmodels/lmcorrmodel.{hpp,cpp} (v1.43).

Concrete subclasses supply the correlation matrix at a given time; the base
derives a pseudo-square-root from it and provides a (deliberately inefficient)
scalar accessor.

C++ overload split (both named ``correlation``):

- ``Matrix correlation(Time, const Array&)`` -> :meth:`LmCorrelationModel.correlation`
- ``Real correlation(Size, Size, Time, const Array&)`` ->
  :meth:`LmCorrelationModel.correlation_scalar`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)
from pquantlib.models.parameter import Parameter


def spectral_pseudo_sqrt(matrix: Matrix) -> Matrix:
    """Full-rank spectral pseudo-square-root, C++ ``pseudoSqrt(m, Spectral)``.

    # C++ parity: ql/math/matrixutilities/pseudosqrt.cpp:377-385 — spectral
    # decomposition, negative eigenvalues clipped to zero, then
    # ``normalizePseudoRoot``.

    PQuantLib has no standalone ``pseudo_sqrt``; it ships
    ``rank_reduced_sqrt(m, max_rank, retained, sa)``, whose body IS the
    ``Spectral`` arm of ``pseudoSqrt`` once no factor is dropped. Asking for
    ``max_rank == size`` and ``component_retained_percentage == 1.0`` retains
    every factor (the C++ code inflates the retention threshold by 1.1
    precisely so numerical glitches cannot discard one), so this returns
    exactly ``eigenvectors * diag(sqrt(max(eig, 0)))`` normalized — i.e. the
    same square matrix ``pseudoSqrt(m, Spectral)`` returns.
    """
    return rank_reduced_sqrt(matrix, matrix.shape[0], 1.0, SalvagingAlgorithm.SPECTRAL)


class LmCorrelationModel(ABC):
    """Abstract LIBOR forward correlation model.

    # C++ parity: ``class LmCorrelationModel`` (lmcorrmodel.hpp:35-57).
    """

    def __init__(self, size: int, n_arguments: int) -> None:
        # C++ parity: lmcorrmodel.cpp:25-26.
        self._size: int = size
        self._arguments: list[Parameter] = [Parameter() for _ in range(n_arguments)]

    # --- inspectors -------------------------------------------------------

    def size(self) -> int:
        """# C++ parity: ``LmCorrelationModel::size`` (lmcorrmodel.cpp:28-30)."""
        return self._size

    def factors(self) -> int:
        """Number of driving Brownian factors.

        # C++ parity: ``LmCorrelationModel::factors`` (lmcorrmodel.cpp:32-34)
        # — full rank by default.
        """
        return self._size

    def params(self) -> list[Parameter]:
        """The calibration arguments (live list, as C++ returns a non-const ref).

        # C++ parity: ``LmCorrelationModel::params`` (lmcorrmodel.cpp:53-55).
        """
        return self._arguments

    def set_params(self, arguments: Sequence[Parameter]) -> None:
        """# C++ parity: ``LmCorrelationModel::setParams`` (lmcorrmodel.cpp:57-61)."""
        self._arguments = list(arguments)
        self._generate_arguments()

    def is_time_independent(self) -> bool:
        """Whether ``correlation(t)`` is the same matrix for every ``t``.

        # C++ parity: ``LmCorrelationModel::isTimeIndependent``
        # (lmcorrmodel.cpp:36-38) — ``false`` in the base. The flag is what
        # lets ``LfmCovarianceProxy`` take its analytic integrated-covariance
        # shortcut.
        """
        return False

    # --- correlation ------------------------------------------------------

    @abstractmethod
    def correlation(self, t: float, x: Array | None = None) -> Matrix:
        """The size x size correlation matrix at time ``t``.

        # C++ parity: pure virtual ``Matrix correlation(Time, const Array&)``
        # (lmcorrmodel.hpp:46).
        """

    def pseudo_sqrt(self, t: float, x: Array | None = None) -> Matrix:
        """A matrix ``B`` with ``B B^T == correlation(t, x)``.

        # C++ parity: ``LmCorrelationModel::pseudoSqrt``
        # (lmcorrmodel.cpp:40-44) — ``QuantLib::pseudoSqrt(correlation(t, x),
        # SalvagingAlgorithm::Spectral)``.
        """
        return spectral_pseudo_sqrt(self.correlation(t, x))

    def correlation_scalar(
        self, i: int, j: int, t: float, x: Array | None = None
    ) -> float:
        """Single entry of the correlation matrix.

        # C++ parity: ``LmCorrelationModel::correlation(Size, Size, Time,
        # const Array&)`` (lmcorrmodel.cpp:46-50) — "inefficient
        # implementation, please overload in derived classes".
        """
        return float(self.correlation(t, x)[i, j])

    # --- protected --------------------------------------------------------

    @abstractmethod
    def _generate_arguments(self) -> None:
        """Rebuild the cached matrices from ``params()``.

        # C++ parity: protected pure virtual ``generateArguments``
        # (lmcorrmodel.hpp:52).
        """


__all__ = ["LmCorrelationModel", "spectral_pseudo_sqrt"]
