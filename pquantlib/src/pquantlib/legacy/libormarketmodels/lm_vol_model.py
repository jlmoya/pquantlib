"""LmVolatilityModel — abstract caplet volatility model for LIBOR market models.

# C++ parity: ql/legacy/libormarketmodels/lmvolmodel.{hpp,cpp} (v1.43).

The model owns a fixed number of ``Parameter`` objects (the calibration
arguments) and exposes the instantaneous forward-rate volatilities at a given
time. Concrete subclasses supply the vector-valued ``volatility(t, x)``, and
optionally the analytic ``integrated_variance``.

C++ overload split (both named ``volatility``):

- ``Array volatility(Time, const Array&)`` -> :meth:`LmVolatilityModel.volatility`
- ``Volatility volatility(Size, Time, const Array&)`` ->
  :meth:`LmVolatilityModel.volatility_scalar`

Python has no overloading, so the scalar form gets a ``_scalar`` suffix — the
same convention ``OneFactorAffineModel.discount_bond_scalar`` already uses in
this port.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.models.parameter import Parameter


class LmVolatilityModel(ABC):
    """Abstract caplet volatility model.

    # C++ parity: ``class LmVolatilityModel`` (lmvolmodel.hpp:33-53).
    """

    def __init__(self, size: int, n_arguments: int) -> None:
        # C++ parity: lmvolmodel.cpp:24-27 — ``arguments_(nArguments)``
        # default-constructs empty Parameters, exactly as
        # ``CalibratedModel`` does.
        self._size: int = size
        self._arguments: list[Parameter] = [Parameter() for _ in range(n_arguments)]

    # --- inspectors -------------------------------------------------------

    def size(self) -> int:
        """Number of forward rates the model covers.

        # C++ parity: ``LmVolatilityModel::size`` (lmvolmodel.cpp:29-31).
        """
        return self._size

    def params(self) -> list[Parameter]:
        """The calibration arguments.

        # C++ parity: ``LmVolatilityModel::params`` (lmvolmodel.cpp:44-46),
        # which returns a NON-const reference — callers mutate the vector in
        # place. The Python list is likewise the live object, not a copy.
        """
        return self._arguments

    def set_params(self, arguments: Sequence[Parameter]) -> None:
        """Replace the arguments and regenerate the derived state.

        # C++ parity: ``LmVolatilityModel::setParams``
        # (lmvolmodel.cpp:48-52) — assignment followed by
        # ``generateArguments()``.
        """
        self._arguments = list(arguments)
        self._generate_arguments()

    # --- volatility -------------------------------------------------------

    @abstractmethod
    def volatility(self, t: float, x: Array | None = None) -> Array:
        """Instantaneous volatilities of every forward rate at time ``t``.

        # C++ parity: pure virtual ``Array volatility(Time, const Array&)``
        # (lmvolmodel.hpp:43). C++ defaults ``x`` to an empty ``Array``; the
        # Python default ``None`` means the same thing.
        """

    def volatility_scalar(self, i: int, t: float, x: Array | None = None) -> float:
        """Instantaneous volatility of forward rate ``i`` at time ``t``.

        # C++ parity: ``LmVolatilityModel::volatility(Size, Time, const
        # Array&)`` (lmvolmodel.cpp:33-37) — the base implementation is
        # deliberately inefficient ("please overload in derived classes") and
        # simply indexes the vector form.
        """
        return float(self.volatility(t, x)[i])

    def integrated_variance(
        self, i: int, j: int, u: float, x: Array | None = None
    ) -> float:
        """Analytic integrated covariance of forwards ``i`` and ``j`` over [0, u].

        # C++ parity: ``LmVolatilityModel::integratedVariance``
        # (lmvolmodel.cpp:39-42) — the base ``QL_FAIL``s. That failure is
        # load-bearing: ``LfmCovarianceProxy::integratedCovariance`` catches it
        # and falls back to numerical integration.
        """
        qassert.fail("integratedVariance() method is not supported")

    # --- protected --------------------------------------------------------

    @abstractmethod
    def _generate_arguments(self) -> None:
        """Rebuild any state cached from ``params()``.

        # C++ parity: private pure virtual ``generateArguments``
        # (lmvolmodel.hpp:52), called by ``setParams``.
        """


__all__ = ["LmVolatilityModel"]
