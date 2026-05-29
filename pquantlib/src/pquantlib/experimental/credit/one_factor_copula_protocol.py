"""OneFactorCopula protocol — minimal slice used by RandomDefaultModel.

# C++ parity: ql/experimental/credit/onefactorcopula.hpp (v1.42.1) — partial.

The full C++ ``OneFactorCopula`` class is a LazyObject with a tabulated
distribution of the systemic factor ``M`` and a convolution-based
``cumulativeY`` / ``density`` interface. W3-B will land the full port;
the W3-D slice exposes only ``correlation()`` because that is the only
method ``GaussianRandomDefaultModel`` calls.

# C++ parity divergence: Python ``Protocol`` replaces concrete-type
# coupling. The full W3-B port will satisfy the protocol structurally.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OneFactorCopulaProtocol(Protocol):
    """Surface used by ``GaussianRandomDefaultModel``.

    Implementations must expose ``correlation()`` returning the single-
    factor copula correlation parameter (``rho`` in the Hull-White
    notation).
    """

    def correlation(self) -> float:
        """Single-factor copula correlation.

        # C++ parity: ``OneFactorCopula::correlation``.
        """
        ...


__all__ = ["OneFactorCopulaProtocol"]
