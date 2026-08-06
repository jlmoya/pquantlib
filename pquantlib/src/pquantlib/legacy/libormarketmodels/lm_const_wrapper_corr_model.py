"""LmConstWrapperCorrelationModel — non-calibratable view of a correlation model.

# C++ parity: ql/legacy/libormarketmodels/lmconstwrappercorrmodel.hpp (v1.43,
# header-only).

Forwards every query to a wrapped :class:`LmCorrelationModel` while presenting
an EMPTY argument vector, so a calibration walking the wrapper's ``params()``
cannot move the wrapped model.
"""

from __future__ import annotations

from pquantlib.legacy.libormarketmodels.lm_corr_model import LmCorrelationModel
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix


class LmConstWrapperCorrelationModel(LmCorrelationModel):
    """Read-only wrapper around another correlation model.

    # C++ parity: ``class LmConstWrapperCorrelationModel``
    # (lmconstwrappercorrmodel.hpp:31-56).
    """

    def __init__(self, corr_model: LmCorrelationModel) -> None:
        # C++ parity: lmconstwrappercorrmodel.hpp:33-37 — size from the
        # wrapped model, zero arguments.
        super().__init__(corr_model.size(), 0)
        self._corr_model: LmCorrelationModel = corr_model

    def correlation_model(self) -> LmCorrelationModel:
        """The wrapped model.

        Python addition: C++ keeps ``corrModel_`` protected with no accessor.
        """
        return self._corr_model

    def factors(self) -> int:
        """# C++ parity: lmconstwrappercorrmodel.hpp:39."""
        return self._corr_model.factors()

    def correlation(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lmconstwrappercorrmodel.hpp:41-43."""
        return self._corr_model.correlation(t, x)

    def pseudo_sqrt(self, t: float, x: Array | None = None) -> Matrix:
        """# C++ parity: lmconstwrappercorrmodel.hpp:44-46."""
        return self._corr_model.pseudo_sqrt(t, x)

    def correlation_scalar(
        self, i: int, j: int, t: float, x: Array | None = None
    ) -> float:
        """# C++ parity: lmconstwrappercorrmodel.hpp:47-49."""
        return self._corr_model.correlation_scalar(i, j, t, x)

    def is_time_independent(self) -> bool:
        """# C++ parity: lmconstwrappercorrmodel.hpp:50."""
        return self._corr_model.is_time_independent()

    def _generate_arguments(self) -> None:
        """# C++ parity: lmconstwrappercorrmodel.hpp:53 — empty body."""


__all__ = ["LmConstWrapperCorrelationModel"]
