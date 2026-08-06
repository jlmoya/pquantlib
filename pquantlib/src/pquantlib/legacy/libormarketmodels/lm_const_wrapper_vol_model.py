"""LmConstWrapperVolatilityModel — non-calibratable view of a volatility model.

# C++ parity: ql/legacy/libormarketmodels/lmconstwrappervolmodel.hpp (v1.43,
# header-only).

Forwards every query to a wrapped :class:`LmVolatilityModel` while presenting
an EMPTY argument vector, so a calibration that walks the wrapper's
``params()`` cannot move the wrapped model. ``_generate_arguments`` is a no-op.
"""

from __future__ import annotations

from pquantlib.legacy.libormarketmodels.lm_vol_model import LmVolatilityModel
from pquantlib.math.array import Array


class LmConstWrapperVolatilityModel(LmVolatilityModel):
    """Read-only wrapper around another volatility model.

    # C++ parity: ``class LmConstWrapperVolatilityModel``
    # (lmconstwrappervolmodel.hpp:32-57).
    """

    def __init__(self, vola_model: LmVolatilityModel) -> None:
        # C++ parity: lmconstwrappervolmodel.hpp:34-38 — size from the wrapped
        # model, zero arguments.
        super().__init__(vola_model.size(), 0)
        self._vola_model: LmVolatilityModel = vola_model

    def volatility_model(self) -> LmVolatilityModel:
        """The wrapped model.

        Python addition: C++ keeps ``volaModel_`` protected with no accessor.
        """
        return self._vola_model

    def volatility(self, t: float, x: Array | None = None) -> Array:
        """# C++ parity: lmconstwrappervolmodel.hpp:40-42."""
        return self._vola_model.volatility(t, x)

    def volatility_scalar(self, i: int, t: float, x: Array | None = None) -> float:
        """# C++ parity: lmconstwrappervolmodel.hpp:43-46.

        C++ divergence, deliberate: that overload is declared non-``const`` and
        is NOT marked ``override``, so through an ``LmVolatilityModel*`` C++
        silently falls back to the base implementation (index into the vector
        form). Both routes return the same number here — the wrapped model's
        own scalar accessor — so the port overrides it properly rather than
        reproducing an accident of C++ overload resolution.
        """
        return self._vola_model.volatility_scalar(i, t, x)

    def integrated_variance(
        self, i: int, j: int, u: float, x: Array | None = None
    ) -> float:
        """# C++ parity: lmconstwrappervolmodel.hpp:47-49."""
        return self._vola_model.integrated_variance(i, j, u, x)

    def _generate_arguments(self) -> None:
        """# C++ parity: lmconstwrappervolmodel.hpp:56 — empty body."""


__all__ = ["LmConstWrapperVolatilityModel"]
