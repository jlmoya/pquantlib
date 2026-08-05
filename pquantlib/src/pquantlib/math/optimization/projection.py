"""Parameter projection: optimize over a subset of a parameter vector.

# C++ parity: ql/math/optimization/projection.{hpp,cpp} (v1.43).

``Projection`` splits a full parameter vector into a FIXED part and a
FREE part, selected by a boolean mask. ``project`` extracts the free
coordinates; ``include`` puts a free vector back into the full vector,
keeping the fixed coordinates at the values supplied to the constructor.

The mask convention is the C++ one: ``fix_parameters[j] is True`` means
coordinate ``j`` is HELD FIXED, so the free count is the number of
``False`` entries. An empty mask means every parameter is free.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib import qassert


class Projection:
    """Fixed/free split of a parameter vector.

    # C++ parity: ``class Projection`` in
    # ql/math/optimization/projection.hpp:33-53 (v1.43).
    """

    __slots__ = (
        "_actual_parameters",
        "_fix_parameters",
        "_fixed_parameters",
        "_number_of_free_parameters",
    )

    def __init__(
        self,
        parameter_values: npt.NDArray[np.float64],
        fix_parameters: list[bool] | None = None,
    ) -> None:
        # C++ parity: projection.cpp:27-41.
        self._fixed_parameters: npt.NDArray[np.float64] = parameter_values.astype(
            np.float64, copy=True
        )
        self._actual_parameters: npt.NDArray[np.float64] = parameter_values.astype(
            np.float64, copy=True
        )
        self._fix_parameters: list[bool] = (
            [False] * int(self._actual_parameters.size)
            if not fix_parameters
            else list(fix_parameters)
        )

        qassert.require(
            self._fixed_parameters.size == len(self._fix_parameters),
            "fixedParameters_.size()!=parametersFreedoms_.size()",
        )
        self._number_of_free_parameters: int = sum(
            1 for fix in self._fix_parameters if not fix
        )
        qassert.require(self._number_of_free_parameters > 0, "numberOfFreeParameters==0")

    @property
    def number_of_free_parameters(self) -> int:
        """Count of coordinates NOT held fixed. # C++ parity: ``numberOfFreeParameters_``."""
        return self._number_of_free_parameters

    @property
    def fixed_parameters(self) -> npt.NDArray[np.float64]:
        """The full parameter vector the projection was built from.

        # C++ parity: ``fixedParameters_``, which is ``protected`` there.
        # Python exposes it read-only so ``ProjectedCostFunction`` and
        # ``ProjectedConstraint`` can reproduce C++'s copy-construction of
        # a ``Projection``.
        """
        return self._fixed_parameters

    @property
    def fix_parameters(self) -> list[bool]:
        """The fixed/free mask (``True`` == held fixed).

        # C++ parity: ``fixParameters_``, ``protected`` there.
        """
        return list(self._fix_parameters)

    def _map_free_parameters(self, parameter_values: npt.NDArray[np.float64]) -> None:
        """Scatter ``parameter_values`` into the free slots of the working vector.

        # C++ parity: projection.cpp:43-52 — ``mapFreeParameters``, which is
        # ``const`` in C++ but mutates the ``mutable actualParameters_``.
        """
        qassert.require(
            parameter_values.size == self._number_of_free_parameters,
            "parameterValues.size()!=numberOfFreeParameters",
        )
        i = 0
        for j in range(self._actual_parameters.size):
            if not self._fix_parameters[j]:
                self._actual_parameters[j] = parameter_values[i]
                i += 1

    def project(
        self, parameters: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Extract the free coordinates of a full parameter vector.

        # C++ parity: projection.cpp:54-65.
        """
        qassert.require(
            parameters.size == len(self._fix_parameters),
            "parameters.size()!=parametersFreedoms_.size()",
        )
        projected_parameters = np.empty(
            self._number_of_free_parameters, dtype=np.float64
        )
        i = 0
        for j in range(len(self._fix_parameters)):
            if not self._fix_parameters[j]:
                projected_parameters[i] = parameters[j]
                i += 1
        return projected_parameters

    def include(
        self, projected_parameters: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Rebuild a full parameter vector from its free coordinates.

        # C++ parity: projection.cpp:67-78 — the fixed coordinates come
        # from the vector the constructor was given, NOT from the working
        # ``actualParameters_``.
        """
        qassert.require(
            projected_parameters.size == self._number_of_free_parameters,
            "projectedParameters.size()!=numberOfFreeParameters",
        )
        y = self._fixed_parameters.astype(np.float64, copy=True)
        i = 0
        for j in range(y.size):
            if not self._fix_parameters[j]:
                y[j] = projected_parameters[i]
                i += 1
        return y
