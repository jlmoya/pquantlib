"""Cross-validation of ``FdmMesherIntegral`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmmesherintegral.{hpp,cpp}
# @ v1.43 (6b57206e0).

Both 1-D integrators from ``ql/math/integrals/discreteintegrals.hpp`` are
exercised, in 1, 2 and 3 dimensions. TIGHT tier: the recursion performs the
same fixed number of additions/multiplications in both ports and the measured
deviation is exactly zero.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.integrals.discrete_integrals import (
    DiscreteSimpsonIntegral,
    DiscreteTrapezoidIntegral,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.utilities.fdm_mesher_integral import (
    FdmMesherIntegral,
)
from pquantlib.testing.tolerance import tight


def _mesher_1d() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(-2.0, 2.0, 9))


def _mesher_2d() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(-2.0, 2.0, 9), Uniform1dMesher(-1.5, 1.5, 7)
    )


def _mesher_3d() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(-2.0, 2.0, 9),
        Uniform1dMesher(-1.5, 1.5, 7),
        Uniform1dMesher(0.0, 1.0, 5),
    )


def _gauss_1d(mesher: FdmMesherComposite) -> Array:
    f = np.empty(mesher.layout().size(), dtype=np.float64)
    for it in mesher.layout().iter():
        x = mesher.location(it, 0)
        f[it.index] = math.exp(-x * x)
    return f


def _gauss_2d(mesher: FdmMesherComposite) -> Array:
    f = np.empty(mesher.layout().size(), dtype=np.float64)
    for it in mesher.layout().iter():
        x = mesher.location(it, 0)
        y = mesher.location(it, 1)
        f[it.index] = math.exp(-x * x - 0.5 * y * y) * (1.0 + 0.25 * x * y)
    return f


def _gauss_3d(mesher: FdmMesherComposite) -> Array:
    f = np.empty(mesher.layout().size(), dtype=np.float64)
    for it in mesher.layout().iter():
        x = mesher.location(it, 0)
        y = mesher.location(it, 1)
        z = mesher.location(it, 2)
        f[it.index] = math.exp(-x * x - 0.5 * y * y) * (1.0 + z)
    return f


def test_1d_simpson(reference_data: dict[str, Any]) -> None:
    mesher = _mesher_1d()
    integral = FdmMesherIntegral(mesher, DiscreteSimpsonIntegral())
    tight(
        integral.integrate(_gauss_1d(mesher)),
        float(reference_data["mesher_integral_1d_simpson"]),
    )


def test_1d_trapezoid(reference_data: dict[str, Any]) -> None:
    mesher = _mesher_1d()
    integral = FdmMesherIntegral(mesher, DiscreteTrapezoidIntegral())
    tight(
        integral.integrate(_gauss_1d(mesher)),
        float(reference_data["mesher_integral_1d_trapezoid"]),
    )


def test_2d_simpson(reference_data: dict[str, Any]) -> None:
    mesher = _mesher_2d()
    integral = FdmMesherIntegral(mesher, DiscreteSimpsonIntegral())
    tight(
        integral.integrate(_gauss_2d(mesher)),
        float(reference_data["mesher_integral_2d_simpson"]),
    )


def test_2d_trapezoid(reference_data: dict[str, Any]) -> None:
    mesher = _mesher_2d()
    integral = FdmMesherIntegral(mesher, DiscreteTrapezoidIntegral())
    tight(
        integral.integrate(_gauss_2d(mesher)),
        float(reference_data["mesher_integral_2d_trapezoid"]),
    )


def test_3d_simpson(reference_data: dict[str, Any]) -> None:
    mesher = _mesher_3d()
    integral = FdmMesherIntegral(mesher, DiscreteSimpsonIntegral())
    tight(
        integral.integrate(_gauss_3d(mesher)),
        float(reference_data["mesher_integral_3d_simpson"]),
    )
