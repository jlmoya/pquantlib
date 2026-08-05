"""Akima cubic interpolation.

# C++ parity: ql/math/interpolations/cubicinterpolation.hpp
#             ``class AkimaCubicInterpolation`` (v1.43).

``AkimaCubicInterpolation`` is one line of C++: ``CubicInterpolation`` with
``DerivativeApprox::Akima``, ``monotonic = false`` and a natural (second
derivative = 0) condition at both ends. This module is that line.

**This used to delegate to** ``scipy.interpolate.Akima1DInterpolator``,
justified by an argument that scipy is the more faithful rendering of
Akima's 1970 paper. That argument is beside the point: the ground truth
for this port is C++ QuantLib v1.43, not the paper. The two agree on the
*interior* slope stencil but not on the end slopes — QuantLib uses its own
non-standard endpoint expressions built from products like
``2*S[0]*S[1]`` and ``4*S[0]^2*S[1]`` (cubicinterpolation.hpp:613-629),
which are not Akima's reflection rule and are not even dimensionally
consistent. Those four slopes (``tmp[0]``, ``tmp[1]``, ``tmp[n-2]``,
``tmp[n-1]``) drive the first two and last two intervals, so the
disagreement was never confined to the boundary: on ``y = x^2`` over
``x = 0..4`` C++ produces a pillar slope of ``2.25`` at ``x = 0`` where
scipy — and calculus — give ``0``. The old module tested that scipy
"recovers the quadratic exactly" and recorded C++ as deviating, i.e. it
asserted the divergence rather than the port.
"""

from __future__ import annotations

from typing import final

from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    DerivativeApprox,
)


@final
class AkimaCubicInterpolation(CubicInterpolation):
    """Akima slopes, natural BCs, unfiltered.

    # C++ parity: ``AkimaCubicInterpolation`` (cubicinterpolation.hpp:260-271).

    Requires at least four points — the slope stencil reads ``S[2]`` and
    ``S[n-4]`` (C++ ``QL_REQUIRE`` at cubicinterpolation.hpp:403-407).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Akima,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


__all__ = ["AkimaCubicInterpolation"]
