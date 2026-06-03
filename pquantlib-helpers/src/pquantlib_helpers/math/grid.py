"""Grid builders for the legacy finite-difference layer.

# Retired-API compat layer — NOT shipped in C++ QuantLib v1.42.1 as a
# standalone class (modern QuantLib builds FD meshes via the Fdm* meshers).

Java parity: ``org.jquantlib.math.Grid`` — the ``CenteredGrid`` / ``BoundedGrid``
/ ``BoundedLogGrid`` static factory helpers consumed by
:class:`~pquantlib_helpers.math.sampled_curve.SampledCurve.set_log_grid` and the
legacy FD vanilla engines.

``bounded_log_grid(x_min, x_max, steps)`` returns ``steps + 1`` nodes
geometrically spaced from ``x_min`` to ``x_max``, built by repeated
multiplication by ``exp((log(x_max) - log(x_min)) / steps)`` — the exact Java
recurrence, NOT ``numpy.geomspace`` (which would accumulate different last-bit
rounding and break the TIGHT cross-validation against the Java FD engines).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pquantlib.math.array import Array


def centered_grid(center: float, dx: float, steps: int) -> Array:
    """Return ``steps + 1`` uniform nodes centred on ``center`` with spacing ``dx``.

    Java parity: ``Grid.CenteredGrid``.
    """
    result: Array = np.empty(steps + 1, dtype=np.float64)
    for i in range(steps + 1):
        result[i] = center + (i - steps / 2.0) * dx
    return result


def bounded_grid(x_min: float, x_max: float, steps: int) -> Array:
    """Return ``steps + 1`` uniform nodes from ``x_min`` to ``x_max``.

    Java parity: ``Grid.BoundedGrid``.
    """
    result: Array = np.empty(steps + 1, dtype=np.float64)
    dx = (x_max - x_min) / steps
    x = x_min
    for i in range(steps + 1):
        result[i] = x
        x += dx
    return result


def bounded_log_grid(x_min: float, x_max: float, steps: int) -> Array:
    """Return ``steps + 1`` geometrically-spaced nodes from ``x_min`` to ``x_max``.

    Java parity: ``Grid.BoundedLogGrid`` — built by the exact Java recurrence
    ``g[0] = x_min; g[j] = g[j-1] * exp((log(x_max) - log(x_min)) / steps)`` so
    the node values are bit-faithful to the Java FD engines.
    """
    result: Array = np.empty(steps + 1, dtype=np.float64)
    grid_log_spacing = (math.log(x_max) - math.log(x_min)) / steps
    edx = math.exp(grid_log_spacing)
    result[0] = x_min
    for j in range(1, steps + 1):
        result[j] = result[j - 1] * edx
    return result


__all__ = ["bounded_grid", "bounded_log_grid", "centered_grid"]
