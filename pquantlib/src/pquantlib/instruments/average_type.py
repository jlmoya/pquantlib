"""Average — averaging-algorithm enumeration for Asian options.

# C++ parity: ql/instruments/averagetype.hpp (v1.42.1) —
# ``struct Average { enum Type { Arithmetic, Geometric }; }``.

The C++ ``struct`` wraps the enum just for namespacing.  The Python
port uses ``IntEnum`` and exposes the enum directly as
``AverageType`` (no wrapping struct needed — Python's enum already
namespaces).
"""

from __future__ import annotations

from enum import IntEnum


class AverageType(IntEnum):
    """Discriminator for Asian-average algorithms.

    # C++ parity: ``Average::Type`` (averagetype.hpp:35).
    """

    Arithmetic = 0
    Geometric = 1


__all__ = ["AverageType"]
