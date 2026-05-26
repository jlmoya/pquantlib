"""Rounding — OMG-style decimal rounding with 5 conventions.

# C++ parity: ql/math/rounding.hpp + ql/math/rounding.cpp (v1.42.1).

Five rounding types selected by a ``Type`` IntEnum (None / Up / Down /
Closest / Floor / Ceiling). The ``digit`` parameter (default 5) selects
the rounding threshold for Closest / Floor / Ceiling. Precision range is
[0, 16] decimal places.

Convenience subclasses (UpRounding, DownRounding, ClosestRounding,
CeilingTruncation, FloorTruncation) pre-bind a single Type.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Final

from pquantlib import qassert

_POW10_LUT: Final[tuple[float, ...]] = (
    1.0e0,
    1.0e1,
    1.0e2,
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
    1.0e7,
    1.0e8,
    1.0e9,
    1.0e10,
    1.0e11,
    1.0e12,
    1.0e13,
    1.0e14,
    1.0e15,
    1.0e16,
)


class Type(IntEnum):
    None_ = 0
    Up = 1
    Down = 2
    Closest = 3
    Floor = 4
    Ceiling = 5


def _fast_pow10(precision: int) -> float:
    """Mirrors C++ ``fast_pow10`` — LUT-backed 10^precision (precision masked to 0-31)."""
    idx = precision & 0x1F
    if idx >= len(_POW10_LUT):
        return 0.0  # C++ behavior beyond the LUT (we expect precision in [0, 16])
    return _POW10_LUT[idx]


class Rounding:
    """Configurable decimal rounding.

    Defaults: ``Rounding()`` returns the no-op rounding (Type.None_).
    """

    def __init__(
        self,
        precision: int = 0,
        type_: Type = Type.None_,
        digit: int = 5,
    ) -> None:
        self._precision: int = precision
        self._type: Type = type_
        self._digit: int = digit

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def type(self) -> Type:
        return self._type

    @property
    def rounding_digit(self) -> int:
        return self._digit

    def __call__(self, value: float) -> float:
        if self._type == Type.None_:
            return value
        mult = _fast_pow10(self._precision)
        neg = value < 0.0
        lvalue = math.fabs(value) * mult
        integral = math.floor(lvalue)
        mod_val = lvalue - integral
        lvalue = integral
        threshold = self._digit / 10.0
        if self._type == Type.Down:
            pass
        elif self._type == Type.Up:
            if mod_val != 0.0:
                lvalue += 1.0
        elif self._type == Type.Closest:
            if mod_val >= threshold:
                lvalue += 1.0
        elif self._type == Type.Floor:
            if not neg and mod_val >= threshold:
                lvalue += 1.0
        elif self._type == Type.Ceiling:
            if neg and mod_val >= threshold:
                lvalue += 1.0
        else:
            qassert.fail("unknown rounding method")
        return -(lvalue / mult) if neg else lvalue / mult


class UpRounding(Rounding):
    def __init__(self, precision: int, digit: int = 5) -> None:
        super().__init__(precision, Type.Up, digit)


class DownRounding(Rounding):
    def __init__(self, precision: int, digit: int = 5) -> None:
        super().__init__(precision, Type.Down, digit)


class ClosestRounding(Rounding):
    def __init__(self, precision: int, digit: int = 5) -> None:
        super().__init__(precision, Type.Closest, digit)


class CeilingTruncation(Rounding):
    def __init__(self, precision: int, digit: int = 5) -> None:
        super().__init__(precision, Type.Ceiling, digit)


class FloorTruncation(Rounding):
    def __init__(self, precision: int, digit: int = 5) -> None:
        super().__init__(precision, Type.Floor, digit)
