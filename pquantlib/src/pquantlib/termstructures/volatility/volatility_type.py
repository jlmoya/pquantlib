"""VolatilityType — Black/Normal enum for smile sections.

# C++ parity: ql/termstructures/volatility/volatilitytype.hpp (v1.42.1).

C++ ``enum VolatilityType { ShiftedLognormal, Normal }``. ShiftedLognormal
is the default — the underlying is assumed to follow a shifted-lognormal
diffusion (``df = sigma (f + s) dW``). Normal selects the Bachelier model
(``df = sigma dW``).
"""

from __future__ import annotations

from enum import IntEnum


class VolatilityType(IntEnum):
    """Volatility-model selector for smile sections."""

    ShiftedLognormal = 0
    Normal = 1

    def __str__(self) -> str:
        return {
            VolatilityType.ShiftedLognormal: "ShiftedLognormal",
            VolatilityType.Normal: "Normal",
        }[self]
