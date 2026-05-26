"""Extrapolator — mixin enabling per-instance extrapolation toggling.

# C++ parity: ql/math/interpolations/extrapolation.hpp (v1.42.1)

C++ places this under `math/interpolations/` though it's only used by
term structures (and a handful of interpolation wrappers). PQuantLib
groups it with the term-structure layer where it's consumed.
"""

from __future__ import annotations


class Extrapolator:
    """Base for objects that may extrapolate past their nominal max date/time."""

    __slots__ = ("_extrapolate",)

    def __init__(self) -> None:
        self._extrapolate: bool = False

    def enable_extrapolation(self, enabled: bool = True) -> None:
        self._extrapolate = enabled

    def disable_extrapolation(self, disabled: bool = True) -> None:
        self._extrapolate = not disabled

    def allows_extrapolation(self) -> bool:
        return self._extrapolate
