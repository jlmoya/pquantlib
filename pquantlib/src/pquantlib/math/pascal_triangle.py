"""Pascal triangle (binomial coefficients per row).

# C++ parity: ql/math/pascaltriangle.hpp + pascaltriangle.cpp (v1.42.1).

``PascalTriangle.get(order)`` returns the row of binomial coefficients
``C(order, 0..order)`` as a tuple. Rows are cached lazily on first request
and extended monotonically — mirrors the C++ ``coefficients_`` static
``std::vector<std::vector<BigNatural>>``.
"""

from __future__ import annotations

from typing import ClassVar

from pquantlib import qassert


class PascalTriangle:
    """Cached lazy generator of Pascal-triangle rows."""

    _coefficients: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def get(cls, order: int) -> tuple[int, ...]:
        # C++ takes a Size (unsigned). Python int can be negative; the
        # cache lookup would silently return the last row via negative
        # indexing. Guard here.
        qassert.require(order >= 0, f"PascalTriangle.get requires order >= 0, got {order}")
        if not cls._coefficients:
            # Bootstrap rows 0..3 verbatim from the C++ implementation.
            cls._coefficients.append((1,))
            cls._coefficients.append((1, 1))
            cls._coefficients.append((1, 2, 1))
            cls._coefficients.append((1, 3, 3, 1))
        while len(cls._coefficients) <= order:
            cls._next_order()
        return cls._coefficients[order]

    @classmethod
    def _next_order(cls) -> None:
        prev = cls._coefficients[-1]
        size = len(prev) + 1
        new_row = [1] * size
        for i in range(1, size - 1):
            new_row[i] = prev[i - 1] + prev[i]
        cls._coefficients.append(tuple(new_row))
