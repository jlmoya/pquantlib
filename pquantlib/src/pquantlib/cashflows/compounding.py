"""Compounding convention enum.

# C++ parity: ql/compounding.hpp (v1.42.1) — ``enum Compounding``.

The five canonical compounding conventions used by ``InterestRate``:

- ``Simple``: 1 + r*t
- ``Compounded``: (1 + r/f)^(f*t) — requires a Frequency
- ``Continuous``: exp(r*t)
- ``SimpleThenCompounded``: Simple up to one period, then Compounded
- ``CompoundedThenSimple``: Compounded up to one period, then Simple
"""

from __future__ import annotations

from enum import IntEnum


class Compounding(IntEnum):
    """Mirrors ``QuantLib::Compounding`` (v1.42.1 ql/compounding.hpp:32-37)."""

    Simple = 0
    Compounded = 1
    Continuous = 2
    SimpleThenCompounded = 3
    CompoundedThenSimple = 4
