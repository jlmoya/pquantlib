"""Interest-rate compounding convention enum.

# C++ parity: ql/compounding.hpp (v1.42.1) — enum Compounding.

Five conventions are supported:

- ``Simple``                → 1 + r*t
- ``Compounded``            → (1 + r/f)^(f*t)
- ``Continuous``            → exp(r*t)
- ``SimpleThenCompounded``  → simple up to 1/f, compounded after
- ``CompoundedThenSimple``  → compounded up to 1/f, simple after

The C++ header places this enum at namespace ``QuantLib`` root
(``ql/compounding.hpp``). PQuantLib puts it under ``pquantlib.time``
because it travels alongside ``Frequency`` semantically and Frequency
already lives there; no Python module sits at the ``pquantlib`` root.
"""

from __future__ import annotations

from enum import IntEnum


class Compounding(IntEnum):
    Simple = 0  # 1 + r*t
    Compounded = 1  # (1 + r/f)^(f*t)
    Continuous = 2  # exp(r*t)
    SimpleThenCompounded = 3  # simple up to 1/f, compounded after
    CompoundedThenSimple = 4  # compounded up to 1/f, simple after
