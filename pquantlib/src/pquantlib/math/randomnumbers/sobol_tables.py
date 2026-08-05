"""Lazy loaders for the Sobol / lattice data tables.

# C++ parity: ql/math/randomnumbers/primitivepolynomials.cpp and the
# anonymous-namespace tables at the top of ql/math/randomnumbers/sobolrsg.cpp
# and ql/math/randomnumbers/latticerules.cpp (v1.43).

The C++ tables are ~7.9 MB of literal data: 21 200 primitive polynomials
modulo two, ten families of Sobol free-direction integers (Joe-Kuo D6 alone
covers 21 200 dimensions) and four lattice-rule generating vectors of 3 600
entries. They live next to this module as plain-text resources under
``data/``, transcribed mechanically by
``migration-harness/generate_sobol_tables.py``; re-running that script against
the pinned C++ checkout reproduces them byte for byte.

Every table is loaded on first use and cached, so importing this module costs
nothing and a program that only ever asks for the 32-dimension Jaeckel family
never parses the 1.4 MB Joe-Kuo D6 file.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

_DATA = "data"


def _read_rows(name: str) -> tuple[tuple[int, ...], ...]:
    """Parse one whitespace-separated integer matrix resource."""
    text = (files(__package__) / _DATA / name).read_text(encoding="utf-8")
    rows: list[tuple[int, ...]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        rows.append(tuple(int(tok) for tok in line.split()))
    return tuple(rows)


@cache
def primitive_polynomials() -> tuple[tuple[int, ...], ...]:
    """``PrimitivePolynomials[degree-1][index]``, ``-1`` terminator dropped.

    # C++ parity: primitivepolynomials.cpp — 18 degrees, 21 200 polynomials.
    """
    return _read_rows("sobol_primitive_polynomials.txt")


@cache
def alt_primitive_polynomials() -> tuple[tuple[int, ...], ...]:
    """``AltPrimitivePolynomials[degree-1][index]``, ``-1`` terminator dropped.

    # C++ parity: sobolrsg.cpp:37-106. Used by the Kuo/Kuo2/Kuo3 and
    # SobolLevitan/SobolLevitanLemieux families, which were generated against
    # a different ordering of the low-degree primitive polynomials.
    """
    return _read_rows("sobol_alt_primitive_polynomials.txt")


@cache
def direction_integer_initializers(family: str) -> tuple[tuple[int, ...], ...]:
    """Free direction integers for ``family``, row ``k`` = dimension ``k + 2``.

    # C++ parity: the ``*initializers`` pointer tables in sobolrsg.cpp; the
    # ``0UL`` end-of-row marker is dropped here and the row length is used
    # instead.
    """
    return _read_rows(f"sobol_init_{family}.txt")


@cache
def lattice_rule_vector(letter: str) -> tuple[int, ...]:
    """Generating vector of lattice rule ``letter`` (``a``/``b``/``c``/``d``).

    # C++ parity: ``latticeA`` .. ``latticeD`` in latticerules.cpp — 3 600
    # entries each.
    """
    return _read_rows(f"lattice_rule_{letter}.txt")[0]


__all__ = [
    "alt_primitive_polynomials",
    "direction_integer_initializers",
    "lattice_rule_vector",
    "primitive_polynomials",
]
