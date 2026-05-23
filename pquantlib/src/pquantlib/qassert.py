"""C++ ``QL_REQUIRE`` / ``QL_FAIL`` analogues as free functions.

# C++ parity: ql/errors.hpp QL_REQUIRE / QL_FAIL macros (v1.42.1).

Usage:

    from pquantlib import qassert
    qassert.require(notional > 0, "notional must be positive")
    qassert.fail("unreachable code path")
"""

from __future__ import annotations

from typing import NoReturn

from pquantlib.exceptions import LibraryException


def require(condition: object, message: str, *, where: str | None = None) -> None:
    """Raise ``LibraryException(message)`` if ``condition`` is falsy."""
    if not condition:
        raise LibraryException(message, where=where)


def fail(message: str, *, where: str | None = None) -> NoReturn:
    """Unconditionally raise ``LibraryException(message)``."""
    raise LibraryException(message, where=where)
