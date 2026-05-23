"""Exception hierarchy.

# C++ parity: ql/errors.hpp (v1.42.1) — class QuantLib::Error.
"""

from __future__ import annotations


class LibraryException(RuntimeError):  # noqa: N818  # parity with jquantlib LibraryException naming (see phase0-design.md decision #11)
    """Base class for all PQuantLib exceptions.

    Carries an optional ``where`` source-location hint (e.g. C++ parity
    file:line) for diagnostics. Construction has NO side effects — no
    stderr print, no logging — guarding against the jquantlib pre-de95bb17
    bug where ``QL.error(this)`` in the ctor leaked caught-for-control-flow
    exceptions to stderr.
    """

    def __init__(self, message: str, *, where: str | None = None) -> None:
        super().__init__(message)
        self.where: str | None = where
