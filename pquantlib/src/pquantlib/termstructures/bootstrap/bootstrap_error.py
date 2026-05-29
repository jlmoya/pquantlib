"""BootstrapError — diagnostic exception for failed bootstrap solves.

# C++ parity: ql/termstructures/bootstraperror.hpp (v1.42.1) — the C++
# ``BootstrapError<Curve>`` template is a *deprecated* (1.40) callable
# functor used by the legacy bootstrap loop ``operator()(guess)``. The
# task spec for Phase 11 W2-B reframes ``BootstrapError`` as a Python
# diagnostic exception carrying the same information the C++ functor
# captures (curve handle, helper, segment, last residual). The
# IterativeBootstrap port at ``iterative_bootstrap.py`` raises this
# exception when its Brent-loop fails to converge after
# ``max_iterations`` passes, when the per-pillar Brent search reports a
# non-bracketing failure, or when a helper's quote becomes invalid
# mid-iteration.

This is the *exception* form — distinct from a deprecated functor —
and is the form callers should ``except`` on when they want
per-helper diagnostics out of a failed curve build.

Usage:

    try:
        bootstrap.calculate()
    except BootstrapError as e:
        # e.helper_index, e.helper, e.last_residual, e.curve, e.message
        ...
"""

from __future__ import annotations

from typing import Any

from pquantlib.exceptions import LibraryException


class BootstrapError(LibraryException):
    """Exception capturing what an iterative bootstrap failed on.

    Attributes:
        helper_index: 0-based index of the bootstrap helper that failed
            (or ``-1`` if the failure was global, e.g. convergence).
        helper: the bootstrap helper itself (``BootstrapHelper`` instance)
            or ``None`` if the failure was not helper-specific.
        last_residual: the helper's ``quote_error()`` at the last attempted
            solver iterate, if known.
        curve: the curve being bootstrapped (typically a
            ``PiecewiseYieldCurve`` or ``PiecewiseDefaultCurve``); kept as
            ``object`` to avoid a hard dep on a concrete TS class.
        accuracy: the convergence accuracy that was being attempted.

    Args:
        message: human-readable diagnostic.
        helper_index: see attributes.
        helper: see attributes.
        last_residual: see attributes.
        curve: see attributes.
        accuracy: see attributes.
        where: optional source location, forwarded to
            :class:`LibraryException`.
    """

    helper_index: int
    helper: object | None
    last_residual: float | None
    curve: object | None
    accuracy: float | None

    def __init__(
        self,
        message: str,
        *,
        helper_index: int = -1,
        helper: Any = None,
        last_residual: float | None = None,
        curve: Any = None,
        accuracy: float | None = None,
        where: str | None = None,
    ) -> None:
        # Compose a richer message that surfaces the diagnostic
        # context — matches the C++ ``QL_FAIL`` style of streaming
        # ``segment_`` into the message text.
        context_bits: list[str] = []
        if helper_index >= 0:
            context_bits.append(f"helper_index={helper_index}")
        if last_residual is not None:
            context_bits.append(f"last_residual={last_residual!r}")
        if accuracy is not None:
            context_bits.append(f"accuracy={accuracy!r}")
        full_message = (
            f"{message} [{', '.join(context_bits)}]" if context_bits else message
        )
        super().__init__(full_message, where=where)
        self.helper_index = helper_index
        self.helper = helper
        self.last_residual = last_residual
        self.curve = curve
        self.accuracy = accuracy


__all__ = ["BootstrapError"]
