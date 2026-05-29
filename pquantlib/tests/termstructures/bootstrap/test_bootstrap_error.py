"""Tests for BootstrapError — diagnostic exception class.

# C++ parity: ql/termstructures/bootstraperror.hpp (v1.42.1) — note the
# task-spec reframe: in PQuantLib BootstrapError is a typed Python
# exception, not a deprecated callable functor (the C++ class is
# deprecated in 1.40 and replaced by lambdas).
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.bootstrap.bootstrap_error import BootstrapError


def test_bootstrap_error_is_library_exception() -> None:
    """BootstrapError subclasses LibraryException for try/except interop."""
    err = BootstrapError("something failed")
    assert isinstance(err, LibraryException)
    assert isinstance(err, RuntimeError)
    assert "something failed" in str(err)


def test_bootstrap_error_carries_helper_diagnostics() -> None:
    """All optional fields are stored as attributes for callers to inspect."""
    sentinel_helper = object()
    sentinel_curve = object()
    err = BootstrapError(
        "convergence not reached",
        helper_index=3,
        helper=sentinel_helper,
        last_residual=1.5e-3,
        curve=sentinel_curve,
        accuracy=1e-12,
    )
    assert err.helper_index == 3
    assert err.helper is sentinel_helper
    assert err.last_residual == 1.5e-3
    assert err.curve is sentinel_curve
    assert err.accuracy == 1e-12


def test_bootstrap_error_message_includes_context() -> None:
    """The exception text surfaces helper_index / residual / accuracy."""
    err = BootstrapError(
        "boom",
        helper_index=2,
        last_residual=1e-4,
        accuracy=1e-12,
    )
    msg = str(err)
    assert "helper_index=2" in msg
    assert "1e-04" in msg or "0.0001" in msg
    assert "1e-12" in msg


def test_bootstrap_error_raises_and_catches() -> None:
    """The exception is throwable and catchable via ``except BootstrapError``."""
    def failing_op() -> None:
        raise BootstrapError("failed", helper_index=0)

    with pytest.raises(BootstrapError) as exc_info:
        failing_op()
    assert exc_info.value.helper_index == 0


def test_bootstrap_error_can_be_caught_as_library_exception() -> None:
    """A caller that catches LibraryException still sees BootstrapError."""
    def failing_op() -> None:
        raise BootstrapError("failed", helper_index=1)

    with pytest.raises(LibraryException) as exc_info:
        failing_op()
    # The actual exception is still a BootstrapError.
    assert isinstance(exc_info.value, BootstrapError)


def test_bootstrap_error_default_attribute_values() -> None:
    """Optional fields default to sentinel values (no helper context)."""
    err = BootstrapError("global failure")
    assert err.helper_index == -1
    assert err.helper is None
    assert err.last_residual is None
    assert err.curve is None
    assert err.accuracy is None
