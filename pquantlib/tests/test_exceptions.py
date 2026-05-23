"""Tests for pquantlib.exceptions.LibraryException."""

from __future__ import annotations

import io
import sys

import pytest

from pquantlib.exceptions import LibraryException


def test_library_exception_is_runtime_error() -> None:
    assert issubclass(LibraryException, RuntimeError)


def test_library_exception_carries_message() -> None:
    exc = LibraryException("boom")
    assert str(exc) == "boom"


def test_library_exception_default_where_is_none() -> None:
    exc = LibraryException("boom")
    assert exc.where is None


def test_library_exception_carries_where() -> None:
    exc = LibraryException("boom", where="ql/math/foo.hpp:42")
    assert exc.where == "ql/math/foo.hpp:42"


def test_library_exception_construction_has_no_side_effect_on_stderr() -> None:
    """Regression guard for the jquantlib pre-de95bb17 bug.

    The original 2007-era jquantlib LibraryException called ``QL.error(this)``
    in its constructors, unconditionally printing every constructed instance
    plus full stack trace to stderr — even ones that were caught immediately
    for control flow. PQuantLib starts clean: ctor must do nothing observable.
    """
    captured = io.StringIO()
    real_stderr = sys.stderr
    sys.stderr = captured
    try:
        _ = LibraryException("constructed without raising")
    finally:
        sys.stderr = real_stderr
    assert captured.getvalue() == ""


def test_library_exception_raisable_and_catchable() -> None:
    with pytest.raises(LibraryException, match="boom"):
        raise LibraryException("boom")
