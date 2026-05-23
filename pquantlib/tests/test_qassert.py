"""Tests for pquantlib.qassert (require / fail helpers)."""

from __future__ import annotations

import pytest

from pquantlib import qassert
from pquantlib.exceptions import LibraryException


def test_require_true_returns_none() -> None:
    assert qassert.require(True, "should not raise") is None


def test_require_truthy_returns_none() -> None:
    assert qassert.require(1, "should not raise") is None
    assert qassert.require("nonempty", "should not raise") is None
    assert qassert.require([0], "should not raise") is None


def test_require_false_raises_library_exception() -> None:
    with pytest.raises(LibraryException, match="negative discount factor"):
        qassert.require(False, "negative discount factor")


def test_require_falsy_raises() -> None:
    falsy_values: tuple[object, ...] = (0, "", [], None, 0.0)
    for falsy in falsy_values:
        with pytest.raises(LibraryException, match="bad"):
            qassert.require(falsy, "bad")


def test_require_passes_where_through() -> None:
    with pytest.raises(LibraryException) as info:
        qassert.require(False, "boom", where="ql/math/foo.hpp:42")
    assert info.value.where == "ql/math/foo.hpp:42"


def test_fail_always_raises() -> None:
    with pytest.raises(LibraryException, match="unreachable"):
        qassert.fail("unreachable")


def test_fail_passes_where_through() -> None:
    with pytest.raises(LibraryException) as info:
        qassert.fail("boom", where="ql/math/foo.hpp:42")
    assert info.value.where == "ql/math/foo.hpp:42"
