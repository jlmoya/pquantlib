"""Cross-validate GammaFunction against the L5-A C++ probe.

Reference: ``migration-harness/references/l5a/foundations.json`` —
``gamma_function`` section.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.gamma_function import GammaFunction
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("l5a/foundations")


def test_log_value_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["gamma_function"]
    xs = [float(x) for x in block["xs"]]
    expected = [float(v) for v in block["log_value"]]
    g = GammaFunction()
    for x, ev in zip(xs, expected, strict=True):
        # Lanczos approximation is bit-identical in pure-Python double
        # arithmetic since we use the same operation order as C++.
        # TIGHT tier: 1e-14 abs / 1e-12 rel.
        tolerance.tight(g.log_value(x), ev)


def test_value_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["gamma_function"]
    xs = [float(x) for x in block["xs"]]
    expected = [float(v) for v in block["value"]]
    g = GammaFunction()
    for x, ev in zip(xs, expected, strict=True):
        tolerance.tight(g.value(x), ev)


def test_log_value_rejects_non_positive() -> None:
    g = GammaFunction()
    with pytest.raises(LibraryException, match="positive argument"):
        g.log_value(0.0)
    with pytest.raises(LibraryException, match="positive argument"):
        g.log_value(-1.5)


def test_gamma_at_integer_is_factorial() -> None:
    # Gamma(n+1) = n!
    g = GammaFunction()
    # The Lanczos approximation accuracy is ~14-15 decimal digits, so
    # n=10 (-> 9! = 362880) is comfortably within TIGHT (1e-12 rel).
    tolerance.tight(g.value(1.0), 1.0)
    tolerance.tight(g.value(2.0), 1.0)
    tolerance.tight(g.value(3.0), 2.0)
    tolerance.tight(g.value(6.0), 120.0)
    # Larger n drifts a few ULPs from the exact integer factorial
    # (n=11 -> 10! = 3,628,800). LOOSE tier here documents the
    # Lanczos truncation error.
    tolerance.loose(
        g.value(11.0),
        3_628_800.0,
        reason="Lanczos approximation truncation at large n",
    )


def test_gamma_at_half() -> None:
    # Gamma(1/2) = sqrt(pi).
    g = GammaFunction()
    tolerance.tight(g.value(0.5), math.sqrt(math.pi))


def test_log_value_camel_alias_round_trip() -> None:
    # The camelCase alias mirrors the C++ method name.
    g = GammaFunction()
    tolerance.exact(g.logValue(2.5), g.log_value(2.5))


def test_value_negative_non_integer() -> None:
    # Reflection formula path: Gamma(-0.5) = -2 * sqrt(pi).
    g = GammaFunction()
    tolerance.tight(g.value(-0.5), -2.0 * math.sqrt(math.pi))
