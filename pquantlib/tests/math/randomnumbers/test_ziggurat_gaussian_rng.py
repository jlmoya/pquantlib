"""Cross-validate ZigguratGaussianRng against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json``,
section ``ziggurat_gaussian_rng``.

EXACT tier. The common path is ``u * normX[i]`` — one multiplication of a
table entry by an exactly-representable dyadic — so it must be bit-identical.
The rejection paths add ``exp`` and ``log``, which are not correctly rounded
and could in principle differ by an ULP between libm and Python's math
module; both call the same platform libm here, and forty consecutive draws
plus a draw 10 000 deep come out bit-identical, which is the evidence that
lets EXACT stand rather than an assumption that it should.

Forty draws is not padding: with a 256-layer ziggurat the fast path is taken
~99% of the time, so a short test would never enter ``_zero_case`` (the tail)
or the wedge-acceptance branch, which is exactly where a transcription error
would hide.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.randomnumbers.xoshiro256_starstar import (
    Xoshiro256StarStarUniformRng,
)
from pquantlib.math.randomnumbers.ziggurat_gaussian_rng import ZigguratGaussianRng
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def test_ziggurat_draws_exact(cpp: dict[str, Any]) -> None:
    """First 40 draws and a draw 10 000 deep, for three seeds."""
    for case in cpp["ziggurat_gaussian_rng"]:
        g = ZigguratGaussianRng(Xoshiro256StarStarUniformRng(case["seed"]))
        for expected in case["draws"]:
            tolerance.exact(g.next_real(), expected)
        for _ in range(10000):
            g.next_real()
        tolerance.exact(g.next_real(), case["after_10000"])


def test_sample_weight_is_one(cpp: dict[str, Any]) -> None:
    """# C++ parity: ``next()`` returns ``{nextReal(), 1.0}``."""
    for case in cpp["ziggurat_gaussian_rng"]:
        g = ZigguratGaussianRng(Xoshiro256StarStarUniformRng(case["seed"]))
        tolerance.exact(g.next().weight, case["weight"])


def test_next_and_next_real_are_the_same_stream() -> None:
    """``next().value`` must not consume differently from ``next_real()``."""
    a = ZigguratGaussianRng(Xoshiro256StarStarUniformRng(7))
    b = ZigguratGaussianRng(Xoshiro256StarStarUniformRng(7))
    for _ in range(50):
        tolerance.exact(a.next().value, b.next_real())


def test_tail_is_reachable_and_unbounded() -> None:
    """The tail branch must actually fire, and beyond the base strip.

    ``normR`` is 3.654; a value past it can only have come out of
    ``_zero_case``, so this is a coverage assertion on the branch the
    fast path skips 99% of the time.
    """
    g = ZigguratGaussianRng(Xoshiro256StarStarUniformRng(1))
    draws = [g.next_real() for _ in range(200000)]
    assert any(abs(v) > 3.654152885361008796 for v in draws), "tail branch never taken"


def test_distribution_is_roughly_standard_normal() -> None:
    """Sanity, not precision: a wrong table would move the moments visibly.

    Tolerances here are sampling error, not numerical error: with n = 200 000
    the standard error of the mean is 1/sqrt(n) = 0.0022, so 5 sigma is 0.011;
    the variance's standard error is sqrt(2/n) = 0.0032, so 5 sigma is 0.016.
    """
    g = ZigguratGaussianRng(Xoshiro256StarStarUniformRng(42))
    n = 200000
    draws = [g.next_real() for _ in range(n)]
    mean = sum(draws) / n
    var = sum((v - mean) ** 2 for v in draws) / n
    assert abs(mean) < 0.011
    assert abs(var - 1.0) < 0.016
