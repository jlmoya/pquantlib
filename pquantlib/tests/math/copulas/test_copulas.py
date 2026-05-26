"""Cross-validate the 12 L1-B copulas against the C++ probe.

Probe key: cluster/b -> "copulas".

For each copula, the JSON encodes ``param`` (theta or alpha) and a grid
of ``{x, y, v}`` test points. We instantiate the Python class with the
same parameter and assert ``copula(x, y) == v`` at TIGHT tolerance —
all formulas are closed-form, so TIGHT is achievable.

Two copulas (Marshall-Olkin, Husler-Reiss) take additional parameters
or evaluate transcendentals; both still pass at TIGHT against the C++
reference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.copulas.ali_mikhail_haq import AliMikhailHaqCopula
from pquantlib.math.copulas.clayton import ClaytonCopula
from pquantlib.math.copulas.farlie_gumbel_morgenstern import FarlieGumbelMorgensternCopula
from pquantlib.math.copulas.frank import FrankCopula
from pquantlib.math.copulas.galambos import GalambosCopula
from pquantlib.math.copulas.gumbel import GumbelCopula
from pquantlib.math.copulas.husler_reiss import HuslerReissCopula
from pquantlib.math.copulas.independent import IndependentCopula
from pquantlib.math.copulas.marshall_olkin import MarshallOlkinCopula
from pquantlib.math.copulas.max import MaxCopula
from pquantlib.math.copulas.min import MinCopula
from pquantlib.math.copulas.plackett import PlackettCopula
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/b")


def _assert_grid(copula: Callable[[float, float], float], values: list[dict[str, float]]) -> None:
    for case in values:
        x = float(case["x"])
        y = float(case["y"])
        expected = float(case["v"])
        tolerance.tight(copula(x, y), expected)


def test_clayton_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["clayton"]
    _assert_grid(ClaytonCopula(theta=float(block["param"])), block["values"])


def test_gumbel_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["gumbel"]
    _assert_grid(GumbelCopula(theta=float(block["param"])), block["values"])


def test_frank_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["frank"]
    _assert_grid(FrankCopula(theta=float(block["param"])), block["values"])


def test_ali_mikhail_haq_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["ali"]
    _assert_grid(AliMikhailHaqCopula(theta=float(block["param"])), block["values"])


def test_farlie_gumbel_morgenstern_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["fgm"]
    _assert_grid(FarlieGumbelMorgensternCopula(theta=float(block["param"])), block["values"])


def test_galambos_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["galambos"]
    _assert_grid(GalambosCopula(theta=float(block["param"])), block["values"])


def test_husler_reiss_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["husler"]
    # Husler-Reiss composes the cumulative-normal CDF (math.erf-backed) with
    # x^Phi(...) * y^Phi(...). Empirically diffs are ~1e-16 vs the C++
    # reference — well under TIGHT — because both routes hit the same
    # IEEE-754 erf within a few ULPs and the outer pow contracts the error.
    _assert_grid(HuslerReissCopula(theta=float(block["param"])), block["values"])


def test_independent_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["indep"]
    _assert_grid(IndependentCopula(), block["values"])


def test_marshall_olkin_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["marshall"]
    # Probe shape: the C++ probe passes ``MarshallOlkinCopula(0.3, 0.4)`` —
    # see migration-harness/cpp/probes/cluster_b/probe.cpp. The JSON's
    # single ``param`` field stores alpha1 only; alpha2=0.4 is hard-coded.
    _assert_grid(MarshallOlkinCopula(alpha1=0.3, alpha2=0.4), block["values"])


def test_max_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["maxc"]
    _assert_grid(MaxCopula(), block["values"])


def test_min_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["minc"]
    _assert_grid(MinCopula(), block["values"])


def test_plackett_matches_cpp(cpp: dict[str, Any]) -> None:
    block = cpp["copulas"]["plackett"]
    _assert_grid(PlackettCopula(theta=float(block["param"])), block["values"])


# --- range-validation guards ---------------------------------------------


def test_clayton_rejects_zero_theta() -> None:
    with pytest.raises(LibraryException):
        ClaytonCopula(theta=0.0)


def test_gumbel_rejects_theta_below_one() -> None:
    with pytest.raises(LibraryException):
        GumbelCopula(theta=0.5)


def test_clayton_rejects_x_above_one() -> None:
    c = ClaytonCopula(theta=0.5)
    with pytest.raises(LibraryException):
        c(1.5, 0.5)
