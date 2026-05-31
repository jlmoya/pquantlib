"""Tests for the W10-C alpha-form calibration primitives.

Cross-validates against ``migration-harness/references/cluster/w10c.json``.

C++ parity:
  ql/math/quadratic.{hpp,cpp}
  ql/models/marketmodels/models/alphaform.hpp
  ql/models/marketmodels/models/alphaformconcrete.{hpp,cpp}
  ql/models/marketmodels/models/alphafinder.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.quadratic import Quadratic
from pquantlib.models.marketmodels.models.alpha_finder import AlphaFinder
from pquantlib.models.marketmodels.models.alpha_form_concrete import (
    AlphaFormInverseLinear,
    AlphaFormLinearHyperbolic,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10c")


# --- Quadratic (math primitive) --------------------------------------------


def test_quadratic_roots_real() -> None:
    # x^2 - 3x + 2 = (x-1)(x-2): roots 1, 2; turning point 1.5; vertex value -0.25
    q = Quadratic(1.0, -3.0, 2.0)
    x, y, real = q.roots()
    assert real
    tight(x, 1.0)
    tight(y, 2.0)
    tight(q.turning_point(), 1.5)
    tight(q.value_at_turning_point(), -0.25)
    tight(q(0.0), 2.0)
    tight(q.discriminant(), 1.0)


def test_quadratic_roots_complex() -> None:
    # x^2 + 1: no real roots, returns turning point (0) twice
    q = Quadratic(1.0, 0.0, 1.0)
    x, y, real = q.roots()
    assert not real
    tight(x, 0.0)
    tight(y, 0.0)


# --- AlphaForm concretes ----------------------------------------------------

_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5]


def test_alpha_form_inverse_linear(ref: dict[str, Any]) -> None:
    form = AlphaFormInverseLinear(_TIMES, 0.3)
    tight(form(0), ref["ail_a0_i0"])
    tight(form(2), ref["ail_a0_i2"])
    tight(form(4), ref["ail_a0_i4"])
    form.set_alpha(-0.2)
    tight(form(3), ref["ail_a1_i3"])


def test_alpha_form_linear_hyperbolic(ref: dict[str, Any]) -> None:
    form = AlphaFormLinearHyperbolic(_TIMES, 0.4)
    tight(form(0), ref["alh_a0_i0"])
    tight(form(2), ref["alh_a0_i2"])
    tight(form(4), ref["alh_a0_i4"])
    form.set_alpha(-0.15)
    tight(form(3), ref["alh_a1_i3"])


# --- AlphaFinder ------------------------------------------------------------


def _make_finder() -> AlphaFinder:
    form = AlphaFormLinearHyperbolic([0.5, 1.0])
    return AlphaFinder(form)


def test_alpha_finder_solve(ref: dict[str, Any]) -> None:
    finder = _make_finder()
    ratetwovols = [0.0, 0.0]
    ok, alpha, a, b = finder.solve(
        0.0,
        0,
        [0.20],
        [0.18, 0.22],
        [0.85],
        0.6,
        0.4,
        0.02,
        1e-12,
        1.0,
        -1.0,
        100,
        ratetwovols,
    )
    assert ok == bool(ref["af_solve_ok"])
    tight(alpha, ref["af_alpha"])
    tight(a, ref["af_a"])
    tight(b, ref["af_b"])
    tight(ratetwovols[0], ref["af_v0"])
    tight(ratetwovols[1], ref["af_v1"])


def test_alpha_finder_solve_with_max_homogeneity(ref: dict[str, Any]) -> None:
    finder = _make_finder()
    ratetwovols = [0.0, 0.0]
    ok, alpha, a, b = finder.solve_with_max_homogeneity(
        0.0,
        0,
        [0.20],
        [0.18, 0.22],
        [0.85],
        0.6,
        0.4,
        0.02,
        1e-12,
        1.0,
        -1.0,
        100,
        ratetwovols,
    )
    # C++ probe reports this synthetic fixture has no feasible solution
    # (afh_solve_ok == 0) -> all outputs left at zero.
    assert ok == bool(ref["afh_solve_ok"])
    tight(alpha, ref["afh_alpha"])
    tight(a, ref["afh_a"])
    tight(b, ref["afh_b"])
