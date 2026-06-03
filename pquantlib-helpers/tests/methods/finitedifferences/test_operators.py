"""Stencil + cross-validation tests for the D-operators and BSMOperator (WS3-FD1).

# Retired-API compat layer — see package docstring.

The BSMOperator coefficient bands + applyTo are cross-validated TIGHT against
``cluster/ws3fd1.json`` (``bsm_operator`` block). The D-operators (DPlus /
DMinus / DZero / DPlusMinus) have no C++ v1.42.1 class, so their matrix
structure is checked against hand-derived first-/second-order stencils with an
inline derivation comment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pquantlib.testing import reference_reader, tolerance
from pquantlib_helpers.methods.finitedifferences.bsm_operator import BSMOperator
from pquantlib_helpers.methods.finitedifferences.d_minus import DMinus
from pquantlib_helpers.methods.finitedifferences.d_plus import DPlus
from pquantlib_helpers.methods.finitedifferences.d_plus_minus import DPlusMinus
from pquantlib_helpers.methods.finitedifferences.d_zero import DZero

_REF: dict[str, Any] = reference_reader.load("cluster/ws3fd1")


def _floats(key: str) -> list[float]:
    return [float(x) for x in _REF["bsm_operator"][key]]


# --- BSMOperator vs C++ -----------------------------------------------------


def test_bsm_coefficient_bands_match_cpp() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    for got, want in zip(op.diagonal(), _floats("diagonal"), strict=True):
        tolerance.tight(float(got), want)
    for got, want in zip(op.lower_diagonal(), _floats("lower"), strict=True):
        tolerance.tight(float(got), want)
    for got, want in zip(op.upper_diagonal(), _floats("upper"), strict=True):
        tolerance.tight(float(got), want)


def test_bsm_apply_to_matches_cpp() -> None:
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    v = np.array(_floats("v"), dtype=np.float64)
    for got, want in zip(op.apply_to(v), _floats("applyTo"), strict=True):
        tolerance.tight(float(got), want)


def test_bsm_first_and_last_diagonal_are_zero() -> None:
    # set_mid_rows leaves the endpoints untouched — faithful to old QuantLib.
    op = BSMOperator(7, 0.1, 0.05, 0.01, 0.20)
    assert op.diagonal()[0] == 0.0
    assert op.diagonal()[-1] == 0.0


# --- D-operator stencils (hand-derived) ------------------------------------
#
# For step h and gridPoints n, the interior rows are:
#   DPlus     (forward  D+):  lower= 0,        diag= -1/h,        upper= +1/h
#   DMinus    (backward D-):  lower= -1/h,     diag= +1/h,        upper= 0
#   DZero     (central  D0):  lower= -1/(2h),  diag= 0,           upper= +1/(2h)
#   DPlusMinus(second   D+D-):lower= +1/h^2,   diag= -2/h^2,      upper= +1/h^2
# All four set first/last rows to the linear-extrapolation stencil (or zero).


def test_d_plus_interior_stencil() -> None:
    h = 0.5
    op = DPlus(5, h)
    # interior row i=2: lower[1], diag[2], upper[2]
    tolerance.tight(float(op.lower_diagonal()[1]), 0.0)
    tolerance.tight(float(op.diagonal()[2]), -1.0 / h)
    tolerance.tight(float(op.upper_diagonal()[2]), 1.0 / h)


def test_d_plus_boundary_rows() -> None:
    h = 0.5
    op = DPlus(5, h)
    tolerance.tight(float(op.diagonal()[0]), -1.0 / h)
    tolerance.tight(float(op.upper_diagonal()[0]), 1.0 / h)
    tolerance.tight(float(op.lower_diagonal()[-1]), -1.0 / h)
    tolerance.tight(float(op.diagonal()[-1]), 1.0 / h)


def test_d_minus_interior_stencil() -> None:
    h = 0.5
    op = DMinus(5, h)
    tolerance.tight(float(op.lower_diagonal()[1]), -1.0 / h)
    tolerance.tight(float(op.diagonal()[2]), 1.0 / h)
    tolerance.tight(float(op.upper_diagonal()[2]), 0.0)


def test_d_zero_interior_stencil() -> None:
    h = 0.5
    op = DZero(5, h)
    tolerance.tight(float(op.lower_diagonal()[1]), -1.0 / (2.0 * h))
    tolerance.tight(float(op.diagonal()[2]), 0.0)
    tolerance.tight(float(op.upper_diagonal()[2]), 1.0 / (2.0 * h))


def test_d_plus_minus_interior_stencil() -> None:
    h = 0.5
    op = DPlusMinus(5, h)
    tolerance.tight(float(op.lower_diagonal()[1]), 1.0 / (h * h))
    tolerance.tight(float(op.diagonal()[2]), -2.0 / (h * h))
    tolerance.tight(float(op.upper_diagonal()[2]), 1.0 / (h * h))


def test_d_plus_minus_boundary_rows_are_zero() -> None:
    op = DPlusMinus(5, 0.5)
    assert op.diagonal()[0] == 0.0
    assert op.upper_diagonal()[0] == 0.0
    assert op.lower_diagonal()[-1] == 0.0
    assert op.diagonal()[-1] == 0.0


def test_d_plus_minus_applied_to_quadratic_recovers_second_derivative() -> None:
    # u(x)=x^2 on a uniform grid has constant second derivative 2; the central
    # second-difference operator recovers 2 on every interior node.
    h = 0.25
    n = 7
    x = np.arange(n, dtype=np.float64) * h
    u = x * x
    op = DPlusMinus(n, h)
    result = op.apply_to(u)
    for i in range(1, n - 1):
        tolerance.tight(float(result[i]), 2.0)
