"""Cross-validate ``ModTripleBandLinearOp`` against C++ v1.43.

Reference: ``migration-harness/references/v143/methods/core.json``, emitted by
``v143_methods_core_probe`` (cases ``tb_dir*_mod_*``).

C++ exposes the three bands of a ``TripleBandLinearOp`` as reference-returning
overloads (``Real& lower(Size i)``); Python spells the lvalue form as an
explicit setter. The probe reads two rows straight out of a
``FirstDerivativeOp``, patches the first row and one interior diagonal, then
re-runs ``apply`` and ``solve_splitting`` so a wrong band-to-index mapping
cannot pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.first_derivative_op import (
    FirstDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.mod_triple_band_linear_op import (
    ModTripleBandLinearOp,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/methods/core")


def _assert_array(actual: Array, expected: list[float]) -> None:
    assert actual.size == len(expected)
    for a, e in zip(actual.tolist(), expected, strict=True):
        tight(float(a), float(e))


def _probe_vector(n: int, seed: float) -> Array:
    """Reproduce the probe's synthetic vector exactly (same closed form)."""
    i = np.arange(n, dtype=np.float64)
    return np.sin(seed + 0.37 * i) + 1.5 + 0.011 * i * i


@pytest.fixture(scope="module")
def mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(-1.0, 2.0, 4),
        Uniform1dMesher(0.5, 3.5, 3),
    )


@pytest.mark.parametrize("direction", [0, 1])
def test_mod_triple_band_matches_cpp(
    cpp: dict[str, Any], mesher: FdmMesherComposite, direction: int
) -> None:
    n = mesher.layout().size()
    v = _probe_vector(n, 0.3)
    d = FirstDerivativeOp(direction, mesher)
    p = f"tb_dir{direction}_"

    mod = ModTripleBandLinearOp(op=d)
    _assert_array(
        np.array([mod.lower(0), mod.diag(0), mod.upper(0)]),
        cpp[p + "mod_row0"],
    )
    _assert_array(
        np.array([mod.lower(n - 1), mod.diag(n - 1), mod.upper(n - 1)]),
        cpp[p + "mod_row_last"],
    )

    mod.set_lower(0, 0.125)
    mod.set_diag(0, -2.5)
    mod.set_upper(0, 3.75)
    mod.set_diag(n // 2, mod.diag(n // 2) + 1.0)
    _assert_array(mod.apply(v), cpp[p + "mod_patched_apply"])
    _assert_array(mod.solve_splitting(v, 0.3, 1.0), cpp[p + "mod_patched_solve"])
