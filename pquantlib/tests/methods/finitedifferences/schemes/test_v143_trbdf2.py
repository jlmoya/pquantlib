"""Cross-validate TRBDF2 (trbdf2.hpp) against the C++ v1.43 probe.

# C++ parity: ql/methods/finitedifferences/trbdf2.hpp @ v1.43 (6b57206e0).

Expected values come from ``block_trbdf2_old()`` in
``migration-harness/cpp/probes/v143_methods_schemes/probe.cpp``, which drives
``TRBDF2<TridiagonalOperator>`` — the only instantiation the C++ library
supports, since the class needs ``identity`` plus the operator algebra.

The conforming Python operator is ``pquantlib_helpers``'
:class:`TridiagonalOperator`: this repository hosts the retired pre-1.0 FD
framework in the helpers package (Phase 12 decision), while
``ql/methods/finitedifferences/trbdf2.hpp`` itself is core C++ and so lands in
core Python. ``TRBDF2`` is therefore written against a structural protocol and
this test supplies the operator — which incidentally cross-validates
``TridiagonalOperator``, ``DirichletBC`` and ``NeumannBC`` against C++ for the
first time (they were previously pinned against Java only).

Four variants, all TIGHT:

* time-constant and time-dependent operator — the latter is what exercises the
  three ``if (L_.isTimeDependent())`` re-derivations inside ``step``;
* with and without boundary conditions — the "with" variants also pin that the
  BDF2 parts are mutated in place by ``applyBeforeApplying`` and carried into
  the next step in that mutated state, which is C++ behaviour.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
import pytest

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.trbdf2 import TRBDF2
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib_helpers.methods.finitedifferences.boundary_condition import (
    DirichletBC,
    NeumannBC,
    Side,
)
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import TridiagonalOperator
from tests.methods.finitedifferences.schemes._probe_fixture import probe_start

REF: dict[str, Any] = reference_reader.load("v143/methods/schemes")

#: # C++ parity: ``const Size TRI_N = 7``.
TRI_N: Final[int] = 7


def _assert_tight(actual: Array, key: str) -> None:
    expected = [float(v) for v in REF[key]]
    assert len(actual) == len(expected), f"{key}: length {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        tight(float(a), e, reason=f"{key}[{i}]")


def _fill(op: TridiagonalOperator, t: float) -> None:
    """# C++ parity: ``TriTimeSetter::setTime`` / ``makeConstantTridiagonal``."""
    op.set_first_row(2.0 + 0.5 * t, -1.0)
    for i in range(1, TRI_N - 1):
        op.set_mid_row(i, -1.0 + 0.0625 * i, 2.0 + 0.5 * t, -1.0 + 0.03125 * i)
    op.set_last_row(-1.0, 2.0 + 0.5 * t)


class _TriTimeSetter:
    """# C++ parity: ``class TriTimeSetter : public TridiagonalOperator::TimeSetter``."""

    def set_time(self, t: float, op: TridiagonalOperator) -> None:
        _fill(op, t)


def _make_constant() -> TridiagonalOperator:
    op = TridiagonalOperator(TRI_N)
    _fill(op, 0.0)
    return op


def _make_time_dependent() -> TridiagonalOperator:
    """# C++ parity: ``class TimeDepTridiagonalOperator``."""
    op = TridiagonalOperator(TRI_N)
    op.time_setter = _TriTimeSetter()
    _fill(op, 0.0)
    return op


def _make_bcs(with_bc: bool) -> list[DirichletBC | NeumannBC]:
    """# C++ parity: ``makeTriBcSet``."""
    if not with_bc:
        return []
    return [DirichletBC(0.5, Side.Lower), NeumannBC(0.25, Side.Upper)]


def _run(op: TridiagonalOperator, with_bc: bool, prefix: str) -> None:
    """# C++ parity: ``runTrbdf2`` — emit the state after step 1 and step 3."""
    evolver = TRBDF2(op, _make_bcs(with_bc))
    evolver.set_step(0.25)

    a = probe_start()
    t = 1.0
    a = evolver.step(a, t)
    _assert_tight(a, f"{prefix}_step1")
    for _ in range(2):
        t -= 0.25
        a = evolver.step(a, t)
    _assert_tight(a, f"{prefix}_step3")


def test_tridiagonal_fixture_matches_cpp() -> None:
    """Pin the operator itself before pinning the evolver that consumes it."""
    op = _make_constant()
    _assert_tight(op.lower_diagonal(), "tri_const_lower")
    _assert_tight(op.diagonal(), "tri_const_diag")
    _assert_tight(op.upper_diagonal(), "tri_const_upper")
    assert op.is_time_dependent() == bool(REF["tri_const_is_time_dependent"])
    assert _make_time_dependent().is_time_dependent() == bool(REF["tri_timedep_is_time_dependent"])


def test_trbdf2_alpha_is_two_minus_sqrt_two() -> None:
    """``alpha`` is hard-coded because the BDF2 half reuses the implicit part.

    # C++ parity: ``alpha_(2.0-sqrt(2.0))`` in the constructor.
    """
    evolver = TRBDF2(_make_constant())
    # Not part of the public C++ surface, so read through the slot rather than
    # inventing an accessor the C++ class does not have.
    tight(evolver._alpha, 2.0 - math.sqrt(2.0))  # pyright: ignore[reportPrivateUsage]


def test_trbdf2_constant_operator() -> None:
    _run(_make_constant(), False, "trbdf2_const")


def test_trbdf2_constant_operator_with_boundary_conditions() -> None:
    _run(_make_constant(), True, "trbdf2_const_bc")


def test_trbdf2_time_dependent_operator() -> None:
    _run(_make_time_dependent(), False, "trbdf2_timedep")


def test_trbdf2_time_dependent_operator_with_boundary_conditions() -> None:
    _run(_make_time_dependent(), True, "trbdf2_timedep_bc")


def test_trbdf2_time_dependence_actually_changes_the_answer() -> None:
    """The constant and time-dependent runs must not coincide.

    Guards the fixture: if the ``is_time_dependent`` branches were skipped the
    two blocks would produce the same numbers and every other assertion in this
    module would still pass.
    """
    assert REF["trbdf2_const_step3"] != REF["trbdf2_timedep_step3"]


def test_trbdf2_does_not_mutate_the_input_array() -> None:
    """The Python port returns a new array instead of mutating in place.

    C++ takes ``array_type&`` and overwrites it; the Python FD package returns
    the new array, so the caller's vector must come back untouched.
    """
    evolver = TRBDF2(_make_constant())
    evolver.set_step(0.25)
    a = probe_start()
    before = np.array(a, dtype=np.float64, copy=True)
    evolver.step(a, 1.0)
    assert np.array_equal(a, before)


def test_trbdf2_rejects_an_operator_of_the_wrong_size() -> None:
    """The identity is built from ``op.size()``; a size-0 operator cannot work."""
    with pytest.raises(Exception, match="identity"):
        TRBDF2(TridiagonalOperator(0))
