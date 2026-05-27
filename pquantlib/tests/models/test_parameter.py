"""Parameter hierarchy behavioral + cross-validation tests.

Cross-validates against C++ probe values at
``migration-harness/references/l4a/foundations.json``.

Tolerance choice: TIGHT for direct round-trip values (Constant /
PiecewiseConstant) — these are simple algebraic identities that should
match the C++ output bit-for-bit modulo floating-point representation
of the input literals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.constraint import NoConstraint, PositiveConstraint
from pquantlib.models.parameter import (
    ConstantParameter,
    NullParameter,
    Parameter,
    PiecewiseConstantParameter,
    TermStructureFittingParameter,
    TermStructureFittingParameterImpl,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return reference_reader.load("l4a/foundations")


# ---------------------------------------------------------------------
# NullParameter
# ---------------------------------------------------------------------


def test_null_parameter_value_is_zero() -> None:
    p = NullParameter()
    tolerance.exact(p(0.0), 0.0)
    tolerance.exact(p(5.0), 0.0)
    tolerance.exact(p(100.0), 0.0)


def test_null_parameter_size_is_zero() -> None:
    p = NullParameter()
    assert p.size == 0
    assert p.params.shape == (0,)


# ---------------------------------------------------------------------
# ConstantParameter
# ---------------------------------------------------------------------


def test_constant_parameter_value_matches_cpp(cpp_refs: dict[str, Any]) -> None:
    cp_ref = cpp_refs["constant_parameter"]
    p = ConstantParameter(0.05, NoConstraint())
    tolerance.tight(p(0.0), float(cp_ref["value_at_0"]))
    tolerance.tight(p(3.25), float(cp_ref["value_at_3_25"]))
    tolerance.tight(p(100.0), float(cp_ref["value_at_100"]))
    assert p.size == int(cp_ref["size"])


def test_constant_parameter_constraint_only_overload() -> None:
    # C++ overload ConstantParameter(constraint) — no initial value.
    p = ConstantParameter(NoConstraint())
    assert p.size == 1
    assert p(0.0) == 0.0  # default-initialized
    p.set_param(0, 0.07)
    tolerance.exact(p(0.0), 0.07)


def test_constant_parameter_rejects_constraint_violator() -> None:
    # PositiveConstraint rejects 0 and negatives, so this should raise.
    with pytest.raises(LibraryException, match="invalid value"):
        ConstantParameter(-1.0, PositiveConstraint())


def test_constant_parameter_set_param_updates_value() -> None:
    p = ConstantParameter(0.05, NoConstraint())
    p.set_param(0, 0.10)
    tolerance.exact(p(0.0), 0.10)


def test_constant_parameter_params_array() -> None:
    p = ConstantParameter(0.05, NoConstraint())
    assert p.params.shape == (1,)
    tolerance.tight(float(p.params[0]), 0.05)


# ---------------------------------------------------------------------
# PiecewiseConstantParameter
# ---------------------------------------------------------------------


def test_piecewise_constant_size_is_times_plus_one() -> None:
    p = PiecewiseConstantParameter([1.0, 3.0, 5.0])
    assert p.size == 4  # 3 breaks => 4 slots


def test_piecewise_constant_matches_cpp(cpp_refs: dict[str, Any]) -> None:
    pcp_ref = cpp_refs["piecewise_constant_parameter"]
    p = PiecewiseConstantParameter([1.0, 3.0, 5.0])
    p.set_param(0, 0.10)
    p.set_param(1, 0.20)
    p.set_param(2, 0.30)
    p.set_param(3, 0.40)
    # The cpp probe uses semantic suffixes (_at_0, _at_1_5, etc.).
    tolerance.tight(p(0.0), float(pcp_ref["value_at_0"]))
    tolerance.tight(p(1.0), float(pcp_ref["value_at_1"]))
    tolerance.tight(p(1.5), float(pcp_ref["value_at_1_5"]))
    tolerance.tight(p(3.0), float(pcp_ref["value_at_3"]))
    tolerance.tight(p(3.5), float(pcp_ref["value_at_3_5"]))
    tolerance.tight(p(5.0), float(pcp_ref["value_at_5"]))
    tolerance.tight(p(6.0), float(pcp_ref["value_at_6"]))


def test_piecewise_constant_boundary_uses_upper_bound() -> None:
    # C++ ``upper_bound`` (bisect_right) returns first index strictly
    # greater than t; at exactly t == break, the lookup advances to the
    # next slot.
    p = PiecewiseConstantParameter([1.0, 3.0])
    p.set_param(0, 1.0)
    p.set_param(1, 2.0)
    p.set_param(2, 3.0)
    # t=1 hits the boundary; upper_bound returns 1 -> params[1]=2.0.
    tolerance.exact(p(1.0), 2.0)


def test_piecewise_constant_t_below_first_break() -> None:
    p = PiecewiseConstantParameter([1.0, 3.0])
    p.set_param(0, 1.0)
    p.set_param(1, 2.0)
    p.set_param(2, 3.0)
    tolerance.exact(p(-100.0), 1.0)


def test_piecewise_constant_empty_times() -> None:
    # Edge case: no breaks => 1-slot piecewise (i.e. effectively constant).
    p = PiecewiseConstantParameter([])
    p.set_param(0, 0.42)
    tolerance.exact(p(0.0), 0.42)
    tolerance.exact(p(100.0), 0.42)


# ---------------------------------------------------------------------
# TermStructureFittingParameter
# ---------------------------------------------------------------------


def test_term_structure_fitting_default_impl() -> None:
    p = TermStructureFittingParameter()
    assert p.size == 0
    # No values set => lookup raises.
    with pytest.raises(RuntimeError, match="fitting parameter not set"):
        p(1.0)


def test_term_structure_fitting_set_change_reset_roundtrip() -> None:
    impl = TermStructureFittingParameterImpl()
    p = TermStructureFittingParameter(impl)
    impl.set(1.0, 0.10)
    impl.set(2.0, 0.20)
    tolerance.exact(p(1.0), 0.10)
    tolerance.exact(p(2.0), 0.20)
    impl.change(0.25)  # overwrites last
    tolerance.exact(p(2.0), 0.25)
    impl.reset()
    with pytest.raises(RuntimeError):
        p(1.0)


# ---------------------------------------------------------------------
# Base class behaviors
# ---------------------------------------------------------------------


def test_parameter_default_ctor_is_empty() -> None:
    # C++ parity: parameter.hpp:48-49 default ctor — no Impl, NoConstraint,
    # empty params array.
    p = Parameter()
    assert p.size == 0
    assert p.impl is None
    assert isinstance(p.constraint, NoConstraint)


def test_parameter_call_without_impl_raises() -> None:
    p = Parameter()
    with pytest.raises(RuntimeError, match="no implementation"):
        p(1.0)


def test_parameter_test_params_uses_constraint() -> None:
    p = ConstantParameter(2.0, PositiveConstraint())
    assert p.test_params(np.array([1.0]))
    assert not p.test_params(np.array([-1.0]))
