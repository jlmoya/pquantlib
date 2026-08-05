"""Cross-validate ``Projection`` / ``ProjectedCostFunction`` / ``ProjectedConstraint``.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block H.

Everything here is EXACT: projection and inclusion only copy doubles between
slots, and the cost-function values are one evaluation of the wrapped
objective on the reassembled vector.

The mask convention is the C++ one and worth stating once: ``True`` in
``fix_parameters`` means the coordinate is HELD FIXED. A consequence pinned
below is that ``include`` restores the fixed coordinates from the vector the
projection was built with, so if one of those violates the inner constraint,
``ProjectedConstraint.test`` returns ``False`` for every free vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.constraint import Constraint, NonhomogeneousBoundaryConstraint
from pquantlib.math.optimization.projected_constraint import ProjectedConstraint
from pquantlib.math.optimization.projected_cost_function import ProjectedCostFunction
from pquantlib.math.optimization.projection import Projection
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import RosenbrockResiduals


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def _arr(values: list[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _assert_all_exact(
    got: npt.NDArray[np.float64], expected: list[float]
) -> None:
    for a, b in zip(got, expected, strict=True):
        tolerance.exact(float(a), float(b))


# --- Projection --------------------------------------------------------------


def test_projection_project_and_include(cpp: dict[str, Any]) -> None:
    proj = Projection(_arr(cpp["projection_parameter_values"]), [False, True, False, True])
    _assert_all_exact(
        proj.project(_arr(cpp["projection_project_in"])), cpp["projection_project_out"]
    )
    _assert_all_exact(
        proj.include(_arr(cpp["projection_include_in"])), cpp["projection_include_out"]
    )
    assert proj.number_of_free_parameters == 2


def test_projection_with_an_empty_mask_frees_everything(cpp: dict[str, Any]) -> None:
    """# C++ parity: projection.cpp:31-33 — an empty vector means all free."""
    proj = Projection(_arr(cpp["projection_parameter_values"]))
    _assert_all_exact(
        proj.project(_arr([10.0, 20.0, 30.0, 40.0])),
        cpp["projection_allfree_project_out"],
    )
    _assert_all_exact(
        proj.include(_arr([5.0, 6.0, 7.0, 8.0])),
        cpp["projection_allfree_include_out"],
    )
    assert proj.number_of_free_parameters == 4


def test_projection_rejects_a_mismatched_mask() -> None:
    with pytest.raises(LibraryException, match="parametersFreedoms_"):
        Projection(_arr([1.0, 2.0]), [False])


def test_projection_rejects_an_all_fixed_mask() -> None:
    with pytest.raises(LibraryException, match="numberOfFreeParameters==0"):
        Projection(_arr([1.0, 2.0]), [True, True])


def test_projection_rejects_a_wrongly_sized_free_vector() -> None:
    proj = Projection(_arr([1.0, 2.0, 3.0]), [False, True, False])
    with pytest.raises(LibraryException, match="numberOfFreeParameters"):
        proj.include(_arr([1.0, 2.0, 3.0]))
    with pytest.raises(LibraryException, match="parametersFreedoms_"):
        proj.project(_arr([1.0, 2.0]))


def test_projection_include_uses_the_constructor_values_for_fixed_slots() -> None:
    """Not the working vector that ``value``/``values`` scribble on."""
    proj = Projection(_arr([10.0, 20.0]), [False, True])
    tolerance.exact(float(proj.include(_arr([1.0]))[1]), 20.0)
    tolerance.exact(float(proj.include(_arr([2.0]))[1]), 20.0)


# --- ProjectedCostFunction ---------------------------------------------------


def test_projected_cost_function_value_and_values(cpp: dict[str, Any]) -> None:
    pcf = ProjectedCostFunction(
        RosenbrockResiduals(), _arr(cpp["pcf_base"]), [False, True]
    )
    free = _arr(cpp["pcf_free"])
    tolerance.exact(pcf.value(free), float(cpp["pcf_value"]))
    _assert_all_exact(pcf.values(free), cpp["pcf_values"])


def test_projected_cost_function_is_also_a_projection(cpp: dict[str, Any]) -> None:
    """C++ inherits from both ``CostFunction`` and ``Projection``."""
    pcf = ProjectedCostFunction(
        RosenbrockResiduals(), _arr(cpp["pcf_base"]), [False, True]
    )
    _assert_all_exact(pcf.include(_arr(cpp["pcf_free"])), cpp["pcf_include"])
    _assert_all_exact(pcf.project(_arr([9.0, 8.0])), cpp["pcf_project"])


def test_projected_cost_function_inherits_the_finite_difference_gradient(
    cpp: dict[str, Any],
) -> None:
    """Only ``value``/``values`` are overridden; ``gradient`` falls through.

    LOOSE tier: the inherited gradient is a central difference with
    ``h = 1e-8``, so its rounding floor is ``|f| * 2^-53 / (2h)``, i.e. about
    ``|f| * 5.5e-9`` in absolute terms. With ``|f|`` around 5 here that is a
    few times 1e-8 relative in the worst case; the observed difference is well
    inside LOOSE.
    """
    pcf = ProjectedCostFunction(
        RosenbrockResiduals(), _arr(cpp["pcf_base"]), [False, True]
    )
    grad = np.zeros(1, dtype=np.float64)
    pcf.gradient(grad, _arr(cpp["pcf_free"]))
    for a, b in zip(grad, cpp["pcf_gradient"], strict=True):
        tolerance.loose(float(a), float(b))


def test_projected_cost_function_from_an_existing_projection(
    cpp: dict[str, Any],
) -> None:
    proj = Projection(_arr([0.3, 0.7]), [True, False])
    pcf = ProjectedCostFunction(RosenbrockResiduals(), projection=proj)
    free = _arr(cpp["pcf2_free"])
    tolerance.exact(pcf.value(free), float(cpp["pcf2_value"]))
    _assert_all_exact(pcf.values(free), cpp["pcf2_values"])


# --- ProjectedConstraint -----------------------------------------------------


def test_projected_constraint_with_an_infeasible_fixed_coordinate(
    cpp: dict[str, Any],
) -> None:
    """``base[1] = 5`` is outside ``[-2, 2]``, so no free vector can satisfy it."""
    inner = NonhomogeneousBoundaryConstraint(
        _arr(cpp["projconstraint_lo"]), _arr(cpp["projconstraint_hi"])
    )
    pc = ProjectedConstraint(
        inner, _arr(cpp["projconstraint_base"]), [False, True, False]
    )
    assert pc.test(_arr([0.5, 0.5])) is bool(
        cpp["projconstraint_test_infeasible_fixed"]
    )
    assert pc.test(_arr([0.5, 0.5])) is False
    _assert_all_exact(pc.upper_bound(_arr([0.5, 0.5])), cpp["projconstraint_upper"])
    _assert_all_exact(pc.lower_bound(_arr([0.5, 0.5])), cpp["projconstraint_lower"])


def test_projected_constraint_from_an_existing_projection(cpp: dict[str, Any]) -> None:
    inner = NonhomogeneousBoundaryConstraint(
        _arr(cpp["projconstraint_lo"]), _arr(cpp["projconstraint_hi"])
    )
    proj = Projection(_arr([0.0, 1.0, 0.0]), [False, True, False])
    pc = ProjectedConstraint(inner, projection=proj)
    assert pc.test(_arr([0.5, 0.5])) is bool(cpp["projconstraint2_test_inside"])
    assert pc.test(_arr([0.5, 9.0])) is bool(cpp["projconstraint2_test_outside"])
    _assert_all_exact(pc.upper_bound(_arr([0.5, 0.5])), cpp["projconstraint2_upper"])
    _assert_all_exact(pc.lower_bound(_arr([0.5, 0.5])), cpp["projconstraint2_lower"])


def test_projected_constraint_is_a_constraint() -> None:
    inner = NonhomogeneousBoundaryConstraint(_arr([-1.0]), _arr([1.0]))
    assert isinstance(
        ProjectedConstraint(inner, _arr([0.0]), [False]), Constraint
    )
