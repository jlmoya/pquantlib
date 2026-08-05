"""Cross-validate ``LineSearch.update`` and ``Constraint.update`` against C++.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block A. Both routines are the same halving loop (linesearch.cpp:26-45 and
constraint.cpp:27-43, duplicated in v1.43); the reference pins the accepted
step and the resulting parameter vector for a direction that overshoots the
feasible set and must be halved several times.

Tolerance: EXACT. Every value here is a repeated multiplication of the input
``beta`` by 0.5 followed by one ``params + diff*direction``, so both sides
perform the identical IEEE-754 operations in the identical order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    NoConstraint,
    PositiveConstraint,
)
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.line_search import LineSearch, dot_product
from pquantlib.math.optimization.problem import Problem
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


class _ProbeLineSearch(LineSearch):
    """Concrete stand-in for the probe's ``ProbeLineSearch``.

    ``LineSearch.update`` is a non-virtual member of an abstract class, so the
    C++ probe subclasses it with a trivial ``operator()`` purely to reach
    ``update``. Same here.
    """

    def __call__(
        self,
        problem: Problem,
        ec_type: Type,
        end_criteria: EndCriteria,
        t_ini: float,
    ) -> tuple[float, Type]:
        del problem, end_criteria
        return t_ini, ec_type


def _arr(values: list[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def test_constraint_update_noconstraint_never_halves(cpp: dict[str, Any]) -> None:
    """With ``NoConstraint`` the first trial is always valid, so ``diff == beta``."""
    c = NoConstraint()
    params = _arr([1.0, 2.0])
    diff = c.update(params, _arr([-1.0, 0.5]), 3.0)
    tolerance.exact(diff, float(cpp["cupdate_noconstraint_diff"]))
    for got, expected in zip(params, cpp["cupdate_noconstraint_params"], strict=True):
        tolerance.exact(float(got), float(expected))


def test_constraint_update_positive_halves_until_feasible(cpp: dict[str, Any]) -> None:
    """``beta = 8`` from (1, 4) along (-1, -1) needs four halvings to stay positive."""
    c = PositiveConstraint()
    params = _arr([1.0, 4.0])
    diff = c.update(params, _arr([-1.0, -1.0]), 8.0)
    tolerance.exact(diff, float(cpp["cupdate_positive_halved_diff"]))
    for got, expected in zip(
        params, cpp["cupdate_positive_halved_params"], strict=True
    ):
        tolerance.exact(float(got), float(expected))


def test_constraint_update_boundary_halves_on_the_upper_side(
    cpp: dict[str, Any],
) -> None:
    c = BoundaryConstraint(0.0, 1.0)
    params = _arr([0.25, 0.25])
    diff = c.update(params, _arr([1.0, 0.0]), 5.0)
    tolerance.exact(diff, float(cpp["cupdate_boundary_halved_diff"]))
    for got, expected in zip(
        params, cpp["cupdate_boundary_halved_params"], strict=True
    ):
        tolerance.exact(float(got), float(expected))


def test_line_search_update_matches_constraint_update(cpp: dict[str, Any]) -> None:
    """``LineSearch.update`` is the same halving loop reached through the base class."""
    ls = _ProbeLineSearch()
    c = PositiveConstraint()
    params = _arr([1.0, 4.0])
    diff = ls.update(params, _arr([-1.0, -1.0]), 8.0, c)
    tolerance.exact(diff, float(cpp["lsupdate_positive_halved_diff"]))
    for got, expected in zip(
        params, cpp["lsupdate_positive_halved_params"], strict=True
    ):
        tolerance.exact(float(got), float(expected))


def test_line_search_update_unconstrained(cpp: dict[str, Any]) -> None:
    ls = _ProbeLineSearch()
    c = NoConstraint()
    params = _arr([-1.0, 0.0, 2.0])
    diff = ls.update(params, _arr([0.5, -0.25, 1.0]), 1.0, c)
    tolerance.exact(diff, float(cpp["lsupdate_noconstraint_diff"]))
    for got, expected in zip(params, cpp["lsupdate_noconstraint_params"], strict=True):
        tolerance.exact(float(got), float(expected))


def test_constraint_update_raises_after_200_halvings() -> None:
    """The C++ guard is ``icount > 200`` then ``QL_FAIL``, not a silent give-up.

    ``BoundaryConstraint(0, 1)`` starting at 0.5 along +1 can never be
    satisfied by halving below 0.5, because the step is added, so the loop
    exhausts its budget.
    """
    class _NeverSatisfied(BoundaryConstraint):
        def test(self, params: npt.NDArray[np.float64]) -> bool:
            del params
            return False

    c = _NeverSatisfied(0.0, 1.0)
    with pytest.raises(LibraryException, match="can't update parameter vector"):
        c.update(_arr([0.5]), _arr([1.0]), 1.0)


def test_line_search_update_raises_with_its_own_message() -> None:
    """v1.43 keeps two copies of the loop with DIFFERENT failure messages."""
    class _NeverSatisfied(NoConstraint):
        def test(self, params: npt.NDArray[np.float64]) -> bool:
            del params
            return False

    ls = _ProbeLineSearch()
    with pytest.raises(LibraryException, match="can't update linesearch"):
        ls.update(_arr([0.5]), _arr([1.0]), 1.0, _NeverSatisfied())


def test_line_search_defaults() -> None:
    """A fresh line search reports empty state and ``succeed == True``.

    # C++ parity: linesearch.hpp:71-77 — ``qt_ = qpt_ = 0.0``,
    # ``succeed_ = true``, and the three Arrays default-constructed empty.
    """
    ls = _ProbeLineSearch()
    assert ls.last_x.size == 0
    assert ls.last_gradient.size == 0
    assert ls.search_direction.size == 0
    assert ls.last_function_value == 0.0
    assert ls.last_gradient_norm2 == 0.0
    assert ls.succeed is True


def test_search_direction_setter_copies() -> None:
    """Assigning the direction must not alias the caller's array.

    C++ assigns through ``Array& searchDirection()``, which copies.
    """
    ls = _ProbeLineSearch()
    d = _arr([1.0, 2.0])
    ls.search_direction = d
    d[0] = 99.0
    assert ls.search_direction[0] == 1.0


def test_dot_product_is_sequential() -> None:
    """``dot_product`` must accumulate left to right, like ``std::inner_product``.

    Constructed so left-to-right and pairwise summation differ: each ``1e-16``
    is below half an ulp of 1.0 and is therefore absorbed one at a time, while
    numpy's pairwise algorithm (which kicks in from 8 elements) adds the small
    terms to each other first and keeps them.
    """
    a = np.array([1.0, *([1e-16] * 14), -1.0], dtype=np.float64)
    ones = np.ones(a.size, dtype=np.float64)
    tolerance.exact(dot_product(a, ones), 0.0)
    # numpy's pairwise sum keeps them, which is the behaviour we must NOT have.
    assert float(np.sum(a)) != 0.0
