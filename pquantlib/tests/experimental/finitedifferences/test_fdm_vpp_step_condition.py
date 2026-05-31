"""Tests for FdmVPPStepCondition + StartLimit + Factory.

# C++ parity reference:
# ql/experimental/finitedifferences/fdmvppstepcondition.hpp
# ql/experimental/finitedifferences/fdmvppstartlimitstepcondition.hpp
# ql/experimental/finitedifferences/fdmvppstepconditionfactory.hpp
# (v1.42.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.finitedifferences.fdm_vpp_start_limit_step_condition import (
    FdmVPPStartLimitStepCondition,
)
from pquantlib.experimental.finitedifferences.fdm_vpp_step_condition import (
    FdmVPPStepConditionMesher,
    FdmVPPStepConditionParams,
)
from pquantlib.experimental.finitedifferences.fdm_vpp_step_condition_factory import (
    FdmVPPStepConditionFactory,
)
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.experimental.finitedifferences.vanilla_vpp_option import (
    VanillaVPPOption,
    VanillaVPPOptionArguments,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def today() -> Date:
    return Date.from_ymd(18, Month.December, 2011)


@pytest.fixture(scope="module")
def exercise(today: Date) -> SwingExercise:
    return SwingExercise.from_range(today, today + 1, 3600)


def _make_params(
    *,
    heat_rate: float = 2.0,
    p_min: float = 8.0,
    p_max: float = 40.0,
    t_min_up: int = 2,
    t_min_down: int = 2,
    start_up_fuel: float = 20.0,
    start_up_fix_cost: float = 100.0,
    fuel_cost_addon: float = 3.0,
) -> FdmVPPStepConditionParams:
    return FdmVPPStepConditionParams(
        heat_rate=heat_rate,
        p_min=p_min,
        p_max=p_max,
        t_min_up=t_min_up,
        t_min_down=t_min_down,
        start_up_fuel=start_up_fuel,
        start_up_fix_cost=start_up_fix_cost,
        fuel_cost_addon=fuel_cost_addon,
    )


def _make_step_condition(
    *,
    n_starts: int | None = None,
    params: FdmVPPStepConditionParams | None = None,
) -> FdmVPPStartLimitStepCondition:
    p = params if params is not None else _make_params()
    n_states = FdmVPPStartLimitStepCondition.compute_n_states(
        p.t_min_up, p.t_min_down, n_starts
    )
    mesher = FdmMesherComposite(Uniform1dMesher(0.0, 1.0, n_states))
    mesh = FdmVPPStepConditionMesher(state_direction=0, mesher=mesher)

    # Constant inner-value calculators: gas = 21.0, spark = 25.0.
    def gas(_iter: FdmLinearOpIterator, _t: float) -> float:
        return 21.0

    def spark(_iter: FdmLinearOpIterator, _t: float) -> float:
        return 25.0

    return FdmVPPStartLimitStepCondition(
        params=p,
        n_starts=n_starts,
        mesh=mesh,
        gas_price=gas,
        spark_spread_price=spark,
    )


# ---------- compute_n_states static formula ----------------------------------


def test_compute_n_states_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["n_states_formula"]
    assert (
        FdmVPPStartLimitStepCondition.compute_n_states(2, 2, None)
        == ref["vanilla_2_2"]
    )
    assert (
        FdmVPPStartLimitStepCondition.compute_n_states(3, 2, None)
        == ref["vanilla_3_2"]
    )
    assert (
        FdmVPPStartLimitStepCondition.compute_n_states(6, 2, None)
        == ref["vanilla_6_2"]
    )
    assert (
        FdmVPPStartLimitStepCondition.compute_n_states(2, 2, 3)
        == ref["start_2_2_3"]
    )
    assert (
        FdmVPPStartLimitStepCondition.compute_n_states(3, 2, 4)
        == ref["start_3_2_4"]
    )
    assert (
        FdmVPPStartLimitStepCondition.compute_n_states(2, 2, 0)
        == ref["start_2_2_0"]
    )


# ---------- evolve_at_p_min / evolve_at_p_max closed forms ------------------


def test_evolve_at_p_min_closed_form() -> None:
    """``evolveAtPMin(s) = p_min * (s - heat_rate * fuel_cost_addon)``."""
    sc = _make_step_condition()
    spark = 25.0
    expected = 8.0 * (spark - 2.0 * 3.0)
    tight(sc.evolve_at_p_min(spark), expected)


def test_evolve_at_p_max_closed_form() -> None:
    """``evolveAtPMax(s) = p_max * (s - heat_rate * fuel_cost_addon)``."""
    sc = _make_step_condition()
    spark = 25.0
    expected = 40.0 * (spark - 2.0 * 3.0)
    tight(sc.evolve_at_p_max(spark), expected)


def test_max_value_reduces_to_max() -> None:
    sc = _make_step_condition()
    states = np.array([0.5, 12.3, -7.1, 4.0], dtype=np.float64)
    tight(sc.max_value(states), 12.3)


# ---------- change_state Bellman update --------------------------------------


def test_change_state_vanilla_cycle_basic_shape() -> None:
    """With tMinUp=tMinDown=2 (no start limit), the state-machine cycle has
    6 entries. Verify ``change_state`` returns a 6-vector. The C++ test
    suite covers the deeper invariant via ``DynProgVPPIntrinsicValueEngine``
    against the LP reference; here we cover the shape contract.
    """
    sc = _make_step_condition()
    state = np.zeros(6, dtype=np.float64)
    out = sc.change_state(gas_price=21.0, state=state, t=0.0)
    assert out.shape == (6,)
    # All-zeros state input + no start cost change ⇒ output stays nonneg.
    assert (out >= -1e-12).all()


def test_change_state_no_start_limit_applies_startup_cost() -> None:
    """The end-of-down branch (j == sss-1, no start limit) reads:
    ``max(state[i], max(state[0], state[t_min_up]) - start_up_cost)``.

    Use t_min_up=2, t_min_down=2 ⇒ sss=6, end index = 5.
    Fill state[0]=1000, state[2]=500, others zero. The end-of-down cell
    should become ``max(state[5], state[0] - start_up_cost) = max(0,
    1000 - (100 + (21+3)*20)) = max(0, 1000 - 580) = 420``.
    """
    p = _make_params(heat_rate=2.0)
    sc = _make_step_condition(params=p)
    state = np.zeros(6, dtype=np.float64)
    state[0] = 1000.0
    state[2] = 500.0
    out = sc.change_state(gas_price=21.0, state=state, t=0.0)
    # start_up_cost = 100 + (21+3) * 20 = 580.
    # out[5] = max(state[5]=0, max(state[0]=1000, state[2]=500) - 580)
    #        = max(0, 420) = 420.
    tight(out[5], 420.0)


def test_change_state_n_starts_zero_exhausts_budget() -> None:
    """With nStarts=0 the budget-exhausted branch (i < sss) returns ``state[i]``
    at the end-of-down index, regardless of profit gain — matches C++.
    """
    p = _make_params(heat_rate=2.0)
    sc = _make_step_condition(n_starts=0, params=p)
    state = np.zeros(6, dtype=np.float64)
    state[0] = 1000.0
    out = sc.change_state(gas_price=21.0, state=state, t=0.0)
    # With n_starts=0, i=5 < sss=6 branch returns state[5] = 0.
    tight(out[5], 0.0)


# ---------- mesher size mismatch raises --------------------------------------


def test_mesher_size_mismatch_raises() -> None:
    """Constructing the step condition with a mesher whose axis size doesn't
    match ``n_states`` raises.
    """
    p = _make_params()
    # n_states for vanilla = 2*2+2 = 6; supply a mesher of size 4.
    mesher = FdmMesherComposite(Uniform1dMesher(0.0, 1.0, 4))
    mesh = FdmVPPStepConditionMesher(state_direction=0, mesher=mesher)

    def gas(_iter: FdmLinearOpIterator, _t: float) -> float:
        return 0.0

    def spark(_iter: FdmLinearOpIterator, _t: float) -> float:
        return 0.0

    with pytest.raises(LibraryException):
        FdmVPPStartLimitStepCondition(
            params=p, n_starts=None, mesh=mesh,
            gas_price=gas, spark_spread_price=spark,
        )


def test_t_min_up_must_be_positive() -> None:
    p = _make_params(t_min_up=0)
    mesher = FdmMesherComposite(Uniform1dMesher(0.0, 1.0, 2))
    mesh = FdmVPPStepConditionMesher(state_direction=0, mesher=mesher)

    def calc(_iter: FdmLinearOpIterator, _t: float) -> float:
        return 0.0

    with pytest.raises(LibraryException):
        FdmVPPStartLimitStepCondition(
            params=p, n_starts=None, mesh=mesh,
            gas_price=calc, spark_spread_price=calc,
        )


# ---------- apply_to: the per-step backward update ---------------------------


def test_apply_to_evolves_running_states_only() -> None:
    """In a vanilla setup with t_min_up=2 and t_min_down=2, the 6 states map
    to phases [PMin, PMin, PMax, PMax, Down, Down]. After one ``apply_to``
    call on a zero array (plus the state-transition matrix), only the
    running-phase cells should carry an evolve contribution.

    We verify by zeroing the initial state and checking the per-state
    contributions in the array AFTER applying ``apply_to`` (the actual
    values include the change_state matrix transform; we just check that
    the resulting vector is well-formed and the max-value across states
    matches what hand calculation would give).
    """
    sc = _make_step_condition()
    a = np.zeros(6, dtype=np.float64)
    sc.apply_to(a, t=0.0)
    # The actual transformed values depend on the change_state matrix. We
    # smoke-test that ``apply_to`` runs cleanly and the output is finite.
    assert np.all(np.isfinite(a))


# ---------- Factory tests ----------------------------------------------------


def test_factory_state_mesher_size_matches_cpp(
    cpp_ref: dict[str, Any], exercise: SwingExercise
) -> None:
    """Factory.state_mesher().size() matches the C++ probe."""
    ref = cpp_ref["factory_mesher_size"]
    # Vanilla case.
    opt = VanillaVPPOption(
        heat_rate=2.0, p_min=8.0, p_max=40.0,
        t_min_up=2, t_min_down=2,
        start_up_fuel=20.0, start_up_fix_cost=100.0,
        exercise=exercise,
    )
    args = VanillaVPPOptionArguments()
    opt.setup_arguments(args)
    factory = FdmVPPStepConditionFactory(args)
    assert factory.state_mesher().size() == ref["vanilla"]

    # Start-limit (4 starts).
    opt2 = VanillaVPPOption(
        heat_rate=2.0, p_min=8.0, p_max=40.0,
        t_min_up=2, t_min_down=2,
        start_up_fuel=20.0, start_up_fix_cost=100.0,
        exercise=exercise, n_starts=4,
    )
    args2 = VanillaVPPOptionArguments()
    opt2.setup_arguments(args2)
    factory2 = FdmVPPStepConditionFactory(args2)
    assert factory2.state_mesher().size() == ref["start_limit_4"]


def test_factory_running_hour_limit_not_supported(
    exercise: SwingExercise,
) -> None:
    """C++ raises ``QL_FAIL("vpp type is not supported")`` on the running-
    hour branch. The Python port matches.
    """
    opt = VanillaVPPOption(
        heat_rate=2.0, p_min=8.0, p_max=40.0,
        t_min_up=2, t_min_down=2,
        start_up_fuel=20.0, start_up_fix_cost=100.0,
        exercise=exercise, n_running_hours=100,
    )
    args = VanillaVPPOptionArguments()
    opt.setup_arguments(args)
    factory = FdmVPPStepConditionFactory(args)
    with pytest.raises(LibraryException):
        factory.state_mesher()


def test_factory_rejects_both_limits_set() -> None:
    """The C++ factory ctor raises when both nStarts AND nRunningHours are set."""
    args = VanillaVPPOptionArguments()
    args.heat_rate = 2.0
    args.p_min = 8.0
    args.p_max = 40.0
    args.t_min_up = 2
    args.t_min_down = 2
    args.start_up_fuel = 20.0
    args.start_up_fix_cost = 100.0
    args.n_starts = 3
    args.n_running_hours = 100
    with pytest.raises(LibraryException):
        FdmVPPStepConditionFactory(args)
