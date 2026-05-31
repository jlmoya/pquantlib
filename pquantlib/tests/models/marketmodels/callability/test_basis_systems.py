"""W11-C basis systems + parametric exercise — deterministic probe checks.

Validates ``SwapBasisSystem`` / ``SwapForwardBasisSystem`` basis-function
values and shapes against the C++ probe ``cluster/w11c.json`` (tier TIGHT), and
exercises ``TriggeredSwapExercise`` + ``ParametricExerciseAdapter``.

C++ parity:
  ql/models/marketmodels/callability/{swapbasissystem,swapforwardbasissystem,
    triggeredswapexercise,parametricexerciseadapter}.cpp @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.models.marketmodels.callability import (
    MarketModelBasisSystem,
    MarketModelParametricExercise,
    ParametricExerciseAdapter,
    SwapBasisSystem,
    SwapForwardBasisSystem,
    TriggeredSwapExercise,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.products import (
    ExerciseStrategy as ProtoExerciseStrategy,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight

_RATE_TIMES = [0.5, 1.0, 1.5, 2.0]
_EXERCISE_TIMES = [0.5, 1.0, 1.5]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w11c")


def _flat_state() -> LMMCurveState:
    cs = LMMCurveState(_RATE_TIMES)
    cs.set_on_forward_rates([0.05] * 3)
    return cs


def _step_and_flatten(
    basis: SwapBasisSystem | SwapForwardBasisSystem, cs: LMMCurveState, n: int
) -> tuple[list[int], list[float]]:
    basis.reset()
    sizes: list[int] = []
    flat: list[float] = []
    for _ in range(n):
        basis.next_step(cs)
        results: list[float] = []
        basis.values(cs, results)
        sizes.append(len(results))
        flat.extend(results)
    return sizes, flat


def test_swap_basis_system(ref: dict[str, Any]) -> None:
    sbs = SwapBasisSystem(_RATE_TIMES, _EXERCISE_TIMES)
    cs = _flat_state()

    assert sbs.number_of_exercises() == int(ref["sbs_num_exercises"])
    assert sbs.number_of_functions() == [int(x) for x in ref["sbs_number_of_functions"]]
    assert sbs.number_of_data() == sbs.number_of_functions()
    assert all(sbs.is_exercise_time())

    sizes, flat = _step_and_flatten(sbs, cs, len(_EXERCISE_TIMES))
    assert sizes == [int(x) for x in ref["sbs_values_sizes"]]
    for a, e in zip(flat, ref["sbs_values_flat"], strict=True):
        tight(a, float(e))


def test_swap_forward_basis_system(ref: dict[str, Any]) -> None:
    sfbs = SwapForwardBasisSystem(_RATE_TIMES, _EXERCISE_TIMES)
    cs = _flat_state()

    assert sfbs.number_of_exercises() == int(ref["sfbs_num_exercises"])
    assert sfbs.number_of_functions() == [
        int(x) for x in ref["sfbs_number_of_functions"]
    ]
    assert all(sfbs.is_exercise_time())

    sizes, flat = _step_and_flatten(sfbs, cs, len(_EXERCISE_TIMES))
    assert sizes == [int(x) for x in ref["sfbs_values_sizes"]]
    for a, e in zip(flat, ref["sfbs_values_flat"], strict=True):
        tight(a, float(e))


def test_basis_systems_are_node_data_providers() -> None:
    sbs = SwapBasisSystem(_RATE_TIMES, _EXERCISE_TIMES)
    sfbs = SwapForwardBasisSystem(_RATE_TIMES, _EXERCISE_TIMES)
    assert isinstance(sbs, MarketModelBasisSystem)
    assert isinstance(sfbs, MarketModelBasisSystem)


def test_basis_system_clone_independent() -> None:
    sbs = SwapBasisSystem(_RATE_TIMES, _EXERCISE_TIMES)
    cs = _flat_state()
    sbs.reset()
    sbs.next_step(cs)
    clone = sbs.clone()
    assert clone is not sbs
    # clone retains the same structural shape
    assert clone.number_of_functions() == sbs.number_of_functions()


def test_triggered_swap_exercise_node_data(ref: dict[str, Any]) -> None:
    strikes = [0.04, 0.045, 0.05]
    tse = TriggeredSwapExercise(_RATE_TIMES, _EXERCISE_TIMES, strikes)
    cs = _flat_state()

    assert tse.number_of_exercises() == 3
    assert tse.number_of_variables() == [1, 1, 1]
    assert tse.number_of_parameters() == [1, 1, 1]
    assert tse.number_of_data() == [1, 1, 1]
    assert all(tse.is_exercise_time())

    # the single node variable is the coterminal swap rate at rate_index[k].
    cot = list(ref["srt_cot_swap_rates"])  # {cotSwapRate(0), (1), (2)}
    tse.reset()
    for k in range(3):
        tse.next_step(cs)
        results: list[float] = []
        tse.values(cs, results)
        assert len(results) == 1
        tight(results[0], float(cot[k]))

    # guess fills the strike
    params: list[float] = []
    tse.guess(1, params)
    assert params == [0.045]

    # exercise rule: variables[0] >= parameters[0]
    assert tse.exercise(0, [0.04], [0.05]) is True
    assert tse.exercise(0, [0.06], [0.05]) is False


def test_parametric_exercise_adapter(ref: dict[str, Any]) -> None:
    strikes = [0.04, 0.045, 0.05]
    tse = TriggeredSwapExercise(_RATE_TIMES, _EXERCISE_TIMES, strikes)
    # parameters: trigger 0.04 at each exercise → fires (swap rate 0.05 >= 0.04).
    parameters = [[0.04], [0.04], [0.04]]
    adapter = ParametricExerciseAdapter(tse, parameters)
    cs = _flat_state()

    assert isinstance(adapter, ProtoExerciseStrategy)
    # exercise_times are the evolution times flagged as exercise times.
    for a, e in zip(adapter.exercise_times(), _EXERCISE_TIMES, strict=True):
        tight(a, float(e))

    adapter.reset()
    flags: list[bool] = []
    for _ in range(len(adapter.relevant_times())):
        adapter.next_step(cs)
        flags.append(adapter.exercise(cs))
    assert all(flags)  # all fire with trigger 0.04 < swap rate 0.05

    # higher trigger → never fires.
    adapter2 = ParametricExerciseAdapter(tse, [[0.10], [0.10], [0.10]])
    adapter2.reset()
    for _ in range(len(adapter2.relevant_times())):
        adapter2.next_step(cs)
        assert adapter2.exercise(cs) is False


def test_triggered_swap_exercise_is_parametric() -> None:
    tse = TriggeredSwapExercise(_RATE_TIMES, _EXERCISE_TIMES, [0.04, 0.045, 0.05])
    assert isinstance(tse, MarketModelParametricExercise)
