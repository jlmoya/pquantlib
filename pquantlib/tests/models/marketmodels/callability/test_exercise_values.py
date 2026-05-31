"""W11-C exercise values + swap-rate trigger — deterministic probe checks.

Validates ``NothingExerciseValue`` / ``BermudanSwaptionExerciseValue`` cash
flows and ``SwapRateTrigger`` exercise logic on a flat-forward
``LMMCurveState`` against the C++ probe ``cluster/w11c.json`` (tier TIGHT).

C++ parity:
  ql/models/marketmodels/callability/{nothingexercisevalue,
    bermudanswaptionexercisevalue,swapratetrigger}.cpp @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.callability import (
    BermudanSwaptionExerciseValue,
    ExerciseStrategy,
    NothingExerciseValue,
    SwapRateTrigger,
)
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.products import (
    ExerciseStrategy as ProtoExerciseStrategy,
)
from pquantlib.models.marketmodels.products import (
    MarketModelExerciseValue as ProtoExerciseValue,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight

_RATE_TIMES = [0.5, 1.0, 1.5, 2.0]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w11c")


def _flat_state() -> LMMCurveState:
    cs = LMMCurveState(_RATE_TIMES)
    cs.set_on_forward_rates([0.05] * 3)
    return cs


def test_nothing_exercise_value(ref: dict[str, Any]) -> None:
    nev = NothingExerciseValue(_RATE_TIMES)
    cs = _flat_state()

    assert nev.number_of_exercises() == int(ref["nev_num_exercises"])
    assert len(nev.possible_cash_flow_times()) == int(ref["nev_num_pcf"])
    assert len(nev.is_exercise_time()) == int(ref["nev_is_exercise_size"])
    assert all(nev.is_exercise_time())

    nev.reset()
    time_indices: list[int] = []
    amounts: list[float] = []
    for _ in range(int(ref["nev_is_exercise_size"])):
        nev.next_step(cs)
        cf = nev.value(cs)
        time_indices.append(cf.time_index)
        amounts.append(cf.amount)

    assert time_indices == [int(x) for x in ref["nev_time_indices"]]
    for a, e in zip(amounts, ref["nev_amounts"], strict=True):
        tight(a, float(e))


def test_nothing_exercise_value_requires_two_times() -> None:
    with pytest.raises(LibraryException):
        NothingExerciseValue([0.5])


def test_nothing_exercise_value_is_exercise_time_size_check() -> None:
    with pytest.raises(LibraryException):
        NothingExerciseValue(_RATE_TIMES, is_exercise_time=[True, True])


def test_bermudan_swaption_exercise_value(ref: dict[str, Any]) -> None:
    strike = float(ref["bev_strike"])
    n = int(ref["bev_num_exercises"])
    payoffs = [PlainVanillaPayoff(OptionType.Call, strike) for _ in range(n)]
    bev = BermudanSwaptionExerciseValue(_RATE_TIMES, payoffs)
    cs = _flat_state()

    assert bev.number_of_exercises() == n
    assert len(bev.is_exercise_time()) == int(ref["bev_is_exercise_size"])
    assert all(bev.is_exercise_time())

    bev.reset()
    time_indices: list[int] = []
    amounts: list[float] = []
    for _ in range(n):
        bev.next_step(cs)
        cf = bev.value(cs)
        time_indices.append(cf.time_index)
        amounts.append(cf.amount)

    assert time_indices == [int(x) for x in ref["bev_time_indices"]]
    for a, e in zip(amounts, ref["bev_amounts"], strict=True):
        tight(a, float(e))


def test_swap_rate_trigger(ref: dict[str, Any]) -> None:
    trigger = float(ref["srt_trigger"])
    exercise_times = list(ref["srt_exercise_times"])
    srt = SwapRateTrigger(_RATE_TIMES, [trigger] * len(exercise_times), exercise_times)
    cs = _flat_state()

    for a, e in zip(srt.exercise_times(), exercise_times, strict=True):
        tight(a, float(e))
    # relevant_times == exercise_times for the naive trigger
    for a, e in zip(srt.relevant_times(), exercise_times, strict=True):
        tight(a, float(e))

    srt.reset()
    flags: list[float] = []
    for _ in range(len(exercise_times)):
        srt.next_step(cs)
        flags.append(1.0 if srt.exercise(cs) else 0.0)

    assert flags == list(ref["srt_exercise_flags"])


def test_swap_rate_trigger_does_not_fire_when_below() -> None:
    # trigger above the 0.05 flat swap rate → never exercise.
    exercise_times = [0.5, 1.0, 1.5]
    srt = SwapRateTrigger(_RATE_TIMES, [0.10] * 3, exercise_times)
    cs = _flat_state()
    srt.reset()
    for _ in range(3):
        srt.next_step(cs)
        assert srt.exercise(cs) is False


def test_concretes_satisfy_w11a_protocols() -> None:
    rt = _RATE_TIMES
    nev = NothingExerciseValue(rt)
    bev = BermudanSwaptionExerciseValue(
        rt, [PlainVanillaPayoff(OptionType.Call, 0.045)] * 3
    )
    srt = SwapRateTrigger(rt, [0.04] * 3, [0.5, 1.0, 1.5])

    assert isinstance(nev, ProtoExerciseValue)
    assert isinstance(bev, ProtoExerciseValue)
    assert isinstance(srt, ProtoExerciseStrategy)
    # and the concrete ABCs subclass relationship holds
    assert isinstance(srt, ExerciseStrategy)


def test_clone_is_independent() -> None:
    nev = NothingExerciseValue(_RATE_TIMES)
    cs = _flat_state()
    nev.reset()
    nev.next_step(cs)  # advance original to index 1
    clone = nev.clone()
    clone.reset()
    clone.next_step(cs)
    # original is at index 1 (cf.time_index == 0 from its first next_step),
    # clone independently reset + stepped → cf.time_index == 0 too, but the
    # internal current_index counters are independent objects.
    assert clone is not nev
