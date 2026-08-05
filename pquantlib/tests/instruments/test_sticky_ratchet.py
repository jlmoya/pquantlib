"""Cross-validate the sticky/ratchet payoff family against C++ v1.43.

Probe: ``v143/inst/stickyratchet``.

Two things are pinned, not one:

* the arithmetic — every payoff evaluated over a forward ladder that crosses
  both effective strikes, so each branch of the nested ``max()`` is reached;
* the **argument slotting** — each of the six named presets is asserted equal
  to the explicit ``DoubleStickyRatchetPayoff`` it is defined to be. The
  single-option variants route their second gearing/spread to slot 3, not
  slot 2; a port that slots them wrong still returns plausible numbers, and
  only this equivalence catches it.

All nine scalars are distinct and none is 0 or 1, so no mis-slotted argument
can coincide with another.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.instruments.sticky_ratchet import (
    DoubleStickyRatchetPayoff,
    RatchetMaxPayoff,
    RatchetMinPayoff,
    RatchetPayoff,
    StickyMaxPayoff,
    StickyMinPayoff,
    StickyPayoff,
)
from pquantlib.payoffs import Payoff
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/stickyratchet")


def _args(cpp: dict[str, Any]) -> dict[str, float]:
    return dict(cpp["inputs"])


def _double(cpp: dict[str, Any], type1: float, type2: float) -> DoubleStickyRatchetPayoff:
    i = _args(cpp)
    return DoubleStickyRatchetPayoff(
        type1,
        type2,
        i["gearing1"],
        i["gearing2"],
        i["gearing3"],
        i["spread1"],
        i["spread2"],
        i["spread3"],
        i["initial_value1"],
        i["initial_value2"],
        i["accrual_factor"],
    )


def _check(cpp: dict[str, Any], key: str, payoff: Payoff) -> None:
    ref = cpp[key]
    assert payoff.name() == ref["name"]
    assert payoff.description() == ref["description"]
    for fwd, expected in zip(cpp["forwards"], ref["values"], strict=True):
        # TIGHT: a handful of multiplications, additions and a max over
        # identical double inputs — no iteration, no transcendentals.
        tight(payoff(fwd), expected)


@pytest.mark.parametrize(
    ("key", "type1", "type2"),
    [
        ("double_m1_m1", -1.0, -1.0),
        ("double_m1_p1", -1.0, +1.0),
        ("double_p1_m1", +1.0, -1.0),
        ("double_p1_p1", +1.0, +1.0),
        ("double_m1_zero", -1.0, 0.0),
        ("double_p1_zero", +1.0, 0.0),
        ("double_zero_zero", 0.0, 0.0),
    ],
)
def test_double_sticky_ratchet(cpp: dict[str, Any], key: str, type1: float, type2: float) -> None:
    _check(cpp, key, _double(cpp, type1, type2))


def _presets(cpp: dict[str, Any]) -> dict[str, Payoff]:
    i = _args(cpp)
    return {
        # Single-option: (gearing1, gearing2, spread1, spread2, initialValue,
        # accrualFactor) — the SECOND gearing/spread is the probe's g3/s3.
        "ratchet": RatchetPayoff(
            i["gearing1"],
            i["gearing3"],
            i["spread1"],
            i["spread3"],
            i["initial_value1"],
            i["accrual_factor"],
        ),
        "sticky": StickyPayoff(
            i["gearing1"],
            i["gearing3"],
            i["spread1"],
            i["spread3"],
            i["initial_value1"],
            i["accrual_factor"],
        ),
        "ratchet_max": RatchetMaxPayoff(
            i["gearing1"],
            i["gearing2"],
            i["gearing3"],
            i["spread1"],
            i["spread2"],
            i["spread3"],
            i["initial_value1"],
            i["initial_value2"],
            i["accrual_factor"],
        ),
        "ratchet_min": RatchetMinPayoff(
            i["gearing1"],
            i["gearing2"],
            i["gearing3"],
            i["spread1"],
            i["spread2"],
            i["spread3"],
            i["initial_value1"],
            i["initial_value2"],
            i["accrual_factor"],
        ),
        "sticky_max": StickyMaxPayoff(
            i["gearing1"],
            i["gearing2"],
            i["gearing3"],
            i["spread1"],
            i["spread2"],
            i["spread3"],
            i["initial_value1"],
            i["initial_value2"],
            i["accrual_factor"],
        ),
        "sticky_min": StickyMinPayoff(
            i["gearing1"],
            i["gearing2"],
            i["gearing3"],
            i["spread1"],
            i["spread2"],
            i["spread3"],
            i["initial_value1"],
            i["initial_value2"],
            i["accrual_factor"],
        ),
    }


@pytest.mark.parametrize(
    "key", ["ratchet", "sticky", "ratchet_max", "ratchet_min", "sticky_max", "sticky_min"]
)
def test_named_presets(cpp: dict[str, Any], key: str) -> None:
    _check(cpp, key, _presets(cpp)[key])


@pytest.mark.parametrize(
    ("preset_key", "double_key"),
    [
        ("ratchet", "double_m1_zero"),
        ("sticky", "double_p1_zero"),
        ("ratchet_max", "double_m1_m1"),
        ("ratchet_min", "double_m1_p1"),
        ("sticky_max", "double_p1_m1"),
        ("sticky_min", "double_p1_p1"),
    ],
)
def test_preset_equals_its_explicit_double(cpp: dict[str, Any], preset_key: str, double_key: str) -> None:
    """Each preset must be the ``DoubleStickyRatchetPayoff`` it claims to be.

    This is the argument-slotting assertion: it fails if a gearing or spread
    reaches a different slot than C++ puts it in, even though the arithmetic
    is otherwise identical.
    """
    preset = _presets(cpp)[preset_key]
    for fwd, expected in zip(cpp["forwards"], cpp[double_key]["values"], strict=True):
        tight(preset(fwd), expected)


_TYPE_GRID = (
    (-1.0, -1.0),
    (-1.0, +1.0),
    (+1.0, -1.0),
    (+1.0, +1.0),
    (-1.0, 0.0),
    (+1.0, 0.0),
    (0.0, 0.0),
)


@pytest.mark.parametrize(
    "scalar",
    [
        "gearing1",
        "gearing2",
        "gearing3",
        "spread1",
        "spread2",
        "spread3",
        "initial_value1",
        "initial_value2",
        "accrual_factor",
    ],
)
def test_every_scalar_reaches_its_slot(cpp: dict[str, Any], scalar: str) -> None:
    """Perturbing any one of the nine scalars must move the payoff somewhere.

    A stored-but-unused argument is the failure mode this port exists to
    guard against, so each one is asserted load-bearing.

    "Somewhere" is deliberate rather than "everywhere": the payoff is a
    nested ``max``, and for a given ``(type1, type2)`` one whole branch can
    be dominated by the other — e.g. at ``(-1, -1)`` with the probe's inputs,
    ``effStrike3 = effStrike2 - swaplet`` dominates ``effStrike1 - swaplet``
    over the entire forward ladder, so ``gearing1`` / ``spread1`` /
    ``initial_value1`` are genuinely inert there. Requiring movement in at
    least one ``(type1, type2)`` configuration is the strongest claim the
    payoff's own algebra supports.
    """
    i = _args(cpp)
    forwards = cpp["forwards"]

    def values(type1: float, type2: float, overrides: dict[str, float]) -> list[float]:
        args = dict(i)
        args.update(overrides)
        payoff = DoubleStickyRatchetPayoff(
            type1,
            type2,
            args["gearing1"],
            args["gearing2"],
            args["gearing3"],
            args["spread1"],
            args["spread2"],
            args["spread3"],
            args["initial_value1"],
            args["initial_value2"],
            args["accrual_factor"],
        )
        return [payoff(f) for f in forwards]

    bumped = {scalar: i[scalar] + 0.05}
    assert any(values(t1, t2, {}) != values(t1, t2, bumped) for t1, t2 in _TYPE_GRID), (
        f"{scalar} is accepted but never used"
    )


def test_type_pairs_are_pinned_against_cpp(cpp: dict[str, Any]) -> None:
    """``type1`` / ``type2`` reach the payoff.

    Deliberately *not* an "every pair is distinguishable" assertion: C++
    itself returns identical ladders for ``(+1, +1)`` and ``(+1, 0)`` with
    these inputs (``effStrike1 - swaplet`` dominates ``effStrike3``
    throughout), so such a claim would be false of the source of truth. What
    is asserted instead is that the seven pairs reproduce the seven distinct
    C++ ladders — which fails if either type is dropped, since five of the
    seven ladders are pairwise different.
    """
    keys = [
        "double_m1_m1",
        "double_m1_p1",
        "double_p1_m1",
        "double_p1_p1",
        "double_m1_zero",
        "double_p1_zero",
        "double_zero_zero",
    ]
    ladders = {tuple(cpp[k]["values"]) for k in keys}
    assert len(ladders) >= 5, "probe grid no longer discriminates between types"
    for key, (t1, t2) in zip(keys, _TYPE_GRID, strict=True):
        _check(cpp, key, _double(cpp, t1, t2))


@pytest.mark.parametrize("key", ["raises_type1_two", "raises_type2_half"])
def test_illegal_types_raise(cpp: dict[str, Any], key: str) -> None:
    ref = cpp[key]
    assert ref["raises"] is True
    payoff = _double(cpp, ref["type1"], ref["type2"])
    with pytest.raises(LibraryException) as excinfo:
        payoff(0.03)
    assert ref["error"] in str(excinfo.value)


def test_zero_type_is_legal(cpp: dict[str, Any]) -> None:
    ref = cpp["raises_type1_zero_ok"]
    assert ref["raises"] is False
    payoff: Callable[[float], float] = _double(cpp, ref["type1"], ref["type2"])
    payoff(0.03)  # must not raise
