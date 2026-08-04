"""Cross-validate the two v1.43 behavioural changes against the C++ probe.

Reference: ``migration-harness/references/v143/trinomial/g2process.json``,
emitted by ``v143_trinomial_g2process_probe``. The probe body is shared
verbatim with JQuantLib's harness so both ports are pinned to the same 21
cases; only its output layer is adapted to PQuantLib's conventions.

1. ``TrinomialTree`` — the gated dx floor. On grid steps far shorter than the
   longest step the natural spacing ``v*sqrt(3)`` collapses and the node count
   explodes. v1.43 caches the per-step variances, forms
   ``dx_floor = sqrt(3 * max_i v2_i)`` and, on steps with
   ``dt < 0.01 * dt_max``, widens dx to ``max(dx_natural, dx_floor)``. Where —
   and only where — the floor actually widened dx, the branching probabilities
   switch from the classical Hull-White/Clewlow form to a general
   moment-matching one.

2. ``G2Process`` / ``G2ForwardProcess`` — term-structure awareness. Both gain
   an optional term structure plus ``term_structure()``, ``phi(t)`` and
   ``short_rate(t, z1, z2)``. The simulated state is shifted to
   ``(x + phi(t), y)`` so ``state[0] + state[1] == r(t)``. With no curve the
   processes must degenerate to exactly the pre-v1.43 pair of zero-mean OU
   processes, which the ``*_empty_*`` cases pin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.shortrate.generalized_ornstein_uhlenbeck_process import (
    GeneralizedOrnsteinUhlenbeckProcess,
)
from pquantlib.methods.lattices.trinomial_tree import TrinomialTree
from pquantlib.processes.g2_forward_process import G2ForwardProcess
from pquantlib.processes.g2_process import G2Process
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/trinomial/g2process")


# --------------------------------------------------------------------------
# TrinomialTree
# --------------------------------------------------------------------------

_TREE_CASES = [
    "tree_uniform_ou",
    "tree_weekend_roll_ou",
    "tree_small_gap_ou",
    "tree_small_gap_short_tail_ou",
    "tree_threshold_below_ou",
    "tree_threshold_above_ou",
    "tree_gate_fires_dx_unchanged",
    "tree_is_positive_bump",
]


def _process_from(inputs: dict[str, Any]) -> StochasticProcess1D:
    if inputs["process"] == "OrnsteinUhlenbeckProcess":
        return OrnsteinUhlenbeckProcess(float(inputs["speed"]), float(inputs["vol"]))
    # Transcendental-free time-dependent-vol process: speed == 0 selects
    # GeneralizedOrnsteinUhlenbeckProcess's algebraic-limit branch, so
    # variance(t, ., dt) == vol(t)^2 * dt and expectation(t, x, dt) == x.
    vol_base = float(inputs["volBase"])
    vol_spike = float(inputs["volSpike"])
    spike_from = float(inputs["volSpikeFrom"])
    spike_to = float(inputs["volSpikeTo"])

    def vol(t: float) -> float:
        return vol_spike if spike_from <= t <= spike_to else vol_base

    return GeneralizedOrnsteinUhlenbeckProcess(
        speed=lambda _t: float(inputs["speed"]),
        vol=vol,
        x0=float(inputs["x0"]),
        level=float(inputs["level"]),
    )


def _grid_from(inputs: dict[str, Any]) -> TimeGrid:
    if inputs["grid"] == "uniform":
        return TimeGrid.regular(float(inputs["end"]), int(inputs["steps"]))
    return TimeGrid.with_mandatory([float(t) for t in inputs["mandatoryTimes"]])


@pytest.mark.parametrize("case", _TREE_CASES)
def test_tree_matches_cpp(cpp: dict[str, Any], case: str) -> None:
    block = cpp[case]
    inputs, expected = block["inputs"], block["expected"]
    tree = TrinomialTree(
        _process_from(inputs), _grid_from(inputs), bool(inputs["isPositive"])
    )

    n_steps = int(expected["nSteps"])
    tight(tree.x0, float(expected["x0"]))
    for i, dx in enumerate(expected["dx"]):
        tight(tree.dx(i), float(dx))
    for i, size in enumerate(expected["sizes"]):
        assert tree.size(i) == int(size)

    # Every node's descendants and branching probabilities.
    for step in expected["steps"]:
        i = int(step["i"])
        for node in step["nodes"]:
            index = int(node["index"])
            tight(tree.underlying(i, index), float(node["underlying"]))
            for branch in range(3):
                assert tree.descendant(i, index, branch) == int(
                    node["descendants"][branch]
                )
                tight(
                    tree.probability(i, index, branch),
                    float(node["probabilities"][branch]),
                )

    for index, underlying in enumerate(expected["terminalUnderlying"]):
        tight(tree.underlying(n_steps, index), float(underlying))


@pytest.mark.parametrize("case", _TREE_CASES)
def test_tree_floor_is_on_exactly_where_cpp_says(cpp: dict[str, Any], case: str) -> None:
    """``dx`` differs from the unfloored ``dx_natural`` on the floored steps only.

    The probe reports both, so this asserts the *state of the gate* rather than
    just the resulting numbers — a port that floored every step, or none, would
    still match many of the dx values above but not this.
    """
    expected = cpp[case]["expected"]
    dx = [float(v) for v in expected["dx"]]
    dx_natural = [float(v) for v in expected["dxNatural"]]
    inputs = cpp[case]["inputs"]
    tree = TrinomialTree(
        _process_from(inputs), _grid_from(inputs), bool(inputs["isPositive"])
    )
    for i, (d, dn) in enumerate(zip(dx, dx_natural, strict=True)):
        tight(tree.dx(i), d)
        if d != dn:
            assert tree.dx(i) > dn, f"step {i} must be floored"


def test_small_gap_tree_would_explode_without_the_floor(cpp: dict[str, Any]) -> None:
    """The pathology the floor exists for: a 1 ms mandatory gap after t = 1.

    Unfloored, the step's natural dx is ~3% of its neighbours', so the next
    slice needs ~30x the nodes to span the same range. The reference records
    the floored node counts; this pins that the port produces them rather than
    the explosion.
    """
    expected = cpp["tree_small_gap_ou"]["expected"]
    assert int(expected["maxNodes"]) < 20
    inputs = cpp["tree_small_gap_ou"]["inputs"]
    tree = TrinomialTree(
        _process_from(inputs), _grid_from(inputs), bool(inputs["isPositive"])
    )
    assert max(tree.size(i) for i in range(len(expected["sizes"]))) == int(
        expected["maxNodes"]
    )


# --------------------------------------------------------------------------
# G2Process / G2ForwardProcess
# --------------------------------------------------------------------------

_CURVE_DATES = [
    (15, Month.January, 2026),
    (15, Month.July, 2026),
    (15, Month.January, 2027),
    (15, Month.January, 2028),
    (15, Month.January, 2031),
    (15, Month.January, 2036),
    (15, Month.January, 2046),
    (15, Month.January, 2056),
]
_CURVE_ZEROS = [0.0180, 0.0215, 0.0245, 0.0290, 0.0335, 0.0360, 0.0372, 0.0368]

# Derived tolerances for the curve-backed cases. Nothing here is a round number
# picked to pass: each is the arithmetic's own amplification bound.
#
# ``phi(t)`` reads the instantaneous forward, which both C++ and this port
# compute as a centred difference of width dt = 1e-4:
# ``log(discount(t - dt/2) / discount(t + dt/2)) / dt``. Each discount factor
# carries at least one ulp of relative error (~1.1e-16 near 0.9), the ratio
# ~2.2e-16, ``log`` maps that to the same absolute error, and dividing by
# dt = 1e-4 amplifies it to ~2.2e-12 absolute in the rate. Add as much again
# for the interpolated zero curve's own last-bit disagreement between the two
# implementations, and 5e-12 is the bound. (Worst observed: 9.6e-14.)
_PHI_ABS_TOL = 5.0e-12

# ``drift`` differentiates phi again — ``(phi(t + h) - phi(t)) / h`` with
# h = 1e-4 — so a phi error of 1e-13 becomes 1e-9 in the drift. That is not a
# defect of either implementation: the reference curve is linear in the zero
# rate with a node at t = 1, where phi'(t) is genuinely discontinuous, and the
# probe deliberately samples it there (the drift is -7.24 at that point).
# (Worst observed: 3.3e-10.)
_DRIFT_ABS_TOL = 1.0e-9

# ``expectation`` uses phi linearly (``phi(t0+dt) - phi(t0)*exp(-a*dt)``), so
# phi's own bound carries over unamplified. (Worst observed: 9.3e-14.)
_EXPECTATION_ABS_TOL = 5.0e-12

_PHI_REASON = (
    "phi reads the instantaneous forward, a dt=1e-4 centred difference that "
    "amplifies a 1-ulp discount error to ~2e-12 in the rate"
)
_DRIFT_REASON = (
    "drift differentiates phi again over h=1e-4, amplifying phi's ~1e-13 to "
    "~1e-9; the probe samples t=1 where the linear zero curve has a node and "
    "phi'(t) is genuinely discontinuous"
)


def _zero_curve() -> InterpolatedZeroCurve:
    return InterpolatedZeroCurve(
        [Date.from_ymd(d, m, y) for d, m, y in _CURVE_DATES],
        _CURVE_ZEROS,
        Actual365Fixed(),
    )


def _g2(inputs: dict[str, Any], *, forward: bool) -> G2Process | G2ForwardProcess:
    curve = _zero_curve() if inputs["hasTermStructure"] else None
    args = (
        float(inputs["a"]),
        float(inputs["sigma"]),
        float(inputs["b"]),
        float(inputs["eta"]),
        float(inputs["rho"]),
        curve,
    )
    if not forward:
        return G2Process(*args)
    fwd = G2ForwardProcess(*args)
    fwd.set_forward_measure_time(float(inputs["forwardMeasureTime"]))
    return fwd


@pytest.mark.parametrize(
    ("case", "forward"),
    [
        ("g2_curve_scalars", False),
        ("g2_empty_scalars", False),
        ("g2fwd_curve_scalars", True),
        ("g2fwd_empty_scalars", True),
    ],
)
def test_g2_scalars_match_cpp(cpp: dict[str, Any], case: str, forward: bool) -> None:
    block = cpp[case]
    process = _g2(block["inputs"], forward=forward)
    expected = block["expected"]

    curve_backed = bool(block["inputs"]["hasTermStructure"])
    for value, want in zip(
        process.initial_values().tolist(), expected["initialValues"], strict=True
    ):
        if curve_backed:
            custom(
                float(value), float(want),
                abs_tol=_PHI_ABS_TOL, rel_tol=0.0, reason=_PHI_REASON,
            )
        else:
            tight(float(value), float(want))

    if expected["phi"] is None:
        # Empty-curve case: phi is undefined and must say so rather than
        # silently returning something.
        with pytest.raises(Exception, match="no term structure"):
            process.phi(1.0)
    else:
        for t, want in zip(expected["phiTimes"], expected["phi"], strict=True):
            custom(
                process.phi(float(t)), float(want),
                abs_tol=_PHI_ABS_TOL, rel_tol=0.0, reason=_PHI_REASON,
            )

    for entry in expected["shortRate"]:
        tight(
            process.short_rate(
                float(entry["t"]), float(entry["z1"]), float(entry["z2"])
            ),
            float(entry["shortRate"]),
        )


@pytest.mark.parametrize(
    ("case", "forward"),
    [
        ("g2_curve_drift", False),
        ("g2_empty_drift", False),
        ("g2fwd_curve_drift", True),
        ("g2fwd_empty_drift", True),
    ],
)
def test_g2_drift_matches_cpp(cpp: dict[str, Any], case: str, forward: bool) -> None:
    block = cpp[case]
    process = _g2(block["inputs"], forward=forward)
    curve_backed = bool(block["inputs"]["hasTermStructure"])
    for entry in block["expected"]:
        z = np.array([float(v) for v in entry["z"]], dtype=np.float64)
        drift = process.drift(float(entry["t"]), z)
        for got, want in zip(drift.tolist(), entry["drift"], strict=True):
            if curve_backed:
                custom(
                    float(got), float(want),
                    abs_tol=_DRIFT_ABS_TOL, rel_tol=0.0, reason=_DRIFT_REASON,
                )
            else:
                tight(float(got), float(want))


@pytest.mark.parametrize(
    ("case", "forward"),
    [
        ("g2_curve_evolution", False),
        ("g2_empty_evolution", False),
        ("g2fwd_curve_evolution", True),
        ("g2fwd_empty_evolution", True),
    ],
)
def test_g2_evolution_matches_cpp(cpp: dict[str, Any], case: str, forward: bool) -> None:
    block = cpp[case]
    process = _g2(block["inputs"], forward=forward)
    curve_backed = bool(block["inputs"]["hasTermStructure"])
    for entry in block["expected"]:
        t0 = float(entry["t0"])
        dt = float(entry["dt"])
        z0 = np.array([float(v) for v in entry["z0"]], dtype=np.float64)

        for got, want in zip(
            process.expectation(t0, z0, dt).tolist(), entry["expectation"], strict=True
        ):
            if curve_backed:
                custom(
                    float(got), float(want),
                    abs_tol=_EXPECTATION_ABS_TOL, rel_tol=0.0, reason=_PHI_REASON,
                )
            else:
                tight(float(got), float(want))
        for got_row, want_row in zip(
            process.std_deviation(t0, z0, dt).tolist(), entry["stdDeviation"], strict=True
        ):
            for got, want in zip(got_row, want_row, strict=True):
                tight(float(got), float(want))
        for got_row, want_row in zip(
            process.covariance(t0, z0, dt).tolist(), entry["covariance"], strict=True
        ):
            for got, want in zip(got_row, want_row, strict=True):
                tight(float(got), float(want))
        for got_row, want_row in zip(
            process.diffusion(t0, z0).tolist(), entry["diffusion"], strict=True
        ):
            for got, want in zip(got_row, want_row, strict=True):
                tight(float(got), float(want))


def test_g2_state_sums_to_the_short_rate(cpp: dict[str, Any]) -> None:
    """The point of the v1.43 shift: ``state[0] + state[1] == r(t)``.

    C++ parity: ``g2_curve_short_rate_identity``. With a curve attached the
    process starts at ``phi(0)`` and its expectation stays curve-consistent,
    so the two simulated components sum to the short rate by construction.
    """
    block = cpp["g2_curve_short_rate_identity"]
    process = _g2(block["inputs"], forward=False)
    expected = block["expected"]

    for value, want in zip(
        process.initial_values().tolist(), expected["initialValues"], strict=True
    ):
        custom(
            float(value), float(want),
            abs_tol=_PHI_ABS_TOL, rel_tol=0.0, reason=_PHI_REASON,
        )

    z0 = process.initial_values()
    for t, row, want_sum in zip(
        expected["times"], expected["expectation"], expected["expectationSum"], strict=True
    ):
        got = process.expectation(0.0, z0, float(t))
        for g, w in zip(got.tolist(), row, strict=True):
            custom(
                float(g), float(w),
                abs_tol=_EXPECTATION_ABS_TOL, rel_tol=0.0, reason=_PHI_REASON,
            )
        # E[state] summed is the short rate the curve implies at t.
        custom(
            float(got[0]) + float(got[1]), float(want_sum),
            abs_tol=_EXPECTATION_ABS_TOL, rel_tol=0.0, reason=_PHI_REASON,
        )
        custom(
            process.short_rate(float(t), float(got[0]), float(got[1])), float(want_sum),
            abs_tol=_EXPECTATION_ABS_TOL, rel_tol=0.0, reason=_PHI_REASON,
        )
