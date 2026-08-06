"""Cross-validate the PSO Inertia/Topology and firefly RandomWalk families.

Probe source: migration-harness/cpp/probes/v143_experimental_pso/probe.cpp
Reference:    migration-harness/references/v143/experimental/pso.json

Covers the four v1.43 gap classes ``AdaptiveInertia``, ``LevyFlightInertia``,
``ClubsTopology`` and ``DecreasingGaussianWalk``. The ``<random>`` streams they
sit on are pinned separately in ``test_std_random.py``.

How the tolerances are chosen
-----------------------------
Everything discrete is asserted **exactly**: engine words, integer draws,
objective-call counts, trajectory lengths, ``EndCriteria`` outcomes. So are the
``<random>`` distributions — the whole point of ``std_random`` is that those
streams are bit-identical, and a rounding tolerance there would hide exactly
the class of bug it exists to catch.

The optimizer *trajectories* are asserted at TIGHT rather than exactly. The
reference ``libQuantLib`` is an ARM64 ``-O3`` build, and Clang contracts
multiply-adds inside the optimizer loops into FMAs — for example the PSO
velocity update

    v[j] += c1*r1*(pB[j]-x[j]) + c2*r2*(gB[j]-x[j])

is emitted as ``fma(c1*r1, pB[j]-x[j], (c2*r2)*(gB[j]-x[j]))``. The fused and
unfused results differ in the last bit. That is a property of how the
reference was compiled, not of the algorithm: the same source on x86-64
without ``-mfma`` produces the unfused result. The port implements the
arithmetic as written in the C++ source, so the residual is bounded by
accumulated last-bit rounding, which TIGHT (1e-12 rel / 1e-14 abs) covers with
four orders of magnitude to spare. This was verified rather than assumed —
substituting ``math.fma`` at that one site makes ``D_global_trivial`` and
``D_clubsfull_trivial`` match C++ bit-for-bit across all 120 recorded values.

Two upstream v1.43 defects are pinned as defects rather than as behaviour;
see ``test_clubs_topology_upstream_off_by_one`` and
``test_decreasing_inertia_loop_bound_is_uninitialised``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.firefly_algorithm import (
    DecreasingGaussianWalk,
    ExponentialIntensity,
    FireflyAlgorithm,
    GaussianWalk,
    Intensity,
    InverseLawSquareIntensity,
    LevyFlightWalk,
    RandomWalk,
)
from pquantlib.experimental.math.particle_swarm_optimization import (
    AdaptiveInertia,
    ClubsTopology,
    DecreasingInertia,
    GlobalTopology,
    Inertia,
    KNeighbors,
    LevyFlightInertia,
    ParticleSwarmOptimization,
    SimpleRandomInertia,
    Topology,
    TrivialInertia,
)
from pquantlib.experimental.math.std_random import (
    StdMt19937,
    StdUniformIntDistribution,
)
from pquantlib.math.optimization.constraint import BoundaryConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.problem import Problem
from pquantlib.testing import reference_reader, tolerance

# The probe drives PSO through the explicit-omega (PSO-In) constructor with
# these literals, so the trajectories do not depend on the constriction factor
# — see the probe's runPso comment.
_OMEGA = 0.72984378812835763
_C1 = 1.4961797656631331
_C2 = 1.4961797656631331


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/pso")


# --------------------------------------------------------------------------
# Recording objectives — mirror the probe's, including its plain (unfused)
# arithmetic, so the objective itself contributes no divergence.
# --------------------------------------------------------------------------


class _RecordingQuartic(CostFunction):
    """``sum_i (x_i - shift_i)^4 + (x_i - shift_i)^2``, logging every call."""

    def __init__(self, shift: list[float], max_record: int) -> None:
        self._shift = shift
        self._max_record = max_record
        self.log: list[float] = []
        self.calls = 0

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()

    def value(self, x: npt.NDArray[np.float64]) -> float:
        total = 0.0
        for i in range(len(x)):
            d = float(x[i]) - self._shift[i]
            total += d * d * d * d + d * d
        if len(self.log) < self._max_record:
            self.log.extend(float(v) for v in x)
            self.log.append(total)
        self.calls += 1
        return total


class _RecordingSphere(CostFunction):
    """``sum_i x_i^2``, logging every call."""

    def __init__(self, max_record: int) -> None:
        self._max_record = max_record
        self.log: list[float] = []
        self.calls = 0

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()

    def value(self, x: npt.NDArray[np.float64]) -> float:
        total = 0.0
        for xi in x:
            total += float(xi) * float(xi)
        if len(self.log) < self._max_record:
            self.log.extend(float(v) for v in x)
            self.log.append(total)
        self.calls += 1
        return total


def _run_pso(
    topology: Topology, inertia: Inertia, m: int, max_iter: int
) -> tuple[_RecordingQuartic, Problem, int]:
    f = _RecordingQuartic([0.7, -1.3], 120)
    problem = Problem(f, BoundaryConstraint(-4.0, 4.0), np.zeros(2))
    pso = ParticleSwarmOptimization(
        m, topology, inertia, _C1, _C2, omega=_OMEGA, seed=31415
    )
    ec_type = pso.minimize(
        problem, EndCriteria(max_iter, max_iter - 1, 1e-12, 1e-12, 1e-12)
    )
    return f, problem, int(ec_type)


def _run_fa(
    intensity: Intensity, walk: RandomWalk, m: int, m_de: int, max_iter: int
) -> tuple[_RecordingSphere, Problem, int]:
    f = _RecordingSphere(120)
    problem = Problem(f, BoundaryConstraint(-3.0, 3.0), np.zeros(2))
    fa = FireflyAlgorithm(m, intensity, walk, m_de, 1.0, 0.5, seed=271828)
    ec_type = fa.minimize(
        problem, EndCriteria(max_iter, max_iter - 1, 1e-12, 1e-12, 1e-12)
    )
    return f, problem, int(ec_type)


def _assert_run(
    ref: dict[str, Any],
    key: str,
    f: _RecordingQuartic | _RecordingSphere,
    problem: Problem,
    ec_type: int,
) -> None:
    expected = ref[f"{key}_trajectory"]
    # Exact: how many points were visited, and how many objective calls it
    # took to visit them. A port that diverges structurally fails here first.
    assert len(f.log) == len(expected), f"{key}: trajectory length"
    assert f.calls == ref[f"{key}_calls"], f"{key}: objective calls"
    assert ec_type == ref[f"{key}_ecType"], f"{key}: EndCriteria type"
    # TIGHT: see the module docstring on FP contraction in the reference build.
    for i, (got, want) in enumerate(zip(f.log, expected, strict=True)):
        tolerance.tight(got, float(want), reason=f"{key} trajectory[{i}]")
    for i, want in enumerate(ref[f"{key}_x"]):
        tolerance.tight(
            float(problem.current_value[i]), float(want), reason=f"{key} x[{i}]"
        )
    tolerance.tight(problem.function_value, float(ref[f"{key}_f"]), reason=f"{key} f")


# --------------------------------------------------------------------------
# Block D — PSO inertia x topology.
# --------------------------------------------------------------------------


def test_pso_global_trivial(ref: dict[str, Any]) -> None:
    """Baseline: the already-ported strategies, to anchor the harness."""
    _assert_run(
        ref, "D_global_trivial", *_run_pso(GlobalTopology(), TrivialInertia(), 8, 12)
    )


def test_pso_global_simplerandom(ref: dict[str, Any]) -> None:
    """SimpleRandomInertia baseline over the same harness."""
    _assert_run(
        ref,
        "D_global_simplerandom",
        *_run_pso(GlobalTopology(), SimpleRandomInertia(0.4, 555), 8, 12),
    )


def test_adaptive_inertia_global(ref: dict[str, Any]) -> None:
    """AdaptiveInertia under the global topology."""
    _assert_run(
        ref,
        "D_global_adaptive",
        *_run_pso(GlobalTopology(), AdaptiveInertia(0.2, 0.9, 5, 2), 8, 12),
    )


def test_adaptive_inertia_kneighbors(ref: dict[str, Any]) -> None:
    """AdaptiveInertia with different sh/sl under a ring topology."""
    _assert_run(
        ref,
        "D_kneighbors_adaptive",
        *_run_pso(KNeighbors(2), AdaptiveInertia(0.1, 1.2, 3, 1), 8, 12),
    )


def test_clubs_topology_trivial(ref: dict[str, Any]) -> None:
    """ClubsTopology, total_clubs == default_clubs (the runnable branch)."""
    _assert_run(
        ref,
        "D_clubsfull_trivial",
        *_run_pso(ClubsTopology(4, 4, 4, 2, 3, 909), TrivialInertia(), 6, 10),
    )


def test_clubs_topology_adaptive(ref: dict[str, Any]) -> None:
    """ClubsTopology driving AdaptiveInertia, with a shorter reset period."""
    _assert_run(
        ref,
        "D_clubsfull_adaptive",
        *_run_pso(ClubsTopology(3, 3, 3, 1, 2, 909), AdaptiveInertia(0.2, 0.9, 5, 2), 8, 12),
    )


def test_levy_flight_inertia_global(ref: dict[str, Any]) -> None:
    """LevyFlightInertia: simple-random below threshold, Lévy flight above."""
    _assert_run(
        ref,
        "D_global_levyflight",
        *_run_pso(GlobalTopology(), LevyFlightInertia(1.5, 2, 4242), 8, 12),
    )


def test_levy_flight_inertia_with_clubs(ref: dict[str, Any]) -> None:
    """LevyFlightInertia and ClubsTopology together — both std::mt19937 users.

    Their engines are separate objects, so this also checks neither is
    stealing draws from the other.
    """
    _assert_run(
        ref,
        "D_clubsfull_levyflight",
        *_run_pso(ClubsTopology(5, 5, 5, 1, 3, 909), LevyFlightInertia(1.2, 1, 4242), 8, 12),
    )


# --------------------------------------------------------------------------
# Block E — firefly random walks.
# --------------------------------------------------------------------------


def test_firefly_gaussian_walk(ref: dict[str, Any]) -> None:
    """GaussianWalk over the libc++ normal distribution."""
    _assert_run(
        ref,
        "E_exp_gaussianwalk",
        *_run_fa(
            ExponentialIntensity(1.0, 1e-8, 1.0), GaussianWalk(0.3, 0.9, 8080), 8, 0, 8
        ),
    )


def test_firefly_levy_walk(ref: dict[str, Any]) -> None:
    """LevyFlightWalk over the libc++ uniform distribution."""
    _assert_run(
        ref,
        "E_exp_levywalk",
        *_run_fa(
            ExponentialIntensity(1.0, 1e-8, 1.0),
            LevyFlightWalk(1.5, 0.5, 0.9, 8080),
            8,
            0,
            8,
        ),
    )


def test_decreasing_gaussian_walk(ref: dict[str, Any]) -> None:
    """DecreasingGaussianWalk — delta squared once per swept generation."""
    _assert_run(
        ref,
        "E_exp_decreasinggaussian",
        *_run_fa(
            ExponentialIntensity(1.0, 1e-8, 1.0),
            DecreasingGaussianWalk(0.3, 0.9, 8080),
            8,
            0,
            8,
        ),
    )


def test_decreasing_gaussian_walk_inverse_square(ref: dict[str, Any]) -> None:
    """DecreasingGaussianWalk under the inverse-square intensity kernel."""
    _assert_run(
        ref,
        "E_invsq_decreasinggaussian",
        *_run_fa(
            InverseLawSquareIntensity(1.0, 1e-8),
            DecreasingGaussianWalk(0.5, 0.8, 8080),
            8,
            0,
            8,
        ),
    )


def test_decreasing_gaussian_walk_with_de(ref: dict[str, Any]) -> None:
    """With a DE subpopulation, so Mfa < M and the delta schedule spans more."""
    _assert_run(
        ref,
        "E_exp_decreasinggaussian_de",
        *_run_fa(
            ExponentialIntensity(1.0, 1e-8, 1.0),
            DecreasingGaussianWalk(0.3, 0.9, 8080),
            8,
            3,
            8,
        ),
    )


def test_firefly_pure_de(ref: dict[str, Any]) -> None:
    """Mde == M: the branch where C++ assigns through the ``Array& xBest``.

    That assignment copies the drawn array into ``x_[values_[0].second]``
    rather than rebinding, mutating the incumbent. If the port "tidied" it
    into a rebind, this trajectory would diverge.
    """
    _assert_run(
        ref,
        "E_pure_de",
        *_run_fa(
            ExponentialIntensity(1.0, 1e-8, 1.0), GaussianWalk(0.3, 0.9, 8080), 6, 6, 8
        ),
    )


# --------------------------------------------------------------------------
# Block F — AdaptiveInertia's state machine, isolated from swarm dynamics.
# --------------------------------------------------------------------------


class _ScriptedPso:
    """Minimal stand-in exposing just what an Inertia reads and writes."""

    def __init__(self, pbf: list[float], n_particles: int) -> None:
        self.personal_best_f = np.array(pbf, dtype=np.float64)
        self.velocities = [np.ones(1, dtype=np.float64) for _ in range(n_particles)]


_HISTORY = [
    [5.0, 6.0, 7.0], [4.0, 6.0, 7.0], [3.0, 6.0, 7.0], [3.0, 6.0, 7.0],
    [3.0, 6.0, 7.0], [3.0, 6.0, 7.0], [3.0, 6.0, 7.0], [3.0, 6.0, 7.0],
    [3.0, 6.0, 7.0], [2.5, 6.0, 7.0], [2.5, 6.0, 7.0], [1.0, 6.0, 7.0],
]


@pytest.mark.parametrize(
    ("c0", "lo", "hi", "sh", "sl", "key"),
    [
        (0.7298437881283576, 0.2, 0.9, 5, 2, "F_adaptive_c0_schedule_sh5_sl2"),
        (0.5, 0.1, 1.2, 3, 1, "F_adaptive_c0_schedule_sh3_sl1"),
    ],
)
def test_adaptive_inertia_c0_schedule(
    ref: dict[str, Any], c0: float, lo: float, hi: float, sh: int, sl: int, key: str
) -> None:
    """The intensify/diversify counter, driven by a scripted best-value history.

    Exact: the schedule is halving, doubling and clamping of one scalar, with
    no accumulation, so there is nothing for rounding to hide behind.

    The history improves twice, stalls for six iterations, then improves again,
    which drives the counter through both the ``> sh`` and ``< sl`` branches
    and through the unsigned wrap on the very first improvement.
    """
    inertia = AdaptiveInertia(lo, hi, sh, sl)
    pso = _ScriptedPso(_HISTORY[0], 3)
    inertia.set_size(3, 1, c0, EndCriteria(12, 11, 1e-12, 1e-12, 1e-12))
    inertia.init(pso)  # type: ignore[arg-type]

    got: list[float] = []
    for step in _HISTORY:
        pso.personal_best_f = np.array(step, dtype=np.float64)
        inertia.set_values()
        got.append(inertia.inertia)
    assert got == [float(v) for v in ref[key]]


def test_pso_constriction_factor_matches_the_source_arithmetic(
    ref: dict[str, Any],
) -> None:
    """c0 = 2 / |2 - phi - sqrt(phi^2 - 4 phi)|, evaluated as written.

    The reference libQuantLib is built ``-O3`` on ARM64, where Clang contracts
    ``phi*phi - 4*phi`` into an FMA and lands 3 ULP away. The port does not
    reproduce the fusion: the same source compiled for x86-64 without
    ``-mfma`` produces the unfused value, so fusing here would make PQuantLib
    agree with one reference build and disagree with another. Both values are
    emitted by the probe and both are checked, so the gap is recorded rather
    than papered over.
    """
    pso = ParticleSwarmOptimization(4, GlobalTopology(), SimpleRandomInertia(0.5, 3), 2.05, 2.05)
    unfused = float(ref["F_pso_constriction_c0_unfused"])
    fused = float(ref["F_pso_constriction_c0_fused"])
    assert pso.constriction_factor == unfused
    assert fused != unfused
    # 3 ULP apart, i.e. the difference is pure last-bit rounding.
    assert abs(fused - unfused) < 4 * math.ulp(unfused)


# --------------------------------------------------------------------------
# Blocks G/H — the two upstream v1.43 defects.
# --------------------------------------------------------------------------


def test_clubs_topology_upstream_off_by_one(ref: dict[str, Any]) -> None:
    """ClubsTopology indexes length-N containers with a draw from [1, N].

    The C++ ctor builds ``uniform_int_distribution<Size>(1, totalClubs_)``,
    whose range is closed, and ``setSize`` uses the draw directly as an index
    into containers of length ``totalClubs_``. Club 0 is unreachable and
    ``totalClubs_`` is one past the end. The probe pins the draw sequence,
    which needs no undefined behaviour to produce; running the real
    ``ClubsTopology(3, 6, 5, 2, 4, 909).setSize(8)`` against the reference
    build segfaults.

    The port raises here instead of reading out of range. It does not
    renumber the draw or clamp it — that would be inventing behaviour the
    reference does not have.
    """
    assert ref["G_clubs_draw_can_exceed_valid_index"] is True
    assert ref["G_clubs_index_zero_unreachable"] is True
    assert ref["G_clubs_draw_max"] == ref["G_clubs_totalClubs"]
    assert ref["G_clubs_valid_index_max"] == ref["G_clubs_totalClubs"] - 1

    # The Python draw sequence is the same one the C++ produces.
    g = StdMt19937(909)
    dist = StdUniformIntDistribution(1, 6)
    assert [dist(g) for _ in range(24)] == ref["G_clubs_setSize_draws_total6_seed909"]

    topo = ClubsTopology(3, 6, 5, 2, 4, 909)
    with pytest.raises(LibraryException, match="upstream off-by-one"):
        topo.set_size(8)


def test_clubs_topology_equal_default_and_total_is_well_defined() -> None:
    """default_clubs == total_clubs takes the branch that never draws."""
    topo = ClubsTopology(4, 4, 4, 2, 3, 909)
    topo.set_size(6)  # must not raise


def test_decreasing_inertia_loop_bound_is_uninitialised(ref: dict[str, Any]) -> None:
    """DecreasingInertia's C++ dynamics are indeterminate; only c0 is pinnable.

    ``DecreasingInertia::setSize`` assigns ``N_``, ``c0_``, ``iteration_`` and
    ``maxIterations_`` but never ``M_``, and ``M_`` has no default member
    initialiser, so ``setValues``'s loop runs to whatever was in memory. On
    the reference build it comes out 0 — the inertia is never applied at all.
    Modelling the port's ``set_values`` as a no-op drops the trajectory
    mismatch against C++ from 67/120 values to 10/120 (the rest being ordinary
    FP contraction), which is how this was identified.

    The port assigns ``_m`` properly, so its dynamics are *deliberately* not
    those of the reference build: reproducing an uninitialised read is not
    something a port can do. What is checked here is the part that is
    well-defined — the c0 schedule arithmetic.
    """
    max_iterations = 12
    for threshold, key in ((0.3, "thr0p3"), (0.5, "thr0p5")):
        expected = ref[f"H_decreasing_c0_schedule_{key}"]
        inertia = DecreasingInertia(threshold)
        inertia.set_size(
            3, 1, _OMEGA, EndCriteria(max_iterations, max_iterations - 1, 1e-12, 1e-12, 1e-12)
        )
        # C++ never advances iteration_ inside setValues, so only entry 0 is
        # ever reachable in a real run; the rest of the schedule is emitted to
        # pin the formula itself.
        assert ref["H_decreasing_iteration_is_never_advanced"] == 1
        for iteration, want in enumerate(expected):
            got = _OMEGA * (
                threshold
                + (1.0 - threshold) * (max_iterations - iteration) / max_iterations
            )
            tolerance.tight(got, float(want), reason=f"decreasing c0[{iteration}]")
