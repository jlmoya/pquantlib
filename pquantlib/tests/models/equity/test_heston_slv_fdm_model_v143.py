"""Cross-validate HestonSLVFDMModel + LogEntry against C++ QuantLib v1.43.

Probe source: migration-harness/cpp/probes/v143_models_hestonslvfdm/probe.cpp
Reference:    migration-harness/references/v143/models/hestonslvfdm.json

C++ classes:
    ``HestonSLVFDMModel``              ql/models/equity/hestonslvfdmmodel.hpp:73
    ``HestonSLVFDMModel::LogEntry``    ql/models/equity/hestonslvfdmmodel.hpp:87
    ``HestonSLVFokkerPlanckFdmParams`` ql/models/equity/hestonslvfdmmodel.hpp:43

The model was a scaffold returning ``L = 1`` on a 5x5 grid; ``LogEntry`` did
not exist, because a scaffold has no density to log.

Order of assertions
-------------------
The mesh comes first. ``performCalculations`` builds one variance mesher per
rescale step of the ``LocalVolRNDCalculator`` and reuses the previous mesher
otherwise, so a port that rescales at different steps builds a different mesh
sequence and nothing downstream can agree for the right reason. Then the
leverage function, then the log entries.

The configuration's working envelope
------------------------------------
``local_vol_eps_prob`` is 1e-3 here rather than the 1e-4 the C++ test-suite
uses, and that is load-bearing. ``v_mesher[0]`` is the DEGENERATE
``Predefined1dMesher([v0] * v_grid)`` — every node at ``v0``, zero spacing —
and is only replaced at a rescale step. If the first rescale step is not 1,
``v_mesher[1]`` stays degenerate and the first ``reshapePDF`` interpolates
from a mesh whose ``y_min`` equals its ``y_max``: every target ``v`` falls
outside the range, the guard zeroes the density, and ``rescalePDF`` then
divides by zero. A 72-point sweep over the fixture found the correspondence
exact — ``rescaleTimeSteps()[0] == 1`` calibrates, ``== 2`` throws "could not
converge". ``test_first_rescale_step_is_one`` pins that precondition
explicitly so a port cannot drift out of the envelope silently.

Tolerance
---------
Mesh geometry and the time grid: TIGHT.

Leverage and densities: a CUSTOM 1e-6 relative tier, derived rather than
chosen. The Rannacher steps run ``ImplicitEulerScheme``, whose linear solve
is BiCGstab converged to a RELATIVE RESIDUAL of 1e-8 — that is the C++
default (``ImplicitEulerScheme(map, bcSet, 1e-8, BiCGstab)``) and the port's.
An iterative solver stopped at a residual, not at a fixed point, does not
produce a bit-reproducible iterate across two linear-algebra stacks; it
produces one accurate to its own tolerance. The calibration then takes about
27 time steps, each with 2 predictor-corrector iterations, and each step's
leverage column is fed straight back into the operator for the next one, so a
1e-8 perturbation propagates rather than averaging out. The reachable bound
is O(n · relTol) with n ≈ 54 solves, i.e. ~5e-7 — which is what is observed
(worst case 4.7e-7, on the Power transformation).

1e-8 (the LOOSE tier) is therefore below the noise floor of the algorithm, not
a standard this port fails to meet.
``test_agreement_is_inside_the_derived_bound`` pins the ACHIEVED worst-case
gap so a regression that merely stays inside 1e-6 still fails.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.finitedifferences.operators.fdm_square_root_fwd_op import (
    TransformationType,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.utilities.fdm_heston_greens_fct import (
    FdmHestonGreensFctAlgorithm,
)
from pquantlib.methods.finitedifferences.utilities.local_vol_rnd_calculator import (
    LocalVolRNDCalculator,
)
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.heston_slv_fdm_model import (
    HestonSLVFDMModel,
    HestonSLVFokkerPlanckFdmParams,
    LogEntry,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.time_grid import TimeGrid


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/models/hestonslvfdm")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date(cpp_ref: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Probe: ``Settings::instance().evaluationDate() = TODAY`` (probe main)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Date(int(cpp_ref["evaluationDate_serial"]))
    yield
    settings.evaluation_date = previous


class _Fixture:
    __slots__ = ("dc", "heston_model", "local_vol", "q_ts", "r_ts", "spot", "today")

    def __init__(self, cpp_ref: dict[str, Any]) -> None:
        self.today = Date(int(cpp_ref["evaluationDate_serial"]))
        self.dc = Actual365Fixed()
        self.spot = SimpleQuote(cpp_ref["s0"])
        self.r_ts = FlatForward(self.today, SimpleQuote(cpp_ref["r"]), self.dc)
        self.q_ts = FlatForward(self.today, SimpleQuote(cpp_ref["q"]), self.dc)

        times = list(cpp_ref["lv_times"])
        strikes = list(cpp_ref["lv_strikes"])
        matrix = np.empty((len(strikes), len(times)), dtype=np.float64)
        for i, k in enumerate(strikes):
            for j, t in enumerate(times):
                # Probe: ``localVolValue``.
                matrix[i][j] = 0.30 + 0.10 * (100.0 / k - 1.0) - 0.02 * t
        surface = FixedLocalVolSurface(
            reference_date=self.today,
            times=times,
            strikes=strikes,
            local_vol_matrix=matrix,
            day_counter=self.dc,
        )
        surface.enable_extrapolation()
        self.local_vol = surface

        process = HestonProcess(
            risk_free_rate=self.r_ts,
            dividend_yield=self.q_ts,
            s0=self.spot,
            v0=cpp_ref["v0"],
            kappa=cpp_ref["kappa"],
            theta=cpp_ref["theta"],
            sigma=cpp_ref["sigma"],
            rho=cpp_ref["rho"],
        )
        self.heston_model = HestonModel(process)


def _params(
    cpp_ref: dict[str, Any],
    trafo_type: TransformationType,
    greens: FdmHestonGreensFctAlgorithm,
    scheme_desc: FdmSchemeDesc,
) -> HestonSLVFokkerPlanckFdmParams:
    """Probe: ``makeParams`` — two predictor-corrector steps throughout."""
    return HestonSLVFokkerPlanckFdmParams(
        x_grid=int(cpp_ref["xGrid"]),
        v_grid=int(cpp_ref["vGrid"]),
        t_max_steps_per_year=100,
        t_min_steps_per_year=25,
        t_step_number_decay=100.0,
        n_rannacher_time_steps=2,
        prediction_correction_steps=2,
        x0_density=0.1,
        local_vol_eps_prob=1e-3,
        max_integration_iterations=10000,
        v_lower_eps=1e-5,
        v_upper_eps=1e-5,
        v_min=0.0000025,
        v0_density=1.0,
        v_lower_bound_density=0.1,
        v_upper_bound_density=0.9,
        leverage_fct_prop_eps=1e-5,
        greens_algorithm=greens,
        trafo_type=trafo_type,
        scheme_desc=scheme_desc,
    )


#: Derived in the module docstring: BiCGstab's own 1e-8 relative residual,
#: compounded over ~54 implicit solves with leverage feedback.
_REL_TOL = 1e-6
_ABS_TOL = 1e-10
_REASON = (
    "Rannacher steps use ImplicitEulerScheme/BiCGstab at relTol 1e-8; "
    "compounded over ~54 solves with leverage feedback the reachable bound "
    "is O(n * relTol) ~ 5e-7 (observed worst case 4.7e-7)"
)

_CASES = {
    "lev_log_pc2": (
        TransformationType.Log,
        FdmHestonGreensFctAlgorithm.ZeroCorrelation,
        FdmSchemeDesc.modified_craig_sneyd,
    ),
    "lev_plain_pc2": (
        TransformationType.Plain,
        FdmHestonGreensFctAlgorithm.Gaussian,
        FdmSchemeDesc.modified_craig_sneyd,
    ),
    "lev_power_pc2": (
        TransformationType.Power,
        FdmHestonGreensFctAlgorithm.Gaussian,
        FdmSchemeDesc.modified_craig_sneyd,
    ),
    "lev_log_hundsdorfer": (
        TransformationType.Log,
        FdmHestonGreensFctAlgorithm.ZeroCorrelation,
        FdmSchemeDesc.hundsdorfer,
    ),
}


# ---------------------------------------------------------------------------
# Section A — the mesh performCalculations is built on
# ---------------------------------------------------------------------------


def _time_grid(cpp_ref: dict[str, Any], fx: _Fixture) -> tuple[list[float], TimeGrid]:
    params = _params(
        cpp_ref,
        TransformationType.Log,
        FdmHestonGreensFctAlgorithm.ZeroCorrelation,
        FdmSchemeDesc.modified_craig_sneyd(),
    )
    t_end = fx.dc.year_fraction(fx.today, Date(int(cpp_ref["finalDate_serial"])))
    max_dt = 1.0 / params.t_max_steps_per_year
    min_dt = 1.0 / params.t_min_steps_per_year
    t_idx = 0.0
    times = [t_idx]
    while t_idx < t_end:
        decay = math.exp(-params.t_step_number_decay * t_idx)
        t_idx += max_dt * decay + min_dt * (1.0 - decay)
        times.append(min(t_end, t_idx))
    return times, TimeGrid.with_mandatory(times)


def test_local_vol_surface_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """The input surface, before the model touches it."""
    fx = _Fixture(cpp_ref)
    got: list[float] = []
    for t in (0.0, 0.1, 0.25, 0.5, 1.0):
        for k in (60.0, 80.0, 100.0, 125.0, 160.0):
            got.append(fx.local_vol.local_vol_at_time(t, k, True))
    for a, b in zip(got, cpp_ref["mesh_localVol_probe"], strict=True):
        tolerance.tight(a, b)


def test_time_grid_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """The exponentially decaying step schedule (hestonslvfdmmodel.cpp:314-333)."""
    fx = _Fixture(cpp_ref)
    times, grid = _time_grid(cpp_ref, fx)
    tolerance.tight(
        fx.dc.year_fraction(fx.today, Date(int(cpp_ref["finalDate_serial"]))),
        cpp_ref["mesh_T"],
    )
    for a, b in zip(times, cpp_ref["mesh_raw_times"], strict=True):
        tolerance.tight(a, b)
    assert grid.size() == cpp_ref["mesh_grid_size"]
    for a, b in zip(grid.times, cpp_ref["mesh_grid_times"], strict=True):
        tolerance.tight(a, b)


def test_first_rescale_step_is_one(cpp_ref: dict[str, Any]) -> None:
    """The precondition the whole calibration rests on.

    If ``rescale_time_steps()[0] != 1`` then ``v_mesher[1]`` stays the
    degenerate ``Predefined1dMesher([v0] * v_grid)`` and the first
    ``_reshape_pdf`` interpolates from a mesh whose ``y_min == y_max``,
    zeroing the density and making ``_rescale_pdf`` divide by zero. C++ fails
    the same way, with "could not converge" out of BiCGstab.
    """
    assert cpp_ref["mesh_rescale_steps"][0] == 1
    fx = _Fixture(cpp_ref)
    _, grid = _time_grid(cpp_ref, fx)
    rnd = LocalVolRNDCalculator(
        fx.spot,
        fx.r_ts,
        fx.q_ts,
        fx.local_vol,
        x_grid=int(cpp_ref["xGrid"]),
        x0_density=0.1,
        local_vol_prob_eps=1e-3,
        max_iter=10000,
        time_grid=grid,
    )
    assert rnd.rescale_time_steps() == cpp_ref["mesh_rescale_steps"]


def test_x_mesher_sequence_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """One x-mesher per time step; the model indexes them by step."""
    fx = _Fixture(cpp_ref)
    _, grid = _time_grid(cpp_ref, fx)
    rnd = LocalVolRNDCalculator(
        fx.spot,
        fx.r_ts,
        fx.q_ts,
        fx.local_vol,
        x_grid=int(cpp_ref["xGrid"]),
        x0_density=0.1,
        local_vol_prob_eps=1e-3,
        max_iter=10000,
        time_grid=grid,
    )
    for i in range(grid.size()):
        locs = rnd.mesher(grid.at(i)).locations()
        tolerance.tight(float(locs[0]), cpp_ref["mesh_x_front"][i])
        tolerance.tight(float(locs[-1]), cpp_ref["mesh_x_back"][i])
    locs1 = rnd.mesher(grid.at(1)).locations()
    for a, b in zip(locs1, cpp_ref["mesh_x1_locations"], strict=True):
        tolerance.tight(float(a), b)


# ---------------------------------------------------------------------------
# Section B — the calibrated leverage function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag", list(_CASES))
def test_leverage_function_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``HestonSLVFDMModel::performCalculations`` (hestonslvfdmmodel.cpp:281-529).

    Four configurations: all three ``TransformationType`` values and two FD
    schemes. ``prediction_correction_steps`` is 0 in NO case — with 0 the C++
    inner loop never runs and the leverage matrix past column 1 is read
    uninitialised (``new Matrix(...)`` does not value-initialise), which the
    probe caught as run-to-run nondeterminism and therefore does not pin.
    """
    assert cpp_ref[f"{tag}_error"] == "", "probe recorded a C++ failure here"
    trafo, greens, scheme = _CASES[tag]
    fx = _Fixture(cpp_ref)
    model = HestonSLVFDMModel(
        fx.local_vol,
        fx.heston_model,
        Date(int(cpp_ref["finalDate_serial"])),
        _params(cpp_ref, trafo, greens, scheme()),
    )
    leverage = model.leverage_function()
    got: list[float] = []
    for t in cpp_ref["lev_times"]:
        for s in cpp_ref["lev_spots"]:
            got.append(leverage.local_vol_at_time(t, s, True))
    for a, b in zip(got, cpp_ref[f"{tag}_leverage"], strict=True):
        tolerance.custom(a, b, abs_tol=_ABS_TOL, rel_tol=_REL_TOL, reason=_REASON)


def test_agreement_is_inside_the_derived_bound(cpp_ref: dict[str, Any]) -> None:
    """1e-6 is the ceiling, not the achieved accuracy.

    Pins the actual worst-case relative gap across every leverage case, so a
    regression that degrades agreement to, say, 8e-7 — still "passing" the
    custom tier — fails here instead of passing quietly.
    """
    worst = 0.0
    for tag, (trafo, greens, scheme) in _CASES.items():
        fx = _Fixture(cpp_ref)
        model = HestonSLVFDMModel(
            fx.local_vol,
            fx.heston_model,
            Date(int(cpp_ref["finalDate_serial"])),
            _params(cpp_ref, trafo, greens, scheme()),
        )
        leverage = model.leverage_function()
        got = [
            leverage.local_vol_at_time(t, s, True)
            for t in cpp_ref["lev_times"]
            for s in cpp_ref["lev_spots"]
        ]
        for a, b in zip(got, cpp_ref[f"{tag}_leverage"], strict=True):
            worst = max(worst, abs(a - b) / abs(b))
    assert worst < 5.0e-7, f"worst relative gap {worst!r} exceeded 5e-7"


def test_leverage_is_not_the_unit_surface(cpp_ref: dict[str, Any]) -> None:
    """The scaffold this replaced returned ``L = 1`` everywhere.

    Asserted on the C++ reference so the fixture's discriminating power is
    itself pinned: a re-scaffolded port must not be able to pass
    ``test_leverage_function_matches_cpp``.
    """
    values = cpp_ref["lev_log_pc2_leverage"]
    assert max(abs(v - 1.0) for v in values) > 0.1


def test_passthroughs_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """``hestonProcess()`` and ``localVol()`` (hestonslvfdmmodel.cpp:266-272)."""
    fx = _Fixture(cpp_ref)
    model = HestonSLVFDMModel(
        fx.local_vol,
        fx.heston_model,
        Date(int(cpp_ref["finalDate_serial"])),
        _params(
            cpp_ref,
            TransformationType.Log,
            FdmHestonGreensFctAlgorithm.ZeroCorrelation,
            FdmSchemeDesc.modified_craig_sneyd(),
        ),
    )
    assert model.heston_process() is fx.heston_model.process()
    assert model.local_vol() is fx.local_vol
    tolerance.tight(model.heston_process().v0, cpp_ref["v0"])
    tolerance.tight(model.heston_process().kappa, cpp_ref["kappa"])


# ---------------------------------------------------------------------------
# Section C — LogEntry
# ---------------------------------------------------------------------------


def _logged_model(cpp_ref: dict[str, Any], logging: bool) -> HestonSLVFDMModel:
    fx = _Fixture(cpp_ref)
    return HestonSLVFDMModel(
        fx.local_vol,
        fx.heston_model,
        Date(int(cpp_ref["finalDate_serial"])),
        _params(
            cpp_ref,
            TransformationType.Log,
            FdmHestonGreensFctAlgorithm.ZeroCorrelation,
            FdmSchemeDesc.modified_craig_sneyd(),
        ),
        logging=logging,
    )


@pytest.fixture(scope="module")
def entries(cpp_ref: dict[str, Any]) -> list[LogEntry]:
    """One calibration shared by every LogEntry assertion (it is not cheap)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Date(int(cpp_ref["evaluationDate_serial"]))
    try:
        return _logged_model(cpp_ref, logging=True).log_entries()
    finally:
        settings.evaluation_date = previous


def test_logging_false_records_nothing(cpp_ref: dict[str, Any]) -> None:
    """C++ guards both push_backs on ``logging_`` (cpp:420 and :521).

    ``logEntries()`` still runs the whole calibration — it calls
    ``performCalculations()`` directly, not ``calculate()`` — and then returns
    an empty list.
    """
    assert cpp_ref["log_quiet_count"] == 0
    assert _logged_model(cpp_ref, logging=False).log_entries() == []


def test_log_entry_times_match_cpp(
    cpp_ref: dict[str, Any], entries: list[LogEntry]
) -> None:
    """One entry per time step from index 1 on (cpp:421 and :523)."""
    assert len(entries) == cpp_ref["log_count"]
    for e, t in zip(entries, cpp_ref["log_t"], strict=True):
        tolerance.tight(e.t, t)


def test_log_entry_meshers_match_cpp(
    cpp_ref: dict[str, Any], entries: list[LogEntry]
) -> None:
    """Each entry carries the mesher its density lives on, not a copy of it."""
    for i, e in enumerate(entries):
        dim = e.mesher.layout().dim()
        assert dim[0] == cpp_ref["log_dim0"][i]
        assert dim[1] == cpp_ref["log_dim1"][i]
        xm = e.mesher.get_fdm_1d_meshers()[0].locations()
        vm = e.mesher.get_fdm_1d_meshers()[1].locations()
        tolerance.tight(float(xm[0]), cpp_ref["log_x_front"][i])
        tolerance.tight(float(xm[-1]), cpp_ref["log_x_back"][i])
        tolerance.tight(float(vm[0]), cpp_ref["log_v_front"][i])
        tolerance.tight(float(vm[-1]), cpp_ref["log_v_back"][i])


def test_log_entry_densities_match_cpp(
    cpp_ref: dict[str, Any], entries: list[LogEntry]
) -> None:
    """The Fokker-Planck density itself — the payload ``LogEntry`` exists for."""
    for i, e in enumerate(entries):
        for got, want in (
            (float(np.sum(e.prob)), cpp_ref["log_prob_sum"][i]),
            (float(np.min(e.prob)), cpp_ref["log_prob_min"][i]),
            (float(np.max(e.prob)), cpp_ref["log_prob_max"][i]),
        ):
            tolerance.custom(
                got, want, abs_tol=_ABS_TOL, rel_tol=_REL_TOL, reason=_REASON
            )


def test_last_log_entry_full_density_matches_cpp(
    cpp_ref: dict[str, Any], entries: list[LogEntry]
) -> None:
    """Every node of the density that has been through every step."""
    last = entries[-1]
    expected = cpp_ref["log_last_prob"]
    assert last.prob.shape[0] == len(expected)
    for a, b in zip(last.prob, expected, strict=True):
        tolerance.custom(
            float(a), b, abs_tol=_ABS_TOL, rel_tol=_REL_TOL, reason=_REASON
        )


def test_log_entry_is_a_snapshot_not_a_view(
    cpp_ref: dict[str, Any], entries: list[LogEntry]
) -> None:
    """C++ stores ``ext::make_shared<Array>(p)`` — a COPY of the density.

    cpp:422 and :523. If the entry aliased the working array instead, every
    entry would end up holding the final density; asserting the entries differ
    is the cheapest way to catch that.
    """
    assert len({float(np.sum(e.prob)) for e in entries}) > 1
    assert entries[0].prob is not entries[-1].prob
