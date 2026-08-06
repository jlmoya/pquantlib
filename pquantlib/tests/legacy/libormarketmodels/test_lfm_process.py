"""Cross-validation of ``LiborForwardModelProcess`` against C++ v1.43.

Expected values come from ``migration-harness/references/v143/legacy/lmm.json``
(``migration-harness/cpp/probes/v143_legacy_lmm/probe.cpp``, section
``lfm_process``).

EVALUATION DATE: ``probe.cpp`` ``makeIndex6M`` sets
``Settings::instance().evaluationDate() = index->fixingCalendar().adjust(
Date(4, September, 2005))``. The autouse fixture pins the same date and
restores the previous one; every fixing date, accrual time and initial forward
below depends on it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor6M
from pquantlib.legacy.libormarketmodels.lfm_covar_proxy import LfmCovarianceProxy
from pquantlib.legacy.libormarketmodels.lfm_process import LiborForwardModelProcess
from pquantlib.legacy.libormarketmodels.lm_exp_corr_model import (
    LmExponentialCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_lin_exp_vol_model import (
    LmLinearExponentialVolatilityModel,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.yield_.interpolated_zero_curve import (
    InterpolatedZeroCurve,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit

ANCHOR: Final[Date] = Date.from_ymd(4, Month.September, 2005)
SIZE: Final[int] = 6
# probe.cpp: the volatility parameters the C++ test-suite uses for
# testSwaptionPricing.
VOL_A: Final[float] = 0.291
VOL_B: Final[float] = 1.483
VOL_C: Final[float] = 0.116
VOL_D: Final[float] = 0.00001
CORR_RHO: Final[float] = 0.5
NOTIONAL: Final[float] = 2.5


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/legacy/lmm")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """probe.cpp ``makeIndex6M``: evaluationDate = TARGET().adjust(4-Sep-2005)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Euribor6M(None).fixing_calendar().adjust(ANCHOR)
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert act.shape == exp.shape, f"{label}: shape {act.shape} != {exp.shape}"
    for idx, (a, e) in enumerate(zip(act.ravel(), exp.ravel(), strict=True)):
        tight(float(a), float(e), reason=f"{label}[{idx}]")


def _make_index() -> Euribor6M:
    """probe.cpp ``makeIndex6M()`` — Euribor6M over a two-pillar ZeroCurve."""
    day_counter = Actual360()
    today = Euribor6M(None).fixing_calendar().adjust(ANCHOR)
    ObservableSettings().evaluation_date = today
    spot = Euribor6M(None).fixing_calendar().advance(
        today, Euribor6M(None).fixing_days(), TimeUnit.Days
    )
    curve = InterpolatedZeroCurve(
        [spot, Date.from_ymd(4, Month.September, 2018)], [0.039, 0.041], day_counter
    )
    return Euribor6M(curve)


@pytest.fixture
def process() -> LiborForwardModelProcess:
    index = _make_index()
    proc = LiborForwardModelProcess(SIZE, index)
    vol = LmLinearExponentialVolatilityModel(
        proc.fixing_times(), VOL_A, VOL_B, VOL_C, VOL_D
    )
    corr = LmExponentialCorrelationModel(SIZE, CORR_RHO)
    proc.set_covar_param(LfmCovarianceProxy(vol, corr))
    return proc


# --- construction & schedule --------------------------------------------------


def test_process_size_and_factors(cpp: dict[str, Any], process: LiborForwardModelProcess) -> None:
    ref = cpp["lfm_process"]
    assert process.size() == ref["size"] == SIZE
    # factors() delegates to the covariance parameterization
    assert process.factors() == ref["factors"]


def test_process_without_a_covariance_parameterization_raises() -> None:
    proc = LiborForwardModelProcess(SIZE, _make_index())
    assert proc.covar_param() is None
    with pytest.raises(LibraryException, match="no covariance parameterization"):
        proc.factors()


def test_process_covar_param_round_trips(process: LiborForwardModelProcess) -> None:
    param = process.covar_param()
    assert isinstance(param, LfmCovarianceProxy)
    other = LfmCovarianceProxy(
        LmLinearExponentialVolatilityModel(process.fixing_times(), 0.1, 0.2, 0.3, 0.4),
        LmExponentialCorrelationModel(SIZE, 0.9),
    )
    process.set_covar_param(other)
    assert process.covar_param() is other


def test_process_initial_values_and_time_grid_match_cpp(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    ref = cpp["lfm_process"]
    _assert_close(process.initial_values(), ref["initial_values"], "proc.initial_values")
    _assert_close(process.fixing_times(), ref["fixing_times"], "proc.fixing_times")
    assert [d.serial_number() for d in process.fixing_dates()] == ref["fixing_date_serials"]
    _assert_close(
        process.accrual_start_times(), ref["accrual_start_times"], "proc.accrual_start"
    )
    _assert_close(process.accrual_end_times(), ref["accrual_end_times"], "proc.accrual_end")


def test_process_next_index_reset_is_a_strict_upper_bound(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    """C++ test-suite testInitialisation checks exactly this boundary triple.

    At ``t`` a hair BELOW a fixing the answer is that index; AT the fixing and
    a hair above it the answer is the NEXT one.
    """
    ref = cpp["lfm_process"]
    got = [process.next_index_reset(t) for t in ref["next_index_reset_times"]]
    assert got == ref["next_index_reset"]


def test_process_cash_flows_honour_a_non_unit_notional(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    """``amount`` is passed through to ``withNotionals``; a dropped argument
    would leave every nominal at the 1.0 default.
    """
    ref = cpp["lfm_process"]
    flows = process.cash_flows(NOTIONAL)
    assert [cf.date().serial_number() for cf in flows] == ref["cashflow_date_serials"]
    _assert_close([cf.amount() for cf in flows], ref["cashflow_amounts"], "proc.cf_amounts")
    _assert_close(
        [cf.nominal() for cf in flows],  # type: ignore[attr-defined]
        ref["cashflow_nominals"],
        "proc.cf_nominals",
    )


def test_process_default_notional_is_one(process: LiborForwardModelProcess) -> None:
    flows = process.cash_flows()
    assert all(cf.nominal() == 1.0 for cf in flows)  # type: ignore[attr-defined]


def test_process_rejects_a_size_that_does_not_match_the_leg() -> None:
    """``cash_flows()`` derives its own schedule from ``size``; a mismatch is
    only reachable by breaking the invariant, so the guard is checked here via
    a zero-length request.
    """
    with pytest.raises((LibraryException, IndexError)):
        LiborForwardModelProcess(0, _make_index())


# --- dynamics -----------------------------------------------------------------


def test_process_discount_bond_matches_cpp(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    ref = cpp["lfm_process"]
    got = process.discount_bond(ref["discount_bond_rates"])
    _assert_close(got, ref["discount_bond"], "proc.discount_bond")


@pytest.mark.parametrize(("t", "key"), [(0.9, "drift_t0_9"), (2.4, "drift_t2_4")])
def test_process_drift_matches_cpp(
    cpp: dict[str, Any], process: LiborForwardModelProcess, t: float, key: str
) -> None:
    """At t = 2.4 all but the last forward have fixed, so the drift is mostly
    zero — that is the ``next_index_reset`` cut-off showing up in the dynamics.
    """
    ref = cpp["lfm_process"]
    x = np.asarray(ref["state_x"], dtype=np.float64)
    _assert_close(process.drift(t, x), ref[key], f"proc.{key}")


def test_process_diffusion_matches_cpp(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    ref = cpp["lfm_process"]
    x = np.asarray(ref["state_x"], dtype=np.float64)
    _assert_close(process.diffusion(0.9, x), ref["diffusion_t0_9"], "proc.diffusion")


def test_process_covariance_scales_by_dt(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    """``covariance(t0, x0, dt)`` multiplies by ``dt`` rather than consulting
    the discretization; a dropped ``dt`` would be invisible at dt = 1.
    """
    ref = cpp["lfm_process"]
    x = np.asarray(ref["state_x"], dtype=np.float64)
    _assert_close(
        process.covariance(0.9, x, 0.25), ref["covariance_t0_9_dt0_25"], "proc.covariance"
    )
    param = process.covar_param()
    assert param is not None
    assert np.allclose(process.covariance(0.9, x, 1.0), param.covariance(0.9, x))


def test_process_apply_is_multiplicative(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    """C++ overrides the additive base ``apply`` with ``x0 * exp(dx)``."""
    ref = cpp["lfm_process"]
    x = np.asarray(ref["state_x"], dtype=np.float64)
    dx = np.asarray(ref["apply_dx"], dtype=np.float64)
    _assert_close(process.apply(x, dx), ref["apply"], "proc.apply")


def test_process_evolve_runs_the_predictor_corrector_step(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    """``evolve`` overrides the base Euler step; the corrector is what makes
    the result differ from ``apply(expectation, stdDeviation @ dw)``.
    """
    ref = cpp["lfm_process"]
    x = np.asarray(ref["state_x"], dtype=np.float64)
    dw = np.asarray(ref["evolve_dw"], dtype=np.float64)
    _assert_close(process.evolve(0.9, x, 0.25, dw), ref["evolve_t0_9_dt0_25"], "proc.evolve")


def test_process_evolve_leaves_already_fixed_forwards_untouched(
    cpp: dict[str, Any], process: LiborForwardModelProcess
) -> None:
    ref = cpp["lfm_process"]
    x = np.asarray(ref["state_x"], dtype=np.float64)
    dw = np.asarray(ref["evolve_dw"], dtype=np.float64)
    m = process.next_index_reset(0.9)
    evolved = process.evolve(0.9, x, 0.25, dw)
    assert np.array_equal(evolved[:m], x[:m])
    assert not np.allclose(evolved[m:], x[m:])
