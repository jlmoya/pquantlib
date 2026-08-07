"""HestonSLVFDMModel structural tests.

C++ parity: ql/models/equity/hestonslvfdmmodel.{hpp,cpp} (v1.43).

Cheap checks that need no calibration: the parameter pack's defaults and the
two passthrough accessors. The model's actual behaviour — the Fokker-Planck
calibration, the leverage function and the LogEntry snapshots — is
cross-validated against C++ v1.43 in ``test_heston_slv_fdm_model_v143.py``,
which is where anything that has to run a calibration belongs.

This file used to assert ``leverage_function()`` returns L = 1 everywhere.
That was true of the scaffold and is false of the model; the assertion is
gone rather than relaxed.

Tolerance choice:
* Public-API round-trips: EXACT — passthrough getters.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.heston_slv_fdm_model import (
    HestonSLVFDMModel,
    HestonSLVFokkerPlanckFdmParams,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_surface import (
    LocalVolSurface,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def fdm_model() -> HestonSLVFDMModel:
    """Build a canonical SLV FDM model on the Heston testbed."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    spot = SimpleQuote(100.0)
    process = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=spot,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
    )
    heston_model = HestonModel(process)
    # Synthetic flat local vol from a constant Black vol.
    bvol = BlackConstantVol(
        reference_date=ref,
        calendar=NullCalendar(),
        volatility=0.20,
        day_counter=dc,
    )
    local_vol = LocalVolSurface(
        black_ts=bvol, risk_free_ts=rf, dividend_ts=div, underlying=spot
    )
    return HestonSLVFDMModel(
        local_vol=local_vol,
        heston_model=heston_model,
        end_date=ref + 365,
    )


def test_default_params_match_cpp_test_defaults() -> None:
    """Default HestonSLVFokkerPlanckFdmParams mirrors the C++ test defaults.

    Except ``prediction_correction_steps``. The C++ struct declares no
    defaults, so these are the port's; 0 is the one value that cannot be a
    sensible default, since it makes the calibration loop body never execute
    and leaves the leverage matrix unwritten past column 1.
    """
    p = HestonSLVFokkerPlanckFdmParams()
    assert p.x_grid == 201
    assert p.v_grid == 51
    assert p.t_max_steps_per_year == 200
    assert p.t_min_steps_per_year == 4
    assert p.prediction_correction_steps == 2


def test_heston_process_passthrough(fdm_model: HestonSLVFDMModel) -> None:
    """``heston_process()`` returns the model's underlying process.

    # C++ parity: hestonslvfdmmodel.hpp:83.
    """
    proc = fdm_model.heston_process()
    exact(proc.v0, 0.04)
    exact(proc.kappa, 2.0)


def test_local_vol_passthrough(fdm_model: HestonSLVFDMModel) -> None:
    """``local_vol()`` returns the input local-vol surface.

    # C++ parity: hestonslvfdmmodel.hpp:84.
    """
    lv = fdm_model.local_vol()
    # Verify at-the-money local vol matches the input constant Black vol.
    v = lv.local_vol_at_time(0.5, 100.0, extrapolate=True)
    # Constant Black vol = 0.20 → constant local vol = 0.20 (Dupire FD
    # introduces float64 round-off, hence TIGHT not EXACT).
    tight(v, 0.20, reason="Dupire FD introduces ~1e-14 float64 noise")


def test_model_starts_uncalculated(fdm_model: HestonSLVFDMModel) -> None:
    """``LazyObject``: nothing is computed until ``leverage_function()`` asks.

    # C++ parity: ``HestonSLVFDMModel : public LazyObject``
    # (hestonslvfdmmodel.hpp:73); ``leverageFunction()`` opens with
    # ``calculate()`` (cpp:274-279).

    The leverage function itself is cross-validated in
    ``test_heston_slv_fdm_model_v143.py``, on a fixture whose configuration is
    inside the calibration's working envelope; this fixture's is not, and
    running one here would prove nothing the v143 file does not.
    """
    assert not fdm_model.is_calculated()


def test_mixing_factor_default_is_one(fdm_model: HestonSLVFDMModel) -> None:
    """Default mixing factor is 1.0."""
    exact(fdm_model.mixing_factor(), 1.0)
