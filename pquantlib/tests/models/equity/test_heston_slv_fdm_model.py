"""HestonSlvFdmModel scaffold tests.

C++ parity: ql/models/equity/hestonslvfdmmodel.{hpp,cpp}.

L11-W1-D ships a structural scaffold; the Fokker-Planck FDM solver is
deferred to Phase 11 W5. These tests confirm that the public API
(constructor + leverage_function + heston_process + local_vol) wires
up correctly and that the unit-leverage fallback is well-defined.

Tolerance choice:
* Public-API round-trips: EXACT — passthrough getters.
* Unit-leverage value: EXACT — L = 1 by construction in the scaffold.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.heston_slv_fdm_model import (
    HestonSlvFdmModel,
    HestonSlvFokkerPlanckFdmParams,
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
def fdm_model() -> HestonSlvFdmModel:
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
    local_vol = LocalVolSurface(black_ts=bvol, underlying=spot)
    return HestonSlvFdmModel(
        local_vol=local_vol,
        heston_model=heston_model,
        end_date=ref + 365,
    )


def test_default_params_match_cpp_test_defaults() -> None:
    """Default HestonSlvFokkerPlanckFdmParams mirrors the C++ test defaults."""
    p = HestonSlvFokkerPlanckFdmParams()
    assert p.x_grid == 201
    assert p.v_grid == 51
    assert p.t_max_steps_per_year == 200
    assert p.t_min_steps_per_year == 4


def test_heston_process_passthrough(fdm_model: HestonSlvFdmModel) -> None:
    """``heston_process()`` returns the model's underlying process.

    # C++ parity: hestonslvfdmmodel.hpp:83.
    """
    proc = fdm_model.heston_process()
    exact(proc.v0, 0.04)
    exact(proc.kappa, 2.0)


def test_local_vol_passthrough(fdm_model: HestonSlvFdmModel) -> None:
    """``local_vol()`` returns the input local-vol surface.

    # C++ parity: hestonslvfdmmodel.hpp:84.
    """
    lv = fdm_model.local_vol()
    # Verify at-the-money local vol matches the input constant Black vol.
    v = lv.local_vol_at_time(0.5, 100.0, extrapolate=True)
    # Constant Black vol = 0.20 → constant local vol = 0.20 (Dupire FD
    # introduces float64 round-off, hence TIGHT not EXACT).
    tight(v, 0.20, reason="Dupire FD introduces ~1e-14 float64 noise")


def test_leverage_function_unit_at_atm(fdm_model: HestonSlvFdmModel) -> None:
    """Scaffold leverage function returns L=1 at ATM.

    # Scaffold-only — see module docstring.
    """
    leverage = fdm_model.leverage_function()
    # L = 1.0 everywhere on the unit-leverage grid.
    v = leverage.local_vol_at_time(0.5, 100.0, extrapolate=True)
    exact(v, 1.0)


def test_leverage_function_is_cached(fdm_model: HestonSlvFdmModel) -> None:
    """Repeated calls return the same surface object (lazy cache)."""
    lev1 = fdm_model.leverage_function()
    lev2 = fdm_model.leverage_function()
    assert lev1 is lev2


def test_mixing_factor_default_is_one(fdm_model: HestonSlvFdmModel) -> None:
    """Default mixing factor is 1.0."""
    exact(fdm_model.mixing_factor(), 1.0)
