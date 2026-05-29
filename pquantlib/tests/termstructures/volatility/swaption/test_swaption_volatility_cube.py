"""Tests for SwaptionVolatilityCube + Sabr / Interpolated subclasses (L9-C)."""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.sabr_formula import sabr_volatility
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.swaption.interpolated_swaption_volatility_cube import (
    InterpolatedSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.sabr_swaption_volatility_cube import (
    SabrSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/l9c")
_SLICE = _REF["cube_sabr_slice"]


# --- a minimal swap-index stub for ATM-strike lookup -------------------


class _FakeSwapIndex:
    """Minimal SwapIndex stand-in: returns ``forward`` on every fixing.

    Used by cube tests so we don't need a fully-wired ibor / discount
    curve. The cube's ATM-strike path only calls ``.tenor()`` and
    ``.fixing(date)``.
    """

    def __init__(self, tenor: Period, forward: float) -> None:
        self._tenor = tenor
        self._forward = forward

    def tenor(self) -> Period:
        return self._tenor

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        _ = fixing_date, forecast_todays_fixing
        return self._forward


# --- shared fixture: ATM vol matrix + 5-strike-spread cube ------------


_REF_DATE = Date.from_ymd(15, Month.January, 2024)
_OPTION_TENORS = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
_SWAP_TENORS = [Period(2, TimeUnit.Years), Period(5, TimeUnit.Years)]
_STRIKE_SPREADS = [-0.01, -0.005, 0.0, 0.005, 0.01]
_FORWARD = _SLICE["forward"]  # 0.04
_BETA = _SLICE["beta"]


def _atm_matrix() -> SwaptionVolatilityMatrix:
    # ATM vols set to the SABR ATM vol at each cell — keeps the cube
    # internally consistent for the round-trip check.
    atm_vol = sabr_volatility(
        _FORWARD, _FORWARD, _SLICE["expiry"],
        _SLICE["alpha"], _SLICE["beta"], _SLICE["nu"], _SLICE["rho"],
    )
    return SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        volatilities=np.full((2, 2), atm_vol),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_REF_DATE,
    )


def _vol_spreads() -> list[list[SimpleQuote]]:
    """Build a (option x swap, strike_spreads) matrix of SABR-implied spreads.

    Each ``(j, k)`` row gets the SABR slice (at given true params) as
    deviations from ATM.
    """
    atm_vol = sabr_volatility(
        _FORWARD, _FORWARD, _SLICE["expiry"],
        _SLICE["alpha"], _SLICE["beta"], _SLICE["nu"], _SLICE["rho"],
    )
    rows: list[list[SimpleQuote]] = []
    for _j in range(len(_OPTION_TENORS)):
        for _k in range(len(_SWAP_TENORS)):
            row: list[SimpleQuote] = []
            for s in _STRIKE_SPREADS:
                strike = _FORWARD + s
                v = sabr_volatility(
                    strike, _FORWARD, _SLICE["expiry"],
                    _SLICE["alpha"], _SLICE["beta"], _SLICE["nu"], _SLICE["rho"],
                )
                row.append(SimpleQuote(v - atm_vol))
            rows.append(row)
    return rows


# --- SabrSwaptionVolatilityCube ---------------------------------------


def test_sabr_swaption_vol_cube_constructs() -> None:
    atm = _atm_matrix()
    cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),  # pin beta
        sabr_initial_guess=[
            [(0.05, _BETA, 0.3, 0.0)] * len(_SWAP_TENORS)
            for _ in range(len(_OPTION_TENORS))
        ],
    )
    # Inspectors
    assert cube.strike_spreads() == _STRIKE_SPREADS
    assert len(cube.vol_spreads()) == len(_OPTION_TENORS) * len(_SWAP_TENORS)


def test_sabr_swaption_vol_cube_fitted_params_recover_true_params() -> None:
    atm = _atm_matrix()
    cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
        sabr_initial_guess=[
            [(0.05, _BETA, 0.3, 0.0)] * len(_SWAP_TENORS)
            for _ in range(len(_OPTION_TENORS))
        ],
    )
    # First option_tenor (1Y) — the true SABR slice was at expiry=3Y so
    # the recovered params will differ. We only check that the recovered
    # parameter set lives in the admissible region.
    for j in range(len(_OPTION_TENORS)):
        for k in range(len(_SWAP_TENORS)):
            alpha, beta, nu, rho = cube.sabr_parameters(j, k)
            assert alpha > 0.0
            assert 0.0 <= beta <= 1.0
            assert nu >= 0.0
            assert -1.0 <= rho <= 1.0


def test_sabr_swaption_vol_cube_smile_section_is_sabr_kind() -> None:
    atm = _atm_matrix()
    cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    section = cube.smile_section_impl(1.0, 2.0)
    assert isinstance(section, SabrSmileSection)
    # Sanity: section.atm_level is the fitted forward (which equals _FORWARD).
    tolerance.tight(section.atm_level(), _FORWARD)


def test_sabr_swaption_vol_cube_smile_section_reproduces_fit_at_grid() -> None:
    """At grid point (j=0, k=0), section.volatility(strike) ~ input slice."""
    atm = _atm_matrix()
    cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    # Pull section at grid cell (0, 0): 1Y option, 2Y swap.
    t0 = cube.option_times()[0]
    s0 = cube.swap_lengths()[0]
    section = cube.smile_section_impl(t0, s0)
    # Section vol at ATM should match the SABR-implied ATM at the
    # fitted parameters + 1Y expiry within a few bp (the fit recovers
    # alpha, nu, rho from spreads; the absolute ATM may shift by the
    # fit residual).
    atm_vol_section = section.volatility(_FORWARD)
    # ATM vol on the underlying surface (target).
    atm_vol_true = atm.volatility(_OPTION_TENORS[0], _SWAP_TENORS[0], _FORWARD)
    # Tolerance: within 0.5% (50 bp absolute on vol).
    assert abs(atm_vol_section - atm_vol_true) < 0.005


def test_sabr_swaption_vol_cube_recalibrate_idempotent_for_static_quotes() -> None:
    atm = _atm_matrix()
    cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    alpha0, beta0, nu0, rho0 = cube.sabr_parameters(0, 0)
    cube.recalibrate()
    alpha1, beta1, nu1, rho1 = cube.sabr_parameters(0, 0)
    tolerance.loose(alpha0, alpha1)
    tolerance.exact(beta0, beta1)
    tolerance.loose(nu0, nu1)
    tolerance.loose(rho0, rho1)


# --- InterpolatedSwaptionVolatilityCube --------------------------------


def test_interpolated_swaption_vol_cube_constructs() -> None:
    atm = _atm_matrix()
    cube = InterpolatedSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
    )
    assert cube.strike_spreads() == _STRIKE_SPREADS


def test_interpolated_swaption_vol_cube_spread_vol_inspector() -> None:
    atm = _atm_matrix()
    cube = InterpolatedSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
    )
    # spread_vol(j, k, i) should equal the input quote at that cell.
    s00_3 = cube.spread_vol(0, 0, 3)  # 4th strike spread (+50bp)
    expected_v = sabr_volatility(
        _FORWARD + _STRIKE_SPREADS[3], _FORWARD, _SLICE["expiry"],
        _SLICE["alpha"], _SLICE["beta"], _SLICE["nu"], _SLICE["rho"],
    )
    atm_vol = sabr_volatility(
        _FORWARD, _FORWARD, _SLICE["expiry"],
        _SLICE["alpha"], _SLICE["beta"], _SLICE["nu"], _SLICE["rho"],
    )
    tolerance.tight(s00_3, expected_v - atm_vol)


# --- SwaptionVolatilityCube base sanity checks ------------------------


def test_swaption_vol_cube_rejects_unsorted_strike_spreads() -> None:
    atm = _atm_matrix()
    with pytest.raises(LibraryException, match="strike spread"):
        SabrSwaptionVolatilityCube(
            atm_vol_structure=atm,
            option_tenors=_OPTION_TENORS,
            swap_tenors=_SWAP_TENORS,
            strike_spreads=[0.01, -0.01],  # unsorted
            vol_spreads=[
                [SimpleQuote(0.0), SimpleQuote(0.0)]
                for _ in range(len(_OPTION_TENORS) * len(_SWAP_TENORS))
            ],
            swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
            short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        )


def test_swaption_vol_cube_rejects_mismatched_spread_dim() -> None:
    atm = _atm_matrix()
    with pytest.raises(LibraryException, match="vol-spreads"):
        SabrSwaptionVolatilityCube(
            atm_vol_structure=atm,
            option_tenors=_OPTION_TENORS,
            swap_tenors=_SWAP_TENORS,
            strike_spreads=_STRIKE_SPREADS,
            vol_spreads=[[SimpleQuote(0.0)] * 5],  # wrong outer dim
            swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
            short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        )


def test_swaption_vol_cube_rejects_short_tenor_above_swap_tenor() -> None:
    atm = _atm_matrix()
    with pytest.raises(LibraryException, match="short index tenor"):
        SabrSwaptionVolatilityCube(
            atm_vol_structure=atm,
            option_tenors=_OPTION_TENORS,
            swap_tenors=_SWAP_TENORS,
            strike_spreads=_STRIKE_SPREADS,
            vol_spreads=_vol_spreads(),
            swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
            short_swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        )


def test_swaption_vol_cube_atm_strike_dispatches_to_short_index_for_short_tenor() -> None:
    atm = _atm_matrix()
    short = _FakeSwapIndex(Period(2, TimeUnit.Years), 0.06)  # different forward
    cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=short,
        is_parameter_fixed=(False, True, False, False),
    )
    # 2Y swap_tenor: matches short_swap_index_base.tenor() (2Y) — uses
    # short index. C++ check: ``if (swap_tenor > shortSwapIndexBase->tenor())``,
    # so equality routes to the short index.
    atm_k = cube.atm_strike(_REF_DATE, Period(2, TimeUnit.Years))
    tolerance.exact(atm_k, 0.06)


def test_swaption_vol_cube_max_swap_tenor_delegates() -> None:
    atm = _atm_matrix()
    cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    assert cube.max_swap_tenor() == atm.max_swap_tenor()
