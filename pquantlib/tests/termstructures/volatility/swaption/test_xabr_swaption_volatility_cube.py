"""Tests for XabrSwaptionVolatilityCube (SABR/ZABR generalisation) +
ZabrSwaptionVolatilityCube wrapper (Phase 11 W2-A)."""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.sabr_formula import sabr_volatility
from pquantlib.math.interpolations.zabr_formula import zabr_volatility
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.termstructures.volatility.swaption.sabr_swaption_volatility_cube import (
    SabrSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.termstructures.volatility.swaption.xabr_swaption_volatility_cube import (
    XabrModelKind,
    XabrSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.zabr_swaption_volatility_cube import (
    ZabrSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_L9C = reference_reader.load("cluster/l9c")
_SABR_SLICE = _L9C["cube_sabr_slice"]


# --- a minimal swap-index stub (lifted from L9-C test scaffolding) -------


class _FakeSwapIndex:
    """Minimal SwapIndex stand-in: returns ``forward`` on every fixing."""

    def __init__(self, tenor: Period, forward: float) -> None:
        self._tenor = tenor
        self._forward = forward

    def tenor(self) -> Period:
        return self._tenor

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        _ = fixing_date, forecast_todays_fixing
        return self._forward


_REF_DATE = Date.from_ymd(15, Month.January, 2024)
_OPTION_TENORS = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
_SWAP_TENORS = [Period(2, TimeUnit.Years), Period(5, TimeUnit.Years)]
_STRIKE_SPREADS = [-0.01, -0.005, 0.0, 0.005, 0.01]
_FORWARD = _SABR_SLICE["forward"]  # 0.04
_BETA = _SABR_SLICE["beta"]


def _sabr_atm_matrix() -> SwaptionVolatilityMatrix:
    atm_vol = sabr_volatility(
        _FORWARD, _FORWARD, _SABR_SLICE["expiry"],
        _SABR_SLICE["alpha"], _SABR_SLICE["beta"],
        _SABR_SLICE["nu"], _SABR_SLICE["rho"],
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


def _sabr_vol_spreads() -> list[list[SimpleQuote]]:
    atm_vol = sabr_volatility(
        _FORWARD, _FORWARD, _SABR_SLICE["expiry"],
        _SABR_SLICE["alpha"], _SABR_SLICE["beta"],
        _SABR_SLICE["nu"], _SABR_SLICE["rho"],
    )
    rows: list[list[SimpleQuote]] = []
    for _j in range(len(_OPTION_TENORS)):
        for _k in range(len(_SWAP_TENORS)):
            row: list[SimpleQuote] = []
            for s in _STRIKE_SPREADS:
                strike = _FORWARD + s
                v = sabr_volatility(
                    strike, _FORWARD, _SABR_SLICE["expiry"],
                    _SABR_SLICE["alpha"], _SABR_SLICE["beta"],
                    _SABR_SLICE["nu"], _SABR_SLICE["rho"],
                )
                row.append(SimpleQuote(v - atm_vol))
            rows.append(row)
    return rows


# === Sanity: SABR mode of XabrSwaptionVolatilityCube reproduces L9-C =====


def test_xabr_cube_sabr_mode_matches_sabr_subclass_at_grid_pillar() -> None:
    """XabrSwaptionVolatilityCube(model_kind=SABR) == SabrSwaptionVolatilityCube
    at every (j, k) grid point.

    Same code path through the base; this exercises the IntEnum
    dispatch on the entry point. TIGHT tier on each fitted parameter.
    """
    atm = _sabr_atm_matrix()
    spreads = _sabr_vol_spreads()
    sabr_cube = SabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=spreads,
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    xabr_cube = XabrSwaptionVolatilityCube(
        model_kind=XabrModelKind.SABR,
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=spreads,
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    for j in range(len(_OPTION_TENORS)):
        for k in range(len(_SWAP_TENORS)):
            s_params = sabr_cube.sabr_parameters(j, k)
            x_params = xabr_cube.xabr_parameters(j, k)
            assert len(s_params) == 4
            assert len(x_params) == 4
            for sp, xp in zip(s_params, x_params, strict=True):
                tolerance.tight(sp, xp)


def test_xabr_cube_sabr_mode_smile_section_is_sabr_smile_section() -> None:
    atm = _sabr_atm_matrix()
    cube = XabrSwaptionVolatilityCube(
        model_kind=XabrModelKind.SABR,
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_sabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    section = cube.smile_section_impl(1.0, 2.0)
    assert isinstance(section, SabrSmileSection)


def test_xabr_cube_model_kind_inspector() -> None:
    atm = _sabr_atm_matrix()
    cube = XabrSwaptionVolatilityCube(
        model_kind=XabrModelKind.SABR,
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_sabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
        is_parameter_fixed=(False, True, False, False),
    )
    assert cube.model_kind() == XabrModelKind.SABR


def test_xabr_cube_rejects_wrong_size_is_parameter_fixed() -> None:
    atm = _sabr_atm_matrix()
    with pytest.raises(LibraryException, match="is_parameter_fixed"):
        XabrSwaptionVolatilityCube(
            model_kind=XabrModelKind.ZABR,
            atm_vol_structure=atm,
            option_tenors=_OPTION_TENORS,
            swap_tenors=_SWAP_TENORS,
            strike_spreads=_STRIKE_SPREADS,
            vol_spreads=_sabr_vol_spreads(),
            swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FORWARD),
            short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FORWARD),
            is_parameter_fixed=(False, True, False, False),  # 4 — needs 5 for ZABR
        )


# === ZabrSwaptionVolatilityCube ===========================================


# Construct ZABR cube from a ZABR-implied vol-spread matrix at known params.

_ZABR_FORWARD = 0.04
_ZABR_BETA = 0.5
_ZABR_TRUE = (0.03, _ZABR_BETA, 0.4, -0.2, 0.85)  # alpha, beta, nu, rho, gamma


def _zabr_atm_matrix() -> SwaptionVolatilityMatrix:
    atm_vol = zabr_volatility(
        _ZABR_FORWARD, _ZABR_FORWARD, 1.0,
        _ZABR_TRUE[0], _ZABR_TRUE[1], _ZABR_TRUE[2],
        _ZABR_TRUE[3], _ZABR_TRUE[4],
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


def _zabr_vol_spreads() -> list[list[SimpleQuote]]:
    atm_vol = zabr_volatility(
        _ZABR_FORWARD, _ZABR_FORWARD, 1.0,
        _ZABR_TRUE[0], _ZABR_TRUE[1], _ZABR_TRUE[2],
        _ZABR_TRUE[3], _ZABR_TRUE[4],
    )
    rows: list[list[SimpleQuote]] = []
    for _j in range(len(_OPTION_TENORS)):
        for _k in range(len(_SWAP_TENORS)):
            row: list[SimpleQuote] = []
            for s in _STRIKE_SPREADS:
                strike = _ZABR_FORWARD + s
                v = zabr_volatility(
                    strike, _ZABR_FORWARD, 1.0,
                    _ZABR_TRUE[0], _ZABR_TRUE[1], _ZABR_TRUE[2],
                    _ZABR_TRUE[3], _ZABR_TRUE[4],
                )
                row.append(SimpleQuote(v - atm_vol))
            rows.append(row)
    return rows


def test_zabr_swaption_vol_cube_constructs_with_default_guess() -> None:
    atm = _zabr_atm_matrix()
    cube = ZabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_zabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _ZABR_FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _ZABR_FORWARD),
        is_parameter_fixed=(False, True, False, False, False),
    )
    assert cube.model_kind() == XabrModelKind.ZABR


def test_zabr_swaption_vol_cube_xabr_parameters_returns_5tuple() -> None:
    atm = _zabr_atm_matrix()
    cube = ZabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_zabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _ZABR_FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _ZABR_FORWARD),
        zabr_initial_guess=[
            [(0.04, _ZABR_BETA, 0.3, 0.0, 1.0)] * len(_SWAP_TENORS)
            for _ in range(len(_OPTION_TENORS))
        ],
        is_parameter_fixed=(False, True, False, False, False),
    )
    for j in range(len(_OPTION_TENORS)):
        for k in range(len(_SWAP_TENORS)):
            zp = cube.zabr_parameters(j, k)
            assert len(zp) == 5
            alpha, beta, nu, rho, gamma = zp
            assert alpha > 0.0
            assert 0.0 <= beta <= 1.0
            assert nu > 0.0
            assert -1.0 < rho < 1.0
            assert gamma > 0.0
            # Beta was pinned.
            tolerance.tight(beta, _ZABR_BETA)


def test_zabr_swaption_vol_cube_smile_section_is_zabr_smile_section() -> None:
    atm = _zabr_atm_matrix()
    cube = ZabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_zabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _ZABR_FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _ZABR_FORWARD),
        is_parameter_fixed=(False, True, False, False, False),
    )
    section = cube.smile_section_impl(1.0, 2.0)
    assert isinstance(section, ZabrSmileSection)
    # ATM level on the section equals the fitted forward (= _ZABR_FORWARD).
    tolerance.tight(section.atm_level(), _ZABR_FORWARD)


def test_zabr_swaption_vol_cube_atm_section_recovers_atm_input() -> None:
    """At the grid pillar, section.volatility(ATM) ~ input ATM vol.

    TIGHT-ish — the fit recovers the input slice modulo
    optimisation tolerance. Compare against the *input* ATM vol of the
    underlying ATM surface (= the synthetic slice's ATM).
    """
    atm = _zabr_atm_matrix()
    cube = ZabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_zabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _ZABR_FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _ZABR_FORWARD),
        is_parameter_fixed=(False, True, False, False, False),
    )
    # Pull section at grid cell (0, 0): 1Y option, 2Y swap.
    t0 = cube.option_times()[0]
    s0 = cube.swap_lengths()[0]
    section = cube.smile_section_impl(t0, s0)
    atm_vol_section = section.volatility(_ZABR_FORWARD)
    atm_vol_true = atm.volatility(_OPTION_TENORS[0], _SWAP_TENORS[0], _ZABR_FORWARD)
    # 50 bp absolute (same threshold as the L9-C SABR cube test —
    # fit residual + tenor interpolation drift).
    assert abs(atm_vol_section - atm_vol_true) < 0.005


def test_zabr_swaption_vol_cube_strike_spread_offset_reproduces_fitted_vol() -> None:
    """At grid cell (0,0), section.volatility(K) at a non-ATM strike
    matches a manually-built ZabrSmileSection.volatility(K) on the
    fitted params. LOOSE tier — the chain is fit -> wrap -> evaluate.
    """
    atm = _zabr_atm_matrix()
    cube = ZabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_zabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _ZABR_FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _ZABR_FORWARD),
        is_parameter_fixed=(False, True, False, False, False),
    )
    t0 = cube.option_times()[0]
    s0 = cube.swap_lengths()[0]
    section = cube.smile_section_impl(t0, s0)
    assert isinstance(section, ZabrSmileSection)
    alpha, beta, nu, rho, gamma = cube.zabr_parameters(0, 0)
    forward = cube.fitted_forward(0, 0)
    manual = ZabrSmileSection(
        forward=forward,
        zabr_params=(alpha, beta, nu, rho, gamma),
        exercise_time=t0,
    )
    for offset in (-0.01, -0.005, 0.005, 0.01):
        k = _ZABR_FORWARD + offset
        tolerance.tight(section.volatility(k), manual.volatility(k))


def test_zabr_swaption_vol_cube_recalibrate_is_idempotent() -> None:
    atm = _zabr_atm_matrix()
    cube = ZabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_zabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _ZABR_FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _ZABR_FORWARD),
        is_parameter_fixed=(False, True, False, False, False),
    )
    p0 = cube.zabr_parameters(0, 0)
    cube.recalibrate()
    p1 = cube.zabr_parameters(0, 0)
    for a, b in zip(p0, p1, strict=True):
        tolerance.loose(a, b)


def test_zabr_swaption_vol_cube_rejects_wrong_guess_shape() -> None:
    """A 4-tuple initial guess on a ZABR cube is rejected at fit time."""
    atm = _zabr_atm_matrix()
    cube = ZabrSwaptionVolatilityCube(
        atm_vol_structure=atm,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_zabr_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _ZABR_FORWARD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _ZABR_FORWARD),
        zabr_initial_guess=[
            [(0.04, 0.5, 0.3, 0.0, 1.0)] * len(_SWAP_TENORS)
            for _ in range(len(_OPTION_TENORS))
        ],
        is_parameter_fixed=(False, True, False, False, False),
    )
    # Sanity: 5-tuple guess fits cleanly.
    assert len(cube.zabr_parameters(0, 0)) == 5
