"""NoArbSabrSmileSection + interpolated smile section + swaption cube (W6-A batch b).

Cross-validation reference: ``migration-harness/references/cluster/w6a.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.volatility.no_arb_sabr import no_arb_sabr_volatility
from pquantlib.experimental.volatility.no_arb_sabr_interpolated_smile_section import (
    NoArbSabrInterpolatedSmileSection,
)
from pquantlib.experimental.volatility.no_arb_sabr_smile_section import (
    NoArbSabrSmileSection,
)
from pquantlib.payoffs import OptionType
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.no_arb_sabr_swaption_volatility_cube import (
    NoArbSabrSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.termstructures.volatility.swaption.xabr_swaption_volatility_cube import (
    XabrModelKind,
    XabrSwaptionVolatilityCube,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def na() -> dict[str, Any]:
    return reference_reader.load("cluster/w6a")["noarbsabr"]


# --- NoArbSabrSmileSection -------------------------------------------------


def test_smile_section_volatility_matches_cpp(na: dict[str, Any]) -> None:
    sec = NoArbSabrSmileSection(
        forward=na["forward"],
        sabr_params=(na["alpha"], na["beta"], na["nu"], na["rho"]),
        exercise_time=na["tau"],
    )
    for k, cpp_vol in zip(na["strikes"], na["noarb_volatility"], strict=True):
        tolerance.loose(sec.volatility(float(k)), float(cpp_vol))


def test_smile_section_prices_match_model(na: dict[str, Any]) -> None:
    sec = NoArbSabrSmileSection(
        forward=na["forward"],
        sabr_params=(na["alpha"], na["beta"], na["nu"], na["rho"]),
        exercise_time=na["tau"],
    )
    for k, op, dg, dn in zip(
        na["strikes"], na["option_price"],
        na["digital_option_price"], na["density"], strict=True,
    ):
        tolerance.tight(sec.option_price(float(k)), float(op))
        tolerance.tight(sec.digital_option_price(float(k)), float(dg))
        tolerance.tight(sec.density(float(k)), float(dn))


def test_smile_section_put_call_parity(na: dict[str, Any]) -> None:
    sec = NoArbSabrSmileSection(
        forward=na["forward"],
        sabr_params=(na["alpha"], na["beta"], na["nu"], na["rho"]),
        exercise_time=na["tau"],
    )
    k = 0.07
    call = sec.option_price(k, option_type=int(OptionType.Call))
    put = sec.option_price(k, option_type=int(OptionType.Put))
    # call - put = forward - strike.
    tolerance.tight(call - put, na["forward"] - k)


def test_smile_section_atm_and_strike_bounds(na: dict[str, Any]) -> None:
    sec = NoArbSabrSmileSection(
        forward=na["forward"],
        sabr_params=(na["alpha"], na["beta"], na["nu"], na["rho"]),
        exercise_time=na["tau"],
    )
    tolerance.exact(sec.atm_level(), na["forward"])
    tolerance.exact(sec.min_strike(), 0.0)
    assert sec.max_strike() == float("inf")


def test_smile_section_rejects_nonzero_shift(na: dict[str, Any]) -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    with pytest.raises(LibraryException):
        NoArbSabrSmileSection(
            forward=na["forward"],
            sabr_params=(na["alpha"], na["beta"], na["nu"], na["rho"]),
            exercise_time=na["tau"],
            shift=0.01,
        )


# --- NoArbSabrInterpolatedSmileSection ------------------------------------


def test_interpolated_smile_recovers_slice(na: dict[str, Any]) -> None:
    """Fit + wrap recovers the generating no-arb slice (alpha + nu free,
    beta + rho fixed → fast + well-determined)."""
    f, tau = na["forward"], na["tau"]
    alpha, beta, nu, rho = na["alpha"], na["beta"], na["nu"], na["rho"]
    strikes = [0.03, f, 0.07, 0.09]
    vols = [no_arb_sabr_volatility(k, f, tau, alpha, beta, nu, rho) for k in strikes]

    reference = Date.from_ymd(15, Month.January, 2024)
    option_date = reference + Period(365, TimeUnit.Days)
    sec = NoArbSabrInterpolatedSmileSection(
        option_date=option_date,
        forward=f,
        strikes=strikes,
        vols=vols,
        alpha=0.03, beta=beta, nu=0.5, rho=rho,
        beta_is_fixed=True, rho_is_fixed=True,
        reference_date=reference,
        day_counter=Actual365Fixed(),
        max_nfev=80,
    )
    assert sec.converged()
    tolerance.exact(sec.min_strike(), strikes[0])
    tolerance.exact(sec.max_strike(), strikes[-1])
    tolerance.exact(sec.atm_level(), f)
    tolerance.loose(sec.alpha(), alpha)
    tolerance.loose(sec.nu(), nu)
    for k, v in zip(strikes, vols, strict=True):
        tolerance.loose(sec.volatility(k), v)


# --- NoArbSabrSwaptionVolatilityCube --------------------------------------


class _FakeSwapIndex:
    """Minimal SwapIndex stand-in returning ``forward`` on every fixing."""

    def __init__(self, tenor: Period, forward: float) -> None:
        self._tenor = tenor
        self._forward = forward

    def tenor(self) -> Period:
        return self._tenor

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        _ = fixing_date, forecast_todays_fixing
        return self._forward


_REF_DATE = Date.from_ymd(15, Month.January, 2024)
# 2x2 grid — SwaptionVolatilityMatrix needs >= 2 pillars per axis.
_OPTION_TENORS = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
_SWAP_TENORS = [Period(2, TimeUnit.Years), Period(5, TimeUnit.Years)]
_STRIKE_SPREADS = [-0.01, 0.0, 0.01]
# Forward chosen so sigmaI = alpha*F^(beta-1) is admissible ([0.05, 1.0]).
_FWD = 0.05
_ALPHA, _BETA, _NU, _RHO = 0.026, 0.5, 0.4, -0.1


def _atm_matrix() -> SwaptionVolatilityMatrix:
    atm = no_arb_sabr_volatility(_FWD, _FWD, 1.0, _ALPHA, _BETA, _NU, _RHO)
    return SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        volatilities=np.full((2, 2), atm),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_REF_DATE,
    )


def _vol_spreads() -> list[list[SimpleQuote]]:
    atm = no_arb_sabr_volatility(_FWD, _FWD, 1.0, _ALPHA, _BETA, _NU, _RHO)
    rows: list[list[SimpleQuote]] = []
    for _j in range(len(_OPTION_TENORS)):
        for _k in range(len(_SWAP_TENORS)):
            row: list[SimpleQuote] = []
            for s in _STRIKE_SPREADS:
                strike = _FWD + s
                v = no_arb_sabr_volatility(
                    strike, _FWD, 1.0, _ALPHA, _BETA, _NU, _RHO
                )
                row.append(SimpleQuote(v - atm))
            rows.append(row)
    return rows


def test_swaption_cube_fits_and_returns_noarb_smile() -> None:
    """NOARB_SABR cube fits a cell + returns a NoArbSabrSmileSection.

    Beta + rho fixed at the generating values so the per-cell fit is
    well-determined and fast.
    """
    cube = NoArbSabrSwaptionVolatilityCube(
        atm_vol_structure=_atm_matrix(),
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FWD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FWD),
        no_arb_sabr_initial_guess=[
            [(0.026, _BETA, 0.45, _RHO), (0.026, _BETA, 0.45, _RHO)],
            [(0.026, _BETA, 0.45, _RHO), (0.026, _BETA, 0.45, _RHO)],
        ],
        is_parameter_fixed=(False, True, False, True),
    )
    params = cube.no_arb_sabr_parameters(0, 0)
    assert len(params) == 4
    # beta + rho pinned.
    tolerance.exact(params[1], _BETA)
    tolerance.exact(params[3], _RHO)
    # alpha + nu recovered near the generating values (LOOSE — cube fit).
    assert abs(params[0] - _ALPHA) < 1e-2
    assert abs(params[2] - _NU) < 1e-1
    section = cube.smile_section_impl(1.0, 5.0)
    assert isinstance(section, NoArbSabrSmileSection)


def test_swaption_cube_model_kind_is_noarb() -> None:
    cube = NoArbSabrSwaptionVolatilityCube(
        atm_vol_structure=_atm_matrix(),
        option_tenors=_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spreads(),
        swap_index_base=_FakeSwapIndex(Period(5, TimeUnit.Years), _FWD),
        short_swap_index_base=_FakeSwapIndex(Period(1, TimeUnit.Years), _FWD),
        no_arb_sabr_initial_guess=[
            [(0.026, _BETA, 0.45, _RHO), (0.026, _BETA, 0.45, _RHO)],
            [(0.026, _BETA, 0.45, _RHO), (0.026, _BETA, 0.45, _RHO)],
        ],
        is_parameter_fixed=(False, True, False, True),
    )
    assert cube.model_kind() == XabrModelKind.NOARB_SABR
    assert isinstance(cube, XabrSwaptionVolatilityCube)
