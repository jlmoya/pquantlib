"""Cross-validate InterpolatedCPICapFloorTermPriceSurface + engine vs W7-D probe.

Probe source: migration-harness/cpp/probes/cluster_w7d/probe.cpp
Reference:    migration-harness/references/cluster/w7d.json

Builds the canonical UKRPI CPI cap/floor price surface from the QuantLib
inflation-cap/floor test-suite (test-suite/inflationcpicapfloor.cpp) and
checks (a) the surface reproduces the input quotes by put/call parity,
(b) interpolated points + ATM rate match the C++ surface, and (c) the
``InterpolatingCPICapFloorEngine`` reprices the 3Y/0.03 Call to 227.6 bps.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest

from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as AAConvention
from pquantlib.experimental.inflation.cpi_cap_floor_engines import (
    InterpolatingCPICapFloorEngine,
)
from pquantlib.experimental.inflation.interpolated_cpi_cap_floor_term_price_surface import (
    InterpolatedCPICapFloorTermPriceSurface,
)
from pquantlib.indexes.inflation.cpi import InterpolationType, lagged_fixing
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.indexes.inflation.uk_rpi import UKRPI
from pquantlib.instruments.cpi_cap_floor import CPICapFloor
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.inflation.inflation_helpers import (
    ZeroCouponInflationSwapHelper,
)
from pquantlib.termstructures.inflation.piecewise_zero_inflation_curve import (
    PiecewiseZeroInflationCurve,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

_EVAL = Date.from_ymd(1, Month.June, 2010)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/w7d")


@pytest.fixture(autouse=True)
def _pin_eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    old = s.evaluation_date
    s.evaluation_date = _EVAL
    yield
    s.evaluation_date = old


_FIX_DATA = [
    206.1, 207.3, 208.0, 208.9, 209.7, 210.9,
    209.8, 211.4, 212.1, 214.0, 215.1, 216.8,
    216.5, 217.2, 218.4, 217.7, 216.0, 212.9,
    210.1, 211.4, 211.3, 211.5, 212.8, 213.4,
    213.4, 214.4, 215.3, 216.0, 216.6, 218.0,
    217.9, 219.2, 220.7, 222.8,
]

_NOMINAL_DATA: list[tuple[Date, float]] = [
    (Date.from_ymd(2, Month.June, 2010), 0.499997),
    (Date.from_ymd(3, Month.June, 2010), 0.524992),
    (Date.from_ymd(8, Month.June, 2010), 0.524974),
    (Date.from_ymd(15, Month.June, 2010), 0.549942),
    (Date.from_ymd(22, Month.June, 2010), 0.549913),
    (Date.from_ymd(1, Month.July, 2010), 0.574864),
    (Date.from_ymd(2, Month.August, 2010), 0.624668),
    (Date.from_ymd(1, Month.September, 2010), 0.724338),
    (Date.from_ymd(16, Month.September, 2010), 0.769461),
    (Date.from_ymd(1, Month.December, 2010), 0.997501),
    (Date.from_ymd(17, Month.March, 2011), 0.916996),
    (Date.from_ymd(16, Month.June, 2011), 0.984339),
    (Date.from_ymd(22, Month.September, 2011), 1.06085),
    (Date.from_ymd(22, Month.December, 2011), 1.141788),
    (Date.from_ymd(1, Month.June, 2012), 1.504426),
    (Date.from_ymd(3, Month.June, 2013), 1.92064),
    (Date.from_ymd(2, Month.June, 2014), 2.290824),
    (Date.from_ymd(1, Month.June, 2015), 2.614394),
    (Date.from_ymd(1, Month.June, 2016), 2.887445),
    (Date.from_ymd(1, Month.June, 2017), 3.122128),
    (Date.from_ymd(1, Month.June, 2018), 3.322511),
    (Date.from_ymd(3, Month.June, 2019), 3.483997),
    (Date.from_ymd(1, Month.June, 2020), 3.616896),
    (Date.from_ymd(1, Month.June, 2022), 3.8281),
    (Date.from_ymd(2, Month.June, 2025), 4.0341),
    (Date.from_ymd(3, Month.June, 2030), 4.070854),
    (Date.from_ymd(1, Month.June, 2035), 4.023202),
    (Date.from_ymd(1, Month.June, 2040), 3.954748),
    (Date.from_ymd(1, Month.June, 2050), 3.870953),
    (Date.from_ymd(1, Month.June, 2060), 3.85298),
    (Date.from_ymd(2, Month.June, 2070), 3.757542),
    (Date.from_ymd(3, Month.June, 2080), 3.651379),
]

_ZCIIS_DATA: list[tuple[Date, float]] = [
    (Date.from_ymd(1, Month.June, 2011), 3.087),
    (Date.from_ymd(1, Month.June, 2012), 3.12),
    (Date.from_ymd(1, Month.June, 2013), 3.059),
    (Date.from_ymd(1, Month.June, 2014), 3.11),
    (Date.from_ymd(1, Month.June, 2015), 3.15),
    (Date.from_ymd(1, Month.June, 2016), 3.207),
    (Date.from_ymd(1, Month.June, 2017), 3.253),
    (Date.from_ymd(1, Month.June, 2018), 3.288),
    (Date.from_ymd(1, Month.June, 2019), 3.314),
    (Date.from_ymd(1, Month.June, 2020), 3.401),
    (Date.from_ymd(1, Month.June, 2022), 3.458),
    (Date.from_ymd(1, Month.June, 2025), 3.52),
    (Date.from_ymd(1, Month.June, 2030), 3.655),
    (Date.from_ymd(1, Month.June, 2035), 3.668),
    (Date.from_ymd(1, Month.June, 2040), 3.695),
    (Date.from_ymd(1, Month.June, 2050), 3.634),
    (Date.from_ymd(1, Month.June, 2060), 3.629),
]

_CF_MAT = [3, 5, 7, 10, 15, 20, 30]  # years
_C_STRIKE = [0.03, 0.04, 0.05, 0.06]
_F_STRIKE = [-0.01, 0.0, 0.01, 0.02]
# cPrice[maturity][strike] (bps)
_C_PRICE = [
    [227.6, 100.27, 38.8, 14.94], [345.32, 127.9, 40.59, 14.11],
    [477.95, 170.19, 50.62, 16.88], [757.81, 303.95, 107.62, 43.61],
    [1140.73, 481.89, 168.4, 63.65], [1537.6, 607.72, 172.27, 54.87],
    [2211.67, 839.24, 184.75, 45.03],
]
_F_PRICE = [
    [15.62, 28.38, 53.61, 104.6], [21.45, 36.73, 66.66, 129.6],
    [24.45, 42.08, 77.04, 152.24], [39.25, 63.52, 109.2, 203.44],
    [36.82, 63.62, 116.97, 232.73], [39.7, 67.47, 121.79, 238.56],
    [41.48, 73.9, 139.75, 286.75],
]

_OBS_LAG = Period(2, TimeUnit.Months)


def _build_index_and_nominal() -> tuple[
    ZeroInflationIndex, YieldTermStructureProtocol
]:
    cal = UnitedKingdom()
    bdc = BusinessDayConvention.ModifiedFollowing
    dc = ActualActual(AAConvention.ISDA)

    ii = UKRPI()
    rpi_schedule = (
        MakeSchedule()
        .from_date(Date.from_ymd(1, Month.July, 2007))
        .to(Date.from_ymd(1, Month.April, 2010))
        .with_frequency(Frequency.Monthly)
        .build()
    )
    for i in range(rpi_schedule.size()):
        ii.add_fixing(rpi_schedule.date(i), _FIX_DATA[i], True)

    nom_dates = [d for d, _ in _NOMINAL_DATA]
    nom_rates = [r / 100.0 for _, r in _NOMINAL_DATA]
    nominal = cast(
        YieldTermStructureProtocol,
        InterpolatedZeroCurve(nom_dates, nom_rates, dc),
    )

    helpers: list[ZeroCouponInflationSwapHelper] = []
    for d, r in _ZCIIS_DATA:
        helpers.append(
            ZeroCouponInflationSwapHelper(
                quote=SimpleQuote(r / 100.0),
                observation_lag=_OBS_LAG,
                maturity=d,
                calendar=cal,
                payment_convention=bdc,
                day_counter=dc,
                index=ii,
                nominal_yts=nominal,
            )
        )
    curve = PiecewiseZeroInflationCurve(
        reference_date=_EVAL,
        calendar=cal,
        day_counter=dc,
        observation_lag=_OBS_LAG,
        frequency=Frequency.Monthly,
        instruments=helpers,
        base_rate=_ZCIIS_DATA[0][1] / 100.0,
        nominal_yts=nominal,
    )
    ii.set_zero_inflation_term_structure(curve)
    return ii, nominal


def _build_surface() -> InterpolatedCPICapFloorTermPriceSurface:
    cal = UnitedKingdom()
    bdc = BusinessDayConvention.ModifiedFollowing
    dc = ActualActual(AAConvention.ISDA)
    ii, nominal = _build_index_and_nominal()

    # c/f price matrices indexed [strike, maturity] / 10000.
    c_m = np.array(
        [[_C_PRICE[j][i] / 10000.0 for j in range(7)] for i in range(4)],
        dtype=np.float64,
    )
    f_m = np.array(
        [[_F_PRICE[j][i] / 10000.0 for j in range(7)] for i in range(4)],
        dtype=np.float64,
    )
    return InterpolatedCPICapFloorTermPriceSurface(
        nominal=1.0,
        start_rate=_ZCIIS_DATA[0][1] / 100.0,
        observation_lag=_OBS_LAG,
        calendar=cal,
        bdc=bdc,
        day_counter=dc,
        zii=ii,
        interpolation_type=InterpolationType.Flat,
        yts=nominal,
        c_strikes=_C_STRIKE,
        f_strikes=_F_STRIKE,
        cf_maturities=[Period(m, TimeUnit.Years) for m in _CF_MAT],
        c_price=c_m,
        f_price=f_m,
    )


def test_surface_reproduces_floor_quotes() -> None:
    # Put/call parity completion must leave the floor quotes intact (1e-7).
    surf = _build_surface()
    for i, qk in enumerate(_F_STRIKE):
        for j, m in enumerate(_CF_MAT):
            tolerance.loose(
                surf.floor_price(Period(m, TimeUnit.Years), qk),
                _F_PRICE[j][i] / 10000.0,
                reason="surface must reproduce input floor quotes (parity).",
            )


def test_surface_reproduces_cap_quotes() -> None:
    surf = _build_surface()
    for i, qk in enumerate(_C_STRIKE):
        for j, m in enumerate(_CF_MAT):
            tolerance.loose(
                surf.cap_price(Period(m, TimeUnit.Years), qk),
                _C_PRICE[j][i] / 10000.0,
                reason="surface must reproduce input cap quotes (parity).",
            )


def test_surface_points_match_cpp(cpp: dict[str, Any]) -> None:
    surf = _build_surface()
    yr = TimeUnit.Years
    tolerance.loose(
        surf.floor_price(Period(3, yr), -0.01), cpp["cpi_surf_floor_3y_fstrike-0.01"]
    )
    tolerance.loose(
        surf.floor_price(Period(7, yr), 0.01), cpp["cpi_surf_floor_7y_fstrike0.01"]
    )
    tolerance.loose(
        surf.cap_price(Period(3, yr), 0.03), cpp["cpi_surf_cap_3y_cstrike0.03"]
    )
    tolerance.loose(
        surf.cap_price(Period(10, yr), 0.05), cpp["cpi_surf_cap_10y_cstrike0.05"]
    )
    tolerance.loose(
        surf.cap_price(Period(4, yr), 0.035), cpp["cpi_surf_cap_4y_cstrike0.035"]
    )


def test_surface_atm_rate_matches_cpp(cpp: dict[str, Any]) -> None:
    surf = _build_surface()
    d = surf.cpi_option_date_from_tenor(Period(3, TimeUnit.Years))
    tolerance.loose(surf.atm_rate(d), cpp["cpi_surf_atm_rate_3y"])


def test_interpolating_engine_reprices_cached(cpp: dict[str, Any]) -> None:
    # 3Y Call @ 0.03 must reprice to the cached 227.6 bps via the engine.
    surf = _build_surface()
    ii = surf.zero_inflation_index()
    cal = UnitedKingdom()
    start = _EVAL
    maturity = start + Period(3, TimeUnit.Years)
    base_cpi = lagged_fixing(ii, start, _OBS_LAG, InterpolationType.AsIndex)
    cap = CPICapFloor(
        type_=OptionType.Call,
        nominal=1.0,
        start_date=start,
        base_cpi=base_cpi,
        maturity=maturity,
        fix_calendar=cal,
        fix_convention=BusinessDayConvention.Unadjusted,
        pay_calendar=cal,
        pay_convention=BusinessDayConvention.ModifiedFollowing,
        strike=0.03,
        inflation_index=ii,
        observation_lag=_OBS_LAG,
        observation_interpolation=InterpolationType.AsIndex,
    )
    cap.set_pricing_engine(InterpolatingCPICapFloorEngine(surf))
    tolerance.loose(cap.npv(), cpp["cpi_engine_cap_3y_0.03_npv"])
    # And the engine reproduces the cached cap price directly.
    tolerance.loose(cap.npv(), _C_PRICE[0][0] / 10000.0)
