"""BlackVolatilitySurfaceDelta — cross-validated against C++ QuantLib v1.43.

References: ``migration-harness/references/v143/ts/blackvoldelta.json``,
emitted verbatim by ``migration-harness/cpp/probes/v143_ts_blackvoldelta/probe.cpp``.

Tolerance: TIGHT throughout. Every number here is a deterministic composition
of BlackVarianceCurve interpolation, a BlackDeltaCalculator inversion and a
strike-dimension interpolation — no optimiser, no quadrature. The largest
observed gap over the whole reference is 9e-15 absolute (cubic-spline
extrapolation at strike 0.90, where the vol itself is ~0.49), which is
accumulation of ~1e-16 rounding through the spline solve, well inside the
TIGHT band.

The probe pins ``Settings::instance().evaluationDate()`` to the surface's own
reference date (probe.cpp:236) even though the surface takes an explicit
reference date; the fixture below does the same and restores the previous
value.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.vanilla.black_delta_calculator import BlackDeltaCalculator
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_surface_delta import (
    BlackVolatilitySurfaceDelta,
    SmileInterpolationMethod,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_time_extrapolation import (
    BlackVolTimeExtrapolation,
)
from pquantlib.termstructures.volatility.equity_fx.delta_vol_quote import (
    AtmType,
    DeltaType,
)
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.interpolated_smile_section import (
    InterpolatedSmileSection,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF: dict[str, Any] = reference_reader.load("v143/ts/blackvoldelta")
_SETUP: dict[str, Any] = _REF["setup"]

_REF_DATE = Date(_SETUP["reference_date_serial"])
_DATES = [Date(s) for s in _SETUP["expiry_date_serials"]]
_PUT_DELTAS: list[float] = _SETUP["put_deltas"]
_CALL_DELTAS: list[float] = _SETUP["call_deltas"]
_STRIKES: list[float] = _SETUP["strikes"]
_TIMES: list[float] = _SETUP["times"]
_SPOT: float = _SETUP["spot"]
_R_DOM: float = _SETUP["domestic_rate"]
_R_FOR: float = _SETUP["foreign_rate"]
_N_COL = 5
_VOL_MATRIX: list[list[float]] = [
    _SETUP["vol_matrix_row_major"][i * _N_COL : (i + 1) * _N_COL] for i in range(len(_DATES))
]
# the matrix with the ATM column (index 2) removed
_VOL_MATRIX_NO_ATM: list[list[float]] = [[r[0], r[1], r[3], r[4]] for r in _VOL_MATRIX]

# C++'s Null<Real>() as it lands in the JSON — float max * 10.
_NULL_REAL = 3.4028234663852886e38


@pytest.fixture(autouse=True)
def _pin_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin Settings to the probe's evaluation date (probe.cpp:236)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _REF_DATE
    yield
    settings.evaluation_date = previous


def _domestic() -> FlatForward:
    return FlatForward.from_rate(_REF_DATE, _R_DOM, Actual365Fixed())


def _foreign() -> FlatForward:
    return FlatForward.from_rate(_REF_DATE, _R_FOR, Actual365Fixed())


def _surface(
    *,
    delta_type: DeltaType = DeltaType.Spot,
    atm_type: AtmType = AtmType.AtmDeltaNeutral,
    atm_delta_type: DeltaType | None = None,
    interpolation_method: SmileInterpolationMethod = SmileInterpolationMethod.Linear,
    flat_strike_extrapolation: bool = False,
    has_atm: bool = True,
    put_deltas: Sequence[float] | None = None,
    call_deltas: Sequence[float] | None = None,
    matrix: Sequence[Sequence[float]] | None = None,
    dates: Sequence[Date] | None = None,
    switch_tenor: Period | None = None,
    long_term_delta_type: DeltaType = DeltaType.Fwd,
    long_term_atm_type: AtmType = AtmType.AtmDeltaNeutral,
    long_term_atm_delta_type: DeltaType | None = None,
) -> BlackVolatilitySurfaceDelta:
    if matrix is None:
        matrix = _VOL_MATRIX if has_atm else _VOL_MATRIX_NO_ATM
    return BlackVolatilitySurfaceDelta(
        reference_date=_REF_DATE,
        dates=list(dates) if dates is not None else _DATES,
        put_deltas=list(put_deltas) if put_deltas is not None else _PUT_DELTAS,
        call_deltas=list(call_deltas) if call_deltas is not None else _CALL_DELTAS,
        has_atm=has_atm,
        black_vol_matrix=matrix,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        spot=SimpleQuote(_SPOT),
        domestic_ts=_domestic(),
        foreign_ts=_foreign(),
        delta_type=delta_type,
        atm_type=atm_type,
        atm_delta_type=atm_delta_type,
        interpolation_method=interpolation_method,
        flat_strike_extrapolation=flat_strike_extrapolation,
        switch_tenor=switch_tenor,
        long_term_delta_type=long_term_delta_type,
        long_term_atm_type=long_term_atm_type,
        long_term_atm_delta_type=long_term_atm_delta_type,
    )


# --------------------------------------------------------------------------
# 1. the delta -> strike inversion, standalone
#    (probe.cpp "strike_from_delta"; blackvolsurfacedelta.cpp:120-158)
# --------------------------------------------------------------------------


def _column_curve(column: int) -> BlackVarianceCurve:
    return BlackVarianceCurve(
        reference_date=_REF_DATE,
        dates=_DATES,
        black_vol_curve=[row[column] for row in _VOL_MATRIX],
        day_counter=Actual365Fixed(),
        force_monotone_variance=False,
        time_extrapolation_type=BlackVolTimeExtrapolation.Type.FlatVolatility,
    )


def test_per_delta_column_vols_match_cpp() -> None:
    """The per-column BlackVarianceCurve reads the surface itself performs."""
    expected = _REF["strike_from_delta"]["column_vols"]
    curves = [_column_curve(c) for c in range(_N_COL)]
    i = 0
    for t in _TIMES:
        for c in range(_N_COL):
            tolerance.tight(curves[c].black_vol_at_time(t, 1.0, True), expected[i])
            i += 1


@pytest.mark.parametrize(
    "delta_type",
    [DeltaType.Spot, DeltaType.Fwd, DeltaType.PaSpot, DeltaType.PaFwd],
)
def test_strike_from_delta_every_convention(delta_type: DeltaType) -> None:
    """Delta -> strike under every DeltaType C++ supports.

    Column 2 is the ATM column, which C++ handles through ``atmStrike`` rather
    than ``strikeFromDelta``; the probe writes Null<Real>() there.
    """
    expected = _REF["strike_from_delta"][f"delta_type_{delta_type.name}"]
    deltas = [_PUT_DELTAS[0], _PUT_DELTAS[1], None, _CALL_DELTAS[0], _CALL_DELTAS[1]]
    curves = [_column_curve(c) for c in range(_N_COL)]
    dom, foreign = _domestic(), _foreign()
    i = 0
    for t in _TIMES:
        sqrt_t = math.sqrt(t)
        d_disc, f_disc = dom.discount(t), foreign.discount(t)
        for c in range(_N_COL):
            delta = deltas[c]
            if delta is None:
                assert expected[i] == _NULL_REAL
                i += 1
                continue
            vol = curves[c].black_vol_at_time(t, 1.0, True)
            bdc = BlackDeltaCalculator(
                OptionType.Put if c < 2 else OptionType.Call,
                delta_type,
                _SPOT,
                d_disc,
                f_disc,
                vol * sqrt_t,
            )
            tolerance.tight(bdc.strike_from_delta(delta), expected[i])
            i += 1


@pytest.mark.parametrize(
    ("delta_type", "atm_type"),
    [
        (dt, at)
        for dt in (DeltaType.Spot, DeltaType.Fwd, DeltaType.PaSpot, DeltaType.PaFwd)
        for at in (
            AtmType.AtmSpot,
            AtmType.AtmFwd,
            AtmType.AtmDeltaNeutral,
            AtmType.AtmVegaMax,
            AtmType.AtmGammaMax,
        )
    ]
    + [(DeltaType.Fwd, AtmType.AtmPutCall50)],
)
def test_atm_strike_every_convention(delta_type: DeltaType, atm_type: AtmType) -> None:
    """ATM strike under every (DeltaType, AtmType) pair C++ accepts."""
    expected = _REF["strike_from_delta"][f"atm_{delta_type.name}_{atm_type.name}"]
    curve = _column_curve(2)
    dom, foreign = _domestic(), _foreign()
    for i, t in enumerate(_TIMES):
        vol = curve.black_vol_at_time(t, 1.0, True)
        bdc = BlackDeltaCalculator(
            OptionType.Put, delta_type, _SPOT, dom.discount(t), foreign.discount(t),
            vol * math.sqrt(t),
        )
        tolerance.tight(bdc.atm_strike(atm_type), expected[i])


def _boundary_calculator(option_type: OptionType, delta_type: DeltaType) -> BlackDeltaCalculator:
    """The probe's boundary fixture: t = 1, stdDev = 0.12 (probe.cpp:313-316)."""
    t = 1.0
    return BlackDeltaCalculator(
        option_type, delta_type, _SPOT, _domestic().discount(t), _foreign().discount(t), 0.12
    )


def test_spot_delta_out_of_range_throws() -> None:
    """|delta| > fDiscount is rejected for the Spot convention."""
    bdc = _boundary_calculator(OptionType.Put, DeltaType.Spot)
    with pytest.raises(LibraryException) as exc:
        bdc.strike_from_delta(-1.5)
    assert _REF["strike_from_delta"]["throw_spot_delta_out_of_range"] in str(exc.value)


def test_fwd_delta_out_of_range_throws() -> None:
    """|delta| > 1 is rejected for the Fwd convention."""
    bdc = _boundary_calculator(OptionType.Put, DeltaType.Fwd)
    with pytest.raises(LibraryException) as exc:
        bdc.strike_from_delta(-1.5)
    assert _REF["strike_from_delta"]["throw_fwd_delta_out_of_range"] in str(exc.value)


def test_incoherent_option_type_and_delta_throws() -> None:
    """A positive delta on a put is rejected before any inversion happens."""
    bdc = _boundary_calculator(OptionType.Put, DeltaType.Fwd)
    with pytest.raises(LibraryException) as exc:
        bdc.strike_from_delta(0.25)
    assert _REF["strike_from_delta"]["throw_incoherent_option_type_and_delta"] in str(exc.value)


def test_atm_putcall50_requires_forward_delta() -> None:
    """AtmPutCall50 is only defined for the Fwd delta convention."""
    bdc = _boundary_calculator(OptionType.Call, DeltaType.Spot)
    with pytest.raises(LibraryException) as exc:
        bdc.atm_strike(AtmType.AtmPutCall50)
    assert _REF["strike_from_delta"]["throw_atm_putcall50_needs_fwd_delta"] in str(exc.value)


# --------------------------------------------------------------------------
# 2. the surfaces
# --------------------------------------------------------------------------

_SURFACE_CASES: dict[str, dict[str, Any]] = {
    "spot_dn_linear": {},
    "spot_dn_natural_cubic": {"interpolation_method": SmileInterpolationMethod.NaturalCubic},
    "spot_dn_financial_cubic": {"interpolation_method": SmileInterpolationMethod.FinancialCubic},
    "spot_dn_cubic_spline": {"interpolation_method": SmileInterpolationMethod.CubicSpline},
    "spot_dn_linear_flat_extrap": {"flat_strike_extrapolation": True},
    "fwd_atmfwd_linear": {"delta_type": DeltaType.Fwd, "atm_type": AtmType.AtmFwd},
    "paspot_dn_linear": {"delta_type": DeltaType.PaSpot},
    "pafwd_dn_linear": {"delta_type": DeltaType.PaFwd},
    "spot_dn_atmdelta_fwd_linear": {"atm_delta_type": DeltaType.Fwd},
    "no_atm_spot_linear": {"has_atm": False},
    "switch_1y_spot_to_fwd": {
        "switch_tenor": Period(1, TimeUnit.Years),
        "long_term_delta_type": DeltaType.Fwd,
        "long_term_atm_type": AtmType.AtmFwd,
    },
}


@pytest.mark.parametrize("case", sorted(_SURFACE_CASES))
def test_surface_black_vol_and_variance(case: str) -> None:
    """blackVol / blackVariance over the (time, strike) grid, all conventions."""
    expected = _REF["surfaces"][case]
    surface = _surface(**_SURFACE_CASES[case])
    i = 0
    for t in _TIMES:
        for k in _STRIKES:
            tolerance.tight(surface.black_vol_at_time(t, k, True), expected["black_vol"][i])
            tolerance.tight(
                surface.black_variance_at_time(t, k, True), expected["black_variance"][i]
            )
            i += 1


@pytest.mark.parametrize("case", sorted(_SURFACE_CASES))
def test_surface_smile_section(case: str) -> None:
    """The per-expiry SmileSection: surviving pillars, ATM level, vols."""
    expected = _REF["surfaces"][case]
    surface = _surface(**_SURFACE_CASES[case])
    i = 0
    for j, t in enumerate(_TIMES):
        smile = surface.black_vol_smile(t)
        assert isinstance(smile, InterpolatedSmileSection)
        tolerance.tight(smile.min_strike(), expected["smile_min_strike"][j])
        tolerance.tight(smile.max_strike(), expected["smile_max_strike"][j])
        tolerance.tight(smile.atm_level(), expected["smile_atm_level"][j])
        for k in _STRIKES:
            tolerance.tight(smile.volatility(k), expected["smile_volatility"][i])
            i += 1


@pytest.mark.parametrize("case", sorted(_SURFACE_CASES))
def test_surface_zero_strike_short_circuit(case: str) -> None:
    """strike == 0 reads the ATM curve directly, or falls back to the forward.

    # C++ parity: blackvolsurfacedelta.cpp:219-227.
    """
    expected = _REF["surfaces"][case]["black_vol_at_zero_strike"]
    surface = _surface(**_SURFACE_CASES[case])
    for j, t in enumerate(_TIMES):
        tolerance.tight(surface.black_vol_at_time(t, 0.0, True), expected[j])


def test_nan_strike_takes_the_same_branch_as_zero() -> None:
    """PQuantLib's null-Real sentinel is NaN, and routes like C++'s Null<Real>.

    # C++ parity: blackvolsurfacedelta.cpp:219 tests
    # ``strike == 0 || strike == Null<Real>()``; this port's null-Real analogue
    # for a level is NaN (see BlackVolTermStructure.atm_level), so the values
    # must agree with the strike-0 column of the reference.
    """
    expected = _REF["surfaces"]["spot_dn_linear"]["black_vol_at_zero_strike"]
    surface = _surface()
    for j, t in enumerate(_TIMES):
        tolerance.tight(surface.black_vol_at_time(t, math.nan, True), expected[j])


def test_smile_section_hook_returns_the_native_smile() -> None:
    """``smile_section_at_time`` hands back the real section, not an adapter."""
    surface = _surface()
    t = _TIMES[4]
    section = surface.smile_section_at_time(t, True)
    assert isinstance(section, InterpolatedSmileSection)
    expected = _REF["surfaces"]["spot_dn_linear"]["smile_volatility"]
    for i, k in enumerate(_STRIKES):
        tolerance.tight(section.volatility(k), expected[4 * len(_STRIKES) + i])


def test_black_vol_smile_by_date_matches_by_time() -> None:
    """``blackVolSmile(Date)`` == ``blackVolSmile(timeFromReference(Date))``."""
    surface = _surface()
    d = _DATES[3]
    t = surface.time_from_reference(d)
    tolerance.exact(
        surface.black_vol_smile_at_date(d).volatility(1.25),
        surface.black_vol_smile(t).volatility(1.25),
    )


# --------------------------------------------------------------------------
# 3. the one-strike (FlatSmileSection) branch
# --------------------------------------------------------------------------


def test_single_column_collapses_to_flat_smile_section() -> None:
    """One quote column -> one strike -> FlatSmileSection (cpp:173-175).

    Divergence, pre-existing and documented at its source: C++'s
    FlatSmileSection reports ``QL_MIN_REAL - shift()`` / ``QL_MAX_REAL`` for
    its strike bounds and ``Null<Rate>()`` for its ATM level, where PQuantLib's
    reports ``-inf`` / ``+inf`` / ``nan`` (flat_smile_section.py:51-62). Only
    the sentinels differ; no volatility does.
    """
    expected = _REF["surfaces"]["atm_only_flat_smile"]
    surface = _surface(
        put_deltas=[], call_deltas=[], has_atm=True, matrix=[[row[2]] for row in _VOL_MATRIX]
    )
    i = 0
    for j, t in enumerate(_TIMES):
        smile = surface.black_vol_smile(t)
        assert isinstance(smile, FlatSmileSection)
        assert smile.min_strike() == -math.inf
        assert expected["smile_min_strike"][j] == -1.7976931348623157e308
        assert smile.max_strike() == math.inf
        assert expected["smile_max_strike"][j] == 1.7976931348623157e308
        assert math.isnan(smile.atm_level())
        assert expected["smile_atm_level"][j] == _NULL_REAL
        for k in _STRIKES:
            tolerance.tight(smile.volatility(k), expected["smile_volatility"][i])
            tolerance.tight(
                surface.black_vol_at_time(t, k, True), expected["black_vol"][i]
            )
            i += 1
        tolerance.tight(
            surface.black_vol_at_time(t, 0.0, True), expected["black_vol_at_zero_strike"][j]
        )


# --------------------------------------------------------------------------
# 4. inspectors
# --------------------------------------------------------------------------


def test_inspectors() -> None:
    expected = _REF["inspectors"]
    surface = _surface()
    assert surface.max_date().serial_number() == expected["max_date_serial"]
    tolerance.exact(surface.min_strike(), expected["min_strike"])
    tolerance.exact(surface.max_strike(), expected["max_strike"])
    assert surface.day_counter().name() == expected["day_counter"]
    assert surface.calendar().name() == expected["calendar"]
    assert surface.reference_date().serial_number() == expected["reference_date_serial"]
    assert [d.serial_number() for d in surface.dates()] == _SETUP["expiry_date_serials"]


def test_atm_level_is_the_forward() -> None:
    """# C++ parity: blackvolsurfacedelta.cpp:210-212."""
    expected = _REF["inspectors"]["atm_level"]
    surface = _surface()
    for j, t in enumerate(_TIMES):
        tolerance.tight(surface.atm_level(t), expected[j])


def test_switch_tenor_resolves_to_the_cpp_date_and_time() -> None:
    """``optionDateFromTenor(1Y)`` under TARGET/Following, then timeFromReference."""
    expected = _REF["inspectors"]
    surface = _surface()
    d = surface.option_date_from_tenor(Period(1, TimeUnit.Years))
    assert d.serial_number() == expected["switch_tenor_1y_date_serial"]
    tolerance.tight(surface.time_from_reference(d), expected["switch_tenor_1y_time"])


def test_zero_switch_tenor_never_switches() -> None:
    """``switch_tenor = 0 * Days`` keeps the short-term conventions forever.

    Pinned by construction: the default surface (no switch) and a surface with
    an explicit zero tenor but *different* long-term conventions must agree at
    every probe time, including past the last pillar.
    """
    default = _surface()
    zero_tenor = _surface(
        switch_tenor=Period(0, TimeUnit.Days),
        long_term_delta_type=DeltaType.PaFwd,
        long_term_atm_type=AtmType.AtmSpot,
    )
    expected = _REF["surfaces"]["spot_dn_linear"]["black_vol"]
    i = 0
    for t in _TIMES:
        for k in _STRIKES:
            tolerance.exact(
                zero_tenor.black_vol_at_time(t, k, True),
                default.black_vol_at_time(t, k, True),
            )
            tolerance.tight(zero_tenor.black_vol_at_time(t, k, True), expected[i])
            i += 1


def test_zero_switch_tenor_in_months_is_also_never() -> None:
    """A zero-length tenor is zero whatever the unit — C++ ``operator==``.

    # C++ parity: ql/time/period.cpp ``operator<`` treats every zero-length
    # Period as equal, so ``Period(0, Months) == 0 * Days`` there. This port's
    # frozen-dataclass ``__eq__`` would say otherwise, so the class tests
    # ``length == 0`` instead; this pins that choice.
    """
    months = _surface(switch_tenor=Period(0, TimeUnit.Months))
    default = _surface()
    for t in _TIMES:
        for k in _STRIKES:
            tolerance.exact(
                months.black_vol_at_time(t, k, True), default.black_vol_at_time(t, k, True)
            )


def test_switch_boundary_uses_the_long_term_conventions() -> None:
    """At exactly switch_time the long-term set applies (cpp:103-111).

    ``close_enough(t, switchTime_)`` counts as "at or past", so the boundary
    time must produce the same smile as a surface whose *short*-term
    conventions are already the long-term ones.
    """
    switch_time = _REF["inspectors"]["switch_tenor_1y_time"]
    switched = _surface(
        switch_tenor=Period(1, TimeUnit.Years),
        long_term_delta_type=DeltaType.Fwd,
        long_term_atm_type=AtmType.AtmFwd,
    )
    long_term_only = _surface(delta_type=DeltaType.Fwd, atm_type=AtmType.AtmFwd)
    short_term_only = _surface()
    for k in _STRIKES:
        tolerance.exact(
            switched.black_vol_at_time(switch_time, k, True),
            long_term_only.black_vol_at_time(switch_time, k, True),
        )
    # just below the boundary the short-term conventions still apply
    below = math.nextafter(switch_time, 0.0) * (1 - 1e-9)
    for k in _STRIKES:
        tolerance.exact(
            switched.black_vol_at_time(below, k, True),
            short_term_only.black_vol_at_time(below, k, True),
        )


# --------------------------------------------------------------------------
# 5. constructor preconditions
# --------------------------------------------------------------------------


def test_requires_more_than_one_date() -> None:
    with pytest.raises(LibraryException) as exc:
        _surface(dates=[_DATES[0]], matrix=[[0.12] * 5])
    assert _REF["throws"]["single_date"] in str(exc.value)


def test_rejects_date_at_or_before_reference() -> None:
    with pytest.raises(LibraryException) as exc:
        _surface(
            dates=[Date.from_ymd(1, _REF_DATE.month(), _REF_DATE.year()), _DATES[0]],
            matrix=[[0.12] * 5, [0.12] * 5],
        )
    assert _REF["throws"]["date_before_reference"] in str(exc.value)


def test_rejects_unsorted_dates() -> None:
    with pytest.raises(LibraryException) as exc:
        _surface(dates=[_DATES[1], _DATES[0]], matrix=[[0.12] * 5, [0.12] * 5])
    assert _REF["throws"]["unsorted_dates"] in str(exc.value)


def test_rejects_wrong_column_count() -> None:
    with pytest.raises(LibraryException) as exc:
        _surface(matrix=[[0.12] * 4 for _ in _DATES])
    assert _REF["throws"]["wrong_columns"] in str(exc.value)


def test_rejects_wrong_row_count() -> None:
    with pytest.raises(LibraryException) as exc:
        _surface(matrix=[[0.12] * 5 for _ in _DATES[:-1]])
    assert _REF["throws"]["wrong_rows"] in str(exc.value)


def test_rejects_empty_delta_set() -> None:
    with pytest.raises(LibraryException) as exc:
        _surface(
            put_deltas=[],
            call_deltas=[],
            has_atm=False,
            matrix=[[] for _ in _DATES],
        )
    assert _REF["throws"]["no_deltas_at_all"] in str(exc.value)


def test_rejects_unknown_interpolation_method() -> None:
    """# C++ parity: blackvolsurfacedelta.cpp:200-202 — ``QL_FAIL("Invalid method ...")``."""
    surface = _surface()
    surface._interpolation_method = 99  # type: ignore[assignment]
    with pytest.raises(LibraryException, match="Invalid method 99"):
        surface.black_vol_smile(1.0)
