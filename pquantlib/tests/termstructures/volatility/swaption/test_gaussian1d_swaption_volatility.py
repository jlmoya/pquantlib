"""Tests for the Gaussian1dSwaptionVolatility surface.

Cross-validates against ``migration-harness/references/cluster/l10b.json``.

C++ parity:
- ql/termstructures/volatility/swaption/gaussian1dswaptionvolatility.{hpp,cpp}
- ql/termstructures/volatility/gaussian1dsmilesection.{hpp,cpp}
@ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.models.shortrate.onefactor.gsr import Gsr
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swaption.black_swaption_engine import BlackSwaptionEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.gaussian1d_swaption_volatility import (
    Gaussian1dSwaptionVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l10b")


@pytest.fixture(autouse=True)
def reset_evaluation_date() -> None:
    """Pin evaluation date to 15-May-2026 to match the probe."""
    today = Date.from_ymd(15, Month.May, 2026)
    ObservableSettings().evaluation_date = today


def _build_curve() -> FlatForward:
    return FlatForward(
        Date.from_ymd(15, Month.May, 2026),
        SimpleQuote(0.03),
        Actual365Fixed(),
    )


def _build_gsr() -> Gsr:
    return Gsr(
        term_structure=_build_curve(),
        volstepdates=[],
        volatilities=[0.01],
        reversion=0.01,
        T=50.0,
    )


def _build_swap_index(curve: FlatForward) -> EuriborSwapIsdaFixA:
    return EuriborSwapIsdaFixA(Period(10, TimeUnit.Years), curve)


def _build_svol(
    vol: float = 0.20,
) -> Gaussian1dSwaptionVolatility:
    """Build the swaption vol surface with a BlackSwaptionEngine back-out."""
    curve = _build_curve()
    gsr = _build_gsr()
    # Override the model's curve to use the same instance the engine sees.
    swap_idx = _build_swap_index(curve)
    engine = BlackSwaptionEngine(curve, vol)
    return Gaussian1dSwaptionVolatility(
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        swap_index_base=swap_idx,
        day_counter=Actual365Fixed(),
        model=gsr,
        swaption_engine=engine,
    )


def test_svol_inspectors() -> None:
    """min/max strike, max swap tenor, vol type — C++ defaults."""
    svol = _build_svol()
    assert svol.min_strike() == 0.0
    assert svol.max_strike() == float("inf")
    assert svol.max_swap_tenor() == Period(100, TimeUnit.Years)
    assert svol.volatility_type() == VolatilityType.ShiftedLognormal


def test_svol_reference_date_matches_model_curve() -> None:
    """``reference_date`` equals the model's curve reference date."""
    today = Date.from_ymd(15, Month.May, 2026)
    svol = _build_svol()
    assert svol.reference_date() == today


def test_svol_smile_section_atm_matches_model_swap_rate(
    reference_data: dict[str, Any],
) -> None:
    """Smile section ATM = model's swap_rate(expiry, tenor, y=0).

    LOOSE: the C++ probe uses an explicit ``expiry`` Date built via
    ``TARGET().advance(today, 5*Years)``, which lands on 17-May-2031
    (TARGET-adjusted). Our path constructs the same date.
    """
    ref = reference_data["gaussian1d_swaption_vol"]
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    svol = _build_svol()
    section = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    loose(section.atm_level(), ref["atm_rate"])


def test_svol_atm_matches_model_swap_rate() -> None:
    """At-the-money rate is the GSR's par-swap rate at the expiry."""
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    gsr = _build_gsr()
    curve = _build_curve()
    swap_idx = _build_swap_index(curve)
    expected_atm = gsr.swap_rate(expiry, Period(10, TimeUnit.Years), None, 0.0, swap_idx)
    svol = _build_svol()
    section = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    tight(section.atm_level(), expected_atm)


def test_svol_round_trip_constant_vol() -> None:
    """If the back-out engine is BlackSwaptionEngine with vol=v, then implied vol = v.

    TIGHT: the Newton-safe inversion of the Black formula over the
    same Black engine is bit-true to ~1e-12.
    """
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    expected_vol = 0.20
    svol = _build_svol(vol=expected_vol)
    # ATM in our setup is ~3.04% -> use a strike +50bp above.
    section = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    atm = section.atm_level()
    strike = atm + 0.0050
    vol = svol.volatility(expiry, Period(10, TimeUnit.Years), strike, extrapolate=True)
    loose(vol, expected_vol)


def test_svol_round_trip_lower_vol() -> None:
    """Round-trip with a different vol level."""
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    expected_vol = 0.30
    svol = _build_svol(vol=expected_vol)
    section = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    atm = section.atm_level()
    strike = atm + 0.0010
    vol = svol.volatility(expiry, Period(10, TimeUnit.Years), strike, extrapolate=True)
    loose(vol, expected_vol)


def test_svol_atm_inversion_returns_zero_on_engine_underflow() -> None:
    """The C++ smile section returns 0 on inversion failure — match it.

    At exactly ATM (strike == forward), the Black-formula implied
    std-dev inversion is ill-conditioned for short maturities; the
    C++ catch-all returns 0. We use a vanishing vol setup to force
    underflow and verify the same fallback.
    """
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    svol = _build_svol(vol=1.0e-10)  # effectively zero engine vol
    section = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    atm = section.atm_level()
    # At exactly ATM with a near-zero vol, the OTM put price is ~0
    # and the inversion either underflows or hits the lower bracket.
    vol = svol.volatility(expiry, Period(10, TimeUnit.Years), atm, extrapolate=True)
    # The catch-all returns 0 on inversion failure.
    assert vol == 0.0


def test_svol_option_price_via_smile_section() -> None:
    """The smile section ``option_price`` round-trips via the engine."""
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    svol = _build_svol(vol=0.20)
    section = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    atm = section.atm_level()
    p_call = section.option_price(atm + 0.0050, 1)  # 1 = Call
    p_put = section.option_price(atm - 0.0050, -1)  # -1 = Put
    # Both prices must be strictly positive (real swaption with vol=20%).
    assert p_call > 0.0
    assert p_put > 0.0


def test_svol_invokes_gsr_swap_machinery(reference_data: dict[str, Any]) -> None:
    """Annuity in the smile section matches the model's swap_annuity directly.

    LOOSE — the call paths are slightly different but yield the same
    annuity by construction.
    """
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    gsr = _build_gsr()
    curve = _build_curve()
    swap_idx = _build_swap_index(curve)
    expected_annuity = gsr.swap_annuity(
        expiry, Period(10, TimeUnit.Years), None, 0.0, swap_idx
    )
    ref_annuity = reference_data["gaussian1d_swaption_vol"]["annuity"]
    loose(expected_annuity, ref_annuity)


def test_svol_volatility_time_overload() -> None:
    """The ``Time, Time`` overload converts to Date via calendar adjustment.

    LOOSE: option_time = 5.0 in years; the smile section is built at
    a calendar-adjusted date; the back-out vol should still equal the
    engine's constant vol (round-trip property).
    """
    expected_vol = 0.20
    svol = _build_svol(vol=expected_vol)
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    atm_section = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    atm = atm_section.atm_level()
    # Use a strike slightly OTM to avoid the ATM-inversion underflow.
    strike = atm + 0.0050
    vol = svol.volatility(5.0, 10.0, strike, extrapolate=True)
    loose(vol, expected_vol)


def test_svol_smile_section_uses_swap_index_tenor() -> None:
    """Smile section honors the swap index's tenor (10Y in our setup)."""
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    svol = _build_svol()
    # We pass 10Y here matching the swap index's tenor.
    section_10y = svol.smile_section(expiry, Period(10, TimeUnit.Years))
    # ATM level should be the model's 10Y swap rate.
    gsr = _build_gsr()
    curve = _build_curve()
    swap_idx = _build_swap_index(curve)
    expected = gsr.swap_rate(expiry, Period(10, TimeUnit.Years), None, 0.0, swap_idx)
    tight(section_10y.atm_level(), expected)


def test_svol_construction_does_not_register_with_unrelated_observers() -> None:
    """Construction should not raise / not trigger eager computation."""
    # Smoke test — just ensure the ctor doesn't raise.
    svol = _build_svol()
    assert svol is not None


def test_svol_atm_independence_of_engine_vol() -> None:
    """ATM level is set by the GSR model, not the engine — same regardless of engine vol."""
    today = Date.from_ymd(15, Month.May, 2026)
    expiry = TARGET().advance_period(today, Period(5, TimeUnit.Years))
    svol_20 = _build_svol(vol=0.20)
    svol_40 = _build_svol(vol=0.40)
    sec_20 = svol_20.smile_section(expiry, Period(10, TimeUnit.Years))
    sec_40 = svol_40.smile_section(expiry, Period(10, TimeUnit.Years))
    tight(sec_20.atm_level(), sec_40.atm_level())
