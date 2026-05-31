"""Cross-validate the FX delta-vol + Vanna-Volga batch (W8-C, batch a).

Probe source: migration-harness/cpp/probes/cluster_w8c/probe.cpp
Reference:    migration-harness/references/cluster/w8c.json

Covers:
  * BlackDeltaCalculator.strike_from_delta round-trips + atm_strike
    (Fwd / Spot conventions) — TIGHT.
  * DeltaVolQuote accessors.
  * VannaVolgaBarrierEngine canonical FX single-barrier values — LOOSE.
  * VannaVolgaDoubleBarrierEngine KnockOut / KnockIn — LOOSE.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierType,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.vanna_volga_barrier_engine import (
    VannaVolgaBarrierEngine,
)
from pquantlib.pricingengines.barrier.vanna_volga_double_barrier_engine import (
    VannaVolgaDoubleBarrierEngine,
)
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.vanilla.black_delta_calculator import (
    BlackDeltaCalculator,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.delta_vol_quote import (
    AtmType,
    DeltaType,
    DeltaVolQuote,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8c")


# --- BlackDeltaCalculator round-trips -----------------------------------


def test_bdc_forward_strike_from_delta_roundtrip(ref: dict[str, Any]) -> None:
    spot, d_disc, f_disc = 1.30265, 0.99663, 0.99965
    std_dev = 0.08925 * math.sqrt(1.0)

    call = BlackDeltaCalculator(OptionType.Call, DeltaType.Fwd, spot, d_disc, f_disc, std_dev)
    kc = call.strike_from_delta(0.25)
    tolerance.tight(kc, ref["bdc_fwd_call_strike"])
    tolerance.tight(call.delta_from_strike(kc), ref["bdc_fwd_call_delta_roundtrip"])

    put = BlackDeltaCalculator(OptionType.Put, DeltaType.Fwd, spot, d_disc, f_disc, std_dev)
    kp = put.strike_from_delta(-0.25)
    tolerance.tight(kp, ref["bdc_fwd_put_strike"])
    tolerance.tight(put.delta_from_strike(kp), ref["bdc_fwd_put_delta_roundtrip"])


def test_bdc_spot_strike_from_delta_roundtrip(ref: dict[str, Any]) -> None:
    spot, d_disc, f_disc = 1.30265, 0.99663, 0.99965
    std_dev = 0.08925 * math.sqrt(1.0)
    call = BlackDeltaCalculator(OptionType.Call, DeltaType.Spot, spot, d_disc, f_disc, std_dev)
    kcs = call.strike_from_delta(0.25)
    tolerance.tight(kcs, ref["bdc_spot_call_strike"])
    tolerance.tight(call.delta_from_strike(kcs), ref["bdc_spot_call_delta_roundtrip"])


def test_bdc_atm_strikes(ref: dict[str, Any]) -> None:
    spot, d_disc, f_disc = 1.30265, 0.99663, 0.99965
    std_dev = 0.08925 * math.sqrt(1.0)
    call = BlackDeltaCalculator(OptionType.Call, DeltaType.Fwd, spot, d_disc, f_disc, std_dev)
    tolerance.tight(call.atm_strike(AtmType.AtmFwd), ref["bdc_atm_fwd"])
    tolerance.tight(call.atm_strike(AtmType.AtmSpot), ref["bdc_atm_spot"])
    tolerance.tight(call.atm_strike(AtmType.AtmDeltaNeutral), ref["bdc_atm_dn"])


def test_bdc_premium_adjusted_strike_roundtrip() -> None:
    # PaSpot / PaFwd require a numerical Brent solve; verify self-consistency.
    spot, d_disc, f_disc = 1.30265, 0.99663, 0.99965
    std_dev = 0.10 * math.sqrt(1.0)
    for dt in (DeltaType.PaSpot, DeltaType.PaFwd):
        for ot, delta in ((OptionType.Call, 0.25), (OptionType.Put, -0.25)):
            bdc = BlackDeltaCalculator(ot, dt, spot, d_disc, f_disc, std_dev)
            k = bdc.strike_from_delta(delta)
            tolerance.loose(bdc.delta_from_strike(k), delta)


# --- DeltaVolQuote ------------------------------------------------------


def test_delta_vol_quote_accessors() -> None:
    q = SimpleQuote(0.10)
    dvq = DeltaVolQuote(-0.25, q, 0.5, DeltaType.Fwd)
    assert dvq.delta() == -0.25
    assert dvq.maturity() == 0.5
    assert dvq.delta_type() == DeltaType.Fwd
    assert dvq.atm_type() == AtmType.AtmNull
    tolerance.exact(dvq.value(), 0.10)
    # Observer propagation: changing the wrapped quote updates value.
    q.set_value(0.12)
    tolerance.exact(dvq.value(), 0.12)


def test_delta_vol_quote_atm() -> None:
    q = SimpleQuote(0.09)
    dvq = DeltaVolQuote.atm(q, DeltaType.Fwd, 1.0, AtmType.AtmDeltaNeutral)
    assert dvq.atm_type() == AtmType.AtmDeltaNeutral
    assert dvq.maturity() == 1.0
    tolerance.exact(dvq.value(), 0.09)


# --- VannaVolga single-barrier ------------------------------------------


def _fx_setup() -> tuple[
    SimpleQuote, FlatForward, FlatForward, DeltaVolQuote, DeltaVolQuote, DeltaVolQuote, Date
]:
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today
    dc = Actual365Fixed()
    spot = SimpleQuote(1.30265)
    q_ts = FlatForward.from_rate(reference_date=today, forward_rate=0.0003541, day_counter=dc)
    r_ts = FlatForward.from_rate(reference_date=today, forward_rate=0.0033871, day_counter=dc)
    atm = DeltaVolQuote.atm(SimpleQuote(0.08925), DeltaType.Fwd, 1.0, AtmType.AtmDeltaNeutral)
    p25 = DeltaVolQuote(-0.25, SimpleQuote(0.10087), 1.0, DeltaType.Fwd)
    c25 = DeltaVolQuote(0.25, SimpleQuote(0.08463), 1.0, DeltaType.Fwd)
    return spot, q_ts, r_ts, atm, p25, c25, today


_VV_ROWS = [
    (BarrierType.UpOut, OptionType.Call, 1.13321, 1.5, 0.11638, "vv_upout_call_k1"),
    (BarrierType.UpOut, OptionType.Call, 1.31179, 1.5, 0.08925, "vv_upout_call_k3"),
    (BarrierType.UpOut, OptionType.Put, 1.38843, 1.5, 0.08463, "vv_upout_put_k4"),
    (BarrierType.UpIn, OptionType.Call, 1.13321, 1.5, 0.11638, "vv_upin_call_k1"),
    (BarrierType.DownOut, OptionType.Put, 1.31179, 1.0, 0.08925, "vv_downout_put_k3"),
]


@pytest.mark.parametrize(("bt", "ot", "strike", "barrier", "smile_vol", "key"), _VV_ROWS)
def test_vanna_volga_single_barrier(
    ref: dict[str, Any],
    bt: BarrierType,
    ot: OptionType,
    strike: float,
    barrier: float,
    smile_vol: float,
    key: str,
) -> None:
    spot, q_ts, r_ts, atm, p25, c25, today = _fx_setup()
    t = 1.0
    fwd = spot.value() * q_ts.discount(t) / r_ts.discount(t)
    bs_vanilla = black_formula(ot, strike, fwd, smile_vol * math.sqrt(t), r_ts.discount(t))
    payoff = PlainVanillaPayoff(ot, strike)
    exercise = EuropeanExercise(today + 365)
    opt = BarrierOption(bt, barrier, 0.0, payoff, exercise)
    eng = VannaVolgaBarrierEngine(atm, p25, c25, spot, r_ts, q_ts, True, bs_vanilla)
    opt.set_pricing_engine(eng)
    # LOOSE: vanna-volga involves FD Greeks; ~1e-6 agreement with C++.
    tolerance.loose(opt.npv(), ref[key])


# --- VannaVolga double-barrier ------------------------------------------


def test_vanna_volga_double_barrier_knockout(ref: dict[str, Any]) -> None:
    spot, q_ts, r_ts, atm, p25, c25, today = _fx_setup()
    payoff = PlainVanillaPayoff(OptionType.Call, 1.30)
    exercise = EuropeanExercise(today + 365)
    opt = DoubleBarrierOption(DoubleBarrierType.KnockOut, 1.1, 1.5, 0.0, payoff, exercise)
    eng = VannaVolgaDoubleBarrierEngine(atm, p25, c25, spot, r_ts, q_ts, False, 0.0, 5)
    opt.set_pricing_engine(eng)
    tolerance.loose(opt.npv(), ref["vv_double_ko_call"])


def test_vanna_volga_double_barrier_knockin(ref: dict[str, Any]) -> None:
    spot, q_ts, r_ts, atm, p25, c25, today = _fx_setup()
    payoff = PlainVanillaPayoff(OptionType.Call, 1.30)
    exercise = EuropeanExercise(today + 365)
    opt = DoubleBarrierOption(DoubleBarrierType.KnockIn, 1.1, 1.5, 0.0, payoff, exercise)
    eng = VannaVolgaDoubleBarrierEngine(atm, p25, c25, spot, r_ts, q_ts, False, 0.0, 5)
    opt.set_pricing_engine(eng)
    tolerance.loose(opt.npv(), ref["vv_double_ki_call"])
