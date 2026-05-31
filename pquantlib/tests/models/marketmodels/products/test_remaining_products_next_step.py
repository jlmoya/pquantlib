"""Deterministic next_time_step generation for the W11-B remaining products.

Each remaining ``MarketModelMultiProduct`` (multistep coinitial swaps / ratchet
/ inverse floater / tarn / swaption / period caplet swaptions, onestep coinitial
/ coterminal swaps) is driven over an ``LMMCurveState`` set on a flat forward
(0.05, rate times {0.5, 1, 1.5, 2}) and its generated cash flows are
cross-validated TIGHT against the C++ probe
``migration-harness/references/cluster/w11b.json``. The TARN and ratchet
path-dependent accumulation is checked over a full deterministic step sequence.

C++ parity:
  ql/models/marketmodels/products/multistep/{multistepcoinitialswaps,
    multistepratchet,multistepinversefloater,multisteptarn,multistepswaption,
    multistepperiodcapletswaptions}
  ql/models/marketmodels/products/onestep/{onestepcoinitialswaps,
    onestepcoterminalswaps}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.multi_product import CashFlow
from pquantlib.models.marketmodels.products import (
    MultiStepCoinitialSwaps,
    MultiStepInverseFloater,
    MultiStepPeriodCapletSwaptions,
    MultiStepRatchet,
    MultiStepSwaption,
    MultiStepTarn,
    OneStepCoinitialSwaps,
    OneStepCoterminalSwaps,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight

_RATE_TIMES = [0.5, 1.0, 1.5, 2.0]
_FLAT = 0.05
_ACCRUALS = [0.5, 0.5, 0.5]
_PAY_TIMES = [1.0, 1.5, 2.0]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w11b")


def _flat_state() -> LMMCurveState:
    cs = LMMCurveState(_RATE_TIMES)
    cs.set_on_forward_rates([_FLAT, _FLAT, _FLAT])
    return cs


def _buffers(n_products: int, max_flows: int) -> tuple[list[int], list[list[CashFlow]]]:
    return (
        [0] * n_products,
        [[CashFlow() for _ in range(max_flows)] for _ in range(n_products)],
    )


# --- MultiStepCoinitialSwaps -------------------------------------------------
def test_multistep_coinitial_swaps_next_step(ref: dict[str, Any]) -> None:
    p = MultiStepCoinitialSwaps(_RATE_TIMES, _ACCRUALS, _ACCRUALS, _PAY_TIMES, 0.045)
    assert p.number_of_products() == int(ref["coin_np"])
    n, gen = _buffers(p.number_of_products(), 2)
    p.reset()
    done0 = p.next_time_step(_flat_state(), n, gen)  # step 0: all products active
    assert done0 is bool(ref["coin_done0"])
    assert n[0] == int(ref["coin_n0_step0"])
    assert n[2] == int(ref["coin_n2_step0"])
    tight(gen[0][0].amount, ref["coin_fixed00"])
    tight(gen[0][1].amount, ref["coin_float00"])
    assert gen[0][0].time_index == int(ref["coin_ti00"])
    p.next_time_step(_flat_state(), n, gen)  # step 1: products 1,2 active
    assert n[0] == int(ref["coin_n0_step1"])
    assert n[1] == int(ref["coin_n1_step1"])
    tight(gen[1][0].amount, ref["coin_fixed11"])


# --- MultiStepInverseFloater -------------------------------------------------
def test_multistep_inverse_floater_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.06, 0.06, 0.06]
    mults = [1.0, 1.0, 1.0]
    spreads = [0.001, 0.001, 0.001]
    p = MultiStepInverseFloater(
        _RATE_TIMES, _ACCRUALS, _ACCRUALS, strikes, mults, spreads, _PAY_TIMES, True
    )
    n, gen = _buffers(1, 1)
    p.reset()
    p.next_time_step(_flat_state(), n, gen)
    assert n[0] == int(ref["invf_n0"])
    assert gen[0][0].time_index == int(ref["invf_ti0"])
    tight(gen[0][0].amount, ref["invf_amt0"])
    p.next_time_step(_flat_state(), n, gen)
    tight(gen[0][0].amount, ref["invf_amt1"])


# --- MultiStepRatchet (path-dependent full-ratchet accumulation) -------------
def test_multistep_ratchet_path(ref: dict[str, Any]) -> None:
    p = MultiStepRatchet(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, 1.0, 1.0, 0.0, 0.0, 0.04, True)
    n, gen = _buffers(1, 1)
    p.reset()
    amts: list[float] = []
    done = False
    for _ in range(3):
        if done:
            break
        done = p.next_time_step(_flat_state(), n, gen)
        amts.append(gen[0][0].amount)
    expected = list(ref["ratchet_amts"])
    assert len(amts) == len(expected)
    for got, exp in zip(amts, expected, strict=True):
        tight(got, exp)
    assert done is bool(ref["ratchet_done"])


# --- MultiStepTarn (path-dependent target redemption) ------------------------
def test_multistep_tarn_path(ref: dict[str, Any]) -> None:
    strikes = [0.06, 0.06, 0.06]
    mults = [1.0, 1.0, 1.0]
    spreads = [0.0, 0.0, 0.0]
    p = MultiStepTarn(
        _RATE_TIMES, _ACCRUALS, _ACCRUALS, _PAY_TIMES, _PAY_TIMES, 0.02, strikes, mults, spreads
    )
    assert p.number_of_products() == int(ref["tarn_np"])
    assert p.max_number_of_cash_flows_per_product_per_step() == int(ref["tarn_ncf"])
    n, gen = _buffers(1, 2)
    p.reset()
    floats: list[float] = []
    fixeds: list[float] = []
    float_ti: list[int] = []
    fixed_ti: list[int] = []
    dones: list[float] = []
    done = False
    for _ in range(3):
        if done:
            break
        done = p.next_time_step(_flat_state(), n, gen)
        floats.append(gen[0][0].amount)
        fixeds.append(gen[0][1].amount)
        float_ti.append(gen[0][0].time_index)
        fixed_ti.append(gen[0][1].time_index)
        dones.append(1.0 if done else 0.0)
    assert len(floats) == int(ref["tarn_steps"])
    for got, exp in zip(floats, ref["tarn_float"], strict=True):
        tight(got, exp)
    for got, exp in zip(fixeds, ref["tarn_fixed"], strict=True):
        tight(got, exp)
    assert float_ti == [int(x) for x in ref["tarn_float_ti"]]
    assert fixed_ti == [int(x) for x in ref["tarn_fixed_ti"]]
    assert dones == list(ref["tarn_done"])


# --- MultiStepSwaption -------------------------------------------------------
def test_multistep_swaption_next_step(ref: dict[str, Any]) -> None:
    payoff: StrikedTypePayoff = PlainVanillaPayoff(OptionType.Call, 0.04)
    p = MultiStepSwaption(_RATE_TIMES, 1, 3, payoff)
    assert p.number_of_products() == int(ref["swpt_np"])
    n, gen = _buffers(1, 1)
    p.reset()
    done0 = p.next_time_step(_flat_state(), n, gen)  # currentIndex 0 != startIndex 1
    assert done0 is bool(ref["swpt_done0"])
    assert n[0] == int(ref["swpt_n0_step0"])
    done1 = p.next_time_step(_flat_state(), n, gen)  # currentIndex 1 == startIndex
    assert done1 is bool(ref["swpt_done1"])
    assert n[0] == int(ref["swpt_n0_step1"])
    tight(gen[0][0].amount, ref["swpt_amt"])
    assert gen[0][0].time_index == int(ref["swpt_ti"])


# --- MultiStepPeriodCapletSwaptions ------------------------------------------
def test_multistep_period_caplet_swaptions_next_step(ref: dict[str, Any]) -> None:
    fwd_pay = [1.0, 1.5, 2.0]
    swp_pay = [1.0, 1.5, 2.0]
    fwd_po: list[StrikedTypePayoff] = [
        PlainVanillaPayoff(OptionType.Call, 0.04) for _ in range(3)
    ]
    swp_po: list[StrikedTypePayoff] = [
        PlainVanillaPayoff(OptionType.Call, 0.04) for _ in range(3)
    ]
    p = MultiStepPeriodCapletSwaptions(_RATE_TIMES, fwd_pay, swp_pay, fwd_po, swp_po, 1, 0)
    assert p.number_of_products() == int(ref["pcs_np"])
    n, gen = _buffers(p.number_of_products(), 1)
    p.reset()
    done0 = p.next_time_step(_flat_state(), n, gen)
    assert done0 is bool(ref["pcs_done0"])
    assert n[0] == int(ref["pcs_caplet0_n"])
    tight(gen[0][0].amount, ref["pcs_caplet0_amt"])
    assert n[3] == int(ref["pcs_swaption0_n"])
    tight(gen[3][0].amount, ref["pcs_swaption0_amt"])


# --- OneStepCoinitialSwaps ---------------------------------------------------
def test_onestep_coinitial_swaps_next_step(ref: dict[str, Any]) -> None:
    p = OneStepCoinitialSwaps(_RATE_TIMES, _ACCRUALS, _ACCRUALS, _PAY_TIMES, 0.045)
    assert p.number_of_products() == int(ref["oscoin_np"])
    assert p.max_number_of_cash_flows_per_product_per_step() == int(ref["oscoin_mx"])
    n, gen = _buffers(p.number_of_products(), p.max_number_of_cash_flows_per_product_per_step())
    p.reset()
    done = p.next_time_step(_flat_state(), n, gen)
    assert done is bool(ref["oscoin_done"])
    assert n[0] == int(ref["oscoin_n0"])
    assert n[2] == int(ref["oscoin_n2"])
    tight(gen[2][0].amount, ref["oscoin_p2_fixed0"])
    tight(gen[2][1].amount, ref["oscoin_p2_float0"])
    assert gen[2][0].time_index == int(ref["oscoin_p2_ti0"])
    assert gen[2][4].time_index == int(ref["oscoin_p2_ti2"])
    tight(gen[0][0].amount, ref["oscoin_p0_fixed0"])


# --- OneStepCoterminalSwaps --------------------------------------------------
def test_onestep_coterminal_swaps_next_step(ref: dict[str, Any]) -> None:
    p = OneStepCoterminalSwaps(_RATE_TIMES, _ACCRUALS, _ACCRUALS, _PAY_TIMES, 0.045)
    assert p.number_of_products() == int(ref["oscot_np"])
    assert p.max_number_of_cash_flows_per_product_per_step() == int(ref["oscot_mx"])
    n, gen = _buffers(p.number_of_products(), p.max_number_of_cash_flows_per_product_per_step())
    p.reset()
    done = p.next_time_step(_flat_state(), n, gen)
    assert done is bool(ref["oscot_done"])
    assert n[0] == int(ref["oscot_n0"])
    assert n[2] == int(ref["oscot_n2"])
    tight(gen[0][0].amount, ref["oscot_p0_fixed0"])
    assert gen[0][0].time_index == int(ref["oscot_p0_ti0"])
    assert gen[0][4].time_index == int(ref["oscot_p0_ti2"])
    tight(gen[2][0].amount, ref["oscot_p2_fixed0"])
    assert gen[2][0].time_index == int(ref["oscot_p2_ti0"])
