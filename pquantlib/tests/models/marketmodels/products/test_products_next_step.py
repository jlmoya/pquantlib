"""Deterministic next_time_step cash-flow generation for the W11-A products.

Each concrete ``MarketModelMultiProduct`` is driven over an ``LMMCurveState``
set on a flat forward (0.05, rate times {0.5, 1, 1.5, 2}) and its generated cash
flows are cross-validated TIGHT against the C++ probe
``migration-harness/references/cluster/w11a.json``. The probe builds the
identical state + products and emits the per-step amounts.

C++ parity:
  ql/models/marketmodels/products/multistep/{multistepforwards,multistepoptionlets,
    multistepswap,multistepcoterminalswaps,multistepcoterminalswaptions,cashrebate}
  ql/models/marketmodels/products/onestep/{onestepforwards,onestepoptionlets}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import CashFlow
from pquantlib.models.marketmodels.products import (
    MarketModelCashRebate,
    MultiStepCoterminalSwaps,
    MultiStepCoterminalSwaptions,
    MultiStepForwards,
    MultiStepNothing,
    MultiStepSwap,
    OneStepForwards,
    OneStepOptionlets,
)
from pquantlib.models.marketmodels.products.multistep_optionlets import (
    MultiStepOptionlets,
)
from pquantlib.payoffs import (
    OptionType,
    Payoff,
    PlainVanillaPayoff,
    StrikedTypePayoff,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight

_RATE_TIMES = [0.5, 1.0, 1.5, 2.0]
_FLAT = 0.05
_ACCRUALS = [0.5, 0.5, 0.5]
_PAY_TIMES = [1.0, 1.5, 2.0]
_STRIKES = [0.04, 0.045, 0.05]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w11a")


def _flat_state() -> LMMCurveState:
    cs = LMMCurveState(_RATE_TIMES)
    cs.set_on_forward_rates([_FLAT, _FLAT, _FLAT])
    return cs


def _buffers(n_products: int, max_flows: int) -> tuple[list[int], list[list[CashFlow]]]:
    return (
        [0] * n_products,
        [[CashFlow() for _ in range(max_flows)] for _ in range(n_products)],
    )


# --- MultiStepForwards -------------------------------------------------------
def test_multistep_forwards_next_step(ref: dict[str, Any]) -> None:
    fwd = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    n, gen = _buffers(fwd.number_of_products(), 1)
    fwd.reset()
    done0 = fwd.next_time_step(_flat_state(), n, gen)
    assert done0 is bool(ref["fwd_done0"])
    assert n[0] == int(ref["fwd_n0"])
    assert gen[0][0].time_index == int(ref["fwd_ti00"])
    tight(gen[0][0].amount, ref["fwd_amt00"])
    fwd.next_time_step(_flat_state(), n, gen)
    tight(gen[1][0].amount, ref["fwd_amt11"])
    done2 = fwd.next_time_step(_flat_state(), n, gen)
    assert done2 is bool(ref["fwd_done2"])
    tight(gen[2][0].amount, ref["fwd_amt22"])


# --- MultiStepOptionlets -----------------------------------------------------
def test_multistep_optionlets_next_step(ref: dict[str, Any]) -> None:
    payoffs: list[Payoff] = [PlainVanillaPayoff(OptionType.Call, k) for k in _STRIKES]
    opt = MultiStepOptionlets(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, payoffs)
    n, gen = _buffers(opt.number_of_products(), 1)
    opt.reset()
    opt.next_time_step(_flat_state(), n, gen)
    assert n[0] == int(ref["opt_n0"])
    tight(gen[0][0].amount, ref["opt_amt00"])
    opt.next_time_step(_flat_state(), n, gen)
    tight(gen[1][0].amount, ref["opt_amt11"])
    done2 = opt.next_time_step(_flat_state(), n, gen)
    assert done2 is bool(ref["opt_done2"])
    tight(gen[2][0].amount, ref["opt_amt22"])


# --- MultiStepSwap -----------------------------------------------------------
def test_multistep_swap_next_step(ref: dict[str, Any]) -> None:
    swap = MultiStepSwap(_RATE_TIMES, _ACCRUALS, _ACCRUALS, _PAY_TIMES, 0.045, True)
    n, gen = _buffers(1, 2)
    swap.reset()
    swap.next_time_step(_flat_state(), n, gen)
    assert n[0] == int(ref["swap_n0"])
    tight(gen[0][0].amount, ref["swap_fixed0"])
    tight(gen[0][1].amount, ref["swap_float0"])

    rec = MultiStepSwap(_RATE_TIMES, _ACCRUALS, _ACCRUALS, _PAY_TIMES, 0.045, False)
    rec.reset()
    rec.next_time_step(_flat_state(), n, gen)
    tight(gen[0][0].amount, ref["swap_rec_fixed0"])
    tight(gen[0][1].amount, ref["swap_rec_float0"])


# --- MultiStepCoterminalSwaptions -------------------------------------------
def test_multistep_coterminal_swaptions_next_step(ref: dict[str, Any]) -> None:
    payoffs: list[StrikedTypePayoff] = [
        PlainVanillaPayoff(OptionType.Call, 0.04) for _ in range(3)
    ]
    swns = MultiStepCoterminalSwaptions(_RATE_TIMES, _PAY_TIMES, payoffs)
    assert swns.number_of_products() == int(ref["swns_np"])
    n, gen = _buffers(swns.number_of_products(), 1)
    swns.reset()
    swns.next_time_step(_flat_state(), n, gen)
    tight(gen[0][0].amount, ref["swns_amt00"])


# --- MultiStepCoterminalSwaps -----------------------------------------------
def test_multistep_coterminal_swaps_next_step(ref: dict[str, Any]) -> None:
    swaps = MultiStepCoterminalSwaps(
        _RATE_TIMES, _ACCRUALS, _ACCRUALS, _PAY_TIMES, 0.045
    )
    assert swaps.number_of_products() == int(ref["cotsw_np"])
    n, gen = _buffers(swaps.number_of_products(), 2)
    swaps.reset()
    swaps.next_time_step(_flat_state(), n, gen)  # step 0: only product 0 active
    assert n[0] == int(ref["cotsw_n0_step0"])
    assert n[1] == int(ref["cotsw_n1_step0"])
    tight(gen[0][0].amount, ref["cotsw_fixed00"])
    tight(gen[0][1].amount, ref["cotsw_float00"])


# --- OneStepForwards ---------------------------------------------------------
def test_onestep_forwards_next_step(ref: dict[str, Any]) -> None:
    osf = OneStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    n, gen = _buffers(osf.number_of_products(), 1)
    osf.reset()
    done = osf.next_time_step(_flat_state(), n, gen)
    assert done is bool(ref["osf_done"])
    assert n[0] == int(ref["osf_n0"])
    tight(gen[0][0].amount, ref["osf_amt00"])
    tight(gen[1][0].amount, ref["osf_amt11"])
    tight(gen[2][0].amount, ref["osf_amt22"])


# --- OneStepOptionlets -------------------------------------------------------
def test_onestep_optionlets_next_step(ref: dict[str, Any]) -> None:
    payoffs: list[Payoff] = [
        PlainVanillaPayoff(OptionType.Call, 0.04),  # ITM
        PlainVanillaPayoff(OptionType.Call, 0.06),  # OTM
        PlainVanillaPayoff(OptionType.Call, 0.045),  # ITM
    ]
    oso = OneStepOptionlets(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, payoffs)
    n, gen = _buffers(oso.number_of_products(), 1)
    oso.reset()
    done = oso.next_time_step(_flat_state(), n, gen)
    assert done is bool(ref["oso_done"])
    assert n[0] == int(ref["oso_n0"])
    assert n[1] == int(ref["oso_n1"])  # OTM -> no cash flow
    assert n[2] == int(ref["oso_n2"])
    tight(gen[0][0].amount, ref["oso_amt00"])
    tight(gen[2][0].amount, ref["oso_amt22"])


# --- MarketModelCashRebate ---------------------------------------------------
def test_cash_rebate_next_step(ref: dict[str, Any]) -> None:
    pay_times = [0.5, 1.0, 1.5]
    evolution = EvolutionDescription(_RATE_TIMES, pay_times)
    amounts = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
    rebate = MarketModelCashRebate(evolution, pay_times, amounts, 2)
    assert rebate.max_number_of_cash_flows_per_product_per_step() == int(
        ref["rebate_maxcf"]
    )
    n, gen = _buffers(2, 1)
    rebate.reset()
    done = rebate.next_time_step(_flat_state(), n, gen)
    assert done is bool(ref["rebate_done"])
    assert n[0] == int(ref["rebate_n0"])
    assert gen[0][0].time_index == int(ref["rebate_ti00"])
    tight(gen[0][0].amount, ref["rebate_amt00"])
    tight(gen[1][0].amount, ref["rebate_amt10"])


# --- MultiStepNothing (no probe needed: produces no cash flows) ---------------
def test_multistep_nothing() -> None:
    evolution = EvolutionDescription(_RATE_TIMES)
    nothing = MultiStepNothing(evolution, number_of_products=2, done_index=2)
    assert nothing.number_of_products() == 2
    assert nothing.possible_cash_flow_times() == []
    assert nothing.max_number_of_cash_flows_per_product_per_step() == 0
    n = [1, 1]
    gen: list[list[CashFlow]] = [[], []]
    nothing.reset()
    # done_index = 2: not done after 1 step, done after 2.
    assert nothing.next_time_step(_flat_state(), n, gen) is False
    assert n == [0, 0]
    assert nothing.next_time_step(_flat_state(), n, gen) is True
