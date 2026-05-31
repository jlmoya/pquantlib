"""Deterministic next_time_step generation for the W11-B pathwise products.

Each pathwise (Greeks-aware) ``MarketModelPathwiseMultiProduct`` is driven over
an ``LMMCurveState`` set on a flat forward (0.05, rate times {0.5, 1, 1.5, 2})
and its generated cash flows -- the value ``amount[0]`` plus the per-forward
derivatives ``amount[1:]`` -- are cross-validated TIGHT against the C++ probe
``migration-harness/references/cluster/w11b.json``. The analytic swaption deltas
are additionally checked to agree with the numerical-finite-difference twin
(both probed). ``MultiProductPathwiseWrapper`` is checked to expose the inner
pathwise value as an ordinary cash flow.

C++ parity:
  ql/models/marketmodels/products/pathwise/{pathwiseproductcaplet,
    pathwiseproductswap,pathwiseproductswaption,pathwiseproductinversefloater,
    pathwiseproductcashrebate}
  ql/models/marketmodels/products/multistep/multisteppathwisewrapper
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import CashFlow
from pquantlib.models.marketmodels.pathwise_multi_product import PathwiseCashFlow
from pquantlib.models.marketmodels.products import (
    MarketModelPathwiseCashRebate,
    MarketModelPathwiseCoterminalSwaptionsDeflated,
    MarketModelPathwiseCoterminalSwaptionsNumericalDeflated,
    MarketModelPathwiseInverseFloater,
    MarketModelPathwiseMultiCaplet,
    MarketModelPathwiseMultiDeflatedCaplet,
    MarketModelPathwiseSwap,
    MultiProductPathwiseWrapper,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight

_RATE_TIMES = [0.5, 1.0, 1.5, 2.0]
_FLAT = 0.05
_ACCRUALS = [0.5, 0.5, 0.5]
_PAY_TIMES = [1.0, 1.5, 2.0]
_RATES = 3


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w11b")


def _flat_state() -> LMMCurveState:
    cs = LMMCurveState(_RATE_TIMES)
    cs.set_on_forward_rates([_FLAT, _FLAT, _FLAT])
    return cs


def _pw_buffers(
    n_products: int, max_flows: int, rates: int = _RATES
) -> tuple[list[int], list[list[PathwiseCashFlow]]]:
    return (
        [0] * n_products,
        [
            [PathwiseCashFlow(amount=[0.0] * (rates + 1)) for _ in range(max_flows)]
            for _ in range(n_products)
        ],
    )


# --- MarketModelPathwiseMultiCaplet (undeflated) -----------------------------
def test_pathwise_multi_caplet_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.04, 0.045, 0.06]  # ITM, ITM, OTM
    p = MarketModelPathwiseMultiCaplet(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, strikes)
    assert p.already_deflated() is bool(ref["pwcap_deflated"])
    assert p.number_of_products() == int(ref["pwcap_np"])
    n, gen = _pw_buffers(p.number_of_products(), 1)
    p.reset()
    p.next_time_step(_flat_state(), n, gen)  # step 0 ITM
    assert n[0] == int(ref["pwcap_n0"])
    tight(gen[0][0].amount[0], ref["pwcap_amt0_val"])
    tight(gen[0][0].amount[1], ref["pwcap_amt0_d1"])
    tight(gen[0][0].amount[2], ref["pwcap_amt0_d2"])
    tight(gen[0][0].amount[3], ref["pwcap_amt0_d3"])
    p.next_time_step(_flat_state(), n, gen)  # step 1 ITM
    tight(gen[1][0].amount[0], ref["pwcap_amt1_val"])
    tight(gen[1][0].amount[2], ref["pwcap_amt1_d2"])
    done2 = p.next_time_step(_flat_state(), n, gen)  # step 2 OTM -> no flow
    assert done2 is bool(ref["pwcap_done2"])
    assert n[2] == int(ref["pwcap_n2"])


# --- MarketModelPathwiseMultiDeflatedCaplet ----------------------------------
def test_pathwise_deflated_caplet_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.04, 0.045, 0.06]
    p = MarketModelPathwiseMultiDeflatedCaplet(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, strikes)
    assert p.already_deflated() is bool(ref["pwdcap_deflated"])
    n, gen = _pw_buffers(p.number_of_products(), 1)
    p.reset()
    p.next_time_step(_flat_state(), n, gen)
    assert n[0] == int(ref["pwdcap_n0"])
    tight(gen[0][0].amount[0], ref["pwdcap_amt0_val"])
    tight(gen[0][0].amount[1], ref["pwdcap_amt0_d1"])
    tight(gen[0][0].amount[2], ref["pwdcap_amt0_d2"])


# --- MarketModelPathwiseSwap -------------------------------------------------
def test_pathwise_swap_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.045, 0.045, 0.045]
    p = MarketModelPathwiseSwap(_RATE_TIMES, _ACCRUALS, strikes, 1.0)
    assert p.already_deflated() is bool(ref["pwswap_deflated"])
    assert p.number_of_products() == int(ref["pwswap_np"])
    n, gen = _pw_buffers(1, 1)
    p.reset()
    p.next_time_step(_flat_state(), n, gen)
    assert n[0] == int(ref["pwswap_n0"])
    assert gen[0][0].time_index == int(ref["pwswap_ti0"])
    tight(gen[0][0].amount[0], ref["pwswap_amt0_val"])
    tight(gen[0][0].amount[1], ref["pwswap_amt0_d1"])
    tight(gen[0][0].amount[2], ref["pwswap_amt0_d2"])
    p.next_time_step(_flat_state(), n, gen)
    tight(gen[0][0].amount[0], ref["pwswap_amt1_val"])
    tight(gen[0][0].amount[2], ref["pwswap_amt1_d2"])


# --- MarketModelPathwiseCoterminalSwaptionsDeflated (analytic) ---------------
def test_pathwise_swaptions_analytic_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.04, 0.04, 0.04]
    p = MarketModelPathwiseCoterminalSwaptionsDeflated(_RATE_TIMES, strikes)
    assert p.number_of_products() == int(ref["pwswpt_np"])
    n, gen = _pw_buffers(p.number_of_products(), 1)
    p.reset()
    p.next_time_step(_flat_state(), n, gen)
    assert n[0] == int(ref["pwswpt_n0"])
    tight(gen[0][0].amount[0], ref["pwswpt_amt0_val"])
    tight(gen[0][0].amount[1], ref["pwswpt_amt0_d1"])
    tight(gen[0][0].amount[2], ref["pwswpt_amt0_d2"])
    tight(gen[0][0].amount[3], ref["pwswpt_amt0_d3"])


# --- numerical-FD swaption twin (matches the analytic deltas) ----------------
def test_pathwise_swaptions_numerical_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.04, 0.04, 0.04]
    p = MarketModelPathwiseCoterminalSwaptionsNumericalDeflated(_RATE_TIMES, strikes, 1e-6)
    n, gen = _pw_buffers(p.number_of_products(), 1)
    p.reset()
    p.next_time_step(_flat_state(), n, gen)
    # The un-bumped value is exact (TIGHT). The derivatives are central finite
    # differences (bumpSize 1e-6): each is a (upValue-downValue)/(2*bump) of two
    # bumped swap-rate/annuity re-prices, so it carries ~bump^2 truncation error
    # AND ~eps/bump rounding noise. Comparing this FD output to the C++ FD output
    # is only meaningful to FD precision (~1e-6 absolute here, since the
    # operation ordering of coterminalSwapRate/Annuity differs between the
    # numpy-backed LMMCurveState and C++). We therefore (1) pin the value TIGHT,
    # (2) compare each FD derivative to its C++ counterpart at a coarse FD
    # tolerance, and (3) below, confirm the FD deltas reproduce the TIGHT
    # analytic deltas to FD precision -- which is the substantive invariant.
    tight(gen[0][0].amount[0], ref["pwswptnum_amt0_val"])
    for k, key in enumerate(
        ("pwswptnum_amt0_d1", "pwswptnum_amt0_d2", "pwswptnum_amt0_d3"), start=1
    ):
        assert abs(gen[0][0].amount[k] - ref[key]) < 1e-5
    # analytic deltas agree with the FD twin to within the FD truncation error.
    analytic = MarketModelPathwiseCoterminalSwaptionsDeflated(_RATE_TIMES, strikes)
    na, ga = _pw_buffers(analytic.number_of_products(), 1)
    analytic.reset()
    analytic.next_time_step(_flat_state(), na, ga)
    for k in range(1, 4):
        assert abs(gen[0][0].amount[k] - ga[0][0].amount[k]) < 1e-6


# --- MarketModelPathwiseInverseFloater ---------------------------------------
def test_pathwise_inverse_floater_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.06, 0.06, 0.06]
    mults = [1.0, 1.0, 1.0]
    spreads = [0.001, 0.001, 0.001]
    p = MarketModelPathwiseInverseFloater(
        _RATE_TIMES, _ACCRUALS, _ACCRUALS, strikes, mults, spreads, _PAY_TIMES, True
    )
    assert p.number_of_products() == int(ref["pwinvf_np"])
    n, gen = _pw_buffers(1, 1)
    p.reset()
    p.next_time_step(_flat_state(), n, gen)
    assert n[0] == int(ref["pwinvf_n0"])
    assert gen[0][0].time_index == int(ref["pwinvf_ti0"])
    tight(gen[0][0].amount[0], ref["pwinvf_amt0_val"])
    tight(gen[0][0].amount[1], ref["pwinvf_amt0_d1"])
    tight(gen[0][0].amount[2], ref["pwinvf_amt0_d2"])


# --- MarketModelPathwiseCashRebate -------------------------------------------
def test_pathwise_cash_rebate_next_step(ref: dict[str, Any]) -> None:
    reb_pay = [0.5, 1.0, 1.5]
    evolution = EvolutionDescription(_RATE_TIMES, reb_pay)
    amounts = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
    p = MarketModelPathwiseCashRebate(evolution, reb_pay, amounts, 2)
    assert p.already_deflated() is bool(ref["pwreb_deflated"])
    n, gen = _pw_buffers(2, 1)
    p.reset()
    done = p.next_time_step(_flat_state(), n, gen)
    assert done is bool(ref["pwreb_done"])
    assert n[0] == int(ref["pwreb_n0"])
    assert gen[0][0].time_index == int(ref["pwreb_ti00"])
    tight(gen[0][0].amount[0], ref["pwreb_amt00_val"])
    tight(gen[0][0].amount[1], ref["pwreb_amt00_d1"])
    tight(gen[1][0].amount[0], ref["pwreb_amt10_val"])


# --- MultiProductPathwiseWrapper(caplet) -------------------------------------
def test_pathwise_wrapper_next_step(ref: dict[str, Any]) -> None:
    strikes = [0.04, 0.045, 0.06]
    inner = MarketModelPathwiseMultiCaplet(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, strikes)
    p = MultiProductPathwiseWrapper(inner)
    assert p.number_of_products() == int(ref["wrap_np"])
    assert p.max_number_of_cash_flows_per_product_per_step() == int(ref["wrap_mx"])
    n = [0] * p.number_of_products()
    gen: list[list[CashFlow]] = [
        [CashFlow() for _ in range(p.max_number_of_cash_flows_per_product_per_step())]
        for _ in range(p.number_of_products())
    ]
    p.reset()
    p.next_time_step(_flat_state(), n, gen)  # step 0 ITM
    assert n[0] == int(ref["wrap_n0"])
    tight(gen[0][0].amount, ref["wrap_amt0"])
    assert gen[0][0].time_index == int(ref["wrap_ti0"])
    p.next_time_step(_flat_state(), n, gen)
    tight(gen[1][0].amount, ref["wrap_amt1"])
