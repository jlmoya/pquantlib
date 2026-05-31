"""Tests for FdmSpreadPayoffInnerValue.

# C++ parity: ql/experimental/finitedifferences/fdmspreadpayoffinnervalue.hpp.

Reference values: migration-harness/references/cluster/w5a.json.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.experimental.finitedifferences.fdm_exp_ext_ou_inner_value_calculator import (
    FdmExpExtOUInnerValueCalculator,
)
from pquantlib.experimental.finitedifferences.fdm_spread_payoff_inner_value import (
    FdmSpreadPayoffInnerValue,
)
from pquantlib.instruments.basket_option import SpreadBasketPayoff
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def refs() -> dict[str, Any]:
    return reference_reader.load("cluster/w5a")


def test_spread_payoff_inner_value(refs: dict[str, Any]) -> None:
    """Spread payoff (Call, K=0) on 2D mesh.

    Spread = exp(u0) - exp(u1). Inner value = max(spread - 0, 0).
    TIGHT.
    """
    m1 = Uniform1dMesher(math.log(50.0), math.log(150.0), 5)
    m2 = Uniform1dMesher(math.log(50.0), math.log(150.0), 5)
    mesher = FdmMesherComposite(m1, m2)

    base_payoff = PlainVanillaPayoff(OptionType.Call, 0.0)
    spread_payoff = SpreadBasketPayoff(base_payoff)

    calc1 = FdmExpExtOUInnerValueCalculator(base_payoff, mesher, direction=0)
    calc2 = FdmExpExtOUInnerValueCalculator(base_payoff, mesher, direction=1)
    spread_calc = FdmSpreadPayoffInnerValue(spread_payoff, calc1, calc2)

    for iter_ in mesher.layout().iter():
        i = iter_.index
        iv = spread_calc.inner_value(iter_, 0.0)
        if i == 0:
            tolerance.tight(iv, refs["spread_payoff_iv_0"])
        if i == 4:
            tolerance.tight(iv, refs["spread_payoff_iv_4"])
        if i == 12:
            tolerance.tight(iv, refs["spread_payoff_iv_12"])
        if i == 20:
            tolerance.tight(iv, refs["spread_payoff_iv_20"])
        if i == 24:
            tolerance.tight(iv, refs["spread_payoff_iv_24"])


def test_spread_payoff_inner_value_avg_equals_inner() -> None:
    """avg_inner_value forwards to inner_value.

    # C++ parity: ``avgInnerValue`` is identical to ``innerValue``.
    """
    m1 = Uniform1dMesher(0.0, 1.0, 3)
    m2 = Uniform1dMesher(0.0, 1.0, 3)
    mesher = FdmMesherComposite(m1, m2)
    base_payoff = PlainVanillaPayoff(OptionType.Call, 0.0)
    spread_payoff = SpreadBasketPayoff(base_payoff)
    calc1 = FdmExpExtOUInnerValueCalculator(base_payoff, mesher, direction=0)
    calc2 = FdmExpExtOUInnerValueCalculator(base_payoff, mesher, direction=1)
    spread_calc = FdmSpreadPayoffInnerValue(spread_payoff, calc1, calc2)
    for iter_ in mesher.layout().iter():
        assert spread_calc.inner_value(iter_, 0.0) == spread_calc.avg_inner_value(iter_, 0.0)


def test_spread_payoff_inner_value_callable() -> None:
    """Callable interface — ``calc(iter, t)`` wraps ``inner_value``."""
    m1 = Uniform1dMesher(0.0, 1.0, 3)
    m2 = Uniform1dMesher(0.0, 1.0, 3)
    mesher = FdmMesherComposite(m1, m2)
    base_payoff = PlainVanillaPayoff(OptionType.Call, 0.0)
    spread_payoff = SpreadBasketPayoff(base_payoff)
    calc1 = FdmExpExtOUInnerValueCalculator(base_payoff, mesher, direction=0)
    calc2 = FdmExpExtOUInnerValueCalculator(base_payoff, mesher, direction=1)
    spread_calc = FdmSpreadPayoffInnerValue(spread_payoff, calc1, calc2)
    for iter_ in mesher.layout().iter():
        assert spread_calc(iter_, 0.0) == spread_calc.inner_value(iter_, 0.0)
