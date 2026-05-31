"""Tests for FdmExpExtOUInnerValueCalculator.

# C++ parity: ql/experimental/finitedifferences/fdmexpextouinnervaluecalculator.hpp.

Reference values: migration-harness/references/cluster/w5a.json.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.experimental.finitedifferences.fdm_exp_ext_ou_inner_value_calculator import (
    FdmExpExtOUInnerValueCalculator,
)
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


def test_exp_ext_ou_iv_at_3_nodes(refs: dict[str, Any]) -> None:
    """Plain-vanilla Call(K=100), mesher Uniform1d([log(50), log(150)], 11).

    Inner value = max(exp(u) - 100, 0).
    Tested at i=0, 5, 10.

    TIGHT.
    """
    mesher_1d = Uniform1dMesher(math.log(50.0), math.log(150.0), 11)
    mesher = FdmMesherComposite(mesher_1d)
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    calc = FdmExpExtOUInnerValueCalculator(payoff, mesher)

    for iter_ in mesher.layout().iter():
        i = iter_.index
        spot = math.exp(mesher.location(iter_, 0))
        iv = calc.inner_value(iter_, 0.0)
        if i == 0:
            tolerance.tight(spot, refs["exp_ou_iv_calc_0_spot"])
            tolerance.tight(iv, refs["exp_ou_iv_calc_0_iv"])
        if i == 5:
            tolerance.tight(spot, refs["exp_ou_iv_calc_5_spot"])
            tolerance.tight(iv, refs["exp_ou_iv_calc_5_iv"])
        if i == 10:
            tolerance.tight(spot, refs["exp_ou_iv_calc_10_spot"])
            tolerance.tight(iv, refs["exp_ou_iv_calc_10_iv"])


def test_exp_ext_ou_iv_avg_equals_inner() -> None:
    """avg_inner_value forwards to inner_value (same single-node semantics).

    # C++ parity: ``avgInnerValue`` is identical to ``innerValue``.
    """
    mesher_1d = Uniform1dMesher(0.0, 1.0, 3)
    mesher = FdmMesherComposite(mesher_1d)
    payoff = PlainVanillaPayoff(OptionType.Call, 1.5)
    calc = FdmExpExtOUInnerValueCalculator(payoff, mesher)
    for iter_ in mesher.layout().iter():
        assert calc.inner_value(iter_, 0.0) == calc.avg_inner_value(iter_, 0.0)


def test_exp_ext_ou_iv_with_seasonal_shape() -> None:
    """Seasonal shape ``f(t)`` shifts the spot before payoff evaluation.

    Shape ``[(0.5, 0.1)]`` -> ``f(0.5) = 0.1`` -> spot becomes exp(0.1 + u).
    TIGHT.
    """
    mesher_1d = Uniform1dMesher(math.log(50.0), math.log(150.0), 5)
    mesher = FdmMesherComposite(mesher_1d)
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    shape = [(0.5, 0.1)]
    calc = FdmExpExtOUInnerValueCalculator(payoff, mesher, shape=shape, direction=0)
    for iter_ in mesher.layout().iter():
        if iter_.index == 2:
            spot = math.exp(0.1 + mesher.location(iter_, 0))
            expected = max(spot - 100.0, 0.0)
            tolerance.tight(calc.inner_value(iter_, 0.5), expected)


def test_exp_ext_ou_iv_callable_interface() -> None:
    """The calculator is callable (``calc(iter, t)``) — wraps inner_value.

    # C++ parity: the C++ class does not implement operator() but the
    # Python port exposes ``__call__`` for ergonomic use as
    # ``InnerValueCalculator = Callable``.
    """
    mesher_1d = Uniform1dMesher(0.0, 1.0, 3)
    mesher = FdmMesherComposite(mesher_1d)
    payoff = PlainVanillaPayoff(OptionType.Call, 1.5)
    calc = FdmExpExtOUInnerValueCalculator(payoff, mesher)
    for iter_ in mesher.layout().iter():
        assert calc(iter_, 0.0) == calc.inner_value(iter_, 0.0)
