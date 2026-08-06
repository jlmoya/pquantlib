"""Shared fixtures for the v1.43 step-conditions cross-validation tests.

Not a test module (leading underscore, so pytest does not collect it).

Everything here mirrors ``migration-harness/cpp/probes/v143_methods_stepconditions/probe.cpp``
one for one: same meshers, same payoffs, same schedules. The seed arrays are
**not** rebuilt here — they are read out of the probe's JSON, so the Python
side operates on the exact IEEE-754 doubles the C++ run used and a mismatch
can only come from the condition arithmetic itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.payoffs import OptionType, Payoff, PlainVanillaPayoff
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight

_REFERENCE_KEY = "v143/methods/stepconditions"


def load() -> dict[str, Any]:
    """The whole probe reference document."""
    return reference_reader.load(_REFERENCE_KEY)


def ref_array(reference: dict[str, Any], key: str) -> Array:
    """A reference entry as a fresh float64 array."""
    values: list[float] = reference[key]
    return np.array(values, dtype=np.float64)


def assert_array_tight(actual: Array, reference: dict[str, Any], key: str) -> None:
    """Element-wise TIGHT comparison against a reference array.

    TIGHT (abs 1e-14 / rel 1e-12) rather than EXACT: the storage and
    arithmetic-average conditions run through an interpolant whose internal
    linear solve is not required to associate its floating-point operations
    the same way QuantLib's ``TridiagonalOperator`` does. Every other block in
    this cluster in fact matches bit for bit.
    """
    expected: list[float] = reference[key]
    assert actual.shape == (len(expected),), f"{key}: length {actual.shape} != {len(expected)}"
    for i, exp in enumerate(expected):
        tight(float(actual[i]), float(exp), reason=f"{key}[{i}]")


class LogInnerValue:
    """Probe-local stand-in for C++ ``FdmLogInnerValue``.

    # C++ parity: ``FdmCellAveragingInnerValue::innerValue`` composed with
    # ``FdmLogInnerValue``'s ``gridMapping = std::exp``, i.e.
    # ``payoff(exp(mesher->location(iter, direction)))``.

    Declared here rather than imported from
    ``pquantlib.methods.finitedifferences.utilities`` so this cluster's tests
    stay independent of that (concurrently developed) module. The tests pin
    this stub against the probe's ``*_inner`` arrays first, so it is itself
    cross-validated before anything is built on top of it.
    """

    def __init__(self, payoff: Payoff, mesher: FdmMesher, direction: int) -> None:
        self._payoff: Payoff = payoff
        self._mesher: FdmMesher = mesher
        self._direction: int = direction

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        del t  # C++ parity: FdmCellAveragingInnerValue::innerValue ignores t.
        return self._payoff(math.exp(self._mesher.location(iterator, self._direction)))


def inner_values(mesher: FdmMesher, calculator: LogInnerValue, t: float) -> Array:
    """Per-node inner values in flat-index order (probe helper ``innerValues``)."""
    out: Array = np.empty(mesher.layout().size(), dtype=np.float64)
    for iterator in mesher.layout().iter():
        out[iterator.index] = calculator.inner_value(iterator, t)
    return out


# --------------------------------------------------------------------------
# Meshers — identical to the probe's fixtures.
# --------------------------------------------------------------------------


def bermudan_mesher() -> FdmMesherComposite:
    """dim (5, 3); direction 0 is log-spot over [50, 150]."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(150.0), 5),
        Uniform1dMesher(0.0, 2.0, 3),
    )


def swing_mesher() -> FdmMesherComposite:
    """dim (5, 4); direction 1 counts exercise rights used."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(150.0), 5),
        Uniform1dMesher(0.0, 3.0, 4),
    )


def storage_mesher() -> FdmMesherComposite:
    """dim (5, 5); direction 0 is log-price over [2, 6], direction 1 volume 0..4."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(2.0), math.log(6.0), 5),
        Uniform1dMesher(0.0, 4.0, 5),
    )


def average_mesher() -> FdmMesherComposite:
    """dim (5, 6); log-equity over [50, 150], log-average over [60, 140]."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(150.0), 5),
        Uniform1dMesher(math.log(60.0), math.log(140.0), 6),
    )


def put_100(mesher: FdmMesher) -> LogInnerValue:
    return LogInnerValue(PlainVanillaPayoff(OptionType.Put, 100.0), mesher, 0)


def call_100(mesher: FdmMesher) -> LogInnerValue:
    return LogInnerValue(PlainVanillaPayoff(OptionType.Call, 100.0), mesher, 0)


def spot_price(mesher: FdmMesher) -> LogInnerValue:
    """A zero-strike call turns the log-inner-value into the spot itself."""
    return LogInnerValue(PlainVanillaPayoff(OptionType.Call, 0.0), mesher, 0)
