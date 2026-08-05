"""Shared reader for the ``v143/math/interp/cubic`` probe reference.

Not a test module (leading underscore, so pytest does not collect it).

The probe emits one JSON object per interpolation *case*, each carrying its
own inputs (``x``, ``y``, whatever parameters the class takes) alongside the
pinned outputs at a common evaluation grid: below the range, every node,
every midpoint, above the range. Tests walk those cases rather than
restating literals, so adding a case to the probe automatically adds
coverage.

Non-finite reference values are carried as the JSON strings ``"nan"``,
``"inf"`` and ``"-inf"`` — QuantLib genuinely produces them in the
FritschButland ``QL_MIN_REAL`` branch, and JSON has no literal for them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.testing import reference_reader

_REFERENCE_KEY = "v143/math/interp/cubic"

_NON_FINITE: dict[str, float] = {
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
}


def load() -> dict[str, Any]:
    """The whole reference document."""
    return reference_reader.load(_REFERENCE_KEY)


def num(value: Any) -> float:
    """Reference scalar as a float, decoding the non-finite string tokens."""
    if isinstance(value, str):
        return _NON_FINITE[value]
    return float(value)


def curve(case: dict[str, Any]) -> tuple[Array, Array]:
    """The ``(x, y)`` pillars of a case."""
    xs: Array = np.asarray([num(v) for v in case["x"]], dtype=np.float64)
    ys: Array = np.asarray([num(v) for v in case["y"]], dtype=np.float64)
    return xs, ys


def names(cases: list[dict[str, Any]]) -> list[str]:
    """Case names, for use as pytest ``ids``."""
    return [str(c["name"]) for c in cases]


#: A single evaluator: reference key -> the call that reproduces it.
Evaluator = tuple[str, Callable[[float], float]]


def evaluators(f: Interpolation) -> list[Evaluator]:
    """The four evaluators, keyed by the reference field they reproduce.

    Extrapolation is enabled on every call because the evaluation grid
    deliberately runs past both ends of the data.
    """
    return [
        ("value", lambda x: f(x, allow_extrapolation=True)),
        ("derivative", lambda x: f.derivative(x, allow_extrapolation=True)),
        (
            "second_derivative",
            lambda x: f.second_derivative(x, allow_extrapolation=True),
        ),
        ("primitive", lambda x: f.primitive(x, allow_extrapolation=True)),
    ]
