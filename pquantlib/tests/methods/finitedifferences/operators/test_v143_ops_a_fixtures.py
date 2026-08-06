"""Shared fixtures for the v1.43 model-family FD-operator cross-validation.

# C++ parity: migration-harness/cpp/probes/v143_methods_operators/probe.cpp
# @ v1.43 (6b57206e0).

Every expected value in the sibling ``test_v143_ops_a_*`` modules comes from
running that probe against C++ QuantLib v1.43; this module only rebuilds the
*inputs* (curves, meshers, synthetic vectors) bit-identically on the Python
side and provides the comparison helpers.

The module name carries the ``test_`` prefix so pytest may import it as part
of the package; it deliberately contains no test functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.sparse import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    csr_matrix,
)

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_composite import (
    FdmLinearOpComposite,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# ``tight`` / ``loose`` / a ``functools.partial`` around ``custom`` all match.
ToleranceFn = Callable[[float, float], None]

REFERENCE_KEY = "v143/methods/operators"


# --- market data (mirrors probe.cpp ``main``) ---------------------------


def day_counter() -> DayCounter:
    return Actual365Fixed()


def today() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


def r_ts() -> FlatForward:
    """Flat 4% risk-free curve. # C++ parity: probe.cpp ``rTS``."""
    return FlatForward.from_rate(today(), 0.04, day_counter())


def q_ts() -> FlatForward:
    """Flat 1.5% dividend curve. # C++ parity: probe.cpp ``qTS``."""
    return FlatForward.from_rate(today(), 0.015, day_counter())


def heston_process(rTS: FlatForward, qTS: FlatForward) -> HestonProcess:  # noqa: N803
    """# C++ parity: probe.cpp ``makeHeston``."""
    return HestonProcess(
        risk_free_rate=rTS,
        dividend_yield=qTS,
        s0=SimpleQuote(100.0),
        v0=0.09,
        kappa=1.5,
        theta=0.04,
        sigma=0.4,
        rho=-0.6,
    )


def heston_mesher() -> FdmMesherComposite:
    """# C++ parity: probe.cpp ``makeHestonMesher``."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(70.0), math.log(140.0), 7),
        Uniform1dMesher(0.005, 0.4, 5),
    )


# --- synthetic test vectors (mirrors probe.cpp) -------------------------


def vec_const(mesher: FdmMesher) -> Array:
    """``v[i] = 1``."""
    return np.ones(mesher.layout().size(), dtype=np.float64)


def vec_lin(mesher: FdmMesher, direction: int) -> Array:
    """``v[i] = location(i, direction)``."""
    return mesher.locations(direction).copy()


def vec_mix(mesher: FdmMesher) -> Array:
    """``v[i] = location(i, 0) * location(i, 1)``."""
    return mesher.locations(0) * mesher.locations(1)


def vec_idx(mesher: FdmMesher) -> Array:
    """``v[i] = 1 + 0.01 * i`` — catches index permutations."""
    n = mesher.layout().size()
    return np.array([1.0 + 0.01 * i for i in range(n)], dtype=np.float64)


# --- comparison helpers -------------------------------------------------


def expected(ref: dict[str, Any], key: str) -> list[float]:
    """The probe's array value for ``key`` as a list of floats."""
    return [float(v) for v in ref[key]]


def assert_array(actual: Array, ref: dict[str, Any], key: str, *, compare: ToleranceFn = tight) -> None:
    """Assert ``actual`` matches the probe array stored under ``key``."""
    exp = expected(ref, key)
    assert actual.shape == (len(exp),), f"{key}: shape {actual.shape} vs {len(exp)}"
    for i, e in enumerate(exp):
        compare(float(actual[i]), e)


def assert_surface(
    ref: dict[str, Any],
    prefix: str,
    op: FdmLinearOpComposite,
    v: Array,
    tag: str,
    n_directions: int,
    dt: float,
    *,
    compare: ToleranceFn = tight,
) -> None:
    """Assert the full composite surface for one synthetic vector.

    Mirrors ``emit_surface`` in probe.cpp: ``apply``, ``apply_mixed``,
    ``apply_direction(d, .)`` and ``solve_splitting(d, ., dt)`` for every
    direction, plus ``preconditioner``.
    """
    p = f"{prefix}_{tag}_"
    assert_array(op.apply(v), ref, p + "apply", compare=compare)
    assert_array(op.apply_mixed(v), ref, p + "apply_mixed", compare=compare)
    for d in range(n_directions):
        assert_array(op.apply_direction(d, v), ref, p + f"apply_dir{d}", compare=compare)
        assert_array(op.solve_splitting(d, v, dt), ref, p + f"solve_dir{d}", compare=compare)
    assert_array(op.preconditioner(v, dt), ref, p + "precond", compare=compare)


def diag_of(matrix: csr_matrix) -> Array:
    """The main diagonal of a sparse operator matrix as a dense Array."""
    dense: Array = np.asarray(matrix.todense(), dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    return np.array([dense[i, i] for i in range(dense.shape[0])], dtype=np.float64)


def all_vectors(mesher: FdmMesher, *, with_direction2: bool = False) -> list[tuple[str, Array]]:
    """The probe's ``(tag, vector)`` list for a 2-D (or 3-D) mesher."""
    out: list[tuple[str, Array]] = [
        ("const", vec_const(mesher)),
        ("lin0", vec_lin(mesher, 0)),
        ("lin1", vec_lin(mesher, 1)),
    ]
    if with_direction2:
        out.append(("lin2", vec_lin(mesher, 2)))
    out.append(("mix", vec_mix(mesher)))
    out.append(("idx", vec_idx(mesher)))
    return out


__all__ = [
    "REFERENCE_KEY",
    "ToleranceFn",
    "all_vectors",
    "assert_array",
    "assert_surface",
    "day_counter",
    "diag_of",
    "expected",
    "heston_mesher",
    "heston_process",
    "q_ts",
    "r_ts",
    "today",
    "vec_const",
    "vec_idx",
    "vec_lin",
    "vec_mix",
]
