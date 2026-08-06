"""Cross-validation of the CIR hybrid FD operator family against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/operators/fdmcirop.{hpp,cpp}
# @ v1.43 (6b57206e0).

Covers ``FdmCIREquityPart``, ``FdmCIRRatesPart``, ``FdmCIRMixedPart`` and
``FdmCIROp``. Expected values come from
``migration-harness/cpp/probes/v143_methods_operators/probe.cpp`` (block 5).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_cir_op import (
    FdmCIREquityPart,
    FdmCIRMixedPart,
    FdmCIROp,
    FdmCIRRatesPart,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.cox_ingersoll_ross_process import CoxIngersollRossProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.testing import reference_reader
from pquantlib.time.calendars.null_calendar import NullCalendar
from tests.methods.finitedifferences.operators.test_v143_ops_a_fixtures import (
    REFERENCE_KEY,
    all_vectors,
    assert_array,
    assert_surface,
    day_counter,
    diag_of,
    q_ts,
    r_ts,
    today,
    vec_const,
    vec_idx,
    vec_mix,
)

_DT = 0.02
_T1 = 0.1
_T2 = 0.35
_RHO = -0.3
_STRIKE = 100.0
# CoxIngersollRossProcess(speed, vol, x0, level) — C++ ctor order.
_CIR_SPEED = 0.6
_CIR_VOL = 0.08
_CIR_X0 = 0.03
_CIR_LEVEL = 0.04


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load(REFERENCE_KEY)


def _mesher() -> FdmMesherComposite:
    """# C++ parity: probe.cpp ``block_cir``."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(70.0), math.log(140.0), 7),
        Uniform1dMesher(0.005, 0.09, 5),
    )


def _bs_process() -> BlackScholesMertonProcess:
    vol_ts = BlackConstantVol(
        reference_date=today(),
        calendar=NullCalendar(),
        volatility=0.22,
        day_counter=day_counter(),
    )
    return BlackScholesMertonProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=q_ts(),
        risk_free_ts=r_ts(),
        black_vol_ts=vol_ts,
    )


def _cir_process() -> CoxIngersollRossProcess:
    return CoxIngersollRossProcess(_CIR_SPEED, _CIR_VOL, _CIR_X0, _CIR_LEVEL)


def _build() -> tuple[FdmCIROp, FdmMesherComposite]:
    mesher = _mesher()
    op = FdmCIROp(mesher, _cir_process(), _bs_process(), _RHO, _STRIKE)
    op.set_time(_T1, _T2)
    return op, mesher


def test_mesher_locations(ref: dict[str, Any]) -> None:
    mesher = _mesher()
    assert_array(mesher.locations(0), ref, "cir_locations0")
    assert_array(mesher.locations(1), ref, "cir_locations1")


def test_equity_part(ref: dict[str, Any]) -> None:
    mesher = _mesher()
    part = FdmCIREquityPart(mesher, _bs_process(), _STRIKE)
    part.set_time(_T1, _T2)
    assert_array(part.get_map().apply(vec_const(mesher)), ref, "cir_equity_part_apply_const")
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "cir_equity_part_apply_idx")
    assert_array(diag_of(part.get_map().to_matrix()), ref, "cir_equity_part_diag")


def test_rates_part(ref: dict[str, Any]) -> None:
    mesher = _mesher()
    part = FdmCIRRatesPart(mesher, _CIR_VOL, _CIR_SPEED, _CIR_LEVEL)
    part.set_time(_T1, _T2)
    assert_array(part.get_map().apply(vec_const(mesher)), ref, "cir_rates_part_apply_const")
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "cir_rates_part_apply_idx")
    assert_array(diag_of(part.get_map().to_matrix()), ref, "cir_rates_part_diag")


def test_mixed_part(ref: dict[str, Any]) -> None:
    mesher = _mesher()
    part = FdmCIRMixedPart(mesher, _cir_process(), _bs_process(), _RHO, _STRIKE)
    part.set_time(_T1, _T2)
    assert_array(part.get_map().apply(vec_const(mesher)), ref, "cir_mixed_part_apply_const")
    assert_array(part.get_map().apply(vec_mix(mesher)), ref, "cir_mixed_part_apply_mix")
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "cir_mixed_part_apply_idx")


def test_size(ref: dict[str, Any]) -> None:
    op, _ = _build()
    assert op.size() == int(ref["cir_size"])


@pytest.mark.parametrize("tag", ["const", "lin0", "lin1", "mix", "idx"])
def test_operator_surface(ref: dict[str, Any], tag: str) -> None:
    op, mesher = _build()
    vectors = dict(all_vectors(mesher))
    assert_surface(ref, "cir", op, vectors[tag], tag, 2, _DT)


def test_to_matrix_decomp(ref: dict[str, Any]) -> None:
    op, _ = _build()
    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(ref["cir_decomp_size"])
    for i in range(3):
        assert_array(diag_of(decomp[i]), ref, f"cir_decomp{i}_diag")


def test_direction_out_of_range_fails() -> None:
    """# C++ parity: ``QL_FAIL("direction too large")``."""
    op, mesher = _build()
    v = vec_const(mesher)
    with pytest.raises(LibraryException, match="direction too large"):
        op.apply_direction(2, v)
    with pytest.raises(LibraryException, match="direction too large"):
        op.solve_splitting(2, v, _DT)
