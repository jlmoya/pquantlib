"""Cross-validation of the Heston FD operator family against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/operators/fdmhestonop.{hpp,cpp}
# @ v1.43 (6b57206e0).

Covers ``FdmHestonEquityPart``, ``FdmHestonVariancePart`` and ``FdmHestonOp``
(plain, mixing-factor != 1, and with a leverage function — both the plain
branch and the ``max(0.01, .)`` clamp). Expected values come from
``migration-harness/cpp/probes/v143_methods_operators/probe.cpp`` (block 3).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_heston_op import (
    FdmHestonEquityPart,
    FdmHestonOp,
    FdmHestonVariancePart,
)
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import (
    FdmQuantoHelper,
)
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_constant_vol import (
    LocalConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from tests.methods.finitedifferences.operators.test_v143_ops_a_fixtures import (
    REFERENCE_KEY,
    all_vectors,
    assert_array,
    assert_surface,
    day_counter,
    diag_of,
    heston_mesher,
    heston_process,
    q_ts,
    r_ts,
    today,
    vec_const,
    vec_idx,
)

_DT = 0.02
_T1 = 0.1
_T2 = 0.35


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load(REFERENCE_KEY)


def test_mesher_locations(ref: dict[str, Any]) -> None:
    mesher = heston_mesher()
    assert_array(mesher.locations(0), ref, "heston_locations0")
    assert_array(mesher.locations(1), ref, "heston_locations1")


def test_equity_part(ref: dict[str, Any]) -> None:
    mesher = heston_mesher()
    part = FdmHestonEquityPart(mesher, r_ts(), q_ts(), None)
    part.set_time(_T1, _T2)
    assert_array(part.get_L(), ref, "heston_equity_part_L")
    assert_array(part.get_map().apply(vec_const(mesher)), ref, "heston_equity_part_apply_const")
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "heston_equity_part_apply_idx")
    assert_array(diag_of(part.get_map().to_matrix()), ref, "heston_equity_part_diag")


def test_variance_part(ref: dict[str, Any]) -> None:
    mesher = heston_mesher()
    part = FdmHestonVariancePart(mesher, r_ts(), 0.4, 1.5, 0.04)
    part.set_time(_T1, _T2)
    assert_array(part.get_map().apply(vec_const(mesher)), ref, "heston_variance_part_apply_const")
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "heston_variance_part_apply_idx")
    assert_array(diag_of(part.get_map().to_matrix()), ref, "heston_variance_part_diag")


def _build() -> tuple[FdmHestonOp, FdmMesherComposite]:
    mesher = heston_mesher()
    op = FdmHestonOp(mesher, heston_process(r_ts(), q_ts()))
    op.set_time(_T1, _T2)
    return op, mesher


def test_size(ref: dict[str, Any]) -> None:
    op, _ = _build()
    assert op.size() == int(ref["heston_size"])


@pytest.mark.parametrize("tag", ["const", "lin0", "lin1", "mix", "idx"])
def test_operator_surface(ref: dict[str, Any], tag: str) -> None:
    op, mesher = _build()
    vectors = dict(all_vectors(mesher))
    assert_surface(ref, "heston", op, vectors[tag], tag, 2, _DT)


def test_to_matrix_decomp(ref: dict[str, Any]) -> None:
    op, _ = _build()
    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(ref["heston_decomp_size"])
    for i in range(3):
        assert_array(diag_of(decomp[i]), ref, f"heston_decomp{i}_diag")


def test_direction_out_of_range_fails() -> None:
    """# C++ parity: ``QL_FAIL("direction too large")``."""
    op, mesher = _build()
    v = vec_const(mesher)
    with pytest.raises(LibraryException, match="direction too large"):
        op.apply_direction(2, v)
    with pytest.raises(LibraryException, match="direction too large"):
        op.solve_splitting(2, v, _DT)


def test_mixing_factor(ref: dict[str, Any]) -> None:
    """``mixingFactor = 0.7`` scales sigma in both the variance and corr maps."""
    mesher = heston_mesher()
    op = FdmHestonOp(mesher, heston_process(r_ts(), q_ts()), None, None, 0.7)
    op.set_time(_T1, _T2)
    idx = vec_idx(mesher)
    assert_array(op.apply(idx), ref, "heston_mix07_idx_apply")
    assert_array(op.apply_mixed(idx), ref, "heston_mix07_idx_apply_mixed")
    assert_array(op.apply_direction(1, idx), ref, "heston_mix07_idx_apply_dir1")
    assert_array(op.solve_splitting(1, idx, _DT), ref, "heston_mix07_idx_solve_dir1")


def test_leverage_function(ref: dict[str, Any]) -> None:
    """Constant leverage 0.25 — the plain branch of ``getLeverageFctSlice``."""
    mesher = heston_mesher()
    lev = LocalConstantVol(reference_date=today(), volatility=0.25, day_counter=day_counter())
    op = FdmHestonOp(mesher, heston_process(r_ts(), q_ts()), None, lev, 1.0)
    op.set_time(_T1, _T2)
    idx = vec_idx(mesher)
    assert_array(op.apply(idx), ref, "heston_lev025_idx_apply")
    assert_array(op.apply_mixed(idx), ref, "heston_lev025_idx_apply_mixed")
    assert_array(op.apply_direction(0, idx), ref, "heston_lev025_idx_apply_dir0")
    assert_array(op.solve_splitting(0, idx, _DT), ref, "heston_lev025_idx_solve_dir0")

    part = FdmHestonEquityPart(mesher, r_ts(), q_ts(), None, lev)
    part.set_time(_T1, _T2)
    assert_array(part.get_L(), ref, "heston_lev025_L")


def test_quanto_adjustment_branch(ref: dict[str, Any]) -> None:
    """``FdmQuantoHelper`` subtracts the quanto drift from the equity map."""
    mesher = heston_mesher()
    f_ts = FlatForward.from_rate(today(), 0.025, day_counter())
    fx_vol = BlackConstantVol(
        reference_date=today(),
        calendar=NullCalendar(),
        volatility=0.12,
        day_counter=day_counter(),
    )
    helper = FdmQuantoHelper(r_ts(), f_ts, fx_vol, -0.35, 1.25)
    tight(
        helper.quanto_adjustment(0.3, _T1, _T2),
        float(ref["heston_quanto_adjustment_scalar"]),
    )

    part = FdmHestonEquityPart(mesher, r_ts(), q_ts(), helper)
    part.set_time(_T1, _T2)
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "heston_quanto_equity_part_apply_idx")
    assert_array(diag_of(part.get_map().to_matrix()), ref, "heston_quanto_equity_part_diag")

    op = FdmHestonOp(mesher, heston_process(r_ts(), q_ts()), helper)
    op.set_time(_T1, _T2)
    idx = vec_idx(mesher)
    assert_array(op.apply(idx), ref, "heston_quanto_idx_apply")
    assert_array(op.apply_direction(0, idx), ref, "heston_quanto_idx_apply_dir0")
    assert_array(op.solve_splitting(0, idx, _DT), ref, "heston_quanto_idx_solve_dir0")
    assert_array(op.preconditioner(idx, _DT), ref, "heston_quanto_idx_precond")

    # quanto AND leverage: the adjustment is asked for volatility * L.
    lev = LocalConstantVol(reference_date=today(), volatility=0.25, day_counter=day_counter())
    part_lev = FdmHestonEquityPart(mesher, r_ts(), q_ts(), helper, lev)
    part_lev.set_time(_T1, _T2)
    assert_array(part_lev.get_map().apply(vec_idx(mesher)), ref, "heston_quanto_lev_apply_idx")


def test_leverage_function_clamp(ref: dict[str, Any]) -> None:
    """Leverage 0.005 < 0.01 — pins the ``max(0.01, .)`` floor."""
    mesher = heston_mesher()
    lev = LocalConstantVol(reference_date=today(), volatility=0.005, day_counter=day_counter())
    part = FdmHestonEquityPart(mesher, r_ts(), q_ts(), None, lev)
    part.set_time(_T1, _T2)
    assert_array(part.get_L(), ref, "heston_levclamp_L")
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "heston_levclamp_apply_idx")
