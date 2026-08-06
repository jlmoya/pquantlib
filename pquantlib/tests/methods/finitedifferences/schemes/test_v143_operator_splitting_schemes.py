"""Cross-validate the v1.43 operator-splitting schemes against the C++ probe.

# C++ parity: ql/methods/finitedifferences/schemes/{boundaryconditionschemehelper,
# craigsneydscheme,douglasscheme,hundsdorferscheme,methodoflinesscheme,
# modifiedcraigsneydscheme,trbdf2scheme} @ v1.43 (6b57206e0).

Every expected value in this module comes from running C++ v1.43 —
``migration-harness/cpp/probes/v143_methods_schemes/probe.cpp`` — not from a
hand-derived formula and not from a Python reimplementation.

Two fixtures:

* the synthetic two-direction :class:`_probe_fixture.ProbeOp`, which is the
  only way to tell Craig-Sneyd, Hundsdorfer and modified Craig-Sneyd apart
  (in 1-D all three collapse onto Douglas) and the only way to reach the
  Krylov branch of ``TrBDF2Scheme``;
* the production 1-D ``FdmBlackScholesOp`` over an ``FdmBlackScholesMesher``,
  which pins that the schemes reproduce C++ when wired to a real operator.

Tolerance: TIGHT throughout. The synthetic fixture's arithmetic is transcribed
operation-for-operation from the probe, and the Black-Scholes fixture is a
short chain of tridiagonal solves, so nothing here has a reason to drift
beyond a few ULP. The two exceptions are called out at their assertions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import FdmMesherComposite
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import FdmBlackScholesOp
from pquantlib.methods.finitedifferences.schemes.boundary_condition_scheme_helper import (
    BoundaryConditionSchemeHelper,
)
from pquantlib.methods.finitedifferences.schemes.craig_sneyd_scheme import CraigSneydScheme
from pquantlib.methods.finitedifferences.schemes.crank_nicolson_scheme import CrankNicolsonScheme
from pquantlib.methods.finitedifferences.schemes.douglas_scheme import DouglasScheme
from pquantlib.methods.finitedifferences.schemes.hundsdorfer_scheme import HundsdorferScheme
from pquantlib.methods.finitedifferences.schemes.method_of_lines_scheme import MethodOfLinesScheme
from pquantlib.methods.finitedifferences.schemes.modified_craig_sneyd_scheme import (
    ModifiedCraigSneydScheme,
)
from pquantlib.methods.finitedifferences.schemes.tr_bdf2_scheme import (
    TrapezoidalScheme,
    TrBDF2Scheme,
    TrBDF2SolverType,
)
from pquantlib.processes.generalized_black_scholes_process import GeneralizedBlackScholesProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from tests.methods.finitedifferences.schemes._probe_fixture import (
    SYN_DT,
    SYN_T,
    ProbeOp,
    make_bc_set,
    probe_start,
)

REF: dict[str, Any] = reference_reader.load("v143/methods/schemes")


def _expected(key: str) -> list[float]:
    return [float(v) for v in REF[key]]


def _assert_tight(actual: Array, key: str) -> None:
    expected = _expected(key)
    assert len(actual) == len(expected), f"{key}: length {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        tight(float(a), e, reason=f"{key}[{i}]")


# --------------------------------------------------------------------------
# Block H -- BoundaryConditionSchemeHelper on its own
# --------------------------------------------------------------------------


def test_boundary_condition_scheme_helper_fans_out_to_every_hook() -> None:
    """All five helper methods, each observed through the probe fixture.

    # C++ parity: ``block_helper()`` in the probe.
    """
    op = ProbeOp()
    op.set_time(0.75, 1.0)

    helper = BoundaryConditionSchemeHelper(make_bc_set(True))
    a = probe_start()

    _assert_tight(op.apply(a), "helper_apply_unmasked")

    helper.set_time(0.5)
    helper.apply_before_applying(op)
    y1 = op.apply(a)
    _assert_tight(y1, "helper_apply_masked")

    helper.apply_after_applying(y1)
    _assert_tight(y1, "helper_after_applying")

    rhs = probe_start()
    helper.apply_before_solving(op, rhs)
    _assert_tight(rhs, "helper_before_solving_rhs")

    solved = op.solve_splitting(0, rhs, -0.25)
    _assert_tight(solved, "helper_solved")

    helper.apply_after_solving(solved)
    _assert_tight(solved, "helper_after_solving")


def test_boundary_condition_scheme_helper_empty_set_is_a_noop() -> None:
    """# C++ parity: the trailing ``empty`` block of ``block_helper()``."""
    empty = BoundaryConditionSchemeHelper()
    untouched = probe_start()
    empty.set_time(0.5)
    empty.apply_after_applying(untouched)
    empty.apply_after_solving(untouched)
    _assert_tight(untouched, "helper_empty_noop")


# --------------------------------------------------------------------------
# Block S -- the six schemes on the synthetic two-direction operator
# --------------------------------------------------------------------------


def _run_synthetic(scheme: TrapezoidalScheme, prefix: str) -> None:
    """Mirror of the probe's ``runSynthetic``: emit step 1 and step 3."""
    a = probe_start()
    scheme.set_step(SYN_DT)
    t = SYN_T
    a = scheme.step(a, t)
    _assert_tight(a, f"{prefix}_step1")
    for _ in range(2):
        t -= SYN_DT
        a = scheme.step(a, t)
    _assert_tight(a, f"{prefix}_step3")


def _suffix(with_bc: bool) -> str:
    return "_bc" if with_bc else ""


def test_douglas_scheme_synthetic() -> None:
    for with_bc in (False, True):
        op = ProbeOp()
        _run_synthetic(DouglasScheme(0.5, op, make_bc_set(with_bc)), f"syn_douglas{_suffix(with_bc)}")


def test_craig_sneyd_scheme_synthetic() -> None:
    for with_bc in (False, True):
        op = ProbeOp()
        _run_synthetic(
            CraigSneydScheme(0.5, 0.5, op, make_bc_set(with_bc)),
            f"syn_craigsneyd{_suffix(with_bc)}",
        )


def test_hundsdorfer_scheme_synthetic() -> None:
    theta = 0.5 + math.sqrt(3.0) / 6.0
    for with_bc in (False, True):
        op = ProbeOp()
        _run_synthetic(
            HundsdorferScheme(theta, 0.5, op, make_bc_set(with_bc)),
            f"syn_hundsdorfer{_suffix(with_bc)}",
        )


def test_modified_craig_sneyd_scheme_synthetic() -> None:
    for with_bc in (False, True):
        op = ProbeOp()
        _run_synthetic(
            ModifiedCraigSneydScheme(1.0 / 3.0, 1.0 / 3.0, op, make_bc_set(with_bc)),
            f"syn_modcraigsneyd{_suffix(with_bc)}",
        )


def test_method_of_lines_scheme_synthetic() -> None:
    for with_bc in (False, True):
        op = ProbeOp()
        _run_synthetic(
            MethodOfLinesScheme(1e-6, 0.1, op, make_bc_set(with_bc)),
            f"syn_mol{_suffix(with_bc)}",
        )


def test_operator_splitting_schemes_are_mutually_distinguishable() -> None:
    """The four ADI schemes must NOT agree on the synthetic 2-D fixture.

    Guards the fixture itself: on a 1-D operator all four are algebraically
    identical, so a test that only used ``FdmBlackScholesOp`` would pass even
    if three of the four ports were copies of the fourth.
    """
    keys = [
        "syn_douglas_step3",
        "syn_craigsneyd_step3",
        "syn_hundsdorfer_step3",
        "syn_modcraigsneyd_step3",
    ]
    vectors = [_expected(k) for k in keys]
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            assert vectors[i] != vectors[j], f"{keys[i]} and {keys[j]} are identical"


def _trbdf2_synthetic(with_bc: bool, solver_type: TrBDF2SolverType, prefix: str) -> None:
    op = ProbeOp()
    # C++ builds a *separate* bc_set for the trapezoidal half, so the two
    # ProbeBC instances carry independent ``v`` state; mirror that.
    trapezoidal = DouglasScheme(0.5, op, make_bc_set(with_bc))
    scheme = TrBDF2Scheme(
        2.0 - math.sqrt(2.0),
        op,
        trapezoidal,
        make_bc_set(with_bc),
        1e-8,
        solver_type,
    )
    # TIGHT for both Krylov branches: the iteration counts match C++ exactly
    # (6 BiCGstab sweeps, 16-17 Arnoldi steps), so the two implementations
    # stop on the same iterate and agree to ~6e-16 relative, four decades
    # inside the tier.
    _run_synthetic(scheme, prefix)
    assert scheme.number_of_iterations() == int(REF[f"{prefix}_iterations"])


def test_trbdf2_scheme_synthetic_bicgstab() -> None:
    """map.size() == 2 -> the BiCGstab branch of ``TrBDF2Scheme::step``."""
    for with_bc in (False, True):
        _trbdf2_synthetic(with_bc, TrBDF2SolverType.BiCGstab, f"syn_trbdf2{_suffix(with_bc)}")


def test_trbdf2_scheme_synthetic_gmres() -> None:
    """Same fixture through the GMRES branch."""
    for with_bc in (False, True):
        _trbdf2_synthetic(
            with_bc, TrBDF2SolverType.GMRES, f"syn_trbdf2_gmres{_suffix(with_bc)}"
        )


# --------------------------------------------------------------------------
# Block B -- the same schemes on the production 1-D FdmBlackScholesOp
# --------------------------------------------------------------------------

BSM_T = 1.0
BSM_DT = 0.05


def _build_bsm() -> tuple[GeneralizedBlackScholesProcess, FdmBlackScholesMesher, FdmMesherComposite]:
    """Same fixture as the probe's ``makeBsmProcess`` / ``block_bsm``."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol)
    bs_mesher = FdmBlackScholesMesher(21, process, 1.0, 100.0)
    return process, bs_mesher, FdmMesherComposite(bs_mesher)


def _bsm_start(bs_mesher: FdmBlackScholesMesher) -> Array:
    return np.array(
        [max(math.exp(float(x)) - 100.0, 0.0) for x in bs_mesher.locations()], dtype=np.float64
    )


def _run_bsm(scheme: TrapezoidalScheme, prefix: str, start: Array) -> None:
    """Mirror of the probe's ``runBsm``: emit step 1 and step 4."""
    a = start
    scheme.set_step(BSM_DT)
    t = BSM_T
    a = scheme.step(a, t)
    _assert_tight(a, f"{prefix}_step1")
    for _ in range(3):
        t -= BSM_DT
        a = scheme.step(a, t)
    _assert_tight(a, f"{prefix}_step4")


def test_bsm_fixture_matches_cpp() -> None:
    """Pin the mesh, the start vector and the raw operator before the schemes.

    Not decoration: it separates "the scheme is wrong" from "the mesher or the
    operator is wrong" if Block B ever goes red.
    """
    _, bs_mesher, mesher = _build_bsm()
    _assert_tight(np.array(bs_mesher.locations(), dtype=np.float64), "bsm_mesher_locations")
    start = _bsm_start(bs_mesher)
    _assert_tight(start, "bsm_start")

    process, _, _ = _build_bsm()
    op = FdmBlackScholesOp(mesher, process, 100.0)
    op.set_time(0.95, 1.0)
    _assert_tight(op.apply(start), "bsm_op_apply_start")
    _assert_tight(op.solve_splitting(0, start, -0.025), "bsm_op_solve_splitting")


def test_douglas_scheme_bsm() -> None:
    process, bs_mesher, mesher = _build_bsm()
    op = FdmBlackScholesOp(mesher, process, 100.0)
    _run_bsm(DouglasScheme(0.5, op), "bsm_douglas", _bsm_start(bs_mesher))


def test_craig_sneyd_scheme_bsm() -> None:
    process, bs_mesher, mesher = _build_bsm()
    op = FdmBlackScholesOp(mesher, process, 100.0)
    _run_bsm(CraigSneydScheme(0.5, 0.5, op), "bsm_craigsneyd", _bsm_start(bs_mesher))


def test_hundsdorfer_scheme_bsm() -> None:
    process, bs_mesher, mesher = _build_bsm()
    op = FdmBlackScholesOp(mesher, process, 100.0)
    _run_bsm(
        HundsdorferScheme(0.5 + math.sqrt(3.0) / 6.0, 0.5, op),
        "bsm_hundsdorfer",
        _bsm_start(bs_mesher),
    )


def test_modified_craig_sneyd_scheme_bsm() -> None:
    process, bs_mesher, mesher = _build_bsm()
    op = FdmBlackScholesOp(mesher, process, 100.0)
    _run_bsm(
        ModifiedCraigSneydScheme(1.0 / 3.0, 1.0 / 3.0, op),
        "bsm_modcraigsneyd",
        _bsm_start(bs_mesher),
    )


def test_method_of_lines_scheme_bsm() -> None:
    process, bs_mesher, mesher = _build_bsm()
    op = FdmBlackScholesOp(mesher, process, 100.0)
    _run_bsm(MethodOfLinesScheme(1e-6, 0.1, op), "bsm_mol", _bsm_start(bs_mesher))


def test_trbdf2_scheme_bsm_single_direction_branch() -> None:
    """map.size() == 1 -> the direct ``solve_splitting`` branch, no Krylov."""
    process, bs_mesher, mesher = _build_bsm()
    op = FdmBlackScholesOp(mesher, process, 100.0)
    trapezoidal = CrankNicolsonScheme(theta=0.5, op=op)
    scheme = TrBDF2Scheme(2.0 - math.sqrt(2.0), op, trapezoidal)
    _run_bsm(scheme, "bsm_trbdf2", _bsm_start(bs_mesher))
    assert scheme.number_of_iterations() == int(REF["bsm_trbdf2_iterations"])


def test_douglas_and_craig_sneyd_coincide_in_one_dimension() -> None:
    """In 1-D ``apply_mixed`` is zero, so Craig-Sneyd degenerates to Douglas.

    Pinned from the probe (the two C++ blocks emit the same numbers), which is
    what makes the synthetic 2-D fixture above necessary rather than optional.
    """
    assert _expected("bsm_douglas_step4") == _expected("bsm_craigsneyd_step4")
