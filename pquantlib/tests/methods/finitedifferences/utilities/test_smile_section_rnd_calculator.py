"""Cross-validate SmileSectionRNDCalculator against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/smilesection/rnd``.

Each probe case carries its smile, grid parameters and abscissa, so the sweeps
below reconstruct the case rather than restating it. Four smiles are covered: a
flat section and three SVI shapes, one of them steeply skewed and off-centre,
where a wrong finite-difference gap or a mishandled wing shows up first.

Two properties the class is easy to get wrong are asserted directly rather than
inferred from agreement: ``pdf``/``cdf`` must be independent of ``n_strikes`` /
``n_std`` because they never build the grid, and ``invcdf`` beyond the surviving
CDF range must clamp onto the retained grid endpoints rather than extrapolate.

Tolerances
----------
``pdf`` and ``cdf`` are finite differences of Black prices, so how closely two
implementations can agree is bounded by cancellation, not by how faithful the
port is. Two cancellations stack:

1. Each Black price is itself a difference of terms of order ``max(F, K)``, so
   its absolute error is about ``eps * max(F, K)`` — for a 2.8 option struck off
   a 100 forward that is ~35x worse than ``eps * |C|``.
2. ``cdf`` divides a first difference by ``gap = 1e-5``; ``density`` divides a
   *second* difference by ``gap**2 = 1e-8``, and ``pdf`` then scales by ``S``.

which gives the floors implemented in :func:`_cdf_bound` and :func:`_pdf_bound`.
At ``S = 116``, ``F = 100`` the pdf floor is ~1.2e-3 on a pdf of ~1.39, and the
disagreement actually observed against C++ is ~1e-6 — both sides evaluate the
same expression and merely lose the same digits differently. Neither is a
blanket tier: the bound is recomputed from each case's own forward and strike,
and a real porting error (wrong gap, wrong sign, wrong smile) moves these by
percent, so nothing is given up.

``invcdf`` propagates the ``cdf`` floor through the quantile slope — see
:func:`_invcdf_bound`. Everything that is not a finite difference (grid
endpoints, ATM levels, exercise times) is asserted TIGHT.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.volatility.svi_smile_section import SviSmileSection
from pquantlib.methods.finitedifferences.utilities.smile_section_rnd_calculator import (
    SmileSectionRNDCalculator,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.volatility.atm_smile_section import AtmSmileSection
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

TODAY = Date.from_ymd(1, Month.March, 2025)
MATURITY = Date.from_ymd(1, Month.March, 2026)

# TIGHT tier, for everything that is not a finite difference.
_REL_TOL = 1.0e-12
_ABS_TOL = 1.0e-14

# The two SmileSection finite-difference gaps, which C++ defaults differently.
_CDF_GAP = 1.0e-5
_PDF_GAP = 1.0e-4
_EPS = sys.float_info.epsilon


def _cdf_bound(forward: float, s: float) -> float:
    """Absolute floor on ``cdf`` at strike ``s`` for a smile with ``forward``."""
    return 2.0 * _EPS * max(forward, s) / _CDF_GAP


def _pdf_bound(forward: float, s: float) -> float:
    """Absolute floor on ``pdf`` at strike ``s`` for a smile with ``forward``."""
    return 4.0 * _EPS * max(forward, s) * s / (_PDF_GAP * _PDF_GAP)


def _invcdf_bound(
    rnd: SmileSectionRNDCalculator, forward: float, p: float, log_strike: float
) -> float:
    """Absolute floor on ``invcdf``: the ``cdf`` floor times the quantile slope.

    The quantile spline runs through CDF values that each carry
    :func:`_cdf_bound`, so an error there becomes a log-strike error multiplied
    by ``d(ln K)/dp``. That derivative is estimated two ways, and the bound has
    to admit both because each is only valid in one regime:

    * ``1 / pdf(ln K)`` — the analytic slope of the *true* quantile function.
      Correct wherever the grid resolves the density, and the only estimate
      available where the spline sits on its clamped plateau and has slope 0.
    * a central difference of ``invcdf`` itself — the slope of the interpolant
      actually being evaluated. On a coarse grid (``n_strikes`` 4 or 8, spanning
      +-5 ATM standard deviations) the spline's local slope is materially
      steeper than the pointwise density, because the perturbed knots are far
      away and out in the tail; the analytic estimate understates the
      sensitivity by ~2.4x there.

    Both are lower bounds on the true sensitivity, so the larger is taken. The
    bound therefore widens by itself in the tails and on coarse grids — where
    the quantile is genuinely ill-conditioned — and stays tight in the middle.
    """
    density = abs(rnd.pdf(log_strike))
    analytic = (
        max(_ABS_TOL, _REL_TOL * abs(log_strike))
        if density < 1.0e-12
        else _cdf_bound(forward, math.exp(log_strike)) / density
    )
    h = 1.0e-7
    lo = max(p - h, 5.0e-324)
    hi = min(p + h, 1.0 - 1.0e-16)
    spline_slope = abs(rnd.invcdf(hi) - rnd.invcdf(lo)) / (hi - lo)
    return max(analytic, _cdf_bound(forward, math.exp(log_strike)) * spline_slope)


def _at_tier(actual: float, expected: float, *, what: str) -> None:
    tolerance.tight(actual, expected, reason=what)


def _within(actual: float, expected: float, bound: float, *, what: str) -> None:
    tolerance.custom(
        actual,
        expected,
        abs_tol=max(_ABS_TOL, bound),
        rel_tol=0.0,
        reason=f"{what}: finite-difference cancellation floor",
    )


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = TODAY


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/smilesection/rnd")


# --- smiles: mirror the probe's definitions exactly --------------------------


def _flat_smile(atm: float) -> SmileSection:
    return FlatSmileSection(
        volatility=0.20,
        exercise_date=MATURITY,
        day_counter=Actual365Fixed(),
        reference_date=TODAY,
        atm_level=atm,
    )


def _flat_smile_without_atm() -> SmileSection:
    return FlatSmileSection(
        volatility=0.20,
        exercise_date=MATURITY,
        day_counter=Actual365Fixed(),
        reference_date=TODAY,
    )


def _svi_smile(
    forward: float, params: tuple[float, float, float, float, float]
) -> SmileSection:
    t = Actual365Fixed().year_fraction(TODAY, MATURITY)
    return SviSmileSection(forward=forward, svi_params=params, exercise_time=t)


def _smile_for(name: str) -> SmileSection:
    match name:
        case "flat":
            return _flat_smile(100.0)
        case "svi1":
            return _svi_smile(100.0, (0.04, 0.10, 0.30, -0.40, 0.0))
        case "svi2":
            return _svi_smile(96.0, (0.02, 0.08, 0.25, -0.30, 0.0))
        case "svi_steep":
            return _svi_smile(100.0, (0.03, 0.25, 0.15, -0.75, -0.10))
        case _:
            raise AssertionError(f"unknown smile: {name}")


def _calculator_for(inputs: dict[str, Any]) -> SmileSectionRNDCalculator:
    return SmileSectionRNDCalculator(
        _smile_for(str(inputs["smile"])),
        int(inputs["n_strikes"]),
        float(inputs["n_std"]),
    )


# --- pdf / cdf ---------------------------------------------------------------


def test_pdf_and_cdf(cpp: dict[str, Any]) -> None:
    """Density and cumulative probability across the strike ladder, per smile."""
    checked = 0
    for name, case in cpp.items():
        if "_pdf_cdf_" not in name:
            continue
        inputs, expected = case["inputs"], case["expected"]
        rnd = _calculator_for(inputs)
        x = float(inputs["x"])
        t = float(inputs["t"])
        fwd = float(inputs["forward"])
        s = math.exp(x)

        _within(rnd.pdf(x, t), float(expected["pdf"]), _pdf_bound(fwd, s), what=f"{name}: pdf")
        _within(rnd.cdf(x, t), float(expected["cdf"]), _cdf_bound(fwd, s), what=f"{name}: cdf")
        # The smile itself is closed-form, so it pins the setup at TIGHT — a
        # pdf/cdf failure can then be read as "the finite difference" rather
        # than "the wrong smile".
        _at_tier(
            _smile_for(str(inputs["smile"])).volatility(s),
            float(expected["smile_volatility"]),
            what=f"{name}: smile volatility",
        )
        checked += 1
    assert checked >= 40, f"expected pdf/cdf cases in the reference, got {checked}"


# --- invcdf ------------------------------------------------------------------


def test_invcdf(cpp: dict[str, Any]) -> None:
    """The quantile function across the probability ladder.

    Includes the extreme tails where the grid clamp bites and the coarse grids
    (``n_strikes`` 4 and 8) where the spline's own algorithm dominates.
    """
    checked = 0
    for name, case in cpp.items():
        if "_invcdf_" not in name or "rejects" in name:
            continue
        inputs, expected = case["inputs"], case["expected"]
        rnd = _calculator_for(inputs)
        q = rnd.invcdf(float(inputs["p"]), float(inputs["t"]))
        fwd = float(inputs["forward"])
        bound = _invcdf_bound(rnd, fwd, float(inputs["p"]), q)

        _within(q, float(expected["invcdf"]), bound, what=f"{name}: invcdf")
        # The strike is exp(invcdf), so its floor scales with the strike.
        _within(
            math.exp(q),
            float(expected["strike"]),
            math.exp(q) * bound,
            what=f"{name}: strike",
        )
        checked += 1
    assert checked >= 40, f"expected invcdf cases in the reference, got {checked}"


# --- grid --------------------------------------------------------------------


def test_grid_endpoints(cpp: dict[str, Any]) -> None:
    """The surviving grid endpoints, reached through the ``invcdf`` clamp.

    The grid is private and is made monotone and then deduplicated, so
    ``invcdf`` at 1e-12 and 1 - 1e-12 returning exactly the first and last
    *retained* strike is the sharpest available pin on how it was built.
    """
    checked = 0
    for name, case in cpp.items():
        if not name.endswith("_grid_endpoints"):
            continue
        inputs, expected = case["inputs"], case["expected"]
        smile = _smile_for(str(inputs["smile"]))
        rnd = _calculator_for(inputs)

        _at_tier(smile.atm_level(), float(expected["atm_level"]), what=f"{name}: atm")
        _at_tier(
            smile.exercise_time(),
            float(expected["exercise_time"]),
            what=f"{name}: exercise time",
        )
        _at_tier(
            smile.volatility(smile.atm_level()),
            float(expected["sigma_atm"]),
            what=f"{name}: sigma atm",
        )

        tiny = 1.0e-12
        _at_tier(
            rnd.invcdf(tiny), float(expected["invcdf_at_p_min"]), what=f"{name}: p min"
        )
        _at_tier(
            rnd.invcdf(1.0 - tiny),
            float(expected["invcdf_at_p_max"]),
            what=f"{name}: p max",
        )
        _at_tier(
            math.exp(rnd.invcdf(tiny)),
            float(expected["strike_at_p_min"]),
            what=f"{name}: strike at p min",
        )
        _at_tier(
            math.exp(rnd.invcdf(1.0 - tiny)),
            float(expected["strike_at_p_max"]),
            what=f"{name}: strike at p max",
        )
        checked += 1
    assert checked >= 4, f"expected grid-endpoint cases in the reference, got {checked}"


def test_pdf_and_cdf_are_grid_independent(cpp: dict[str, Any]) -> None:
    """``pdf``/``cdf`` never build the grid, so its parameters cannot matter.

    A port that eagerly initialised and then read off the spline would pass
    every other case here and fail this one.
    """
    checked = 0
    for name, case in cpp.items():
        if not name.endswith("_grid_independent_pdf_cdf"):
            continue
        inputs, expected = case["inputs"], case["expected"]
        smile_name = str(inputs["smile"])
        grid_a, grid_b = inputs["grid_a"], inputs["grid_b"]
        x = float(inputs["x"])
        t = float(inputs["t"])
        a = SmileSectionRNDCalculator(
            _smile_for(smile_name), int(grid_a["n_strikes"]), float(grid_a["n_std"])
        )
        b = SmileSectionRNDCalculator(
            _smile_for(smile_name), int(grid_b["n_strikes"]), float(grid_b["n_std"])
        )
        fwd = float(inputs["forward"])
        s = math.exp(x)
        pdf_bound = _pdf_bound(fwd, s)
        cdf_bound = _cdf_bound(fwd, s)

        _within(a.pdf(x, t), float(expected["pdf_grid_a"]), pdf_bound, what=f"{name}: pdf a")
        _within(b.pdf(x, t), float(expected["pdf_grid_b"]), pdf_bound, what=f"{name}: pdf b")
        _within(a.cdf(x, t), float(expected["cdf_grid_a"]), cdf_bound, what=f"{name}: cdf a")
        _within(b.cdf(x, t), float(expected["cdf_grid_b"]), cdf_bound, what=f"{name}: cdf b")

        # The claim of this case: the grid parameters must not touch pdf/cdf
        # at all. Asserted EXACT between two wildly different grids — this is
        # a property of the port, independent of agreement with C++.
        tolerance.exact(a.pdf(x, t), b.pdf(x, t))
        tolerance.exact(a.cdf(x, t), b.cdf(x, t))
        checked += 1
    assert checked >= 4, f"expected grid-independence cases, got {checked}"


# --- overloads ---------------------------------------------------------------


def test_overloads_agree(cpp: dict[str, Any]) -> None:
    """The time-defaulting overloads equal the explicit ones at the smile's own time."""
    checked = 0
    for name, case in cpp.items():
        if not name.endswith("_overloads_agree"):
            continue
        inputs, expected = case["inputs"], case["expected"]
        rnd = _calculator_for(inputs)
        x = float(inputs["x"])
        p = float(inputs["p"])
        t = float(inputs["t"])
        fwd = float(inputs["forward"])
        s = math.exp(x)
        pdf_bound = _pdf_bound(fwd, s)
        cdf_bound = _cdf_bound(fwd, s)

        _within(rnd.pdf(x), float(expected["pdf_1arg"]), pdf_bound, what=f"{name}: pdf 1-arg")
        _within(rnd.pdf(x, t), float(expected["pdf_2arg"]), pdf_bound, what=f"{name}: pdf 2-arg")
        _within(rnd.cdf(x), float(expected["cdf_1arg"]), cdf_bound, what=f"{name}: cdf 1-arg")
        _within(rnd.cdf(x, t), float(expected["cdf_2arg"]), cdf_bound, what=f"{name}: cdf 2-arg")
        inv_bound = _invcdf_bound(rnd, fwd, p, rnd.invcdf(p))
        _within(
            rnd.invcdf(p), float(expected["invcdf_1arg"]), inv_bound, what=f"{name}: inv 1-arg"
        )
        _within(
            rnd.invcdf(p, t), float(expected["invcdf_2arg"]), inv_bound, what=f"{name}: inv 2-arg"
        )

        # The two forms must agree bit for bit — that is the actual claim of
        # this case, and it is unaffected by how well either matches C++.
        tolerance.exact(rnd.pdf(x), rnd.pdf(x, t))
        tolerance.exact(rnd.cdf(x), rnd.cdf(x, t))
        tolerance.exact(rnd.invcdf(p), rnd.invcdf(p, t))
        checked += 1
    assert checked == 4, f"expected one overload case per smile, got {checked}"


# --- AtmSmileSection wrapping ------------------------------------------------


def test_atm_smile_section_wrapping_matches_direct(cpp: dict[str, Any]) -> None:
    """Wrapping an ATM-less smile must reproduce the directly-built one exactly.

    That is the documented way to satisfy the calculator's ATM requirement, so
    a discrepancy would make the documentation wrong rather than merely
    imprecise.
    """
    case = cpp["flat_via_atmsmilesection_matches_direct"]
    inputs, expected = case["inputs"], case["expected"]
    n_strikes = int(inputs["n_strikes"])
    n_std = float(inputs["n_std"])
    x = float(inputs["x"])
    p = float(inputs["p"])
    fwd = float(inputs["forward"])

    wrapper = AtmSmileSection(base=_flat_smile_without_atm(), atm=fwd)
    wrapped = SmileSectionRNDCalculator(wrapper, n_strikes, n_std)
    direct = SmileSectionRNDCalculator(_flat_smile(fwd), n_strikes, n_std)

    # The wrapper must expose the base's exercise time and the supplied ATM.
    _at_tier(
        wrapper.exercise_time(),
        float(expected["wrapped_exercise_time"]),
        what="wrapped exercise time",
    )
    _at_tier(
        wrapper.atm_level(), float(expected["wrapped_atm_level"]), what="wrapped atm level"
    )

    pdf_bound = _pdf_bound(fwd, math.exp(x))
    cdf_bound = _cdf_bound(fwd, math.exp(x))
    _within(wrapped.pdf(x), float(expected["pdf_wrapped"]), pdf_bound, what="pdf wrapped")
    _within(direct.pdf(x), float(expected["pdf_direct"]), pdf_bound, what="pdf direct")
    _within(wrapped.cdf(x), float(expected["cdf_wrapped"]), cdf_bound, what="cdf wrapped")
    _within(direct.cdf(x), float(expected["cdf_direct"]), cdf_bound, what="cdf direct")
    inv_bound = _invcdf_bound(direct, fwd, p, direct.invcdf(p))
    _within(wrapped.invcdf(p), float(expected["invcdf_wrapped"]), inv_bound, what="inv wrapped")
    _within(direct.invcdf(p), float(expected["invcdf_direct"]), inv_bound, what="inv direct")

    # Whatever the agreement with C++, wrapping must reproduce the direct
    # construction exactly — that is the claim this case actually makes.
    tolerance.exact(wrapped.pdf(x), direct.pdf(x))
    tolerance.exact(wrapped.cdf(x), direct.cdf(x))
    tolerance.exact(wrapped.invcdf(p), direct.invcdf(p))


# --- guards ------------------------------------------------------------------


def _expected_throws(cpp: dict[str, Any], case_name: str) -> bool:
    return bool(cpp[case_name]["expected"]["throws"])


def _throws(f: Callable[[], object]) -> bool:
    try:
        f()
    except LibraryException:
        return True
    return False


@pytest.mark.parametrize(
    ("case_name", "action"),
    [
        ("ctor_rejects_n_strikes_3", lambda: SmileSectionRNDCalculator(_flat_smile(100.0), 3, 5.0)),
        ("ctor_accepts_n_strikes_4", lambda: SmileSectionRNDCalculator(_flat_smile(100.0), 4, 5.0)),
        ("ctor_rejects_n_std_zero", lambda: SmileSectionRNDCalculator(_flat_smile(100.0), 200, 0.0)),
        (
            "ctor_rejects_n_std_negative",
            lambda: SmileSectionRNDCalculator(_flat_smile(100.0), 200, -1.0),
        ),
        ("invcdf_rejects_p_zero", lambda: SmileSectionRNDCalculator(_flat_smile(100.0)).invcdf(0.0)),
        ("invcdf_rejects_p_one", lambda: SmileSectionRNDCalculator(_flat_smile(100.0)).invcdf(1.0)),
        (
            "invcdf_rejects_p_negative",
            lambda: SmileSectionRNDCalculator(_flat_smile(100.0)).invcdf(-0.25),
        ),
        (
            "pdf_rejects_time_mismatch",
            lambda: SmileSectionRNDCalculator(_flat_smile(100.0)).pdf(math.log(100.0), 0.5),
        ),
        (
            "cdf_rejects_time_mismatch",
            lambda: SmileSectionRNDCalculator(_flat_smile(100.0)).cdf(math.log(100.0), 0.5),
        ),
        (
            "invcdf_rejects_time_mismatch",
            lambda: SmileSectionRNDCalculator(_flat_smile(100.0)).invcdf(0.5, 2.0),
        ),
        (
            "invcdf_rejects_missing_atm_level",
            lambda: SmileSectionRNDCalculator(_flat_smile_without_atm()).invcdf(0.5),
        ),
        (
            "cdf_rejects_missing_atm_level",
            lambda: SmileSectionRNDCalculator(_flat_smile_without_atm()).cdf(math.log(100.0)),
        ),
        (
            "pdf_rejects_missing_atm_level",
            lambda: SmileSectionRNDCalculator(_flat_smile_without_atm()).pdf(math.log(100.0)),
        ),
    ],
)
def test_guards(cpp: dict[str, Any], case_name: str, action: Callable[[], object]) -> None:
    """Each guard behaves exactly as the probe recorded C++ behaving.

    Tying the assertion to the reference rather than to a hard-coded ``True``
    means the test tracks C++ if upstream changes its mind — and it is what
    makes ``ctor_accepts_n_strikes_4`` (which must *not* throw) part of the
    same sweep.
    """
    assert _throws(action) is _expected_throws(cpp, case_name)


def test_null_smile_guard_has_no_python_analogue(cpp: dict[str, Any]) -> None:
    """C++ rejects a null SmileSection; the typed parameter here cannot be one.

    The probe records the case, so it is acknowledged rather than silently
    dropped: an ``ext::shared_ptr<SmileSection>`` can be empty and the class
    dereferences it, whereas a ``SmileSection`` parameter statically excludes
    ``None``. This port therefore has no runtime check to exercise.
    """
    assert _expected_throws(cpp, "ctor_rejects_null_smile")


def test_guard_messages(cpp: dict[str, Any]) -> None:
    """The probe also pins the message substrings C++ produces."""
    with pytest.raises(LibraryException, match="at least 4 strikes required"):
        SmileSectionRNDCalculator(_flat_smile(100.0), 3, 5.0)
    with pytest.raises(LibraryException, match="nStd must be positive"):
        SmileSectionRNDCalculator(_flat_smile(100.0), 200, 0.0)
    with pytest.raises(LibraryException, match=r"p must be in \(0, 1\)"):
        SmileSectionRNDCalculator(_flat_smile(100.0)).invcdf(0.0)
    with pytest.raises(LibraryException, match="does not match smile exercise time"):
        SmileSectionRNDCalculator(_flat_smile(100.0)).cdf(math.log(100.0), 0.5)
    with pytest.raises(LibraryException, match="wrap with AtmSmileSection"):
        SmileSectionRNDCalculator(_flat_smile_without_atm()).invcdf(0.5)
    with pytest.raises(LibraryException, match="smile section must provide atm level"):
        SmileSectionRNDCalculator(_flat_smile_without_atm()).cdf(math.log(100.0))
    assert cpp["invcdf_rejects_missing_atm_level"]["expected"][
        "message_contains_substring"
    ]


def test_invcdf_atm_check_precedes_p_check(cpp: dict[str, Any]) -> None:
    """``invcdf`` builds the grid before range-checking ``p``.

    So on an ATM-less smile an out-of-range probability is reported as the
    missing ATM level, not as a bad probability. The ordering is the only
    reason this is observable at all.
    """
    expected = cpp["invcdf_atm_check_precedes_p_check"]["expected"]
    assert expected["throws"] is True
    assert expected["message_contains_atm_substring"] is True
    assert expected["message_contains_p_range_substring"] is False

    with pytest.raises(LibraryException) as excinfo:
        SmileSectionRNDCalculator(_flat_smile_without_atm()).invcdf(-1.0)
    message = str(excinfo.value)
    assert "wrap with AtmSmileSection" in message
    assert "p must be in (0, 1)" not in message
