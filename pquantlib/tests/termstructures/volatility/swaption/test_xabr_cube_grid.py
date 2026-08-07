"""Cross-validation of ``Cube`` and ``PrivateObserver`` against C++ v1.43.

Reference: ``migration-harness/references/v143/ts/xabrerror.json``, produced by
``migration-harness/cpp/probes/v143_ts_xabrerror/probe.cpp``.

Both C++ classes are PRIVATE nested classes of
``XabrSwaptionVolatilityCube<Model>``, so the probe reaches them through the
public surface of ``SabrSwaptionVolatilityCube``. The whole reference is
optimiser-free: every cube in it is built with all four SABR parameters
FIXED, which makes ``XABRInterpolationImpl::calculate`` short-circuit at
``xabrinterpolation.hpp:162-169`` ("there is nothing to optimize") and leave
the parameters at the guess — and the guess is
``parametersGuess_(optionTimes[j], swapLengths[k])``, a raw
``Cube::operator()`` evaluation (sabrswaptionvolatilitycube.hpp:448-449).
So the numbers pinned here are Cube arithmetic, not fit output, and every
tier below is TIGHT or EXACT.

Evaluation date: the probe sets ``Settings::instance().evaluationDate() =
Date(15, January, 2024)`` (probe.cpp, ``main``); the ``_eval_date`` fixture
pins the same value and restores the previous one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.matrix import Matrix
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.sabr_swaption_volatility_cube import (
    SabrSwaptionVolatilityCube,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_matrix import (
    SwaptionVolatilityMatrix,
)
from pquantlib.termstructures.volatility.swaption.xabr_swaption_volatility_cube import (
    Cube,
    PrivateObserver,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("v143/ts/xabrerror")
_SETUP = _REF["setup"]

_Y = TimeUnit.Years
_M = TimeUnit.Months

_UNIT_LETTER = {
    TimeUnit.Days: "D",
    TimeUnit.Weeks: "W",
    TimeUnit.Months: "M",
    TimeUnit.Years: "Y",
}


def _period_str(p: Period) -> str:
    """C++ ``operator<<(ostream&, const Period&)`` short form ("2Y", "6M")."""
    return f"{p.length}{_UNIT_LETTER[p.units]}"


# --- the probe's fixture, rebuilt in Python ---------------------------------

_EVAL_DATE = Date(int(_SETUP["eval_date_serial"]))
_ATM_OPTION_TENORS = [Period(6, _M), Period(1, _Y), Period(2, _Y), Period(3, _Y)]
_ATM_SWAP_TENORS = [Period(1, _Y), Period(2, _Y), Period(3, _Y), Period(5, _Y)]
_CUBE_OPTION_TENORS = [Period(1, _Y), Period(3, _Y)]
_SWAP_TENORS = [Period(2, _Y), Period(5, _Y)]
_STRIKE_SPREADS: list[float] = list(_SETUP["strike_spreads"])

# (option-date serial, swap-tenor string) -> C++ ``atmStrike``. Read straight
# off the probe so that ``SwaptionVolatilityCube::atmStrike``'s swap rebuild
# (swaptionvolcube.cpp:89-144) never has to be reproduced here: this suite is
# about the Cube, not about the ATM-strike machinery.
_ATM_STRIKES: dict[tuple[int, str], float] = {
    (int(e["date_serial"]), str(e["swap_tenor"])): float(e["forward"])
    for e in _SETUP["atm_strike_table"]
}


class _TableStrikeCube(SabrSwaptionVolatilityCube):
    """SABR cube whose ATM strikes come from the probe's own table."""

    def atm_strike(self, option_date: Date, swap_tenor: Period) -> float:
        return _ATM_STRIKES[(option_date.serial, _period_str(swap_tenor))]


class _StubSwapIndex:
    """Tenor-only swap-index stub — ``fixing`` is never reached.

    ``_TableStrikeCube`` overrides ``atm_strike``, which is the sole consumer
    of the index; the base cube still needs ``tenor()`` for its short-vs-long
    sanity check.
    """

    def __init__(self, tenor: Period) -> None:
        self._tenor = tenor

    def tenor(self) -> Period:
        return self._tenor

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        raise AssertionError("atm_strike is overridden; fixing must not be called")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the probe's evaluation date; restore whatever was there before.

    # Probe: v143_ts_xabrerror/probe.cpp ``main`` —
    # ``Settings::instance().evaluationDate() = Date(15, January, 2024)``.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _EVAL_DATE
    yield
    settings.evaluation_date = previous


def _atm_matrix() -> SwaptionVolatilityMatrix:
    return SwaptionVolatilityMatrix(
        business_day_convention=BusinessDayConvention.Following,
        option_tenors=_ATM_OPTION_TENORS,
        swap_tenors=_ATM_SWAP_TENORS,
        volatilities=np.asarray(_SETUP["atm_vols"], dtype=np.float64),
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_EVAL_DATE,
    )


def _vol_spread_quotes() -> list[list[SimpleQuote]]:
    return [[SimpleQuote(float(v)) for v in row] for row in _SETUP["vol_spreads"]]


def _guess_quotes() -> list[list[list[SimpleQuote]]]:
    """The probe's ``parametersGuess``, reshaped from flat ``[j*nSwap+k]`` to ``[j][k]``."""
    n_swap = len(_SWAP_TENORS)
    flat = _SETUP["parameters_guess"]
    return [
        [[SimpleQuote(float(v)) for v in flat[j * n_swap + k]] for k in range(n_swap)]
        for j in range(len(_CUBE_OPTION_TENORS))
    ]


def _make_cube(
    *,
    backward_flat: bool,
    guess: list[list[list[SimpleQuote]]] | None = None,
) -> _TableStrikeCube:
    return _TableStrikeCube(
        atm_vol_structure=_atm_matrix(),
        option_tenors=_CUBE_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spread_quotes(),
        swap_index_base=_StubSwapIndex(Period(5, _Y)),
        short_swap_index_base=_StubSwapIndex(Period(1, _Y)),
        sabr_initial_guess=_guess_quotes() if guess is None else guess,
        is_parameter_fixed=(True, True, True, True),
        backward_flat=backward_flat,
    )


def _assert_matrix_tight(actual: Matrix, expected: list[list[float]], what: str) -> None:
    exp = np.asarray(expected, dtype=np.float64)
    assert actual.shape == exp.shape, f"{what}: shape {actual.shape} != {exp.shape}"
    for i in range(exp.shape[0]):
        for j in range(exp.shape[1]):
            tolerance.tight(
                float(actual[i, j]), float(exp[i, j]), reason=f"{what}[{i}][{j}]"
            )


# =====================================================================
# 1. Cube::operator() — the interpolation composition, optimiser-free
# =====================================================================

_KERNEL = _REF["cube_kernel"]


def _kernel_cube(backward_flat: bool) -> Cube:
    option_times = [float(t) for t in _KERNEL["option_times"]]
    swap_lengths = [float(s) for s in _KERNEL["swap_lengths"]]
    # Dates/tenors are placeholders here: Cube::operator() reads only the
    # times and lengths (hpp:1204-1211).
    dates = [Date.from_ymd(1, Month.January, 2024 + i) for i in range(len(option_times))]
    tenors = [Period(i + 1, _Y) for i in range(len(swap_lengths))]
    cube = Cube(dates, tenors, option_times, swap_lengths, 1, True, backward_flat)
    cube.set_layer(0, np.asarray(_KERNEL["layer"], dtype=np.float64))
    cube.update_interpolators()
    return cube


@pytest.mark.parametrize("backward_flat", [False, True])
def test_cube_call_matches_cpp_kernel(backward_flat: bool) -> None:
    """``Cube::operator()`` at pillars, strictly between, and outside both ends.

    TIGHT. Closed-form 2-D interpolation; no optimiser anywhere in the path.
    The 14 query points cover: three grid pillars; three strictly-interior
    points; two points on one axis' pillar only; and out-of-range points past
    the low and high end of each axis and of both at once.
    """
    key = "backwardflat" if backward_flat else "bilinear"
    cube = _kernel_cube(backward_flat)
    for query, expected in zip(_KERNEL["queries"], _KERNEL[key], strict=True):
        got = cube(float(query[0]), float(query[1]))
        assert len(got) == 1
        tolerance.tight(got[0], float(expected), reason=f"{key} at {query}")


def test_cube_interpolators_are_built_on_the_transposed_layer() -> None:
    """The 2-D interpolation sees ``transpose(points_[k])`` (hpp:1010, 1225).

    EXACT — this is a reshuffle, not arithmetic. Pinning it separately means
    a transposition bug is reported as a transposition bug rather than as a
    wrong interpolated value.
    """
    cube = _kernel_cube(backward_flat=False)
    expected = np.asarray(_KERNEL["transposed"], dtype=np.float64)
    got = cube.transposed_points()[0]
    assert got.shape == expected.shape
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            tolerance.exact(float(got[i, j]), float(expected[i, j]))


def test_cube_call_is_flat_outside_the_grid_not_linear() -> None:
    """Out-of-range queries clamp (FlatExtrapolator2D, hpp:1030-1032).

    EXACT: the value past an edge must be bit-identical to the value ON that
    edge, which a linear extrapolation off the last cell would not be.
    """
    cube = _kernel_cube(backward_flat=False)
    corner = cube(0.5, 1.0)[0]
    tolerance.exact(cube(0.1, 0.25)[0], corner)
    far = cube(3.0, 10.0)[0]
    tolerance.exact(cube(5.0, 20.0)[0], far)


def test_cube_requires_at_least_two_pillars_per_axis() -> None:
    """# C++ parity: the two QL_REQUIREs at hpp:998-999."""
    d = [Date.from_ymd(1, Month.January, 2024), Date.from_ymd(1, Month.January, 2025)]
    t = [Period(1, _Y), Period(2, _Y)]
    with pytest.raises(LibraryException, match="optionTimes"):
        Cube(d[:1], t, [1.0], [1.0, 2.0], 1)
    with pytest.raises(LibraryException, match="swapLengths"):
        Cube(d, t[:1], [1.0, 2.0], [1.0], 1)


def test_cube_rejects_mismatched_dates_and_tenors() -> None:
    """# C++ parity: the parallel-vector QL_REQUIREs at hpp:1001-1004."""
    d = [Date.from_ymd(1, Month.January, 2024), Date.from_ymd(1, Month.January, 2025)]
    t = [Period(1, _Y), Period(2, _Y)]
    with pytest.raises(LibraryException, match="optionTimes/optionDates"):
        Cube(d, t, [1.0, 2.0, 3.0], [1.0, 2.0], 1)
    with pytest.raises(LibraryException, match="swapTenors/swapLengths"):
        Cube(d, t, [1.0, 2.0], [1.0, 2.0, 3.0], 1)


def test_cube_extrapolation_flag_is_inert() -> None:
    """``extrapolation_`` is stored and never read (hpp:178, 1030-1032).

    EXACT: both settings must give bit-identical out-of-range values, because
    every layer is wrapped in an extrapolation-enabled ``FlatExtrapolator2D``
    regardless.
    """
    on = _kernel_cube(backward_flat=False)
    off = Cube(
        on.option_dates(),
        on.swap_tenors(),
        on.option_times(),
        on.swap_lengths(),
        1,
        False,
        False,
    )
    off.set_layer(0, np.asarray(_KERNEL["layer"], dtype=np.float64))
    off.update_interpolators()
    assert on.extrapolation() is True
    assert off.extrapolation() is False
    for query in _KERNEL["queries"]:
        tolerance.exact(off(float(query[0]), float(query[1]))[0], on(float(query[0]), float(query[1]))[0])


# =====================================================================
# 2. The real private C++ Cube, through SabrSwaptionVolatilityCube
# =====================================================================


def test_python_grid_matches_the_probe_grid() -> None:
    """Sanity gate: same option dates / times / swap lengths as C++.

    TIGHT. Everything downstream is meaningless if the grid differs, so this
    failing first is the useful failure.
    """
    cube = _make_cube(backward_flat=False)
    for got, exp in zip(cube.option_times(), _SETUP["cube_option_times"], strict=True):
        tolerance.tight(got, float(exp))
    for got, exp in zip(cube.swap_lengths(), _SETUP["cube_swap_lengths"], strict=True):
        tolerance.tight(got, float(exp))
    serials = [d.serial for d in cube.option_dates()]
    assert serials == [int(s) for s in _SETUP["cube_option_date_serials"]]

    atm = _atm_matrix()
    for got, exp in zip(atm.option_times(), _SETUP["atm_option_times"], strict=True):
        tolerance.tight(got, float(exp))
    for got, exp in zip(atm.swap_lengths(), _SETUP["atm_swap_lengths"], strict=True):
        tolerance.tight(got, float(exp))


@pytest.mark.parametrize(
    ("backward_flat", "block"),
    [(False, "cube_bilinear_sparse"), (True, "cube_backwardflat_sparse")],
)
def test_cube_browse_matches_cpp_market_vol_cube(backward_flat: bool, block: str) -> None:
    """``Cube::browse()``, verbatim, on the market-vol cube.

    TIGHT. ``marketVolCube_``'s layers are pure quote arithmetic
    (``atmVol + volSpread``, hpp:375-385) — no fit, no ATM strike, since
    ``SwaptionVolatilityMatrix`` ignores the strike argument. So this pins
    the browse LAYOUT (swap length in column 0, option time in column 1,
    ``points_[k][j][i]`` in column ``2+k``, rows ordered by swap length
    first) against C++ with nothing else in the way.
    """
    cube = _make_cube(backward_flat=backward_flat)
    _assert_matrix_tight(
        cube.market_vol_cube(), _REF[block]["market_vol_cube_browse"], "market browse"
    )


@pytest.mark.parametrize(
    ("backward_flat", "block"),
    [(False, "cube_bilinear_sparse"), (True, "cube_backwardflat_sparse")],
)
def test_cube_points_match_cpp_market_vol_cube_layers(
    backward_flat: bool, block: str
) -> None:
    """``Cube::points()[i]`` — rows are option times, columns swap lengths.

    TIGHT. C++ exposes this directly as ``marketVolCube(Size i)`` (hpp:215-217).
    """
    cube = _make_cube(backward_flat=backward_flat)
    for i in range(len(_STRIKE_SPREADS)):
        _assert_matrix_tight(
            cube.market_vol_cube(i),
            _REF[block]["market_vol_cube_points"][i],
            f"market points layer {i}",
        )


@pytest.mark.parametrize(
    ("backward_flat", "block"),
    [(False, "cube_bilinear_sparse"), (True, "cube_backwardflat_sparse")],
)
def test_sparse_parameter_cube_browse_matches_cpp(backward_flat: bool, block: str) -> None:
    """``sparseSabrParameters()`` — the whole parameter cube, verbatim.

    TIGHT. Columns 2..5 are the SABR parameters, which with every parameter
    fixed are ``parametersGuess_(t, s)`` evaluated at the pillars; column 6
    is the ATM forward; 7 and 8 the fit residuals; 9 the end-criteria code
    (0 == ``EndCriteria::None``, i.e. "there was nothing to optimise").
    """
    cube = _make_cube(backward_flat=backward_flat)
    _assert_matrix_tight(
        cube.sparse_xabr_parameters(), _REF[block]["sparse_browse"], "sparse browse"
    )


@pytest.mark.parametrize(
    ("backward_flat", "block"),
    [(False, "cube_bilinear_dense"), (True, "cube_backwardflat_dense")],
)
def test_parameters_guess_cube_interpolates_like_cpp_between_pillars(
    backward_flat: bool, block: str
) -> None:
    """``Cube::operator()`` off the pillars, read out of the REAL private C++ Cube.

    TIGHT, and this is the assertion the whole exercise exists for.

    C++'s ATM-calibrated pass calibrates on a DENSE grid — the union of the
    ATM surface's pillars with the cube's — and, with every parameter fixed,
    each dense cell's fitted parameters are exactly
    ``parametersGuess_(denseOptionTime, denseSwapLength)``
    (sabrswaptionvolatilitycube.hpp:448-449 + xabrinterpolation.hpp:162-169).
    The dense grid contains points strictly BETWEEN the sparse pillars
    (2Y option, 3Y swap) and BELOW them (6M option, 1Y swap), so
    ``denseSabrParameters()`` is a direct readout of the private Cube's
    interpolation at interior and out-of-range arguments.

    The nearest-grid-cell shortcut this port used to take reproduces the
    pillar rows and nothing else.
    """
    cube = _make_cube(backward_flat=backward_flat)
    guess = cube.parameters_guess()
    assert guess is not None
    n_params = cube.n_params()
    for row in _REF[block]["dense_browse"]:
        swap_length = float(row[0])
        option_time = float(row[1])
        got = guess(option_time, swap_length)
        for p in range(n_params):
            tolerance.tight(
                got[p],
                float(row[2 + p]),
                reason=f"param {p} at (t={option_time}, s={swap_length})",
            )


@pytest.mark.parametrize(
    ("backward_flat", "block"),
    [(False, "cube_bilinear_sparse"), (True, "cube_backwardflat_sparse")],
)
def test_smile_section_uses_interpolated_cube_parameters(
    backward_flat: bool, block: str
) -> None:
    """End-to-end: ``volatility(optionTime, swapLength, strike)`` off the pillars.

    TIGHT. ``smileSectionImpl`` reads the parameter cube at ``(t, s)``, takes
    the forward from layer ``nParams`` and the model parameters from layers
    ``0..nParams-1`` (hpp:860-884). Six of the ten query points are NOT grid
    pillars, including four that are strictly interior; the last four are
    outside the grid in one direction each. Every one of them is a value the
    nearest-grid-cell shortcut got wrong.
    """
    cube = _make_cube(backward_flat=backward_flat)
    ref = _REF[block]
    queries = ref["smile_queries"]
    for i, query in enumerate(queries):
        section = cube.smile_section_impl(float(query[0]), float(query[1]))
        atm_level = float(ref["smile_atm_level"][i])
        tolerance.tight(section.atm_level(), atm_level, reason=f"atm_level at {query}")
        tolerance.tight(
            section.volatility(atm_level),
            float(ref["smile_vol_atm"][i]),
            reason=f"vol(atm) at {query}",
        )
        tolerance.tight(
            section.volatility(atm_level - 0.01),
            float(ref["smile_vol_atm_minus_100bp"][i]),
            reason=f"vol(atm-100bp) at {query}",
        )
        tolerance.tight(
            section.volatility(atm_level + 0.01),
            float(ref["smile_vol_atm_plus_100bp"][i]),
            reason=f"vol(atm+100bp) at {query}",
        )


def test_smile_section_between_pillars_differs_from_every_grid_cell() -> None:
    """Guard against a silent regression to nearest-cell lookup.

    The parameter vector read at a midpoint must differ from the vector at
    EVERY one of the four grid cells; a nearest-cell implementation would
    reproduce one of them exactly, whatever it then did with the time.
    """
    cube = _make_cube(backward_flat=False)
    params = cube.sparse_parameters()
    t0, t1 = cube.option_times()
    s0, s1 = cube.swap_lengths()
    mid = params(0.5 * (t0 + t1), 0.5 * (s0 + s1))
    for j, t in enumerate((t0, t1)):
        for k, s in enumerate((s0, s1)):
            assert mid[: cube.n_params()] != list(cube.xabr_parameters(j, k)), (
                f"midpoint parameters equal grid cell ({j},{k}) — nearest-cell lookup?"
            )
            pillar = params(t, s)
            assert mid != pillar


# =====================================================================
# 3. Cube::setPoint and Cube::expandLayers
# =====================================================================


def test_set_point_at_an_existing_pillar_does_not_expand() -> None:
    """``setPoint`` on a (time, length) already present writes in place.

    TIGHT against the C++ ``recalibration`` block, which drives
    ``sabrCalibrationSection`` -> ``Cube::setPoint`` at pillars that ARE
    present (hpp:638-641): the grid must keep its 2x2 shape and only the beta
    layer of the 5Y column may move.
    """
    block = _REF["recalibration"]
    before = np.asarray(block["sparse_browse_before"], dtype=np.float64)
    after = np.asarray(block["sparse_browse_after"], dtype=np.float64)
    assert before.shape == after.shape, "C++ setPoint at a present pillar must not expand"

    cube = _make_cube(backward_flat=False)
    _assert_matrix_tight(cube.sparse_xabr_parameters(), block["sparse_browse_before"], "before")

    # Replay it on the parameter cube itself: overwrite the two 5Y rows with
    # the C++ post-recalibration values via set_point, at (date, tenor) pairs
    # that are already pillars.
    params = cube.sparse_parameters()
    option_dates = params.option_dates()
    option_times = params.option_times()
    swap_tenors = params.swap_tenors()
    swap_lengths = params.swap_lengths()
    n_opt = len(option_times)
    k5 = swap_lengths.index(5.0)
    for j in range(n_opt):
        row = after[k5 * n_opt + j]
        params.set_point(
            option_dates[j], swap_tenors[k5], option_times[j], swap_lengths[k5], list(row[2:])
        )
    params.update_interpolators()
    assert len(params.option_times()) == n_opt, "set_point at a present pillar expanded the grid"
    _assert_matrix_tight(params.browse(), block["sparse_browse_after"], "after")


def test_set_point_at_a_new_pillar_expands_and_shifts() -> None:
    """``setPoint`` -> ``expandLayers``, replaying C++'s ``fillVolatilityCube``.

    TIGHT. C++ starts ``volCubeAtmCalibrated_`` as a copy of ``marketVolCube_``
    (2 option times x 2 swap lengths) and then walks the union grid
    (hpp:685-715), calling ``setPoint`` at every cell whose option time or
    swap length is new. The result is the 4x4 cube pinned in
    ``vol_cube_atm_calibrated_browse``.

    This test replays exactly that loop, feeding the C++ VALUES in (they come
    from ``spreadVolInterpolation``, which is not ported) so that the only
    thing under test is the grid mechanics: where ``expandLayers`` inserts the
    new row/column and where it shifts the existing entries to. All four
    (expandOptionTimes, expandSwapLengths) combinations occur:
    (True, True) at the very first call — 6M option AND 1Y swap are both new —
    then (True, False), (False, True) and (False, False).

    If ``expandLayers`` shifted the wrong way, the four original market-vol
    cells would land in the wrong rows/columns and this would fail.
    """
    cube = _make_cube(backward_flat=False)
    atm = _atm_matrix()
    expected = np.asarray(
        _REF["cube_bilinear_dense"]["vol_cube_atm_calibrated_browse"], dtype=np.float64
    )
    n_strikes = len(_STRIKE_SPREADS)

    # volCubeAtmCalibrated_ = marketVolCube_ (hpp:392).
    market = _make_cube(backward_flat=False)
    working = Cube(
        cube.option_dates(),
        cube.swap_tenors(),
        cube.option_times(),
        cube.swap_lengths(),
        n_strikes,
    )
    for i in range(n_strikes):
        working.set_layer(i, market.market_vol_cube(i))
    working.update_interpolators()

    # The captured "before" copies the C++ loop tests against (hpp:652, 660).
    sparse_times = list(working.option_times())
    sparse_lengths = list(working.swap_lengths())

    # The union grids (hpp:651-681), all four vectors index-parallel.
    atm_times = sorted(set(atm.option_times()) | set(sparse_times))
    atm_lengths = sorted(set(atm.swap_lengths()) | set(sparse_lengths))
    atm_dates = sorted(
        {d.serial for d in atm.option_dates()} | {d.serial for d in cube.option_dates()}
    )
    atm_tenors = sorted(
        set(_ATM_SWAP_TENORS) | set(_SWAP_TENORS), key=lambda p: (p.units, p.length)
    )
    assert len(atm_times) == len(atm_dates)
    assert len(atm_lengths) == len(atm_tenors)

    # Index the C++ result by (swap length, option time) so the replay can
    # look up the value C++ wrote at each expanded cell.
    n_opt_expected = len({float(r[1]) for r in expected})
    by_cell = {
        (float(r[0]), float(r[1])): [float(x) for x in r[2:]] for r in expected
    }

    combos_seen: set[tuple[bool, bool]] = set()
    for j, t in enumerate(atm_times):
        for k, s in enumerate(atm_lengths):
            expand_option_times = t not in sparse_times
            expand_swap_lengths = s not in sparse_lengths
            if not (expand_option_times or expand_swap_lengths):
                continue
            combos_seen.add(
                (t not in working.option_times(), s not in working.swap_lengths())
            )
            working.set_point(
                Date(atm_dates[j]), atm_tenors[k], t, s, by_cell[(s, t)]
            )
    working.update_interpolators()

    assert (True, True) in combos_seen, "the (T, T) expandLayers arm was never taken"
    assert (True, False) in combos_seen
    assert (False, True) in combos_seen
    assert (False, False) in combos_seen
    assert len(working.option_times()) == n_opt_expected
    _assert_matrix_tight(working.browse(), _REF["cube_bilinear_dense"][
        "vol_cube_atm_calibrated_browse"
    ], "expanded browse")


@pytest.mark.parametrize(
    ("expand_rows", "expand_cols"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_expand_layers_shifts_exactly_the_entries_at_or_after_the_insert(
    expand_rows: bool, expand_cols: bool
) -> None:
    """``expandLayers`` mechanics in isolation (hpp:1185-1195).

    EXACT. An old entry moves to ``u+1`` iff ``u >= i`` AND the option axis is
    being expanded, and to ``v+1`` iff ``v >= j`` AND the swap axis is. The
    inserted row/column is zero, the inserted time/length is 0.0 and the
    inserted date/tenor are the null ``Date()`` / ``Period()``.
    """
    dates = [Date.from_ymd(1, Month.January, 2024 + n) for n in range(3)]
    tenors = [Period(n + 1, _Y) for n in range(3)]
    times = [1.0, 2.0, 3.0]
    lengths = [1.0, 2.0, 3.0]
    cube = Cube(dates, tenors, times, lengths, 1)
    original = np.arange(1.0, 10.0).reshape(3, 3)
    cube.set_layer(0, original)

    i, j = 1, 2
    cube.expand_layers(i, expand_rows, j, expand_cols)

    rows = 3 + (1 if expand_rows else 0)
    cols = 3 + (1 if expand_cols else 0)
    got = cube.points()[0]
    assert got.shape == (rows, cols)
    assert len(cube.option_times()) == rows
    assert len(cube.swap_lengths()) == cols
    assert len(cube.option_dates()) == rows
    assert len(cube.swap_tenors()) == cols
    if expand_rows:
        tolerance.exact(cube.option_times()[i], 0.0)
        assert cube.option_dates()[i] == Date()
    if expand_cols:
        tolerance.exact(cube.swap_lengths()[j], 0.0)
        assert cube.swap_tenors()[j] == Period()

    for u in range(3):
        row = u + 1 if (u >= i and expand_rows) else u
        for v in range(3):
            col = v + 1 if (v >= j and expand_cols) else v
            tolerance.exact(float(got[row, col]), float(original[u, v]))


def test_expand_layers_rejects_an_out_of_range_insert_position() -> None:
    """# C++ parity: hpp:1170-1171. Note ``<=``: inserting at the end is legal."""
    dates = [Date.from_ymd(1, Month.January, 2024 + n) for n in range(2)]
    tenors = [Period(n + 1, _Y) for n in range(2)]
    cube = Cube(dates, tenors, [1.0, 2.0], [1.0, 2.0], 1)
    cube.expand_layers(2, True, 2, True)  # at the end: allowed
    with pytest.raises(LibraryException, match="incompatible size 1"):
        cube.expand_layers(99, True, 0, False)
    with pytest.raises(LibraryException, match="incompatible size 2"):
        cube.expand_layers(0, False, 99, True)


def test_set_element_rejects_out_of_range_indices() -> None:
    """# C++ parity: the three QL_REQUIREs at hpp:1099-1104."""
    dates = [Date.from_ymd(1, Month.January, 2024 + n) for n in range(2)]
    tenors = [Period(n + 1, _Y) for n in range(2)]
    cube = Cube(dates, tenors, [1.0, 2.0], [1.0, 2.0], 2)
    with pytest.raises(LibraryException, match="IndexOfLayer"):
        cube.set_element(2, 0, 0, 1.0)
    with pytest.raises(LibraryException, match="IndexOfRow"):
        cube.set_element(0, 2, 0, 1.0)
    with pytest.raises(LibraryException, match="IndexOfColumn"):
        cube.set_element(0, 0, 2, 1.0)


def test_set_layer_and_set_points_reject_wrong_shapes() -> None:
    """# C++ parity: hpp:1110-1115 (setPoints) and hpp:1122-1127 (setLayer)."""
    dates = [Date.from_ymd(1, Month.January, 2024 + n) for n in range(2)]
    tenors = [Period(n + 1, _Y) for n in range(2)]
    cube = Cube(dates, tenors, [1.0, 2.0], [1.0, 2.0], 2)
    with pytest.raises(LibraryException, match="incompatible number of layer"):
        cube.set_layer(5, np.zeros((2, 2)))
    with pytest.raises(LibraryException, match="incompatible size 1"):
        cube.set_layer(0, np.zeros((3, 2)))
    with pytest.raises(LibraryException, match="incompatible size 2"):
        cube.set_layer(0, np.zeros((2, 3)))
    with pytest.raises(LibraryException, match="incompatible number of layers"):
        cube.set_points([np.zeros((2, 2))])


def test_cube_copy_is_independent_and_has_its_own_interpolators() -> None:
    """# C++ parity: the copy constructor (hpp:1037-1061) rebuilds interpolators."""
    original = _kernel_cube(backward_flat=False)
    clone = original.copy()
    for query in _KERNEL["queries"]:
        tolerance.exact(clone(float(query[0]), float(query[1]))[0], original(float(query[0]), float(query[1]))[0])
    original.set_element(0, 0, 0, -999.0)
    original.update_interpolators()
    tolerance.exact(clone(0.5, 1.0)[0], float(_KERNEL["layer"][0][0]))


# =====================================================================
# 4. PrivateObserver
# =====================================================================


def test_private_observer_reseeds_the_guess_cube_and_moves_the_volatility() -> None:
    """A parameter-guess Quote move must propagate all the way to a volatility.

    TIGHT against the probe's ``observer`` block, which reads a smile at an
    interior point, moves ``parametersGuessQuotes_[0][0]`` (alpha of cell
    (0,0)) by +100bp, and reads the same smile again
    (probe.cpp ``emitObserver``). C++ routes that through
    ``PrivateObserver::update`` -> ``setParameterGuess()`` -> ``update()``
    (sabrswaptionvolatilitycube.hpp:272-275).

    The ATM level must NOT move: it comes from the forwards layer, which
    alpha does not touch. That asymmetry is what distinguishes a real
    re-seed from a blanket recompute.
    """
    block = _REF["observer"]
    guess = _guess_quotes()
    cube = _make_cube(backward_flat=False, guess=guess)
    t = float(block["query_option_time"])
    s = float(block["query_swap_length"])
    strike = float(block["strike"])

    before = cube.smile_section_impl(t, s)
    tolerance.tight(before.volatility(strike), float(block["vol_before"]))
    tolerance.tight(before.atm_level(), float(block["atm_level_before"]))

    alpha_quote = guess[0][0][0]
    tolerance.tight(alpha_quote.value(), float(block["alpha_old"]))
    alpha_quote.set_value(float(block["alpha_new"]))

    after = cube.smile_section_impl(t, s)
    tolerance.tight(after.volatility(strike), float(block["vol_after"]))
    tolerance.tight(after.atm_level(), float(block["atm_level_after"]))
    assert before.volatility(strike) != after.volatility(strike)

    _assert_matrix_tight(
        cube.sparse_xabr_parameters(), block["sparse_browse_after"], "sparse after quote move"
    )


def test_private_observer_is_registered_with_every_guess_quote() -> None:
    """# C++ parity: ``registerWithParametersGuess`` (hpp:341-347).

    Every one of the ``nParams * nOptionTenors * nSwapTenors`` quotes must be
    wired, not just the first cell: moving any of them changes the fit.
    """
    guess = _guess_quotes()
    cube = _make_cube(backward_flat=False, guess=guess)
    baseline = cube.sparse_xabr_parameters().copy()
    for j in range(len(_CUBE_OPTION_TENORS)):
        for k in range(len(_SWAP_TENORS)):
            for i in range(cube.n_params()):
                quote = guess[j][k][i]
                old = quote.value()
                quote.set_value(old + 0.01)
                moved = cube.sparse_xabr_parameters()
                assert not np.array_equal(moved, baseline), (
                    f"guess quote ({j},{k}) param {i} is not observed"
                )
                quote.set_value(old)
                restored = cube.sparse_xabr_parameters()
                assert np.array_equal(restored, baseline)


def test_private_observer_holds_the_cube_weakly() -> None:
    """C++ holds ``v_`` as a raw, non-owning pointer (hpp:277-278).

    A strong back-reference would make cube and observer an uncollectable
    cycle, so the port uses a weakref; once the cube is gone, ``update()``
    is a no-op rather than a resurrection.
    """
    cube = _make_cube(backward_flat=False)
    observer = cube._private_observer  # pyright: ignore[reportPrivateUsage]
    assert isinstance(observer, PrivateObserver)
    del cube
    import gc  # noqa: PLC0415

    gc.collect()
    observer.update()  # must not raise


def test_vol_spread_quote_move_invalidates_the_calculation() -> None:
    """The OTHER observer path: vol spreads notify the cube directly.

    # C++ parity: ``SwaptionVolatilityCube::registerWithVolatilitySpread``
    # (swaptionvolcube.cpp:79-87) registers the cube itself, not the
    # PrivateObserver, because a vol-spread move needs no guess re-seed —
    # only a LazyObject invalidation.

    Every SABR parameter is fixed here, so the fitted parameters must NOT
    move; the market vol cube must.
    """
    spreads = _vol_spread_quotes()
    cube = _TableStrikeCube(
        atm_vol_structure=_atm_matrix(),
        option_tenors=_CUBE_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=spreads,
        swap_index_base=_StubSwapIndex(Period(5, _Y)),
        short_swap_index_base=_StubSwapIndex(Period(1, _Y)),
        sabr_initial_guess=_guess_quotes(),
        is_parameter_fixed=(True, True, True, True),
    )
    before_market = cube.market_vol_cube().copy()
    before_params = cube.sparse_xabr_parameters().copy()

    spreads[0][0].set_value(spreads[0][0].value() + 0.001)

    after_market = cube.market_vol_cube()
    after_params = cube.sparse_xabr_parameters()
    assert not np.array_equal(after_market, before_market), "vol-spread move not observed"
    # Columns 2..5 are the SABR parameters; all four are pinned.
    n_params = cube.n_params()
    assert np.array_equal(after_params[:, 2 : 2 + n_params], before_params[:, 2 : 2 + n_params])


def test_cube_without_a_guess_has_no_guess_cube_but_still_fits() -> None:
    """``initial_guess=None`` is a PQuantLib extension; C++ requires the quotes.

    The guess cube is then absent and each cell falls back to the fitter's
    own default initial guess — which is why nothing is pinned against C++
    here beyond the shapes.
    """
    cube = _TableStrikeCube(
        atm_vol_structure=_atm_matrix(),
        option_tenors=_CUBE_OPTION_TENORS,
        swap_tenors=_SWAP_TENORS,
        strike_spreads=_STRIKE_SPREADS,
        vol_spreads=_vol_spread_quotes(),
        swap_index_base=_StubSwapIndex(Period(5, _Y)),
        short_swap_index_base=_StubSwapIndex(Period(1, _Y)),
        sabr_initial_guess=None,
    )
    assert cube.parameters_guess() is None
    browse = cube.sparse_xabr_parameters()
    assert browse.shape == (4, cube.n_params() + 4 + 2)
    assert len(cube.sabr_parameters(0, 0)) == 4
