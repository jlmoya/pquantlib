"""Cross-validated tests for MultiCurve / MultiCurveBootstrapProvider.

# C++ parity: ql/termstructures/multicurve.{hpp,cpp} (v1.43).

Reference values come from the ``multi_curve_cycle`` section of
``migration-harness/cpp/probes/v143_ts_globalbootstrap/probe.cpp``, stored
at ``migration-harness/references/v143/ts/globalbootstrap.json``.

The case is the one ``MultiCurve`` exists for: a genuine dependency cycle.
A 3M forwarding curve is bootstrapped from par swaps that DISCOUNT off an
OIS curve, and that OIS curve is a constant zero spread over the 3M curve
being bootstrapped. Neither curve can be built before the other, so both
are handed to one optimizer.

pquantlib has no ``Handle`` / ``RelinkableHandle`` (both allowlisted in
``migration-harness/check_coverage.py``), so the forward reference that C++
spells as an empty relinkable handle is
:class:`~pquantlib.termstructures.multi_curve.InternalCurveLink` here — see
that module's docstring for the full divergence note.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final, cast

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.global_bootstrap import GlobalBootstrap
from pquantlib.termstructures.multi_curve import (
    InternalCurveLink,
    MultiCurve,
    MultiCurveBootstrapProvider,
)
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.swap_rate_helper import SwapRateHelper
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.termstructures.yield_.zero_spreaded_term_structure import (
    ZeroSpreadedTermStructure,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

REF: Final[dict[str, Any]] = reference_reader.load("v143/ts/globalbootstrap")
CYCLE: Final[dict[str, Any]] = REF["multi_curve_cycle"]

# probe.cpp:87 — ``const Date kToday(23, October, 2025);``
_TODAY: Final[Date] = Date.from_ymd(23, Month.October, 2025)
# probe.cpp:581 — ``const Real accuracy = 1.0e-10``.
_ACCURACY: Final[float] = 1.0e-10
_QUOTE: Final[float] = 0.03  # probe.cpp:572
_SPREAD: Final[float] = -0.01  # probe.cpp:573
_N_SWAPS: Final[int] = 5  # probe.cpp:576
# The back-reference MultiCurve._add_curve injects; see multi_curve.py.
_OWNER_ATTR: Final[str] = "_multi_curve_owner"


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    prev = s.evaluation_date
    s.evaluation_date = _TODAY  # probe.cpp:87
    yield
    s.evaluation_date = prev


class _Cycle:
    """The built cycle, kept together so a test can reach every member."""

    def __init__(
        self,
        multi_curve: MultiCurve,
        curve_3m: PiecewiseYieldCurve,
        curve_ois: ZeroSpreadedTermStructure,
        bootstrap: GlobalBootstrap,
        helpers: list[SwapRateHelper],
    ) -> None:
        self.multi_curve = multi_curve
        self.curve_3m = curve_3m
        self.curve_ois = curve_ois
        self.bootstrap = bootstrap
        self.helpers = helpers


def _build_cycle() -> _Cycle:
    """probe.cpp:567-590, in the same order.

    The order matters and is the C++ order: the spreaded curve is built
    only AFTER ``add_bootstrapped_curve`` has linked the 3M link, because
    until then the link has nothing to forward to (C++ builds the
    ``ZeroSpreadedTermStructure`` after ``addBootstrappedCurve`` for
    exactly the same reason — an empty handle cannot be dereferenced).
    """
    link_3m = InternalCurveLink()
    link_ois = InternalCurveLink()

    # probe.cpp:570-571.
    index_3m = IborIndex(
        "EUR3M",
        Period(3, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,  # end_of_month - positional in the C++ ctor too
        Actual360(),
        link_3m,
    )
    quote = SimpleQuote(_QUOTE)
    spread = SimpleQuote(_SPREAD)

    # probe.cpp:575-579 — par swaps discounting off the (not yet existing)
    # OIS curve.
    helpers = [
        SwapRateHelper(
            quote,
            tenor=Period(i, TimeUnit.Years),
            calendar=TARGET(),
            fixed_frequency=Frequency.Annual,
            fixed_convention=BusinessDayConvention.Following,
            fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
            ibor_index=index_3m,
            discount_curve=link_ois,
            evaluation_date=_TODAY,
        )
        for i in range(1, _N_SWAPS + 1)
    ]

    # probe.cpp:582-584.
    curve_3m = PiecewiseYieldCurve(Discount, _TODAY, helpers, Actual360())
    bootstrap = GlobalBootstrap(Discount, accuracy=_ACCURACY)
    bootstrap.setup(curve_3m)

    multi_curve = MultiCurve(accuracy=_ACCURACY)
    # probe.cpp:585-586.
    multi_curve.add_bootstrapped_curve(link_3m, curve_3m, contributor=bootstrap)
    # probe.cpp:587-590.
    curve_ois = ZeroSpreadedTermStructure(link_3m, spread)
    multi_curve.add_non_bootstrapped_curve(link_ois, curve_ois)

    # C++ triggers the joint solve lazily through the curve's LazyObject;
    # pquantlib's PiecewiseYieldCurve hard-codes IterativeBootstrap as its
    # own bootstrap, so the GlobalBootstrap is driven explicitly. It
    # delegates to the MultiCurveBootstrap (globalbootstrap.hpp:408-411).
    bootstrap.calculate()
    return _Cycle(multi_curve, curve_3m, curve_ois, bootstrap, helpers)


@pytest.fixture(scope="module")
def cycle() -> Iterator[_Cycle]:
    """Built once — the joint solve is the expensive part of this module."""
    s = ObservableSettings()
    prev = s.evaluation_date
    s.evaluation_date = _TODAY  # probe.cpp:87
    yield _build_cycle()
    s.evaluation_date = prev


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_internal_link_is_empty_until_added() -> None:
    """# C++ parity: ``Handle::empty()`` — multicurve.cpp:35 / :48."""
    link = InternalCurveLink()
    assert link.empty()
    with pytest.raises(LibraryException, match="not linked yet"):
        link.reference_date()


def test_adding_a_curve_twice_is_rejected() -> None:
    """# C++ parity: multicurve.cpp:35-36 and :48-49."""
    link = InternalCurveLink()
    curve = PiecewiseYieldCurve(
        Discount,
        _TODAY,
        [
            SwapRateHelper(
                SimpleQuote(_QUOTE),
                tenor=Period(1, TimeUnit.Years),
                calendar=TARGET(),
                fixed_frequency=Frequency.Annual,
                fixed_convention=BusinessDayConvention.Following,
                fixed_day_count=Thirty360(Thirty360Convention.BondBasis),
                ibor_index=IborIndex(
                    "EUR3M",
                    Period(3, TimeUnit.Months),
                    2,
                    EURCurrency(),
                    TARGET(),
                    BusinessDayConvention.ModifiedFollowing,
                    False,  # end_of_month
                    Actual360(),
                ),
                evaluation_date=_TODAY,
            )
        ],
        Actual360(),
    )
    mc = MultiCurve(accuracy=_ACCURACY)
    mc.add_non_bootstrapped_curve(link, curve)
    with pytest.raises(LibraryException, match="was the curve added already"):
        mc.add_non_bootstrapped_curve(link, curve)


def test_add_bootstrapped_curve_requires_a_provider_or_contributor() -> None:
    """# C++ parity: multicurve.cpp:37-40 — the dynamic_pointer_cast guard.

    pquantlib's ``PiecewiseYieldCurve`` is not a
    ``MultiCurveBootstrapProvider`` (it hard-codes ``IterativeBootstrap``),
    so without an explicit contributor the same rejection fires.
    """
    mc = MultiCurve(accuracy=_ACCURACY)
    with pytest.raises(LibraryException, match="MultiCurveBootstrapProvider"):
        mc.add_bootstrapped_curve(None, object())


def test_provider_returning_none_is_rejected() -> None:
    """# C++ parity: multicurve.cpp:40 — ``nullptr`` from a non-global bootstrap.

    ``PiecewiseYieldCurve::multiCurveBootstrapContributor`` returns
    ``nullptr`` when its bootstrap is not a contributor
    (piecewiseyieldcurve.hpp:148-154).
    """

    class _NonGlobal(MultiCurveBootstrapProvider):
        def multi_curve_bootstrap_contributor(self) -> None:
            return None

    mc = MultiCurve(accuracy=_ACCURACY)
    with pytest.raises(LibraryException, match="valid multi curve bootstrap"):
        mc.add_bootstrapped_curve(None, _NonGlobal())


def test_multi_curve_registers_every_member(cycle: _Cycle) -> None:
    """# C++ parity: multicurve.cpp:69 — ``curves_.push_back(curve)``.

    Both members are held, and the joint bootstrap has one contributor
    (the 3M curve) and one observer (the spreaded curve).
    """
    assert cycle.multi_curve.curves() == [cycle.curve_3m, cycle.curve_ois]
    assert cycle.multi_curve.bootstrap().contributors() == [cycle.bootstrap]


def test_cycle_members_stay_alive_through_any_member(cycle: _Cycle) -> None:
    """The C++ external handle co-owns the MultiCurve — multicurve.hpp:73-76.

    Reproduced by the back-reference set in ``MultiCurve._add_curve``: from
    either curve you can reach the MultiCurve, and through it every other
    member, so nothing in the cycle can be collected while any one of them
    is referenced.
    """
    for member in (cycle.curve_3m, cycle.curve_ois):
        # ``_multi_curve_owner`` is the aliasing-handle stand-in, see
        # MultiCurve._add_curve. Read through getattr with a non-literal name:
        # the attribute is injected onto the curve at registration time and is
        # therefore invisible to the type checker.
        owner = cast("MultiCurve", getattr(member, _OWNER_ATTR))
        assert owner is cycle.multi_curve
        assert cycle.curve_3m in owner.curves()
        assert cycle.curve_ois in owner.curves()


def test_multi_curve_update_fans_out_without_recursing(cycle: _Cycle) -> None:
    """# C++ parity: multicurve.cpp:73-76.

    C++ terminates because each member is a LazyObject that stops
    re-notifying on the second lap; pquantlib's TermStructure notifies
    unconditionally, so ``MultiCurve.update`` carries an explicit latch. If
    the latch were missing this call would blow the stack.
    """
    cycle.multi_curve.update()


# ---------------------------------------------------------------------------
# the solved cycle
# ---------------------------------------------------------------------------


def test_cycle_pillar_dates_match_cpp(cycle: _Cycle) -> None:
    """Precondition for the value checks: identical swap schedules.

    # C++ parity: probe section ``multi_curve_cycle.pillar_dates``.
    Tier: EXACT (dates).
    """
    assert cycle.curve_3m.reference_date().serial_number() == CYCLE["reference_date"]
    assert [h.pillar_date().serial_number() for h in cycle.helpers] == (
        CYCLE["pillar_dates"]
    )


def test_cycle_spread_relation_is_exact(cycle: _Cycle) -> None:
    """The OIS curve is the 3M curve plus a constant continuous zero spread.

    # C++ parity: probe section ``multi_curve_cycle.zero_spread_1y``, and
    # the corresponding QL_CHECK_CLOSE in
    # test-suite/piecewiseyieldcurve.cpp:1729-1730.

    This is the one property of the cycle that is INDEPENDENT of the
    optimizer: whatever the 3M curve converges to, the spreaded curve sits
    a fixed distance above it. Tier: TIGHT.
    """
    zero_3m = cycle.curve_3m.zero_rate(1.0, Compounding.Continuous).rate()
    zero_ois = cycle.curve_ois.zero_rate(1.0, Compounding.Continuous).rate()
    tight(zero_ois - zero_3m, _SPREAD)
    tight(zero_ois - zero_3m, CYCLE["zero_spread_1y"])


def test_cycle_reprices_every_helper(cycle: _Cycle) -> None:
    """Every par swap reprices at its quote through the solved cycle.

    # C++ parity: probe section ``multi_curve_cycle.quote_errors`` (all
    # ~1e-16) and test-suite/piecewiseyieldcurve.cpp:1734-1742, which
    # checks the same thing as ``swap.NPV() == 0`` to 1e-10.

    Asserted against an absolute bound rather than against the C++ residual,
    which is rounding noise with no reproducible digits. 1e-10 is the
    accuracy the cycle was built with (probe.cpp:581).
    """
    for helper in cycle.helpers:
        assert abs(helper.quote_error()) < 1e-10


def test_cycle_curve_values_reproduce_cpp(cycle: _Cycle) -> None:
    """The converged cycle, at LOOSE.

    # C++ parity: probe section ``multi_curve_cycle.discounts_3m`` /
    # ``discounts_ois``.

    **Tolerance rationale.** These are the only optimizer-dependent numbers
    in this module. C++ runs the MINPACK ``lmdif`` translation embedded in
    ql/math/optimization/levenbergmarquardt.cpp; pquantlib delegates to
    ``scipy.optimize.least_squares(method='lm')``, whose C++-parity tests
    are xfailed (tests/math/optimization/
    test_levenberg_marquardt_cpp_parity.py). Both minimise the same
    exactly-determined system (5 residuals, 5 unknowns) whose residual norm
    at the solution is ~1e-16, so they land on the same root.

    Measured agreement on this fixture is 5.2e-16 relative on the worst 3M
    discount factor, 4.9e-16 on the worst OIS one and 7.4e-15 on the 1Y
    zero rate — the two optimizers reach the same doubles. The assertion is
    LOOSE all the same, deliberately: those digits come from where scipy's
    MINPACK stops on a flat objective, which this port does not control
    across scipy / BLAS versions. LOOSE is the contract; the measured
    margin is recorded so that a future 1e-9 drift reads as a regression
    rather than as normal optimizer noise.
    """
    # The pillar dates are already pinned exactly by
    # ``test_cycle_pillar_dates_match_cpp``, so reading them off the helpers
    # evaluates the curves at the same dates the probe did.
    pillars = [h.pillar_date() for h in cycle.helpers]
    for d, want in zip(pillars, CYCLE["discounts_3m"], strict=True):
        loose(cycle.curve_3m.discount(d), want)
    for d, want in zip(pillars, CYCLE["discounts_ois"], strict=True):
        loose(cycle.curve_ois.discount(d), want)
    loose(
        cycle.curve_3m.zero_rate(1.0, Compounding.Continuous).rate(),
        CYCLE["zero_3m_1y"],
    )
    loose(
        cycle.curve_ois.zero_rate(1.0, Compounding.Continuous).rate(),
        CYCLE["zero_ois_1y"],
    )
