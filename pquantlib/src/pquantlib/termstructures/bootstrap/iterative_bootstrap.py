"""IterativeBootstrap — generic piecewise-term-structure bootstrapper.

# C++ parity: ql/termstructures/iterativebootstrap.hpp (v1.43) — the
   ``IterativeBootstrap<Curve>`` template that drives the per-pillar
   Brent root-find used by ``PiecewiseYieldCurve`` /
   ``PiecewiseDefaultCurve`` / ``PiecewiseZeroInflationCurve`` /
   ``PiecewiseYoYInflationCurve``.

Design notes (Python-specific):

- C++ uses ``template <class Curve>`` to template the bootstrapper on the
  concrete curve type. PEP 695 generics give the same type-level
  discipline: ``class IterativeBootstrap[TS, Traits]``.
- The C++ template knows about ``Curve::Traits`` because the template
  parameter is the curve type. In Python, we make ``Traits`` a separate
  type parameter so the trait class can be plugged in at call site
  (Zero / YoY / Survival / Hazard / Discount / ...).
- The curve must satisfy ``BootstrapCurveProtocol``. The protocol is
  structural so any Piecewise curve subclass that exposes the right
  methods will work without an explicit ``isinstance`` check.

The algorithm (mirroring C++ ``IterativeBootstrap::initialize`` +
``::calculate``):

1. Sort instruments by pillar date and skip the expired ones — the first
   surviving index is ``firstAliveHelper`` (iterativebootstrap.hpp:164-167).
2. Build the (alive+1) dates/times/data grid: pillar 0 is the curve base
   date, and EVERY data slot is seeded with ``traits.initial_value(curve)``
   (iterativebootstrap.hpp:213-217 — the seed is not a ``guess`` call).
3. Wire each helper to the curve.
4. Outer loop: for each pillar i, take the bracket from
   ``traits.min_value_after`` / ``max_value_after``, the start point from
   ``traits.guess``, and Brent-solve ``helper.quote_error() == 0``.
5. After each pillar solve, ``traits.update_guess(data, level, i)``
   installs the solution.
6. Repeat until ``improvement <= accuracy``.

The interpolation is extended ONE NODE AT A TIME on the first pass
(iterativebootstrap.hpp:296-311) and stays scoped to nodes ``0..i`` for
the whole of pillar ``i``'s root-find, because ``error()`` only calls
``interpolation_.update()`` (iterativebootstrap.hpp:313-317). That scoping is
observable: the trait ``guess`` for pillar ``i`` reads the curve at
``t_i``, which is an extrapolation off nodes ``0..i-1``, not a lookup of
the not-yet-solved node ``i``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.time.date import Date


@runtime_checkable
class BootstrapCurveProtocol(Protocol):
    """Structural surface a curve must expose for ``IterativeBootstrap``.

    The curve is mutated in place during the bootstrap: ``data_live()``
    returns the live ``_data`` array (NOT a defensive copy — must alias
    the curve's internal storage), ``set_data_at(i, level)`` updates
    a single pillar, and ``refresh_interpolation_through(i)`` rebuilds
    the underlying interpolation over the first ``i + 1`` nodes.

    Note: ``data_live`` is distinct from the public ``data()`` accessor
    (which returns a defensive copy per the interpolated-curve API).
    The bootstrap *must* see the live array so trait ``update_guess``
    writes through to curve state; the traits, mirroring C++'s
    ``c->data()``, read the public accessor.
    """

    def reference_date(self) -> Date: ...
    def base_date(self) -> Date: ...
    def day_counter(self) -> DayCounter: ...
    def times(self) -> list[float]: ...
    def dates(self) -> list[Date]: ...
    def data(self) -> list[float]: ...
    def data_live(self) -> list[float]: ...
    def set_data_at(self, i: int, level: float) -> None: ...
    def refresh_interpolation_through(self, up_to: int) -> None: ...

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float]
    ) -> None: ...

    def set_max_date(self, d: Date) -> None:
        """Extend the curve's valid range out to ``d``.

        # C++ parity: ``ts_->maxDate_ = maxDate`` in
        # ``IterativeBootstrap::initialize`` (iterativebootstrap.hpp:209).
        # The curve reaches as far as the furthest date any helper actually
        # needs, which is *not* the same as its last pillar — a helper whose
        # pillar sits at its maturity date can still read the curve at a
        # payment date days later.
        """
        ...

    def time_from_reference(self, d: Date) -> float: ...


@runtime_checkable
class BootstrapTraitsProtocol[TS](Protocol):
    """Structural surface a traits class must expose.

    Mirrors the C++ ``Traits`` template parameter (e.g.
    ``ZeroInflationTraits`` / ``YoYInflationTraits`` / ``Discount`` /
    ``ZeroYield`` / ``ForwardRate`` / ``SimpleZeroYield`` /
    ``SurvivalProbability``).

    ``guess`` / ``min_value_after`` / ``max_value_after`` take the CURVE,
    not a bare data list — C++ passes ``const C* c`` so the trait can read
    ``c->times()``, ``c->data()``, ``c->dates()`` and call back into the
    curve's rate accessors (bootstraptraits.hpp:62-102 and friends).
    ``first_alive_helper`` is the index of the first non-expired helper;
    the standard traits ignore it, but it is part of the signature.
    """

    def initial_date(self, ts: TS) -> Date: ...
    def initial_value(self, ts: TS) -> float: ...
    def guess(
        self, i: int, c: TS, valid_data: bool, first_alive_helper: int
    ) -> float: ...
    def min_value_after(
        self, i: int, c: TS, valid_data: bool, first_alive_helper: int
    ) -> float: ...
    def max_value_after(
        self, i: int, c: TS, valid_data: bool, first_alive_helper: int
    ) -> float: ...
    def update_guess(self, data: list[float], level: float, i: int) -> None: ...
    def max_iterations(self) -> int: ...


@runtime_checkable
class TransformingBootstrapTraitsProtocol[TS](BootstrapTraitsProtocol[TS], Protocol):
    """Traits that additionally map to/from an unconstrained variable.

    # C++ parity: ``Traits::transformDirect`` / ``transformInverse``.
    # Required by ``GlobalBootstrap`` (globalbootstrap.hpp:101-103, 371,
    # 383), which optimises over the unconstrained image of the curve
    # values. The yield traits (bootstraptraits.hpp) and the inflation
    # traits (inflationtraits.hpp:105-112) define them; the credit
    # probability traits (probabilitytraits.hpp) deliberately do not, so
    # they are split out of the base protocol rather than forced onto it.
    """

    def transform_direct(self, x: float, i: int, c: TS) -> float: ...
    def transform_inverse(self, x: float, i: int, c: TS) -> float: ...


class IterativeBootstrap[TS, Traits]:
    """Generic piecewise-term-structure bootstrapper.

    Construct with the curve, the bootstrap helpers, and the traits
    instance; call :meth:`calculate` to run the iterative Brent loop.

    The curve is mutated in place — at the end ``curve.data()[i]`` holds
    the bootstrapped value for pillar ``i``.

    The class is type-parameterised purely for documentation. The actual
    structural contract is enforced at runtime via duck-typed method
    lookups; the parameters ``TS`` and ``Traits`` carry the binding type
    for the caller's convenience.
    """

    def __init__(
        self,
        curve: TS,
        instruments: Sequence[BootstrapHelper[TS]],
        traits: Traits,
        accuracy: float = 1.0e-12,
    ) -> None:
        qassert.require(
            len(instruments) > 0,
            "no helpers provided to IterativeBootstrap",
        )
        self._curve: TS = curve
        # Defensive copy — caller may pass a sequence that mutates.
        self._instruments: list[BootstrapHelper[TS]] = list(instruments)
        self._traits: Traits = traits
        self._accuracy: float = accuracy
        self._valid_curve: bool = False
        # C++ parity: ``firstAliveHelper_`` / ``alive_``
        # (iterativebootstrap.hpp:118). Filled in by ``_initialize``.
        self._first_alive_helper: int = 0
        self._alive: int = len(instruments)

    # -- setup ------------------------------------------------------------

    def _initialize(self, curve: Any, traits: Any) -> None:
        """Sort the helpers, skip expired ones, build the grid, size the curve.

        # C++ parity: ``IterativeBootstrap::initialize`` at
        # iterativebootstrap.hpp:155-220.
        """
        n = len(self._instruments)
        self._instruments.sort(key=lambda h: h.pillar_date())

        first_date: Date = traits.initial_date(curve)
        # C++ parity: iterativebootstrap.hpp:161-163.
        qassert.require(
            self._instruments[n - 1].pillar_date() > first_date,
            "all instruments expired",
        )
        # C++ parity: iterativebootstrap.hpp:164-167 — skip expired helpers.
        first_alive = 0
        while self._instruments[first_alive].pillar_date() <= first_date:
            first_alive += 1
        self._first_alive_helper = first_alive
        self._alive = n - first_alive

        dates: list[Date] = [first_date]
        times: list[float] = [curve.time_from_reference(first_date)]

        # C++ seeds ``maxDate`` with the first grid date and grows it to the
        # furthest date any helper actually needs, then writes it onto the
        # curve. The curve therefore extends past its last pillar whenever a
        # helper's latest relevant date does — an OIS helper with a payment
        # lag pillars at its accrual end but still reads the curve at the
        # payment date.
        max_date: Date = first_date

        for j in range(first_alive, n):
            helper = self._instruments[j]
            pillar = helper.pillar_date()
            dates.append(pillar)
            times.append(curve.time_from_reference(pillar))

            # Pillar uniqueness — C++ parity iterativebootstrap.hpp:189-191.
            # Compared against the previous *grid* entry, so a helper whose
            # pillar lands on the curve's own base date is caught too.
            qassert.require(
                dates[-2] != dates[-1],
                f"more than one instrument with pillar {pillar}",
            )

            # Helpers sorted by pillar must also be sorted by latest relevant
            # date — otherwise a helper does not extend the curve at all.
            # C++ parity: iterativebootstrap.hpp:193-201.
            latest_relevant_date = helper.latest_relevant_date()
            qassert.require(
                latest_relevant_date > max_date,
                f"{j + 1}th instrument (pillar: {pillar}) has "
                f"latestRelevantDate ({latest_relevant_date}) before or equal "
                f"to previous instrument's latestRelevantDate ({max_date})",
            )
            max_date = max(pillar, latest_relevant_date)

            # C++ additionally sets ``loopRequired_`` when a pillar precedes
            # its latest relevant date, so a *local* interpolator still gets a
            # second pass. This port has no single-pass exit — the convergence
            # check in ``calculate`` always compares against the previous
            # pass — so the loop is unconditionally run. Nothing to force.

        # C++ parity: iterativebootstrap.hpp:213-217 — every data slot is
        # seeded with ``Traits::initialValue(ts_)``, NOT with ``guess``.
        # Only ``data[0]`` is meaningful; the rest just have to be numbers
        # the interpolator's early checks will accept.
        data: list[float] = [traits.initial_value(curve)] * len(dates)

        curve.bootstrap_install_grid(dates, times, data)
        # C++ parity: iterativebootstrap.hpp:209 — ``ts_->maxDate_ = maxDate``.
        curve.set_max_date(max_date)

        # Wire helpers. C++ parity: iterativebootstrap.hpp:234-245.
        for j in range(first_alive, n):
            self._instruments[j].set_term_structure(curve)

    # -- main entry -------------------------------------------------------

    def calculate(self) -> None:
        """Run the iterative Brent loop until convergence (or maxIterations).

        # C++ parity: ``IterativeBootstrap::calculate`` at
        # iterativebootstrap.hpp:222-389.
        """
        # Duck-typed curve view — see class docstring on TS being purely
        # for caller-side typing; we treat the curve as an Any internally
        # because pyright cannot prove a TypeVar bound at construction time.
        curve: Any = self._curve
        traits: Any = self._traits

        self._initialize(curve, traits)
        alive = self._alive
        first_alive = self._first_alive_helper

        # Steps 3-4 — outer iteration loop.
        # C++ parity: iterativebootstrap.hpp:257-387.
        max_iterations = traits.max_iterations()
        brent = Brent()

        for iteration in range(max_iterations):
            # Snapshot the previous pass's solved values (defensive copy).
            previous_data = list(curve.data_live())

            # Per-pillar inner loop. C++ parity: iterativebootstrap.hpp:266.
            for i in range(1, alive + 1):
                instrument = self._instruments[first_alive + i - 1]
                # Live alias — trait update writes through.
                live_data = curve.data_live()
                valid_data = self._valid_curve or iteration > 0

                # Bracket first: C++ takes min/max BEFORE the guess
                # (iterativebootstrap.hpp:274-280) so the guess can be
                # clamped into it.
                min_v = traits.min_value_after(i, curve, valid_data, first_alive)
                max_v = traits.max_value_after(i, curve, valid_data, first_alive)
                guess = traits.guess(i, curve, valid_data, first_alive)

                # C++ parity: iterativebootstrap.hpp:289-293 — a guess outside
                # the bracket is pulled a FIFTH of the way in, not recentred.
                if guess >= max_v:
                    guess = max_v - (max_v - min_v) / 5.0
                elif guess <= min_v:
                    guess = min_v + (max_v - min_v) / 5.0

                # First pass: extend interpolation one node at a time so
                # extrapolation never reaches an unsolved pillar. The extent
                # stays at ``i`` for the whole of this pillar's root-find —
                # C++ only calls ``interpolation_.update()`` inside ``error``
                # (iterativebootstrap.hpp:296-317).
                if not valid_data:
                    curve.refresh_interpolation_through(i)
                    extent = i
                else:
                    extent = len(live_data) - 1

                # The error function the Brent solver drives to zero.
                def error_fn(
                    x: float,
                    i: int = i,
                    h: BootstrapHelper[TS] = instrument,
                    c: Any = curve,
                    t: Any = traits,
                    up_to: int = extent,
                ) -> float:
                    # Live mutation — ``data_live`` returns the curve's
                    # internal ``_data`` list, not a defensive copy.
                    t.update_guess(c.data_live(), x, i)
                    c.refresh_interpolation_through(up_to)
                    return h.quote_error()

                root = brent.solve(error_fn, self._accuracy, guess, min_v, max_v)
                traits.update_guess(curve.data_live(), root, i)

            # Refresh the full interpolation at end of pass.
            curve.refresh_interpolation_through(len(curve.data_live()) - 1)

            # Single-pass convergence check.
            # For non-global interpolators (Linear / BackwardFlat / ...) one
            # pass suffices — no convergence loop. C++ checks
            # ``interpolator_.global()`` which is ``false`` for these.
            # # C++ parity: iterativebootstrap.hpp:363-374.
            # PQuantLib LinearInterpolation is not global — break after pass 1.
            improvement = 0.0
            cur = curve.data_live()
            for i in range(1, alive + 1):
                improvement = max(improvement, abs(cur[i] - previous_data[i]))

            if improvement <= self._accuracy:
                self._valid_curve = True
                return

        # If we exit the loop without convergence, raise.
        qassert.fail(
            f"convergence not reached after {max_iterations} iterations; "
            f"accuracy = {self._accuracy}",
        )


__all__ = [
    "BootstrapCurveProtocol",
    "BootstrapTraitsProtocol",
    "IterativeBootstrap",
    "TransformingBootstrapTraitsProtocol",
]
