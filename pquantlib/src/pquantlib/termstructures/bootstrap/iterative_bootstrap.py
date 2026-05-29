"""IterativeBootstrap — generic piecewise-term-structure bootstrapper.

# C++ parity: ql/termstructures/iterativebootstrap.hpp (v1.42.1) — the
   ``IterativeBootstrap<Curve>`` template that drives the per-pillar
   Brent root-find used by ``PiecewiseYieldCurve`` /
   ``PiecewiseZeroInflationCurve`` / ``PiecewiseYoYInflationCurve``.

Closes the L2-B carve-out (yield bootstrap) and the L7-Bb carve-out
(inflation bootstrap) — both share the same algorithm.

Design notes (Python-specific):

- C++ uses ``template <class Curve>`` to template the bootstrapper on the
  concrete curve type. PEP 695 generics give the same type-level
  discipline: ``class IterativeBootstrap[TS, Traits]``.
- The C++ template knows about ``Curve::Traits`` because the template
  parameter is the curve type. In Python, we make ``Traits`` a separate
  type parameter so the trait class can be plugged in at call site
  (Zero / YoY / Survival / Hazard / etc.).
- The curve must satisfy ``BootstrapCurveProtocol`` (a small surface of
  data/time/interpolation mutators). The protocol is structural so any
  Piecewise curve subclass that exposes the right methods will work
  without an explicit ``isinstance`` check.

The algorithm (mirroring C++ ``IterativeBootstrap::calculate``):

1. Sort instruments by pillar date.
2. Build the (n+1) dates/times/data grid: pillar 0 is the curve base
   date with ``traits.initial_value(curve)``; pillar i+1 is
   ``instruments[i].pillar_date()`` with ``traits.guess(i+1, data,
   valid_data=False)`` as initial guess.
3. Wire each helper to the curve.
4. Run the per-iteration Brent loop, solving each pillar's data slot
   such that the corresponding helper's ``quote_error()`` reaches zero.
5. After each pillar solve, ``traits.update_guess(data, level, i)``
   installs the solution.
6. After every pillar passes once, the loop exits (non-global
   interpolators) or repeats until ``improvement <= accuracy`` (global
   interpolators).

The Brent solver from L1-C is used for the per-pillar root find; the
bracket is given by ``traits.min_value_after`` / ``max_value_after``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.time.date import Date


@runtime_checkable
class BootstrapCurveProtocol(Protocol):
    """Structural surface a curve must expose for ``IterativeBootstrap``.

    The curve is mutated in place during the bootstrap: ``data_live()``
    returns the live ``_data`` array (NOT a defensive copy — must aliase
    the curve's internal storage), ``set_data_at(i, level)`` updates
    a single pillar, and ``refresh_interpolation_through(i)`` rebuilds
    the underlying interpolation over the first ``i + 1`` nodes.

    Note: ``data_live`` is distinct from the public ``data()`` accessor
    (which returns a defensive copy per the interpolated-curve API).
    The bootstrap *must* see the live array so trait ``update_guess``
    writes through to curve state.
    """

    def reference_date(self) -> Date: ...
    def base_date(self) -> Date: ...
    def times(self) -> list[float]: ...
    def data_live(self) -> list[float]: ...
    def set_data_at(self, i: int, level: float) -> None: ...
    def refresh_interpolation_through(self, up_to: int) -> None: ...

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float]
    ) -> None: ...

    def time_from_reference(self, d: Date) -> float: ...


@runtime_checkable
class BootstrapTraitsProtocol[TS](Protocol):
    """Structural surface a traits class must expose.

    Mirrors the C++ ``Traits`` template parameter (e.g.
    ``ZeroInflationTraits`` / ``YoYInflationTraits`` /
    ``Discount`` / ``ZeroYield`` / ``ForwardRate``).
    """

    def initial_date(self, ts: TS) -> Date: ...
    def initial_value(self, ts: TS) -> float: ...
    def guess(self, i: int, data: list[float], valid_data: bool) -> float: ...
    def min_value_after(
        self, i: int, data: list[float], valid_data: bool
    ) -> float: ...
    def max_value_after(
        self, i: int, data: list[float], valid_data: bool
    ) -> float: ...
    def update_guess(self, data: list[float], level: float, i: int) -> None: ...
    def max_iterations(self) -> int: ...


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

    # -- main entry -------------------------------------------------------

    def calculate(self) -> None:
        """Run the iterative Brent loop until convergence (or maxIterations).

        # C++ parity: ``IterativeBootstrap::calculate`` at
        # iterativebootstrap.hpp:184-368. Algorithm:
        # 1. Sort helpers by pillar date; check pillar uniqueness.
        # 2. Build dates/times/data arrays (n + 1 each, with pillar 0 = base).
        # 3. Wire each helper to the curve.
        # 4. Outer loop: for each pillar i in 1..n, Brent-solve for
        #    ``data[i]`` s.t. helper[i-1].quote_error() == 0.
        # 5. Repeat until either non-global interpolator (no second pass
        #    needed) or improvement <= accuracy.
        """
        n = len(self._instruments)
        # Duck-typed curve view — see class docstring on TS being purely
        # for caller-side typing; we treat the curve as an Any internally
        # because pyright cannot prove a TypeVar bound at construction time.
        curve: Any = self._curve
        traits: Any = self._traits

        # Step 1 — sort helpers by pillar date.
        self._instruments.sort(key=lambda h: h.pillar_date())

        # Pillar uniqueness check — # C++ parity iterativebootstrap.hpp:191-196.
        for i in range(1, n):
            qassert.require(
                self._instruments[i - 1].pillar_date()
                != self._instruments[i].pillar_date(),
                "two instruments have the same pillar date",
            )

        # Step 2 — build dates/times/data grid.
        # C++ parity: iterativebootstrap.hpp:204-220.
        dates: list[Date] = [traits.initial_date(curve)]
        times: list[float] = [curve.time_from_reference(dates[0])]
        data: list[float] = [traits.initial_value(curve)]

        for i in range(n):
            dates.append(self._instruments[i].pillar_date())
            times.append(curve.time_from_reference(dates[i + 1]))
            data.append(traits.guess(i + 1, data, valid_data=False))

        # Install grid on the curve.
        curve.bootstrap_install_grid(dates, times, data)

        # Step 3 — wire helpers.
        # C++ parity: iterativebootstrap.hpp:225-227.
        for i in range(n):
            self._instruments[i].set_term_structure(curve)

        # Steps 4-5 — outer iteration loop.
        # C++ parity: iterativebootstrap.hpp:229-368.
        max_iterations = traits.max_iterations()
        brent = Brent()

        for iteration in range(max_iterations):
            # Snapshot the previous pass's solved values (defensive copy).
            previous_data = list(curve.data_live())

            # Per-pillar inner loop.
            for i in range(1, n + 1):
                instrument = self._instruments[i - 1]
                # Live alias — trait update writes through.
                live_data = curve.data_live()
                valid_data = self._valid_curve or iteration > 0

                # Compute the initial guess.
                if valid_data:
                    guess = live_data[i]
                elif i == 1:
                    guess = traits.guess(i, live_data, valid_data=False)
                else:
                    guess = live_data[i - 1]

                min_v = traits.min_value_after(i, live_data, valid_data)
                max_v = traits.max_value_after(i, live_data, valid_data)
                if guess <= min_v or guess >= max_v:
                    guess = (min_v + max_v) / 2.0

                # First pass: extend interpolation one node at a time so
                # extrapolation never reaches an unsolved pillar.
                if not self._valid_curve and iteration == 0:
                    curve.refresh_interpolation_through(i)

                # The error function the Brent solver minimises.
                def error_fn(
                    x: float,
                    i: int = i,
                    h: BootstrapHelper[TS] = instrument,
                    c: Any = curve,
                    t: Any = traits,
                ) -> float:
                    # Live mutation — ``data_live`` returns the curve's
                    # internal ``_data`` list, not a defensive copy.
                    t.update_guess(c.data_live(), x, i)
                    c.refresh_interpolation_through(len(c.data_live()) - 1)
                    return h.quote_error()

                root = brent.solve(error_fn, self._accuracy, guess, min_v, max_v)
                traits.update_guess(curve.data_live(), root, i)

            # Refresh the full interpolation at end of pass.
            curve.refresh_interpolation_through(len(curve.data_live()) - 1)

            # Single-pass convergence check.
            # For non-global interpolators (Linear / BackwardFlat / ...) one
            # pass suffices — no convergence loop. C++ checks
            # ``interpolator_.global()`` which is ``false`` for these.
            # # C++ parity: iterativebootstrap.hpp:325-329.
            # PQuantLib LinearInterpolation is not global — break after pass 1.
            improvement = 0.0
            cur = curve.data_live()
            for i in range(1, n + 1):
                improvement = max(improvement, abs(cur[i] - previous_data[i]))

            if improvement <= self._accuracy:
                self._valid_curve = True
                return

        # If we exit the loop without convergence, raise.
        qassert.fail(
            f"convergence not reached after {max_iterations} iterations; "
            f"accuracy = {self._accuracy}",
        )
