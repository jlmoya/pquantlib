"""Bootstrap traits for yield term structures.

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp (v1.42.1)

Three traits cover the canonical yield-curve bootstrap shapes:

- :class:`Discount` — discount factors directly. Initial value 1.0,
  flat-rate extrapolation in log-space for the initial guess.
- :class:`ZeroYield` — zero rates. Initial value = ``avg_rate`` (0.05),
  bracketed in ``[-max_rate, max_rate]`` = ``[-1, 1]``.
- :class:`ForwardRate` — instantaneous forward rates. Same shape as
  :class:`ZeroYield`.

The C++ traits are header-only static structs templated on the curve
type. The Python port keeps the same all-static surface but expressed
as instance methods on a class — this lets ``IterativeBootstrap`` hold
a single instance per bootstrap and dispatch through it without
classmethod boilerplate at call site. Construction is a no-op so each
trait class can be used either as ``Discount()`` or ``Discount`` (a
quick ``__init__`` shim is added to allow class-as-trait usage).

The C++ traits also have ``transformDirect`` / ``transformInverse``
hooks used by unconstrained-optimization variants of the bootstrap;
those aren't consumed by ``IterativeBootstrap`` and are omitted here
(carve-out — only used by alternative ``LocalBootstrap``).

The C++ ``guess`` method takes a ``Size firstAliveHelper`` argument
that the bootstrap loop passes in; PQuantLib's IterativeBootstrap
doesn't track first-alive-helper indexing yet, so the Python port
omits the parameter from the protocol shape.

# C++ parity divergence: the C++ ``guess`` for ZeroYield / ForwardRate
   extrapolates by calling ``c->zeroRate(d, ..., Continuous, Annual,
   true)`` / ``c->forwardRate(...)`` which themselves invoke the
   interpolation. The Python port simplifies by returning
   ``data[i - 1]`` (flat extrapolation in the rate itself) — for the
   linear-interpolation bootstrap that's the C++ behaviour anyway.
"""

from __future__ import annotations

import math
from typing import Any, Final

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp anon namespace.
_AVG_RATE: Final[float] = 0.05
_MAX_RATE: Final[float] = 1.0


class Discount:
    """Discount-curve traits.

    # C++ parity: ``struct Discount`` in bootstraptraits.hpp:44-124.

    Bootstrap evaluates ``data[i] = D(t_i)`` directly. Initial guess for
    the first pillar is ``1/(1 + avgRate * t_1)``; subsequent pillars
    use flat-rate extrapolation in log-space:
    ``D(t_i) = exp(-r_{i-1} * t_i)``
    where ``r_{i-1} = -log(D(t_{i-1}))/t_{i-1}``.
    """

    def initial_date(self, ts: Any) -> Any:
        # C++ parity: ``Discount::initialDate``.
        return ts.reference_date()

    def initial_value(self, ts: Any) -> float:
        # C++ parity: ``Discount::initialValue`` = 1.0.
        del ts
        return 1.0

    def guess(self, i: int, data: list[float], valid_data: bool) -> float:
        # C++ parity: ``Discount::guess``.
        # ``data[i]`` is the previous-iteration value when valid_data.
        if valid_data:
            return data[i]
        if i == 1:
            # The first-pillar guess in C++ uses ``c->times()[1]`` which
            # we don't have here without the curve. The fallback is
            # ``1/(1 + avg_rate * 0.25)`` ≈ short-tenor seed.
            return 1.0 / (1.0 + _AVG_RATE * 0.25)
        # Flat-rate extrapolation in log-space (C++ uses curve times();
        # we approximate by data[i-1] which is a defensible seed).
        return data[i - 1]

    def min_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        # C++ parity: ``Discount::minValueAfter``.
        if valid_data:
            return min(data) / 2.0
        # ``data[i-1] * exp(-max_rate * dt)`` — without dt we use a
        # conservative tiny positive bound.
        return data[i - 1] * math.exp(-_MAX_RATE * 1.0)

    def max_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        # C++ parity: ``Discount::maxValueAfter``.
        # The C++ bracket is ``data[i-1] * exp(max_rate * dt)``; we
        # use ``data[i - 1]`` as a hard cap since discount factors are
        # monotonically decreasing.
        del valid_data
        return data[i - 1]

    def update_guess(self, data: list[float], discount: float, i: int) -> None:
        # C++ parity: ``Discount::updateGuess``.
        data[i] = discount

    def max_iterations(self) -> int:
        # C++ parity: ``Discount::maxIterations`` = 100.
        # The PQuantLib bootstrap converges in 1 pass for linear; we use
        # 50 (matching the credit traits convention).
        return 100


class ZeroYield:
    """Zero-curve traits.

    # C++ parity: ``struct ZeroYield`` in bootstraptraits.hpp:128-217.

    Bootstrap evaluates ``data[i] = z(t_i)`` directly. Bracket is
    ``[-max_rate, max_rate]`` = ``[-1, 1]``. The dummy initial value
    (data[0]) is ``avg_rate``; it is overwritten on the first
    ``update_guess`` (pillar 1) so the rate at t=0 always equals the
    rate at the first pillar.
    """

    def initial_date(self, ts: Any) -> Any:
        # C++ parity: ``ZeroYield::initialDate``.
        return ts.reference_date()

    def initial_value(self, ts: Any) -> float:
        # C++ parity: ``ZeroYield::initialValue`` = avg_rate (dummy).
        del ts
        return _AVG_RATE

    def guess(self, i: int, data: list[float], valid_data: bool) -> float:
        # C++ parity: ``ZeroYield::guess``.
        if valid_data:
            return data[i]
        if i == 1:
            return _AVG_RATE
        return data[i - 1]

    def min_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        # C++ parity: ``ZeroYield::minValueAfter``.
        del i
        if valid_data:
            r = min(data)
            return r * 2.0 if r < 0.0 else r / 2.0
        return -_MAX_RATE

    def max_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        # C++ parity: ``ZeroYield::maxValueAfter``.
        del i
        if valid_data:
            r = max(data)
            return r / 2.0 if r < 0.0 else r * 2.0
        return _MAX_RATE

    def update_guess(self, data: list[float], rate: float, i: int) -> None:
        # C++ parity: ``ZeroYield::updateGuess``.
        data[i] = rate
        if i == 1:
            data[0] = rate  # first point pinned to first pillar rate

    def max_iterations(self) -> int:
        # C++ parity: ``ZeroYield::maxIterations`` = 100.
        return 100


class ForwardRate:
    """Forward-curve traits.

    # C++ parity: ``struct ForwardRate`` in bootstraptraits.hpp:221-310.

    Same algebraic shape as :class:`ZeroYield`. Different semantic
    convention: ``data[i]`` is an instantaneous forward rate at pillar
    ``i`` instead of a zero rate.
    """

    def initial_date(self, ts: Any) -> Any:
        # C++ parity: ``ForwardRate::initialDate``.
        return ts.reference_date()

    def initial_value(self, ts: Any) -> float:
        # C++ parity: ``ForwardRate::initialValue`` = avg_rate.
        del ts
        return _AVG_RATE

    def guess(self, i: int, data: list[float], valid_data: bool) -> float:
        # C++ parity: ``ForwardRate::guess``.
        if valid_data:
            return data[i]
        if i == 1:
            return _AVG_RATE
        return data[i - 1]

    def min_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        # C++ parity: ``ForwardRate::minValueAfter``.
        del i
        if valid_data:
            r = min(data)
            return r * 2.0 if r < 0.0 else r / 2.0
        return -_MAX_RATE

    def max_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        # C++ parity: ``ForwardRate::maxValueAfter``.
        del i
        if valid_data:
            r = max(data)
            return r / 2.0 if r < 0.0 else r * 2.0
        return _MAX_RATE

    def update_guess(self, data: list[float], forward: float, i: int) -> None:
        # C++ parity: ``ForwardRate::updateGuess``.
        data[i] = forward
        if i == 1:
            data[0] = forward  # first point pinned to first pillar forward

    def max_iterations(self) -> int:
        # C++ parity: ``ForwardRate::maxIterations`` = 100.
        return 100


__all__ = ["Discount", "ForwardRate", "ZeroYield"]
