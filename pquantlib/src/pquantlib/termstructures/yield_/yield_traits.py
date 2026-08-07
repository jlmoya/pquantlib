"""Bootstrap traits for yield term structures.

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp (v1.43)

Four traits cover the canonical yield-curve bootstrap shapes:

- :class:`Discount` — discount factors directly (bootstraptraits.hpp:44).
- :class:`ZeroYield` — continuously-compounded zero rates
  (bootstraptraits.hpp:127).
- :class:`ForwardRate` — instantaneous forward rates
  (bootstraptraits.hpp:220).
- :class:`SimpleZeroYield` — *simple* (1 / (1 + R t)) zero rates
  (bootstraptraits.hpp:313).

The C++ traits are header-only structs whose members are templated on the
curve type ``C``; every one of ``guess`` / ``minValueAfter`` /
``maxValueAfter`` receives ``const C* c`` and reads ``c->times()``,
``c->data()``, ``c->dates()``, ``c->dayCounter()`` — and, for the rate
traits, calls back into ``c->zeroRate()`` / ``c->forwardRate()``.  The
Python port takes the curve too: it is the only way to reproduce the C++
arithmetic, and every substitute for it (a hard-coded ``0.25`` for
``times()[1]``, ``dt = 1``, ``data[i-1]`` in place of the extrapolation)
is a silent divergence.  The port keeps the all-static C++ surface
expressed as instance methods so ``IterativeBootstrap`` can hold one
instance per bootstrap; construction is a no-op, so ``Discount()`` and
``Discount`` are interchangeable at a call site.

``transformDirect`` / ``transformInverse`` map an unconstrained
optimiser variable to a curve value and back.  They are consumed by
``GlobalBootstrap`` (globalbootstrap.hpp:371, 383), which is why every
C++ yield trait defines them.

``firstAliveHelper`` is the index of the first non-expired helper.  The
standard traits ignore it (C++ declares the parameter unnamed), but it is
part of the protocol and ``IterativeBootstrap`` computes it
(iterativebootstrap.hpp:164-167), so it is threaded through here.
"""

from __future__ import annotations

import math
from typing import Any, Final

from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp:38-41
# (``namespace detail`` — avgRate at :39, maxRate at :40).
_AVG_RATE: Final[float] = 0.05
_MAX_RATE: Final[float] = 1.0

# C++ parity: bootstraptraits.hpp:366, 387, 392 — the offset that keeps a
# simple zero rate above the value at which ``1 + R t`` hits zero, i.e. the
# "no negative discount factor" constraint.
_SIMPLE_ZERO_EPS: Final[float] = 1e-8


def _simple_zero_floor(c: Any, i: int) -> float:
    """``-1 / times()[i] + 1e-8`` — the SimpleZeroYield lower barrier.

    # C++ parity: bootstraptraits.hpp:366 / 387 / 392. ``1 + R t == 0``
    # at ``R = -1/t``; staying strictly above it is what stops the curve
    # producing a negative (or infinite) discount factor at pillar ``i``.
    """
    return -1.0 / c.times()[i] + _SIMPLE_ZERO_EPS


class Discount:
    """Discount-curve traits.

    # C++ parity: ``struct Discount`` in bootstraptraits.hpp:43-124.

    Bootstrap state is ``data[i] = D(t_i)``.
    """

    def initial_date(self, ts: Any) -> Any:
        # C++ parity: bootstraptraits.hpp:53-56.
        return ts.reference_date()

    def initial_value(self, ts: Any) -> float:
        # C++ parity: bootstraptraits.hpp:57-60 — 1.0.
        del ts
        return 1.0

    def guess(self, i: int, c: Any, valid_data: bool, first_alive_helper: int) -> float:
        """# C++ parity: ``Discount::guess`` (bootstraptraits.hpp:62-78)."""
        del first_alive_helper
        if valid_data:
            # Previous-iteration value.
            return c.data()[i]
        if i == 1:
            # First pillar — bootstraptraits.hpp:72-73. The time is the
            # curve's own ``times()[1]``, NOT a fixed short-tenor stand-in.
            return 1.0 / (1.0 + _AVG_RATE * c.times()[1])
        # Flat-rate extrapolation — bootstraptraits.hpp:75-77.
        r = -math.log(c.data()[i - 1]) / c.times()[i - 1]
        return math.exp(-r * c.times()[i])

    def min_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: ``Discount::minValueAfter`` (bootstraptraits.hpp:80-93)."""
        del first_alive_helper
        if valid_data:
            return min(c.data()) / 2.0
        # bootstraptraits.hpp:91-92 — the step uses the ACTUAL time gap.
        dt = c.times()[i] - c.times()[i - 1]
        return c.data()[i - 1] * math.exp(-_MAX_RATE * dt)

    def max_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: ``Discount::maxValueAfter`` (bootstraptraits.hpp:94-102).

        Unlike ``minValueAfter`` there is no ``validData`` branch: the cap
        is ``data[i-1] * exp(maxRate * dt)`` either way, and it is well
        above ``data[i-1]`` — discount factors are *not* clamped to be
        monotonically decreasing here.
        """
        del valid_data, first_alive_helper
        dt = c.times()[i] - c.times()[i - 1]
        return c.data()[i - 1] * math.exp(_MAX_RATE * dt)

    def transform_direct(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:104-109 — ``exp(x)``."""
        del i, c
        return math.exp(x)

    def transform_inverse(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:110-114 — ``log(x)``."""
        del i, c
        return math.log(x)

    def update_guess(self, data: list[float], discount: float, i: int) -> None:
        # C++ parity: bootstraptraits.hpp:116-121.
        data[i] = discount

    def max_iterations(self) -> int:
        # C++ parity: bootstraptraits.hpp:123 — 100.
        return 100


class ZeroYield:
    """Zero-curve traits.

    # C++ parity: ``struct ZeroYield`` in bootstraptraits.hpp:127-217.

    Bootstrap state is ``data[i] = z(t_i)``, continuously compounded.
    """

    def initial_date(self, ts: Any) -> Any:
        # C++ parity: bootstraptraits.hpp:137-140.
        return ts.reference_date()

    def initial_value(self, ts: Any) -> float:
        # C++ parity: bootstraptraits.hpp:141-144 — avgRate (a dummy).
        del ts
        return _AVG_RATE

    def guess(self, i: int, c: Any, valid_data: bool, first_alive_helper: int) -> float:
        """# C++ parity: ``ZeroYield::guess`` (bootstraptraits.hpp:146-163)."""
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        if i == 1:
            return _AVG_RATE
        # bootstraptraits.hpp:159-162 — extrapolate off the CURVE, which
        # runs the interpolation, rather than copying ``data[i-1]``.
        d = c.dates()[i]
        return c.zero_rate(
            d,
            Compounding.Continuous,
            Frequency.Annual,
            True,
            c.day_counter(),
        ).rate()

    def min_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: ``ZeroYield::minValueAfter`` (bootstraptraits.hpp:165-179)."""
        del i, first_alive_helper
        if valid_data:
            r = min(c.data())
            return r * 2.0 if r < 0.0 else r / 2.0
        return -_MAX_RATE

    def max_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: ``ZeroYield::maxValueAfter`` (bootstraptraits.hpp:180-193)."""
        del i, first_alive_helper
        if valid_data:
            r = max(c.data())
            return r / 2.0 if r < 0.0 else r * 2.0
        return _MAX_RATE

    def transform_direct(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:196-200 — identity."""
        del i, c
        return x

    def transform_inverse(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:201-205 — identity."""
        del i, c
        return x

    def update_guess(self, data: list[float], rate: float, i: int) -> None:
        # C++ parity: bootstraptraits.hpp:207-214.
        data[i] = rate
        if i == 1:
            data[0] = rate  # first point is updated as well

    def max_iterations(self) -> int:
        # C++ parity: bootstraptraits.hpp:216 — 100.
        return 100


class ForwardRate:
    """Forward-curve traits.

    # C++ parity: ``struct ForwardRate`` in bootstraptraits.hpp:220-310.

    Same algebra as :class:`ZeroYield`, but ``data[i]`` is an
    instantaneous forward rate and the extrapolating guess calls
    ``forwardRate(d, d, ...)``.
    """

    def initial_date(self, ts: Any) -> Any:
        # C++ parity: bootstraptraits.hpp:230-233.
        return ts.reference_date()

    def initial_value(self, ts: Any) -> float:
        # C++ parity: bootstraptraits.hpp:234-237.
        del ts
        return _AVG_RATE

    def guess(self, i: int, c: Any, valid_data: bool, first_alive_helper: int) -> float:
        """# C++ parity: ``ForwardRate::guess`` (bootstraptraits.hpp:239-256)."""
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        if i == 1:
            return _AVG_RATE
        # bootstraptraits.hpp:252-255.
        d = c.dates()[i]
        return c.forward_rate(
            d,
            d,
            Compounding.Continuous,
            Frequency.Annual,
            True,
            c.day_counter(),
        ).rate()

    def min_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: bootstraptraits.hpp:258-272."""
        del i, first_alive_helper
        if valid_data:
            r = min(c.data())
            return r * 2.0 if r < 0.0 else r / 2.0
        return -_MAX_RATE

    def max_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: bootstraptraits.hpp:273-286."""
        del i, first_alive_helper
        if valid_data:
            r = max(c.data())
            return r / 2.0 if r < 0.0 else r * 2.0
        return _MAX_RATE

    def transform_direct(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:289-293 — identity."""
        del i, c
        return x

    def transform_inverse(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:294-298 — identity."""
        del i, c
        return x

    def update_guess(self, data: list[float], forward: float, i: int) -> None:
        # C++ parity: bootstraptraits.hpp:300-307.
        data[i] = forward
        if i == 1:
            data[0] = forward  # first point is updated as well

    def max_iterations(self) -> int:
        # C++ parity: bootstraptraits.hpp:309 — 100.
        return 100


class SimpleZeroYield:
    """Simple-zero-curve traits.

    # C++ parity: ``struct SimpleZeroYield`` in bootstraptraits.hpp:312-405.

    Bootstrap state is ``data[i] = R(t_i)`` where the discount factor is
    ``1 / (1 + R t)``, so the underlying curve is
    :class:`~pquantlib.termstructures.yield_.interpolated_simple_zero_curve.InterpolatedSimpleZeroCurve`.

    What makes this trait different from :class:`ZeroYield` is the
    ``-1/t + 1e-8`` barrier: ``1 + R t`` vanishes at ``R = -1/t``, so both
    ``minValueAfter`` and the transform pair are shifted to keep the
    solver strictly on the positive-discount side of it.  The transforms
    are therefore NOT the identity.
    """

    def initial_date(self, ts: Any) -> Any:
        # C++ parity: bootstraptraits.hpp:322-325.
        return ts.reference_date()

    def initial_value(self, ts: Any) -> float:
        # C++ parity: bootstraptraits.hpp:326-329 — avgRate (a dummy).
        del ts
        return _AVG_RATE

    def guess(self, i: int, c: Any, valid_data: bool, first_alive_helper: int) -> float:
        """# C++ parity: ``SimpleZeroYield::guess`` (bootstraptraits.hpp:331-348).

        The extrapolating branch asks the curve for a ``Simple`` zero
        rate — not ``Continuous`` as :class:`ZeroYield` does.
        """
        del first_alive_helper
        if valid_data:
            return c.data()[i]
        if i == 1:
            return _AVG_RATE
        d = c.dates()[i]
        return c.zero_rate(
            d,
            Compounding.Simple,
            Frequency.Annual,
            True,
            c.day_counter(),
        ).rate()

    def min_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: bootstraptraits.hpp:350-367.

        Same body as :class:`ZeroYield` up to the final clamp against
        ``-1/times()[i] + 1e-8`` (bootstraptraits.hpp:366).
        """
        del first_alive_helper
        if valid_data:
            r = min(c.data())
            result = r * 2.0 if r < 0.0 else r / 2.0
        else:
            result = -_MAX_RATE
        return max(result, _simple_zero_floor(c, i))

    def max_value_after(
        self, i: int, c: Any, valid_data: bool, first_alive_helper: int
    ) -> float:
        """# C++ parity: bootstraptraits.hpp:368-381.

        No barrier here — ``1 + R t`` only vanishes for negative ``R``.
        """
        del i, first_alive_helper
        if valid_data:
            r = max(c.data())
            return r / 2.0 if r < 0.0 else r * 2.0
        return _MAX_RATE

    def transform_direct(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:384-388.

        ``exp(x) + (-1/t_i + 1e-8)`` — maps the whole real line onto the
        admissible half-line ``(-1/t_i + 1e-8, +inf)``.
        """
        return math.exp(x) + _simple_zero_floor(c, i)

    def transform_inverse(self, x: float, i: int, c: Any) -> float:
        """# C++ parity: bootstraptraits.hpp:389-393 — ``log(x - floor)``."""
        return math.log(x - _simple_zero_floor(c, i))

    def update_guess(self, data: list[float], rate: float, i: int) -> None:
        # C++ parity: bootstraptraits.hpp:395-402.
        data[i] = rate
        if i == 1:
            data[0] = rate  # first point is updated as well

    def max_iterations(self) -> int:
        # C++ parity: bootstraptraits.hpp:404 — 100.
        return 100


__all__ = ["Discount", "ForwardRate", "SimpleZeroYield", "ZeroYield"]
