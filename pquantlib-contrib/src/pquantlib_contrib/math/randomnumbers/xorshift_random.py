"""Marsaglia xorshift64 pseudo-random number generator.

# C++ parity: none — XorShiftRandom is a JQuantLib-original class
# (org.jquantlib.math.randomnumbers.XorShiftRandom) with no equivalent
# in C++ QuantLib v1.42.1.  Ground truth is Java output, cross-validated
# via migration-harness/references/cluster/ws1.json.
#
# Reference: Marsaglia, G. (2003). "Xorshift RNGs".
# Journal of Statistical Software, 8(14), 1-6.
# https://www.jstatsoft.org/v08/i14/paper/
#
# Java source: jquantlib-contrib/src/main/java/org/jquantlib/math/
#              randomnumbers/XorShiftRandom.java (tag jquantlib-final)
#
# Semantics reproduced exactly from Java:
#   - Internal state ``_x`` is a 64-bit unsigned quantity masked to
#     [0, 2^64) at every step, mirroring Java ``long`` two's-complement
#     wrap-at-64-bits arithmetic.
#   - ``<<`` masks at 64 bits; ``>>>`` is Java unsigned right-shift
#     (plain ``>>`` in Python on the masked unsigned value).
#   - ``next_long()`` returns the signed 64-bit view of the new state,
#     exactly as Java's signed-long return of ``nextLong()``.
#   - ``next_double()`` divides the **unsigned** shifted state by 2^53,
#     matching Java's ``(nextLong() >>> 11) / (double)(1L << 53)``.
#
# Deliberate omission: the Java no-arg constructor ``XorShiftRandom()``
# seeds from ``new Random().nextLong()`` — a nondeterministic path that
# cannot be cross-validated.  Only the seeded constructor is ported.
"""

from __future__ import annotations

__all__ = ["XorShiftRandom"]

# 2^64 - 1: mask to keep all arithmetic within 64 bits, reproducing
# Java signed-long overflow (which wraps modulo 2^64).
_MASK64: int = (1 << 64) - 1


class XorShiftRandom:
    """Marsaglia xorshift64 RNG — JQuantLib-original, no C++ QuantLib equivalent.

    Parameters
    ----------
    seed:
        Initial state.  Any 64-bit integer; negative values are accepted
        (masked to 64 bits).  A zero seed is a degenerate fixed point:
        ``0 XOR (0 << k) = 0`` at every step, so every call to
        ``next_long()`` returns 0 and every call to ``next_double()``
        returns 0.0 — matching Java's behavior (no guard is raised).
        Callers should provide a nonzero seed for a non-trivial sequence.

    Examples
    --------
    >>> rng = XorShiftRandom(42)
    >>> rng.next_long()
    45454805674
    >>> rng2 = XorShiftRandom(42)
    >>> 0.0 <= rng2.next_double() < 1.0
    True
    """

    __slots__ = ("_x",)

    def __init__(self, seed: int) -> None:
        # C++ parity: none — mirrors Java ``XorShiftRandom(final long seed)``
        # constructor: ``x = seed``.  We mask to keep Python ints in the
        # 64-bit unsigned range; Java sign-wrapping is then reproduced by
        # the masking in next_long_unsigned().
        self._x: int = seed & _MASK64

    def _next_long_unsigned(self) -> int:
        """Advance state; return unsigned 64-bit new state.

        Internal helper — reproduces Java xorshift steps with explicit
        64-bit masking in place of Java's implicit signed-long overflow.
        """
        # C++ parity: none — Java ``nextLong()`` body:
        #   x ^= x << 13;   (signed-long overflow ↔ mask at 64 bits)
        #   x ^= x >>> 7;   (unsigned right shift ↔ plain >> on unsigned value)
        #   return x ^= (x << 17);
        x = self._x
        x ^= (x << 13) & _MASK64
        x ^= x >> 7  # unsigned right-shift: x is already non-negative (masked)
        x ^= (x << 17) & _MASK64
        self._x = x & _MASK64
        return self._x

    def next_long(self) -> int:
        """Return the next pseudo-random 64-bit **signed** integer.

        Mirrors Java ``nextLong()`` semantics: the return value is in
        [-2^63, 2^63).  Values >= 2^63 in the unsigned state are returned
        as their two's-complement negative equivalent.
        """
        u = self._next_long_unsigned()
        return u - (1 << 64) if u >= (1 << 63) else u

    def next_double(self) -> float:
        """Return the next pseudo-random double in [0.0, 1.0).

        Mirrors Java ``nextDouble()`` override:
        ``(nextLong() >>> 11) / (double)(1L << 53)``
        where ``>>>`` is unsigned right-shift.  The unsigned state is
        used (before sign conversion), so the shift operates on the
        full 64-bit unsigned value.
        """
        # _next_long_unsigned() advances state and returns unsigned value,
        # matching the Java "unsigned shift >>> 11" on the 64-bit state.
        return (self._next_long_unsigned() >> 11) / float(1 << 53)
