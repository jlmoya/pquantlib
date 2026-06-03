"""Cross-validation tests for XorShiftRandom against Java reference output.

# C++ parity: none — XorShiftRandom is a JQuantLib-original (no C++ QuantLib
# equivalent).  Reference values are Java output, emitted by
# migration-harness/java/Ws1Emitter.java and stored in
# migration-harness/references/cluster/ws1.json.

Tolerance rationale
-------------------
- ``next_long`` values: exact Python ``int`` equality.  The xorshift
  algebra is bitwise; there is no floating-point rounding, so any Python
  implementation divergence would be an outright bug, not a rounding
  artefact.  We do NOT use the float ``tolerance.exact`` helper because
  the 64-bit signed range [-2^63, 2^63) is not representable losslessly
  as IEEE-754 doubles (mantissa only 52 bits).
- ``next_double`` values: EXACT tier (``tolerance.exact``).  The Java
  formula ``(state >>> 11) / 2^53`` is a single IEEE-754 division with
  a power-of-two denominator — the result is a faithfully rounded double.
  Python reproduces the same bit pattern given identical unsigned state,
  so bit-exact agreement is achievable and required.  Reference bits are
  recovered from the stored ``doubleToLongBits`` integers via
  ``struct.unpack('!d', struct.pack('!q', bits))[0]``; all reference
  doubles are in [0, 1) so bit 63 of the stored long is always 0.
"""

from __future__ import annotations

import struct

import pytest

from pquantlib.testing import reference_reader, tolerance
from pquantlib_contrib.math.randomnumbers.xorshift_random import XorShiftRandom

# ---------------------------------------------------------------------------
# Load reference
# ---------------------------------------------------------------------------

_REF: dict[str, object] = reference_reader.load("cluster/ws1")
_SEED: int = int(_REF["seed"])  # type: ignore[arg-type]
_NEXT_LONG: list[int] = [int(v) for v in _REF["next_long"]]  # type: ignore[union-attr]
_NEXT_DOUBLE_BITS: list[int] = [int(v) for v in _REF["next_double_bits"]]  # type: ignore[union-attr]

# Reconstruct expected doubles from IEEE-754 bit patterns (signed longs from
# Java's Double.doubleToLongBits; all are positive since doubles are in [0,1)).
_NEXT_DOUBLE: list[float] = [
    struct.unpack("!d", struct.pack("!q", bits))[0] for bits in _NEXT_DOUBLE_BITS
]


# ---------------------------------------------------------------------------
# next_long: exact integer equality
# ---------------------------------------------------------------------------


class TestNextLong:
    """XorShiftRandom.next_long() matches Java nextLong() — bit-exact ints."""

    def test_sequence_length(self) -> None:
        """Reference has 16 values — sanity check before comparing."""
        assert len(_NEXT_LONG) == 16

    @pytest.mark.exact
    @pytest.mark.parametrize("i", range(16))
    def test_next_long_value(self, i: int) -> None:
        """next_long()[i] == Java nextLong()[i] for i in 0..15.

        Integer comparison — not tolerance.exact (which is IEEE-754 double
        only and lossy for 64-bit ints outside the 53-bit mantissa range).
        """
        rng = XorShiftRandom(_SEED)
        for _ in range(i):
            rng.next_long()
        actual = rng.next_long()
        expected = _NEXT_LONG[i]
        assert actual == expected, (
            f"next_long()[{i}]: actual={actual!r}, expected={expected!r}"
        )

    @pytest.mark.exact
    def test_full_sequence_via_single_instance(self) -> None:
        """Single RNG instance produces the full 16-element sequence in order."""
        rng = XorShiftRandom(_SEED)
        for i, expected in enumerate(_NEXT_LONG):
            actual = rng.next_long()
            assert actual == expected, (
                f"next_long()[{i}]: actual={actual!r}, expected={expected!r}"
            )


# ---------------------------------------------------------------------------
# next_double: EXACT tier (bit-identical IEEE-754)
# ---------------------------------------------------------------------------


class TestNextDouble:
    """XorShiftRandom.next_double() matches Java nextDouble() — IEEE-754 exact."""

    def test_sequence_length(self) -> None:
        """Reference has 16 values — sanity check before comparing."""
        assert len(_NEXT_DOUBLE) == 16

    @pytest.mark.exact
    @pytest.mark.parametrize("i", range(16))
    def test_next_double_value(self, i: int) -> None:
        """next_double()[i] == Java nextDouble()[i] — EXACT tier.

        EXACT tier: bit-identical via tolerance.exact (struct.pack '!d').
        The Java formula is (state >>> 11) / 2^53 — a power-of-two
        division yielding a faithfully rounded double that Python must
        reproduce identically.
        """
        rng = XorShiftRandom(_SEED)
        for _ in range(i):
            rng.next_double()
        actual = rng.next_double()
        expected = _NEXT_DOUBLE[i]
        tolerance.exact(actual, expected)

    @pytest.mark.exact
    def test_full_double_sequence_via_single_instance(self) -> None:
        """Single RNG instance produces the full 16-element double sequence."""
        rng = XorShiftRandom(_SEED)
        for i, expected in enumerate(_NEXT_DOUBLE):
            actual = rng.next_double()
            tolerance.exact(actual, expected, reason=f"next_double()[{i}]")


# ---------------------------------------------------------------------------
# Behavioural / contract tests
# ---------------------------------------------------------------------------


class TestBehavioural:
    """Unit-level behavioural tests not dependent on the Java reference."""

    def test_next_double_in_unit_interval(self) -> None:
        """next_double() always returns a value in [0.0, 1.0)."""
        rng = XorShiftRandom(_SEED)
        for _ in range(64):
            d = rng.next_double()
            assert 0.0 <= d < 1.0, f"next_double() out of [0, 1): {d!r}"

    def test_next_long_in_signed_range(self) -> None:
        """next_long() always returns a value in [-2^63, 2^63)."""
        rng = XorShiftRandom(_SEED)
        lo, hi = -(1 << 63), (1 << 63)
        for i in range(64):
            v = rng.next_long()
            assert lo <= v < hi, f"next_long()[{i}]={v!r} out of signed 64-bit range"

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Two different seeds diverge immediately on first call."""
        rng_a = XorShiftRandom(42)
        rng_b = XorShiftRandom(99)
        assert rng_a.next_long() != rng_b.next_long()

    def test_same_seed_reproduces_sequence(self) -> None:
        """Re-seeding (new instance) replays the same sequence."""
        rng_a = XorShiftRandom(7)
        rng_b = XorShiftRandom(7)
        for _ in range(32):
            assert rng_a.next_long() == rng_b.next_long()

    def test_negative_seed_masked_to_64_bits(self) -> None:
        """Negative seed is accepted; -1 and 2^64-1 produce the same sequence."""
        rng_neg = XorShiftRandom(-1)
        rng_pos = XorShiftRandom((1 << 64) - 1)
        for _ in range(16):
            assert rng_neg.next_long() == rng_pos.next_long()
