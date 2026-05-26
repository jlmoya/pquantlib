"""M. Luescher's "luxury" subtract-with-carry RNG.

# C++ parity: ql/math/randomnumbers/ranluxuniformrng.hpp (v1.42.1) —
# template ``Ranlux64UniformRng<P, R>`` wrapping the standard
# ``std::subtract_with_carry_engine<uint_fast64_t, 48, 10, 24>`` filtered
# through ``std::discard_block_engine<base, P, R>``.

This port mirrors the libstdc++ implementation of those two C++
standard library engine adaptors precisely:

* ``subtract_with_carry_engine<UIntType, w, s, r>`` produces ``w``-bit
  outputs by computing ``X[k] = (X[ps] - X[pr] - c) mod 2^w`` where
  ``ps = (k - s) mod r`` and ``pr = k`` (the "short lag" is k itself
  on libstdc++ — equivalently ``(k - r) mod r``). The carry ``c`` is
  the borrow bit from the previous subtraction.
* ``discard_block_engine<Engine, P, R>`` exposes only the first ``R``
  outputs of every ``P``-long block of base outputs, discarding the
  remaining ``P - R``.

Seeding follows libstdc++: a ``linear_congruential_engine<uint32, 40014,
0, 2147483563>`` is used as the bootstrap stream, and exactly
``r * ceil(w / 32)`` 32-bit words are drawn — here ``24 * 2 = 48``
words. Each pair of words is combined as ``(hi << 32) | lo`` and
masked to ``w = 48`` bits to form one element of the initial state
``X[0..r-1]``. The carry bit ``c`` is initialised to 1 iff ``X[r-1]
== 0``, else 0.

Ranlux3 specifically is ``Ranlux64UniformRng<223, 24>`` — 24 outputs
used per 223 base steps, giving Luescher's "level-3" luxury parameter.

This deeply-specified seeding contract is exactly what gives bit-
identical sequences against the C++ reference; if any of those details
diverge the sequence diverges within the first call.
"""

from __future__ import annotations

from typing import Final

from pquantlib.math.randomnumbers.random_number_generator import Sample

# subtract_with_carry_engine parameters: <uint_fast64_t, 48, 10, 24>
_W: Final[int] = 48  # output width in bits
_S: Final[int] = 10  # short lag
_R: Final[int] = 24  # long lag
_W_MASK: Final[int] = (1 << _W) - 1
_W_MOD: Final[int] = 1 << _W
_INV_2_POW_48: Final[float] = 1.0 / (1 << 48)
# libstdc++ seeder: linear_congruential_engine<uint_least32_t, 40014, 0, 2147483563>.
_SEED_A: Final[int] = 40014
_SEED_M: Final[int] = 2147483563
_SEED_WORDS_PER_X: Final[int] = (_W + 31) // 32  # 2 for w=48


class Ranlux3UniformRng:
    """Ranlux3 (subtract-with-carry + discard-block <223, 24>) uniform RNG.

    # C++ parity: ``Ranlux3UniformRng`` typedef in
    # ql/math/randomnumbers/ranluxuniformrng.hpp (v1.42.1) — i.e.
    # ``Ranlux64UniformRng<223, 24>``.

    Seed 0 maps to libstdc++'s seeder fallback of 1 (because the
    LCG period demands ``s != 0``). The C++ template defaults to
    seed 19780503; pquantlib makes the seed explicit at the constructor
    site to avoid hidden-defaults bugs.
    """

    # discard_block parameters for level 3
    _P: Final[int] = 223
    _R_USED: Final[int] = 24

    __slots__ = ("_carry", "_index", "_state", "_used")

    def __init__(self, seed: int) -> None:
        # libstdc++ ``linear_congruential_engine::seed`` clamps zero
        # seeds to 1 (the LCG cannot start at 0). We replicate that
        # rather than refusing seed 0 because the C++ template's
        # default of 19780503 is itself a nonzero constant, and tests
        # legitimately pass seed = 0 to exercise the fallback path.
        s = seed % _SEED_M
        if s == 0:
            s = 1
        # Generate r * ceil(w / 32) = 48 seed words.
        words: list[int] = []
        for _ in range(_R * _SEED_WORDS_PER_X):
            s = (_SEED_A * s) % _SEED_M
            words.append(s & 0xFFFFFFFF)
        # Pack each consecutive pair of 32-bit words into one 48-bit X[i].
        self._state: list[int] = []
        for i in range(_R):
            val = 0
            for j in range(_SEED_WORDS_PER_X):
                val |= words[i * _SEED_WORDS_PER_X + j] << (32 * j)
            self._state.append(val & _W_MASK)
        # libstdc++ initial carry: c = (X[r-1] == 0) ? 1 : 0.
        self._carry: int = 1 if self._state[_R - 1] == 0 else 0
        self._index: int = 0
        # discard_block: ``used`` counts emissions in the current
        # P-long block of base outputs. Once it reaches R_USED, the
        # engine discards (P - R_USED) base outputs before resuming.
        self._used: int = 0

    def _swc_next(self) -> int:
        """One base-engine output (48-bit), mutates state and carry."""
        # ps = (k - s) mod r ; pr = k (libstdc++ short lag).
        ps = (self._index + _R - _S) % _R
        pr = self._index
        val = self._state[ps] - self._state[pr] - self._carry
        if val < 0:
            val += _W_MOD
            self._carry = 1
        else:
            self._carry = 0
        val &= _W_MASK
        self._state[self._index] = val
        self._index = (self._index + 1) % _R
        return val

    def _next_int(self) -> int:
        """One discard-block-filtered base output."""
        if self._used >= self._R_USED:
            # Drop the unused tail of the block.
            for _ in range(self._P - self._R_USED):
                self._swc_next()
            self._used = 0
        val = self._swc_next()
        self._used += 1
        return val

    def next(self) -> Sample:
        """One sample uniformly drawn from [0.0, 1.0) with weight 1.0."""
        # C++ parity: ranluxuniformrng.hpp:54 — ``ranlux_() * nx`` with
        # ``nx = 1.0 / (1ULL << 48)``.
        return Sample(value=self._next_int() * _INV_2_POW_48, weight=1.0)

    def dimension(self) -> int:
        """Scalar RNG — dimension is always 1."""
        return 1
