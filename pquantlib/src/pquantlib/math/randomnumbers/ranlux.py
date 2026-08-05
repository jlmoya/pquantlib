"""M. Luescher's "luxury" subtract-with-carry RNG.

# C++ parity: ql/math/randomnumbers/ranluxuniformrng.hpp (v1.43) —
# ``template <std::size_t P, std::size_t R> class Ranlux64UniformRng``,
# wrapping ``std::subtract_with_carry_engine<uint_fast64_t, 48, 10, 24>``
# behind ``std::discard_block_engine<base, P, R>``.

Reference: M. Luescher, "A portable high-quality random number generator for
lattice field theory simulations", Comp. Phys. Comm. 79 (1994) 100.

The two engine adaptors are specified by the C++ standard, not by a
particular standard library, so libc++ and libstdc++ agree bit for bit and
this port targets the specification:

* ``subtract_with_carry_engine<UIntType, w, s, r>`` produces ``w``-bit outputs
  ``X[k] = (X[k-s] - X[k-r] - c) mod 2^w``, with ``c`` the borrow from the
  previous step.
* ``discard_block_engine<Engine, P, R>`` exposes the first ``R`` outputs of
  every ``P``-long block of base outputs and throws the remaining ``P - R``
  away. ``P`` is the luxury level: 223 for Ranlux3, 389 for Ranlux4.
* Seeding: a ``linear_congruential_engine<uint32, 40014, 0, 2147483563>``
  bootstrap stream produces ``r * ceil(w / 32) == 48`` words; each consecutive
  pair becomes ``(hi << 32) | lo`` masked to 48 bits. The carry starts at 1
  iff ``X[r-1] == 0``.

.. rubric:: Divergence repaired

The previous implementation mapped ``seed == 0`` to 1, on the stated premise
that this was "libstdc++'s seeder fallback". [rand.eng.sub] says otherwise:
``e(value == 0u ? default_seed : value)``, and ``default_seed`` for
``subtract_with_carry_engine`` is **19780503**, not 1. The C++ probe confirms
it — ``Ranlux64UniformRng(0)`` and ``Ranlux64UniformRng(19780503)`` produce
the same stream. The 0-maps-to-1 path produced a completely different stream
for the one seed most likely to be passed by accident.
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
#: # C++ parity: ``nx = 1.0/(std::uint_fast64_t(1) << 48)``.
_INV_2_POW_48: Final[float] = 1.0 / (1 << 48)
# Seeder: linear_congruential_engine<uint_least32_t, 40014, 0, 2147483563>.
_SEED_A: Final[int] = 40014
_SEED_M: Final[int] = 2147483563
_SEED_WORDS_PER_X: Final[int] = (_W + 31) // 32  # 2 for w = 48
#: ``subtract_with_carry_engine::default_seed`` ([rand.eng.sub]/5).
_DEFAULT_SEED: Final[int] = 19780503


class Ranlux64UniformRng:
    """Luxury subtract-with-carry uniform RNG over [0, 1).

    # C++ parity: ``Ranlux64UniformRng<P, R>`` (ranluxuniformrng.hpp:47-63).

    C++ makes ``P`` and ``R`` template parameters; Python takes them as
    constructor arguments, so ``Ranlux64UniformRng(223, 24, seed)`` is the
    C++ ``Ranlux64UniformRng<223, 24>(seed)``.

    Args:
        p: block length of the discard-block adaptor (the luxury level).
        r: number of outputs used per block.
        seed: 0 selects ``default_seed`` (19780503), as the standard requires.
    """

    __slots__ = ("_carry", "_index", "_p", "_r_used", "_state", "_used")

    def __init__(self, p: int, r: int, seed: int = _DEFAULT_SEED) -> None:
        self._p: int = p
        self._r_used: int = r
        # [rand.req.eng]: the LCG bootstrap is seeded with default_seed when
        # the requested seed is 0, and the LCG itself clamps a zero residue
        # to 1 (its multiplier has no additive term, so 0 is a fixed point).
        s = (_DEFAULT_SEED if seed == 0 else seed) % _SEED_M
        if s == 0:
            s = 1
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
        # Initial carry: c = (X[r-1] == 0) ? 1 : 0.
        self._carry: int = 1 if self._state[_R - 1] == 0 else 0
        self._index: int = 0
        # ``used`` counts emissions in the current P-long block.
        self._used: int = 0

    def _swc_next(self) -> int:
        """One base-engine output (48-bit); mutates state and carry."""
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
        if self._used >= self._r_used:
            for _ in range(self._p - self._r_used):
                self._swc_next()
            self._used = 0
        val = self._swc_next()
        self._used += 1
        return val

    def next(self) -> Sample:
        """One sample uniformly drawn from [0.0, 1.0) with weight 1.0.

        # C++ parity: ranluxuniformrng.hpp:54 — ``ranlux_() * nx``.
        """
        return Sample(value=self._next_int() * _INV_2_POW_48, weight=1.0)

    def next_real(self) -> float:
        """The raw value of :meth:`next`."""
        return self._next_int() * _INV_2_POW_48

    def dimension(self) -> int:
        """Scalar RNG — dimension is always 1."""
        return 1


class Ranlux3UniformRng(Ranlux64UniformRng):
    """Luxury level 3 — ``Ranlux64UniformRng<223, 24>``.

    # C++ parity: ``typedef Ranlux64UniformRng<223, 24> Ranlux3UniformRng``
    # (ranluxuniformrng.hpp:65). "Any theoretically possible correlations
    # have very small chance of being observed."
    """

    __slots__ = ()

    def __init__(self, seed: int = _DEFAULT_SEED) -> None:
        super().__init__(223, 24, seed)


class Ranlux4UniformRng(Ranlux64UniformRng):
    """Luxury level 4 — ``Ranlux64UniformRng<389, 24>``.

    # C++ parity: ``typedef Ranlux64UniformRng<389, 24> Ranlux4UniformRng``
    # (ranluxuniformrng.hpp:66). Highest possible luxury.
    """

    __slots__ = ()

    def __init__(self, seed: int = _DEFAULT_SEED) -> None:
        super().__init__(389, 24, seed)


__all__ = ["Ranlux3UniformRng", "Ranlux4UniformRng", "Ranlux64UniformRng"]
