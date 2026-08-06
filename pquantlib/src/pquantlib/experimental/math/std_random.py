"""C++ ``<random>`` primitives, reproduced stream-for-stream.

# C++ parity: none of these is a QuantLib class. They are the C++ standard
# library facilities that ql/experimental/math/{particleswarmoptimization,
# fireflyalgorithm,isotropicrandomwalk,levyflightdistribution}.hpp reach for
# directly, reproduced here so those QuantLib classes can be cross-validated
# against C++ rather than merely checked distributionally.

Why this module exists
----------------------
Most QuantLib randomness flows through
:class:`~pquantlib.math.randomnumbers.mersenne_twister.MersenneTwisterUniformRng`,
which the port already matches bit-for-bit. But three classes in
``experimental/math`` bypass it and use ``<random>`` directly:

* ``ClubsTopology`` — ``std::mt19937`` + ``std::uniform_int_distribution<Size>``
* ``LevyFlightInertia`` — ``std::mt19937`` feeding ``LevyFlightDistribution``,
  which internally draws from ``std::uniform_real_distribution<Real>(0,1)``
* ``GaussianWalk`` / ``DecreasingGaussianWalk`` — ``std::normal_distribution``

The *engine* is portable: ``std::mt19937`` and QuantLib's
``MersenneTwisterUniformRng`` are the same generator with the same seeding, so
``std::mt19937(s)()`` equals ``MersenneTwisterUniformRng(s).next_int32()``.
That equality is not assumed — it is asserted directly from the probe (block
A1 of ``v143/experimental/pso.json``, which emits both sequences side by side).

The *distributions* are not portable: ``[rand.dist]`` specifies their
statistics, not their algorithms, so every standard library is free to consume
a different number of engine draws and combine them differently. Reproducing
QuantLib's behaviour therefore means reproducing the particular implementation
the reference binary was built against — libc++, as shipped by the Apple
toolchain that builds ``migration-harness/cpp``. Each class below transcribes
that algorithm, and each is pinned against probe output that was produced by
calling the real ``std::`` template.

If the reference binary is ever rebuilt against a different standard library,
the block-A tests are what will fail, and they will fail first — before the
QuantLib-level tests that depend on them.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng

_UINT32_MAX: int = 0xFFFFFFFF
_UINT64_MAX: int = 0xFFFFFFFFFFFFFFFF

# Engine traits for std::mt19937: min() == 0, max() == 2^32 - 1, so the
# generated range R = max - min + 1 is 2^32 and log2(R) == 32.
_ENGINE_R: int = 1 << 32
_ENGINE_LOG2_R: int = 32
_ENGINE_DIGITS: int = 32

# std::numeric_limits<std::size_t>::digits on every platform QuantLib targets.
_SIZE_T_DIGITS: int = 64

# std::numeric_limits<double>::digits.
_DOUBLE_DIGITS: int = 53


class StdMt19937:
    """``std::mt19937`` — the raw 32-bit engine, unwrapped.

    Delegates to :class:`MersenneTwisterUniformRng`, which implements the same
    recurrence with the same ``init_genrand`` seeding. Probe block A1 pins the
    two output sequences against each other for three seeds; the delegation is
    only sound because that assertion passes.

    Note this exposes ``__call__`` (the engine's raw 32-bit output), *not* a
    ``next_real``. Converting engine output to a real is the job of a
    distribution, and doing it the QuantLib way (``(u + 0.5) / 2^32``, one
    draw) rather than the ``<random>`` way (``generate_canonical``, two draws)
    is exactly the mistake this module exists to prevent.
    """

    __slots__ = ("_mt",)

    def __init__(self, seed: int) -> None:
        # C++ std::mt19937(0) is a perfectly ordinary seed, but PQuantLib's MT
        # maps seed 0 onto the clock-seeded SeedGenerator (mirroring
        # QuantLib's own convention), which would silently produce a different
        # stream. Refuse it rather than diverge quietly.
        qassert.require(seed != 0, "StdMt19937 requires a nonzero explicit seed")
        self._mt = MersenneTwisterUniformRng(seed)

    def __call__(self) -> int:
        """One raw engine output in ``[0, 2**32 - 1]``."""
        return self._mt.next_int32()

    @staticmethod
    def min() -> int:
        """``std::mt19937::min()``."""
        return 0

    @staticmethod
    def max() -> int:
        """``std::mt19937::max()``."""
        return _UINT32_MAX


def generate_canonical(engine: StdMt19937, bits: int = _DOUBLE_DIGITS) -> float:
    """``std::generate_canonical<double, bits>(engine)`` as libc++ implements it.

    For ``std::mt19937`` and ``bits == 53`` this consumes exactly two engine
    draws and returns ``(g0 + g1 * 2**32) / 2**64``.

    libc++::

        const size_t _Dt = numeric_limits<_RealType>::digits;   // 53
        const size_t __b = _Dt < __bits ? _Dt : __bits;
        const size_t __logR = __log2<...>::value;               // 32
        const size_t __k = __b / __logR + (__b % __logR != 0) + (__b == 0);
        const _RealType _Rp = _URNG::max() - _URNG::min() + 1;  // 2**32
        _RealType __base = _Rp;
        _RealType _Sp = __g() - _URNG::min();
        for (size_t __i = 1; __i < __k; ++__i, __base *= _Rp)
            _Sp += (__g() - _URNG::min()) * __base;
        return _Sp / __base;

    Note ``__base`` is advanced by the loop's third clause, so after the loop
    it has been multiplied ``__k - 1`` times.
    """
    b = min(_DOUBLE_DIGITS, bits)
    k = b // _ENGINE_LOG2_R + (1 if b % _ENGINE_LOG2_R else 0) + (1 if b == 0 else 0)
    base = float(_ENGINE_R)
    total = float(engine() - StdMt19937.min())
    for _ in range(1, k):
        total += float(engine() - StdMt19937.min()) * base
        base *= _ENGINE_R
    return total / base


class StdUniformRealDistribution:
    """``std::uniform_real_distribution<double>(a, b)``.

    libc++ implements ``operator()`` as
    ``(b - a) * generate_canonical<double, numeric_limits<double>::digits>(g) + a``.
    """

    __slots__ = ("_a", "_b")

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        self._a = a
        self._b = b

    def __call__(self, engine: StdMt19937) -> float:
        """Draw one variate (consumes two engine outputs)."""
        return (self._b - self._a) * generate_canonical(engine) + self._a


class StdUniformIntDistribution:
    """``std::uniform_int_distribution<std::size_t>(a, b)`` — closed range.

    This is the fiddly one. libc++ routes the draw through
    ``__independent_bits_engine``, which assembles exactly as many engine words
    as the range needs, rejecting the words that would bias the result, and
    then rejection-samples the assembled value down to ``[0, b - a]``. Both
    rejection loops matter: they change how many engine draws are consumed, so
    getting them wrong desynchronises every subsequent draw even when the
    returned values happen to look plausible.

    libc++ ``uniform_int_distribution::operator()``::

        const _UIntType _Rp = b - a + 1;
        if (_Rp == 1) return a;                        // degenerate: no draw
        const size_t _Dt = numeric_limits<_UIntType>::digits;   // 64
        if (_Rp == 0) return _Eng(g, _Dt)();           // full-width range
        size_t __w = _Dt - countl_zero(_Rp) - 1;
        if ((_Rp & (numeric_limits<_UIntType>::max() >> (_Dt - __w))) != 0) ++__w;
        _Eng __e(g, __w);
        _UIntType __u;
        do { __u = __e(); } while (__u >= _Rp);
        return __u + a;

    ``_Rp == 0`` is the wrapped-to-zero case that arises only for a full 64-bit
    range; ``Size``-typed QuantLib callers never hit it, but it is transcribed
    for completeness.
    """

    __slots__ = ("_a", "_b")

    def __init__(self, a: int = 0, b: int = _UINT32_MAX) -> None:
        self._a = a
        self._b = b

    def __call__(self, engine: StdMt19937, a: int | None = None, b: int | None = None) -> int:
        """Draw one variate in ``[a, b]``.

        ``a`` / ``b`` override the stored parameters for this call only — the
        C++ ``operator()(URNG&, const param_type&)`` overload, which
        ``ClubsTopology`` uses to reparametrise a shared distribution object.
        """
        lo = self._a if a is None else a
        hi = self._b if b is None else b
        span = (hi - lo + 1) & _UINT64_MAX
        if span == 1:
            return lo
        if span == 0:
            return _IndependentBitsEngine(engine, _SIZE_T_DIGITS)() + lo
        # w = position of the highest set bit of span, bumped by one when span
        # is not an exact power of two.
        w = span.bit_length() - 1
        if span & (_UINT64_MAX >> (_SIZE_T_DIGITS - w)):
            w += 1
        e = _IndependentBitsEngine(engine, w)
        while True:
            u = e()
            if u < span:
                return u + lo


class _IndependentBitsEngine:
    """libc++'s ``__independent_bits_engine<std::mt19937, std::uint64_t>``.

    Assembles a ``w``-bit value out of ``n`` engine words, taking ``w0`` bits
    from the first ``n0`` words and ``w0 + 1`` bits from the rest, rejecting
    any word at or above the corresponding ``y`` threshold so that the retained
    bits stay uniform.
    """

    __slots__ = ("_engine", "_mask0", "_mask1", "_n", "_n0", "_w", "_w0", "_y0", "_y1")

    def __init__(self, engine: StdMt19937, w: int) -> None:
        self._engine = engine
        self._w = w
        # __n_ = w / m + (w % m != 0), with m = log2(engine range) = 32
        n = w // _ENGINE_LOG2_R + (1 if w % _ENGINE_LOG2_R else 0)
        w0 = w // n
        y0 = (_ENGINE_R >> w0) << w0 if w0 < _SIZE_T_DIGITS else 0
        # Rebalance when the leftover mass would be rejected too often.
        if _ENGINE_R - y0 > y0 // n:
            n += 1
            w0 = w // n
            y0 = (_ENGINE_R >> w0) << w0 if w0 < _SIZE_T_DIGITS else 0
        self._n = n
        self._w0 = w0
        self._y0 = y0
        self._n0 = n - w % n
        self._y1 = (_ENGINE_R >> (w0 + 1)) << (w0 + 1) if w0 < _SIZE_T_DIGITS - 1 else 0
        self._mask0 = (_UINT32_MAX >> (_ENGINE_DIGITS - w0)) if w0 > 0 else 0
        self._mask1 = (
            _UINT32_MAX >> (_ENGINE_DIGITS - (w0 + 1))
            if w0 < _ENGINE_DIGITS - 1
            else _UINT32_MAX
        )

    def __call__(self) -> int:
        """Assemble one ``w``-bit value."""
        # C++ parity: __independent_bits_engine::__eval(true_type), the
        # _Rp != 0 branch (std::mt19937's range is 2**32, never 0).
        total = 0
        for _ in range(self._n0):
            while True:
                u = self._engine() - StdMt19937.min()
                if u < self._y0:
                    break
            total = (total << self._w0) if self._w0 < _SIZE_T_DIGITS else 0
            total += u & self._mask0
        for _ in range(self._n0, self._n):
            while True:
                u = self._engine() - StdMt19937.min()
                if u < self._y1:
                    break
            total = (total << (self._w0 + 1)) if self._w0 < _SIZE_T_DIGITS - 1 else 0
            total += u & self._mask1
        return total & _UINT64_MAX


class StdNormalDistribution:
    """``std::normal_distribution<double>(mean, stddev)``.

    libc++ uses the Marsaglia polar method, which produces variates **in
    pairs**: the first call draws until it lands inside the unit disc and
    caches the second variate, the next call returns the cache and draws
    nothing. That statefulness is observable — it decides how many engine
    outputs each call consumes — so the cache is part of the port, not an
    optimisation.

    libc++::

        if (_V_hot_) { _V_hot_ = false; _Up = _V_; }
        else {
            uniform_real_distribution<result_type> _Uni(-1, 1);
            do { __u = _Uni(g); __v = _Uni(g); __s = __u*__u + __v*__v; }
            while (__s > 1 || __s == 0);
            result_type _Fp = sqrt(-2 * log(__s) / __s);
            _V_ = __v * _Fp;
            _V_hot_ = true;
            _Up = __u * _Fp;
        }
        return _Up * stddev + mean;

    Note the cached ``_V_`` is stored *unscaled*; ``stddev``/``mean`` are
    applied on the way out of both branches.

    Floating-point contraction
    --------------------------
    ``__s`` is computed here as ``math.fma(u, u, v * v)``, not as
    ``u * u + v * v``. The reference binary is built ``-O3`` with Clang's
    default ``-ffp-contract=on``, which fuses that multiply-add into a single
    FMA, and the fused result differs from the separately-rounded one by up to
    1 ULP. That ULP is not cosmetic: it propagates through
    ``sqrt(-2 log s / s)`` into both variates of the pair, and at the disc
    boundary it can flip the ``s > 1`` rejection test and desynchronise the
    engine entirely. Probe block A4 disagrees at samples 8 and 9 for seed 5
    without the FMA and agrees on all 20 with it, which is how the contraction
    was identified.
    """

    __slots__ = ("_mean", "_stddev", "_uni", "_v", "_v_hot")

    def __init__(self, mean: float = 0.0, stddev: float = 1.0) -> None:
        self._mean = mean
        self._stddev = stddev
        self._uni = StdUniformRealDistribution(-1.0, 1.0)
        self._v: float = 0.0
        self._v_hot: bool = False

    def reset(self) -> None:
        """Drop the cached second variate (``std::normal_distribution::reset``)."""
        self._v_hot = False

    def __call__(self, engine: StdMt19937) -> float:
        """Draw one normal variate."""
        if self._v_hot:
            self._v_hot = False
            up = self._v
        else:
            while True:
                u = self._uni(engine)
                v = self._uni(engine)
                # Fused, matching the reference binary — see the class
                # docstring's "Floating-point contraction" note.
                s = math.fma(u, u, v * v)
                if s <= 1.0 and s != 0.0:
                    break
            f = math.sqrt(-2.0 * math.log(s) / s)
            self._v = v * f
            self._v_hot = True
            up = u * f
        return up * self._stddev + self._mean
