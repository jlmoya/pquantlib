"""Knuth lagged-Fibonacci uniform random number generator.

# C++ parity: ql/math/randomnumbers/knuthuniformrng.hpp +
# knuthuniformrng.cpp (v1.42.1) — QuantLib's wrapper around Knuth's
# Seminumerical Algorithms (3rd edition, Section 3.6) lagged-Fibonacci
# generator with TT=70 quality cycles and a 1009-entry shuffle buffer.

The numerical constants (KK=100, LL=37, TT=70, QUALITY=1009, ULP=2^-52)
and the floating-point ``mod_sum``/``ranf_array`` logic mirror the C++
implementation verbatim — sequences are bit-identical to the C++ probe.
"""

from __future__ import annotations

from typing import Final

from pquantlib.math.randomnumbers.random_number_generator import Sample

_KK: Final[int] = 100
_LL: Final[int] = 37
_TT: Final[int] = 70
_QUALITY: Final[int] = 1009
# 2^-52 == one ULP at 1.0 for IEEE-754 double — chosen by Knuth as the
# "uniform spacing" granularity for the lagged-Fibonacci buffer entries.
_ULP: Final[float] = (1.0 / (1 << 30)) / (1 << 22)


def _mod_sum(x: float, y: float) -> float:
    """``(x + y) - floor(x + y)`` — C++ inline ``mod_sum``."""
    return (x + y) - int(x + y)


def _is_odd(s: int) -> bool:
    """C++ inline ``is_odd``."""
    return (s & 1) != 0


class KnuthUniformRng:
    """Knuth's ranf-array lagged-Fibonacci uniform RNG.

    # C++ parity: ``KnuthUniformRng`` in
    # ql/math/randomnumbers/knuthuniformrng.{hpp,cpp} (v1.42.1).

    The buffer is filled lazily on first ``next()`` call (matching C++
    ``ranf_arr_ptr == ranf_arr_sentinel`` initial state). Seed 0 is
    rejected (the C++ version falls back to ``SeedGenerator``, which is
    deferred to a later cluster — pquantlib refuses seed 0 explicitly).
    """

    __slots__ = (
        "_ran_u",
        "_ranf_arr_buf",
        "_ranf_arr_ptr",
        "_ranf_arr_sentinel",
    )

    def __init__(self, seed: int) -> None:
        if seed == 0:
            raise ValueError(
                "KnuthUniformRng requires nonzero seed (C++ SeedGenerator clock fallback not yet ported)"
            )
        # Mutable state. C++ marks all as ``mutable``; Python plain attrs.
        self._ran_u: list[float] = [0.0] * _KK
        self._ranf_arr_buf: list[float] = [0.0] * _QUALITY
        # Initial pointers point past the end of the buffer; the first
        # ``next()`` call triggers a ``ranf_arr_cycle``.
        self._ranf_arr_ptr: int = _QUALITY
        self._ranf_arr_sentinel: int = _QUALITY
        self._ranf_start(seed)

    def _ranf_start(self, seed: int) -> None:
        # C++ parity: knuthuniformrng.cpp:36-73.
        u: list[float] = [0.0] * (_KK + _KK - 1)
        ul: list[float] = [0.0] * (_KK + _KK - 1)
        ulp = _ULP
        ss = 2.0 * ulp * ((seed & 0x3FFFFFFF) + 2)

        for j in range(_KK):
            u[j] = ss
            ul[j] = 0.0
            ss += ss
            if ss >= 1.0:
                ss -= 1.0 - 2 * ulp  # cyclic shift of 51 bits
        # j now KK; bootstrap zeros for the tail of the work arrays
        for j in range(_KK, _KK + _KK - 1):
            u[j] = 0.0
            ul[j] = 0.0
        u[1] += ulp
        ul[1] = ulp  # make u[1] (and only u[1]) "odd"

        s = seed & 0x3FFFFFFF
        t = _TT - 1
        while t != 0:
            # "square"
            for j in range(_KK - 1, 0, -1):
                ul[j + j] = ul[j]
                u[j + j] = u[j]
            for j in range(_KK + _KK - 2, _KK - _LL, -2):
                ul[_KK + _KK - 1 - j] = 0.0
                u[_KK + _KK - 1 - j] = u[j] - ul[j]
            for j in range(_KK + _KK - 2, _KK - 1, -1):
                if ul[j] != 0.0:
                    ul[j - (_KK - _LL)] = ulp - ul[j - (_KK - _LL)]
                    u[j - (_KK - _LL)] = _mod_sum(u[j - (_KK - _LL)], u[j])
                    ul[j - _KK] = ulp - ul[j - _KK]
                    u[j - _KK] = _mod_sum(u[j - _KK], u[j])
            if _is_odd(s):
                # "multiply by z" — shift the buffer cyclically by one.
                for j in range(_KK, 0, -1):
                    ul[j] = ul[j - 1]
                    u[j] = u[j - 1]
                ul[0] = ul[_KK]
                u[0] = u[_KK]
                if ul[_KK] != 0.0:
                    ul[_LL] = ulp - ul[_LL]
                    u[_LL] = _mod_sum(u[_LL], u[_KK])
            if s != 0:
                s >>= 1
            else:
                t -= 1

        for j in range(_LL):
            self._ran_u[j + _KK - _LL] = u[j]
        for j in range(_LL, _KK):
            self._ran_u[j - _LL] = u[j]

    def _ranf_array(self, aa: list[float], n: int) -> None:
        # C++ parity: knuthuniformrng.cpp:75-82.
        for j in range(_KK):
            aa[j] = self._ran_u[j]
        for j in range(_KK, n):
            aa[j] = _mod_sum(aa[j - _KK], aa[j - _LL])
        i = 0
        j = n
        for _ in range(_LL):
            self._ran_u[i] = _mod_sum(aa[j - _KK], aa[j - _LL])
            i += 1
            j += 1
        while i < _KK:
            self._ran_u[i] = _mod_sum(aa[j - _KK], self._ran_u[i - _LL])
            i += 1
            j += 1

    def _ranf_arr_cycle(self) -> float:
        # C++ parity: knuthuniformrng.cpp:84-89.
        self._ranf_array(self._ranf_arr_buf, _QUALITY)
        self._ranf_arr_ptr = 1
        self._ranf_arr_sentinel = 100
        return self._ranf_arr_buf[0]

    def next(self) -> Sample:
        """One sample uniformly drawn from (0.0, 1.0) with weight 1.0."""
        # C++ parity: knuthuniformrng.hpp:68-73.
        if self._ranf_arr_ptr != self._ranf_arr_sentinel:
            result = self._ranf_arr_buf[self._ranf_arr_ptr]
            self._ranf_arr_ptr += 1
        else:
            result = self._ranf_arr_cycle()
        return Sample(value=result, weight=1.0)

    def dimension(self) -> int:
        """Scalar RNG — dimension is always 1."""
        return 1
