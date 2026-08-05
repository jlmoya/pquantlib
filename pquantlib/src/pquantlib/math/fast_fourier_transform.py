"""Radix-2 FFT.

# C++ parity: ql/math/fastfouriertransform.hpp (v1.43) — header-only, based
#             on public-domain code by Christopher Diggins.

Not ``numpy.fft``. Three things about the C++ class are part of its contract
and none of them is numpy's:

* the transform is over ``2^order`` points and the input is *scattered* into
  the output through a bit-reversal permutation, with any output slots past
  the input length left at their default-constructed value — so a shorter
  input is zero-padded at the positions bit-reversal sends it to, which is
  the same set as trailing zeros but only because the untouched slots start
  at zero;
* the twiddle factors come from a downward recurrence
  (``cs[i-1] = cs[i]^2 - sn[i]^2``) seeded once at the finest angle, not from
  a fresh ``cos``/``sin`` per level, so they carry that recurrence's error;
* ``inverse_transform`` conjugates the twiddle but does **not** divide by
  ``N``, so it is the unnormalised adjoint, not the inverse.

``order == 0`` is a legal degenerate case (a single point, copied).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert


class FastFourierTransform:
    """Radix-2 decimation-in-time FFT over ``2^order`` points.

    # C++ parity: ``class FastFourierTransform`` —
    # fastfouriertransform.hpp:38-130.
    """

    __slots__ = ("_cs", "_sn")

    def __init__(self, order: int) -> None:
        # C++ parity: fastfouriertransform.hpp:48-65.
        self._cs: list[float] = [0.0] * order
        self._sn: list[float] = [0.0] * order
        # For order == 0 the transform is over a single element and reduces to
        # a copy; no twiddle factors need to be computed.
        if order == 0:
            return
        m = 1 << order
        self._cs[order - 1] = math.cos(2 * math.pi / m)
        self._sn[order - 1] = math.sin(2 * math.pi / m)
        for i in range(order - 1, 0, -1):
            self._cs[i - 1] = self._cs[i] * self._cs[i] - self._sn[i] * self._sn[i]
            self._sn[i - 1] = 2 * self._sn[i] * self._cs[i]

    @staticmethod
    def min_order(input_size: int) -> int:
        """Smallest ``order`` whose ``2^order`` covers ``input_size``.

        # C++ parity: fastfouriertransform.hpp:41-45 — ``ceil(log(n)/ln 2)``
        # computed in floating point, so it is the floating-point ceiling and
        # not an exact integer computation.
        """
        return math.ceil(math.log(float(input_size)) / math.log(2.0))

    def output_size(self) -> int:
        # C++ parity: fastfouriertransform.hpp:67-70.
        return 1 << len(self._cs)

    def transform(self, values: Sequence[complex]) -> list[complex]:
        """Forward FFT.

        # C++ parity: ``transform`` — fastfouriertransform.hpp:74-78. C++ takes
        # a user-allocated output range; Python returns a fresh list of length
        # :meth:`output_size`.
        """
        return self._transform_impl(values, inverse=False)

    def inverse_transform(self, values: Sequence[complex]) -> list[complex]:
        """Unnormalised inverse FFT (conjugated twiddle, no 1/N).

        # C++ parity: ``inverse_transform`` — fastfouriertransform.hpp:81-85.
        """
        return self._transform_impl(values, inverse=True)

    def _transform_impl(self, values: Sequence[complex], *, inverse: bool) -> list[complex]:
        # C++ parity: ``transform_impl`` — fastfouriertransform.hpp:90-121.
        order = len(self._cs)
        n_points = 1 << order
        out: list[complex] = [0j] * n_points

        i = 0
        for i, value in enumerate(values, start=1):
            out[self._bit_reverse(i - 1, order)] = complex(value)
        qassert.require(i <= n_points, "FFT order is too small")

        for s in range(1, order + 1):
            m = 1 << s
            w = complex(1.0)
            wm = complex(self._cs[s - 1], self._sn[s - 1] if inverse else -self._sn[s - 1])
            for j in range(m // 2):
                for k in range(j, n_points, m):
                    t = w * out[k + m // 2]
                    u = out[k]
                    out[k] = u + t
                    out[k + m // 2] = u - t
                w *= wm
        return out

    @staticmethod
    def _bit_reverse(x: int, order: int) -> int:
        # C++ parity: fastfouriertransform.hpp:123-130.
        n = 0
        for _ in range(order):
            n <<= 1
            n |= x & 1
            x >>= 1
        return n


__all__ = ["FastFourierTransform"]
