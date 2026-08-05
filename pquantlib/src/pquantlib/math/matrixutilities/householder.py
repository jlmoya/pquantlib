"""Householder transformation and Householder reflection.

# C++ parity: ql/math/matrixutilities/householder.{hpp,cpp} (v1.43).

:class:`HouseholderTransformation` is the plain reflection ``x -> x - 2 (v.x) v``
for a given ``v``.

:class:`HouseholderReflection` is the interesting one: given a unit direction
``e`` it produces the ``v`` whose reflection maps ``a`` onto ``|a| e``. The
naive formula ``v = (a - |a| e) / |a - |a| e|`` cancels catastrophically as
``a`` approaches ``e``, so C++ switches on

    eps = |a2|^2 / (a.e)^2   with a1 = (a.e) e, a2 = a - a1

into three branches: return the zero vector below ``QL_EPSILON**2``, a
fourth-order series in ``eps`` below ``1e-4``, and the direct formula above it.
The series coefficients (``eps/2 - eps^2/8 + eps^3/16 - 5/128 eps^4`` over
``(a.e) sqrt(eps + eps^2/4 - eps^3/8 + 5/64 eps^4)``) are what make the small-
angle case accurate, and no numpy routine has them — this is transcribed.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix

_QL_EPSILON: float = float(np.finfo(np.float64).eps)


def _dot(v1: Array, v2: Array) -> float:
    """``DotProduct`` — C++ parity: ql/math/array.hpp:541-546."""
    return float(np.dot(v1, v2))


def _norm2(v: Array) -> float:
    """``Norm2`` — C++ parity: ql/math/array.hpp:548-550."""
    return math.sqrt(_dot(v, v))


class HouseholderTransformation:
    """The reflection ``x -> x - 2 (v.x) v``.

    # C++ parity: ``class HouseholderTransformation`` (householder.hpp:35-44,
    # householder.cpp:24-42).

    Args:
        v: the reflection vector. It need not be normalised for
            :meth:`__call__` (which reproduces the C++ formula verbatim), but
            :meth:`get_matrix` normalises it first.
    """

    __slots__ = ("_v",)

    def __init__(self, v: Array) -> None:
        self._v: Array = np.asarray(v, dtype=np.float64)

    def __call__(self, x: Array) -> Array:
        """Apply the transformation to ``x``.

        # C++ parity: ``operator()`` (householder.cpp:28-30) —
        # ``x - (2.0*DotProduct(v_, x))*v_``.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        return x_arr - (2.0 * _dot(self._v, x_arr)) * self._v

    def get_matrix(self) -> Matrix:
        """The dense ``I - 2 y y^T`` matrix with ``y = v / |v|``.

        # C++ parity: ``getMatrix()`` (householder.cpp:32-42).
        """
        y = self._v / _norm2(self._v)
        n = int(y.shape[0])

        m: Matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                m[i, j] = (1.0 if i == j else 0.0) - 2 * float(y[i]) * float(y[j])
        return m


class HouseholderReflection:
    """The reflection taking ``a`` onto ``|a| e``.

    # C++ parity: ``class HouseholderReflection`` (householder.hpp:47-56,
    # householder.cpp:44-77).

    Args:
        e: the unit target direction.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Array) -> None:
        self._e: Array = np.asarray(e, dtype=np.float64)

    def reflection_vector(self, a: Array) -> Array:
        """The normalised ``v`` for which ``H(v) a == |a| e``.

        # C++ parity: ``reflectionVector`` (householder.cpp:47-72).

        Raises:
            LibraryException: if ``a`` is the zero vector.
        """
        a_arr = np.asarray(a, dtype=np.float64)
        na = _norm2(a_arr)
        qassert.require(na > 0, "vector of length zero given")

        a_dot_e = _dot(a_arr, self._e)
        a1 = a_dot_e * self._e
        a2 = a_arr - a1

        eps = _dot(a2, a2) / (a_dot_e * a_dot_e)
        if eps < _QL_EPSILON * _QL_EPSILON:
            return np.zeros(int(a_arr.shape[0]), dtype=np.float64)
        if eps < 1e-4:
            eps2 = eps * eps
            eps3 = eps * eps2
            eps4 = eps2 * eps2
            return (a2 - a1 * (eps / 2.0 - eps2 / 8.0 + eps3 / 16.0 - 5 / 128.0 * eps4)) / (
                a_dot_e * math.sqrt(eps + eps2 / 4.0 - eps3 / 8.0 + 5 / 64.0 * eps4)
            )
        c = a_arr - na * self._e
        return c / _norm2(c)

    def __call__(self, a: Array) -> Array:
        """Reflect ``a`` onto ``|a| e``.

        # C++ parity: ``operator()`` (householder.cpp:74-77).
        """
        v = self.reflection_vector(a)
        return HouseholderTransformation(v)(np.asarray(a, dtype=np.float64))
