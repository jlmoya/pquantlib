"""SVD — singular value decomposition (Golub-Reinsch, via JAMA/TNT).

# C++ parity: ql/math/matrixutilities/svd.{hpp,cpp} (v1.43).

**This is not ``scipy.linalg.svd`` / ``numpy.linalg.svd``.** C++ carries the
JAMA/TNT transcription of the Golub-Reinsch algorithm, and its callers depend on
conventions LAPACK's ``dgesdd``/``dgesvd`` do not guarantee:

* the **thin** decomposition is returned — for an ``m x n`` input with
  ``m >= n``, ``U`` is ``m x n``, ``V`` is ``n x n`` and there are ``n``
  singular values (LAPACK's default is the full ``m x m`` ``U``);
* for ``m < n`` the C++ decomposes ``M.T`` and **swaps the accessors**, so
  ``U()`` is ``m x m`` and ``V()`` is ``n x m``;
* the sign of each singular *vector pair* comes out of the specific sequence of
  Householder reflections and Givens rotations below, not from any
  normalisation LAPACK applies;
* :meth:`rank` uses the ``m * s[0] * eps`` threshold with a **strict** ``>``,
  and :meth:`solve_for` builds the pseudo-inverse explicitly as
  ``V @ W @ U.T``.

So the arithmetic is transcribed, in order, including the internal ``hypot``
(which is *not* ``math.hypot``: it is ``|a| * sqrt(1 + (b/a)**2)``, a different
rounding) and the ``2**-52`` deflation epsilon.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix

_QL_EPSILON: float = float(np.finfo(np.float64).eps)


def _hypot(a: float, b: float) -> float:
    """Hypotenuse of ``a`` and ``b`` avoiding under/overflow.

    # C++ parity: anonymous-namespace ``hypot`` in svd.cpp:48-55. Deliberately
    # **not** ``math.hypot``/``std::hypot``: this is
    # ``|a| * sqrt(1 + (b/a)**2)``, which rounds differently.
    """
    if a == 0:
        return abs(b)
    c = b / a
    return abs(a) * math.sqrt(1 + c * c)


class SVD:
    """Singular value decomposition of a dense matrix.

    # C++ parity: ``class SVD`` (svd.hpp:54-73, svd.cpp:60-483).

    Args:
        m: an ``(m, n)`` matrix. ``m < n`` is handled by decomposing the
            transpose and swapping :meth:`u` and :meth:`v`.
    """

    __slots__ = ("_m", "_n", "_s", "_transpose", "_u", "_v")

    def __init__(self, m: Matrix) -> None:  # noqa: PLR0915
        # C++ parity: svd.cpp:60-483.
        src = np.asarray(m, dtype=np.float64)

        # The implementation requires that rows > columns. If this is not the
        # case, we decompose M^T instead; swapping the resulting U and V gives
        # the desired result for M (see svd.cpp:66-80).
        # Both branches copy: the reduction below overwrites ``a`` in place, and
        # a transposed *view* of a single-row input is already C-contiguous, so
        # anything short of an explicit copy would clobber the caller's matrix.
        if src.shape[0] >= src.shape[1]:
            a: Matrix = src.astype(np.float64, copy=True)
            self._transpose: bool = False
        else:
            a = np.array(src.T, dtype=np.float64, order="C", copy=True)
            self._transpose = True

        m_ = int(a.shape[0])
        n_ = int(a.shape[1])
        self._m: int = m_
        self._n: int = n_

        # we're sure that m_ >= n_
        s: Array = np.zeros(n_, dtype=np.float64)
        u: Matrix = np.zeros((m_, n_), dtype=np.float64)
        v: Matrix = np.zeros((n_, n_), dtype=np.float64)
        # C++ leaves ``e`` and ``work`` uninitialised; every element read below
        # is written first, so zeros are an exact stand-in.
        e: Array = np.zeros(n_, dtype=np.float64)
        work: Array = np.zeros(m_, dtype=np.float64)

        # Reduce A to bidiagonal form, storing the diagonal elements in s and
        # the super-diagonal elements in e.
        nct = min(m_ - 1, n_)
        nrt = max(0, n_ - 2)
        for k in range(max(nct, nrt)):
            if k < nct:
                # Compute the transformation for the k-th column and place the
                # k-th diagonal in s[k]. 2-norm without under/overflow.
                s[k] = 0
                for i in range(k, m_):
                    s[k] = _hypot(float(s[k]), float(a[i, k]))
                if float(s[k]) != 0.0:
                    if float(a[k, k]) < 0.0:
                        s[k] = -float(s[k])
                    for i in range(k, m_):
                        a[i, k] /= float(s[k])
                    a[k, k] += 1.0
                s[k] = -float(s[k])
            for j in range(k + 1, n_):
                if (k < nct) and (float(s[k]) != 0.0):
                    # Apply the transformation.
                    t = 0.0
                    for i in range(k, m_):
                        t += float(a[i, k]) * float(a[i, j])
                    t = -t / float(a[k, k])
                    for i in range(k, m_):
                        a[i, j] += t * float(a[i, k])

                # Place the k-th row of A into e for the subsequent calculation
                # of the row transformation.
                e[j] = a[k, j]
            if k < nct:
                # Place the transformation in U for subsequent back
                # multiplication.
                for i in range(k, m_):
                    u[i, k] = a[i, k]
            if k < nrt:
                # Compute the k-th row transformation and place the k-th
                # super-diagonal in e[k]. 2-norm without under/overflow.
                e[k] = 0
                for i in range(k + 1, n_):
                    e[k] = _hypot(float(e[k]), float(e[i]))
                if float(e[k]) != 0.0:
                    if float(e[k + 1]) < 0.0:
                        e[k] = -float(e[k])
                    for i in range(k + 1, n_):
                        e[i] /= float(e[k])
                    e[k + 1] += 1.0
                e[k] = -float(e[k])
                if (k + 1 < m_) and (float(e[k]) != 0.0):
                    # Apply the transformation.
                    for i in range(k + 1, m_):
                        work[i] = 0.0
                    for j in range(k + 1, n_):
                        for i in range(k + 1, m_):
                            work[i] += float(e[j]) * float(a[i, j])
                    for j in range(k + 1, n_):
                        t = -float(e[j]) / float(e[k + 1])
                        for i in range(k + 1, m_):
                            a[i, j] += t * float(work[i])

                # Place the transformation in V for subsequent back
                # multiplication.
                for i in range(k + 1, n_):
                    v[i, k] = e[i]

        # Set up the final bidiagonal matrix of order n.
        if nct < n_:
            s[nct] = a[nct, nct]
        if nrt + 1 < n_:
            e[nrt] = a[nrt, n_ - 1]
        e[n_ - 1] = 0.0

        # generate U
        for j in range(nct, n_):
            for i in range(m_):
                u[i, j] = 0.0
            u[j, j] = 1.0
        for k in range(nct - 1, -1, -1):
            if float(s[k]) != 0.0:
                for j in range(k + 1, n_):
                    t = 0.0
                    for i in range(k, m_):
                        t += float(u[i, k]) * float(u[i, j])
                    t = -t / float(u[k, k])
                    for i in range(k, m_):
                        u[i, j] += t * float(u[i, k])
                for i in range(k, m_):
                    u[i, k] = -float(u[i, k])
                u[k, k] = 1.0 + float(u[k, k])
                for i in range(k - 1):
                    u[i, k] = 0.0
            else:
                for i in range(m_):
                    u[i, k] = 0.0
                u[k, k] = 1.0

        # generate V
        for k in range(n_ - 1, -1, -1):
            if (k < nrt) and (float(e[k]) != 0.0):
                for j in range(k + 1, n_):
                    t = 0.0
                    for i in range(k + 1, n_):
                        t += float(v[i, k]) * float(v[i, j])
                    t = -t / float(v[k + 1, k])
                    for i in range(k + 1, n_):
                        v[i, j] += t * float(v[i, k])
            for i in range(n_):
                v[i, k] = 0.0
            v[k, k] = 1.0

        # Main iteration loop for the singular values.
        p = n_
        pp = p - 1
        eps = math.pow(2.0, -52.0)
        while p > 0:
            # This section of the program inspects for negligible elements in
            # the s and e arrays. On completion the variables kase and k are
            # set as follows.
            #   kase = 1  if s(p) and e[k-1] are negligible and k<p
            #   kase = 2  if s(k) is negligible and k<p
            #   kase = 3  if e[k-1] is negligible, k<p, and s(k), ..., s(p) are
            #             not negligible (qr step)
            #   kase = 4  if e(p-1) is negligible (convergence)
            k = p - 2
            while k >= -1:
                if k == -1:
                    break
                if abs(float(e[k])) <= eps * (abs(float(s[k])) + abs(float(s[k + 1]))):
                    e[k] = 0.0
                    break
                k -= 1
            if k == p - 2:
                kase = 4
            else:
                ks = p - 1
                while ks >= k:
                    if ks == k:
                        break
                    t = (abs(float(e[ks])) if ks != p else 0.0) + (
                        abs(float(e[ks - 1])) if ks != k + 1 else 0.0
                    )
                    if abs(float(s[ks])) <= eps * t:
                        s[ks] = 0.0
                        break
                    ks -= 1
                if ks == k:
                    kase = 3
                elif ks == p - 1:
                    kase = 1
                else:
                    kase = 2
                    k = ks
            k += 1

            # Perform the task indicated by kase.
            if kase == 1:
                # Deflate negligible s(p).
                f = float(e[p - 2])
                e[p - 2] = 0.0
                for j in range(p - 2, k - 1, -1):
                    t = _hypot(float(s[j]), f)
                    cs = float(s[j]) / t
                    sn = f / t
                    s[j] = t
                    if j != k:
                        f = -sn * float(e[j - 1])
                        e[j - 1] = cs * float(e[j - 1])
                    for i in range(n_):
                        t = cs * float(v[i, j]) + sn * float(v[i, p - 1])
                        v[i, p - 1] = -sn * float(v[i, j]) + cs * float(v[i, p - 1])
                        v[i, j] = t
            elif kase == 2:
                # Split at negligible s(k).
                f = float(e[k - 1])
                e[k - 1] = 0.0
                for j in range(k, p):
                    t = _hypot(float(s[j]), f)
                    cs = float(s[j]) / t
                    sn = f / t
                    s[j] = t
                    f = -sn * float(e[j])
                    e[j] = cs * float(e[j])
                    for i in range(m_):
                        t = cs * float(u[i, j]) + sn * float(u[i, k - 1])
                        u[i, k - 1] = -sn * float(u[i, j]) + cs * float(u[i, k - 1])
                        u[i, j] = t
            elif kase == 3:
                # Perform one qr step: calculate the shift.
                scale = max(
                    abs(float(s[p - 1])),
                    abs(float(s[p - 2])),
                    abs(float(e[p - 2])),
                    abs(float(s[k])),
                    abs(float(e[k])),
                )
                sp = float(s[p - 1]) / scale
                spm1 = float(s[p - 2]) / scale
                epm1 = float(e[p - 2]) / scale
                sk = float(s[k]) / scale
                ek = float(e[k]) / scale
                b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0
                c = (sp * epm1) * (sp * epm1)
                shift = 0.0
                if (b != 0.0) or (c != 0.0):
                    shift = math.sqrt(b * b + c)
                    if b < 0.0:
                        shift = -shift
                    shift = c / (b + shift)
                f = (sk + sp) * (sk - sp) + shift
                g = sk * ek

                # Chase zeros.
                for j in range(k, p - 1):
                    t = _hypot(f, g)
                    cs = f / t
                    sn = g / t
                    if j != k:
                        e[j - 1] = t
                    f = cs * float(s[j]) + sn * float(e[j])
                    e[j] = cs * float(e[j]) - sn * float(s[j])
                    g = sn * float(s[j + 1])
                    s[j + 1] = cs * float(s[j + 1])
                    for i in range(n_):
                        t = cs * float(v[i, j]) + sn * float(v[i, j + 1])
                        v[i, j + 1] = -sn * float(v[i, j]) + cs * float(v[i, j + 1])
                        v[i, j] = t
                    t = _hypot(f, g)
                    cs = f / t
                    sn = g / t
                    s[j] = t
                    f = cs * float(e[j]) + sn * float(s[j + 1])
                    s[j + 1] = -sn * float(e[j]) + cs * float(s[j + 1])
                    g = sn * float(e[j + 1])
                    e[j + 1] = cs * float(e[j + 1])
                    if j < m_ - 1:
                        for i in range(m_):
                            t = cs * float(u[i, j]) + sn * float(u[i, j + 1])
                            u[i, j + 1] = -sn * float(u[i, j]) + cs * float(u[i, j + 1])
                            u[i, j] = t
                e[p - 2] = f
            else:
                # Convergence: make the singular values positive.
                if float(s[k]) <= 0.0:
                    s[k] = -float(s[k]) if float(s[k]) < 0.0 else 0.0
                    for i in range(pp + 1):
                        v[i, k] = -float(v[i, k])

                # Order the singular values.
                while k < pp:
                    if float(s[k]) >= float(s[k + 1]):
                        break
                    s[k], s[k + 1] = float(s[k + 1]), float(s[k])
                    if k < n_ - 1:
                        for i in range(n_):
                            v[i, k], v[i, k + 1] = float(v[i, k + 1]), float(v[i, k])
                    if k < m_ - 1:
                        for i in range(m_):
                            u[i, k], u[i, k + 1] = float(u[i, k + 1]), float(u[i, k])
                    k += 1
                p -= 1

        self._s: Array = s
        self._u: Matrix = u
        self._v: Matrix = v

    def u(self) -> Matrix:
        """Left singular vectors as columns.

        # C++ parity: ``SVD::U()`` (svd.cpp:485-487) — returns the *internal*
        # ``V_`` when the input was decomposed transposed.
        """
        return self._v if self._transpose else self._u

    def v(self) -> Matrix:
        """Right singular vectors as columns.

        # C++ parity: ``SVD::V()`` (svd.cpp:489-491).
        """
        return self._u if self._transpose else self._v

    def singular_values(self) -> Array:
        """The ``min(m, n)`` singular values, descending.

        # C++ parity: ``SVD::singularValues()`` (svd.cpp:493-495).
        """
        return self._s

    def s(self) -> Matrix:
        """The singular values on the diagonal of an ``n x n`` matrix.

        # C++ parity: ``SVD::S()`` (svd.cpp:497-506).
        """
        out: Matrix = np.zeros((self._n, self._n), dtype=np.float64)
        for i in range(self._n):
            out[i, i] = self._s[i]
        return out

    def norm2(self) -> float:
        """The 2-norm, i.e. the largest singular value.

        # C++ parity: ``SVD::norm2()`` (svd.cpp:508-510).
        """
        return float(self._s[0])

    def cond(self) -> float:
        """The 2-norm condition number ``s[0] / s[n-1]``.

        # C++ parity: ``SVD::cond()`` (svd.cpp:512-514). Returns ``inf`` for a
        # numerically singular matrix, exactly as the C++ division does —
        # Python's float division would raise ``ZeroDivisionError`` instead, so
        # the division is routed through numpy to keep IEEE-754 semantics.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.float64(self._s[0]) / np.float64(self._s[self._n - 1]))

    def rank(self) -> int:
        """Number of singular values strictly above ``m * s[0] * eps``.

        # C++ parity: ``SVD::rank()`` (svd.cpp:516-526).
        """
        tol = self._m * float(self._s[0]) * _QL_EPSILON
        r = 0
        for value in self._s:
            if float(value) > tol:
                r += 1
        return r

    def solve_for(self, b: Array) -> Array:
        """Least-squares / pseudo-inverse solve of ``M x = b``.

        # C++ parity: ``SVD::solveFor`` (svd.cpp:528-536) — builds the
        # pseudo-inverse ``V @ W @ U.T`` explicitly, with ``W`` the reciprocal
        # of the first :meth:`rank` singular values and zero elsewhere.
        """
        w: Matrix = np.zeros((self._n, self._n), dtype=np.float64)
        numerical_rank = self.rank()
        for i in range(numerical_rank):
            w[i, i] = 1.0 / float(self._s[i])

        inverse: Matrix = self.v() @ w @ self.u().T
        return inverse @ np.asarray(b, dtype=np.float64)
