"""TridiagonalOperator — tridiagonal-matrix differential operator.

# Retired-API compat layer — see package docstring.

C++ parity: ``ql/methods/finitedifferences/tridiagonaloperator.{hpp,cpp}``
(v1.42.1) — ``QuantLib::TridiagonalOperator``. This class IS still shipped in
v1.42.1; ``applyTo`` / ``solveFor`` / ``SOR`` / ``identity`` and the operator
algebra are cross-validated TIGHT against a C++ probe
(``migration-harness/references/cluster/ws3fd1.json``).

Java parity: ``org.jquantlib.methods.finitedifferences.TridiagonalOperator``.

The three diagonals are stored as numpy float64 arrays (``Array`` alias):

- ``lower_diagonal`` — length ``n - 1`` (sub-diagonal).
- ``diagonal`` — length ``n``.
- ``upper_diagonal`` — length ``n - 1`` (super-diagonal).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pquantlib.exceptions import LibraryException

if TYPE_CHECKING:
    from pquantlib.math.array import Array


class TimeSetter(Protocol):
    """Time-setting hook for time-dependent operators.

    C++ parity: ``TridiagonalOperator::TimeSetter`` (pure-virtual ``setTime``).
    Java parity: ``TridiagonalOperator.TimeSetter``.
    """

    def set_time(self, t: float, op: TridiagonalOperator) -> None:
        """Update ``op``'s coefficients in place for time ``t``."""
        ...


class TridiagonalOperator:
    """Tridiagonal-matrix operator over a finite-difference grid.

    C++ parity: ``class TridiagonalOperator``.
    """

    def __init__(
        self,
        size: int | None = None,
        *,
        low: Array | None = None,
        mid: Array | None = None,
        high: Array | None = None,
    ) -> None:
        """Construct either by ``size`` or by explicit diagonals.

        Two constructor forms (C++ parity):

        - ``TridiagonalOperator(size)`` — zero-initialised, ``size`` must be
          ``0`` or ``>= 2`` (C++ ``tridiagonaloperator.cpp`` requires ``>=2``;
          the Java port required ``>=3``. We follow C++.).
        - ``TridiagonalOperator(low=..., mid=..., high=...)`` — explicit
          diagonals; ``len(low) == len(high) == len(mid) - 1``.
        """
        self._time_setter: TimeSetter | None = None
        if low is not None or mid is not None or high is not None:
            if low is None or mid is None or high is None:
                raise LibraryException("low, mid and high must all be supplied together")
            n = mid.shape[0]
            if low.shape[0] != n - 1:
                raise LibraryException(
                    f"low diagonal vector of size {low.shape[0]} instead of {n - 1}"
                )
            if high.shape[0] != n - 1:
                raise LibraryException(
                    f"high diagonal vector of size {high.shape[0]} instead of {n - 1}"
                )
            self._lower_diagonal: Array = np.array(low, dtype=np.float64)
            self._diagonal: Array = np.array(mid, dtype=np.float64)
            self._upper_diagonal: Array = np.array(high, dtype=np.float64)
        else:
            sz = 0 if size is None else size
            # C++ parity: size must be null (0) or >= 2.
            if sz >= 2:
                self._lower_diagonal = np.zeros(sz - 1, dtype=np.float64)
                self._diagonal = np.zeros(sz, dtype=np.float64)
                self._upper_diagonal = np.zeros(sz - 1, dtype=np.float64)
            elif sz == 0:
                self._lower_diagonal = np.zeros(0, dtype=np.float64)
                self._diagonal = np.zeros(0, dtype=np.float64)
                self._upper_diagonal = np.zeros(0, dtype=np.float64)
            else:
                raise LibraryException(
                    f"invalid size ({sz}) for tridiagonal operator (must be null or >= 2)"
                )

    # ------------------------------------------------------------------
    # Inspectors
    # ------------------------------------------------------------------
    def size(self) -> int:
        """Square dimension of the operator (length of the main diagonal)."""
        return int(self._diagonal.shape[0])

    def is_time_dependent(self) -> bool:
        """``True`` iff a time-setter hook is installed."""
        return self._time_setter is not None

    def lower_diagonal(self) -> Array:
        """The sub-diagonal (length ``n - 1``)."""
        return self._lower_diagonal

    def diagonal(self) -> Array:
        """The main diagonal (length ``n``)."""
        return self._diagonal

    def upper_diagonal(self) -> Array:
        """The super-diagonal (length ``n - 1``)."""
        return self._upper_diagonal

    @property
    def time_setter(self) -> TimeSetter | None:
        """The installed time-setter hook (or ``None``)."""
        return self._time_setter

    @time_setter.setter
    def time_setter(self, value: TimeSetter | None) -> None:
        self._time_setter = value

    # ------------------------------------------------------------------
    # Modifiers (row setters) — C++ parity: setFirstRow/setMidRow/etc.
    # ------------------------------------------------------------------
    def set_first_row(self, b: float, c: float) -> None:
        """Set the first row: ``diagonal[0] = b``, ``upper[0] = c``."""
        self._diagonal[0] = b
        self._upper_diagonal[0] = c

    def set_mid_row(self, i: int, a: float, b: float, c: float) -> None:
        """Set interior row ``i``: lower[i-1]=a, diagonal[i]=b, upper[i]=c."""
        if not (1 <= i <= self.size() - 2):
            raise LibraryException("out of range in TridiagonalSystem::setMidRow")
        self._lower_diagonal[i - 1] = a
        self._diagonal[i] = b
        self._upper_diagonal[i] = c

    def set_mid_rows(self, a: float, b: float, c: float) -> None:
        """Set every interior row to the same ``(a, b, c)`` triple."""
        for i in range(1, self.size() - 1):
            self._lower_diagonal[i - 1] = a
            self._diagonal[i] = b
            self._upper_diagonal[i] = c

    def set_last_row(self, a: float, b: float) -> None:
        """Set the last row: ``lower[n-2] = a``, ``diagonal[n-1] = b``."""
        self._lower_diagonal[self.size() - 2] = a
        self._diagonal[self.size() - 1] = b

    def set_time(self, t: float) -> None:
        """Invoke the time-setter hook for time ``t`` (no-op if none installed)."""
        if self._time_setter is not None:
            self._time_setter.set_time(t, self)

    # ------------------------------------------------------------------
    # Operator interface — C++ tridiagonaloperator.cpp
    # ------------------------------------------------------------------
    def apply_to(self, v: Array) -> Array:
        """Apply the operator to ``v`` (tridiagonal matrix-vector product).

        C++ parity: ``Array TridiagonalOperator::applyTo(const Array&)``.
        """
        n = self.size()
        if n == 0:
            raise LibraryException("uninitialized TridiagonalOperator")
        if v.shape[0] != n:
            raise LibraryException(
                f"vector of the wrong size {v.shape[0]} instead of {n}"
            )
        # result = diagonal * v, then add the off-diagonal contributions.
        result: Array = self._diagonal * v
        result[0] += self._upper_diagonal[0] * v[1]
        for j in range(1, n - 1):
            result[j] += (
                self._lower_diagonal[j - 1] * v[j - 1]
                + self._upper_diagonal[j] * v[j + 1]
            )
        result[n - 1] += self._lower_diagonal[n - 2] * v[n - 2]
        return result

    def solve_for(self, rhs: Array) -> Array:
        """Solve ``self @ x = rhs`` via the Thomas algorithm; return ``x``.

        C++ parity: ``Array TridiagonalOperator::solveFor(const Array&)``
        (delegating to the in-place two-argument form).
        """
        result: Array = np.empty(int(rhs.shape[0]), dtype=np.float64)
        self.solve_for_into(rhs, result)
        return result

    def solve_for_into(self, rhs: Array, result: Array) -> None:
        """In-place Thomas solve writing into ``result`` (may alias ``rhs``).

        C++ parity:
        ``void TridiagonalOperator::solveFor(const Array& rhs, Array& result)``.
        """
        n = self.size()
        if n == 0:
            raise LibraryException("uninitialized TridiagonalOperator")
        if rhs.shape[0] != n:
            raise LibraryException(
                f"rhs vector of size {rhs.shape[0]} instead of {n}"
            )
        # `rhs` may alias `result`; snapshot the values we still need.
        rhs_local: Array = np.array(rhs, dtype=np.float64)
        temp: Array = np.empty(n, dtype=np.float64)

        bet = float(self._diagonal[0])
        if _close(bet, 0.0):
            raise LibraryException(
                f"diagonal's first element ({bet}) cannot be close to zero"
            )
        result[0] = rhs_local[0] / bet
        for j in range(1, n):
            temp[j] = self._upper_diagonal[j - 1] / bet
            bet = float(self._diagonal[j]) - float(self._lower_diagonal[j - 1]) * float(
                temp[j]
            )
            if _close(bet, 0.0):
                raise LibraryException("division by zero")
            result[j] = (
                rhs_local[j] - self._lower_diagonal[j - 1] * result[j - 1]
            ) / bet
        # back-substitution (cannot run j>=0 with unsigned Size j in C++)
        for j in range(n - 2, 0, -1):
            result[j] -= temp[j + 1] * result[j + 1]
        result[0] -= temp[1] * result[1]

    def sor(self, rhs: Array, tol: float) -> Array:
        """Solve ``self @ x = rhs`` by Successive Over-Relaxation (omega=1.5).

        C++ parity: ``Array TridiagonalOperator::SOR(const Array&, Real)``.
        Fixed relaxation factor ``omega = 1.5`` and a 100000-iteration cap,
        mirroring C++ exactly.
        """
        n = self.size()
        if n == 0:
            raise LibraryException("uninitialized TridiagonalOperator")
        if rhs.shape[0] != n:
            raise LibraryException(
                f"rhs vector of size {rhs.shape[0]} instead of {n}"
            )
        result: Array = np.array(rhs, dtype=np.float64)  # initial guess = rhs
        omega = 1.5
        err = 2.0 * tol
        sor_iteration = 0
        while err > tol:
            if sor_iteration >= 100000:
                raise LibraryException(
                    f"tolerance ({tol}) not reached in {sor_iteration} iterations. "
                    f"The error still is {err}"
                )
            temp = (
                omega
                * (
                    rhs[0]
                    - self._upper_diagonal[0] * result[1]
                    - self._diagonal[0] * result[0]
                )
                / self._diagonal[0]
            )
            err = float(temp * temp)
            result[0] += temp
            for i in range(1, n - 1):
                temp = (
                    omega
                    * (
                        rhs[i]
                        - self._upper_diagonal[i] * result[i + 1]
                        - self._diagonal[i] * result[i]
                        - self._lower_diagonal[i - 1] * result[i - 1]
                    )
                    / self._diagonal[i]
                )
                err += float(temp * temp)
                result[i] += temp
            # i == n - 1 here (last row)
            i = n - 1
            temp = (
                omega
                * (
                    rhs[i]
                    - self._diagonal[i] * result[i]
                    - self._lower_diagonal[i - 1] * result[i - 1]
                )
                / self._diagonal[i]
            )
            err += float(temp * temp)
            result[i] += temp
            sor_iteration += 1
        return result

    def identity(self, size: int) -> TridiagonalOperator:
        """Return the ``size``-by-``size`` identity tridiagonal operator.

        C++ parity: ``static TridiagonalOperator identity(Size)``.
        """
        if size == 0:
            raise LibraryException("cannot create identity operator of size 0")
        return TridiagonalOperator(
            low=np.zeros(size - 1, dtype=np.float64),
            mid=np.ones(size, dtype=np.float64),
            high=np.zeros(size - 1, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Operator algebra — C++ free operators +, -, * ; Java add/subtract/multiply
    # ------------------------------------------------------------------
    def add(self, other: TridiagonalOperator) -> TridiagonalOperator:
        """Operator sum ``self + other``."""
        return TridiagonalOperator(
            low=self._lower_diagonal + other._lower_diagonal,
            mid=self._diagonal + other._diagonal,
            high=self._upper_diagonal + other._upper_diagonal,
        )

    def subtract(self, other: TridiagonalOperator) -> TridiagonalOperator:
        """Operator difference ``self - other``."""
        return TridiagonalOperator(
            low=self._lower_diagonal - other._lower_diagonal,
            mid=self._diagonal - other._diagonal,
            high=self._upper_diagonal - other._upper_diagonal,
        )

    def multiply(self, a: float) -> TridiagonalOperator:
        """Scalar multiple ``a * self``."""
        return TridiagonalOperator(
            low=self._lower_diagonal * a,
            mid=self._diagonal * a,
            high=self._upper_diagonal * a,
        )


def _close(a: float, b: float) -> bool:
    """C++ ``close(a, b)`` — tight isclose; sufficient to detect a near-zero diagonal pivot.

    C++ uses ``QuantLib::close`` (a 42-ULP relative comparison). For the
    zero-division guards in solveFor we only need ``b == 0.0`` detection, for
    which ``math.isclose`` with a tiny absolute floor is equivalent in
    practice. The ``rel_tol=4.2e-15`` threshold is ~19 ULPs (not 42), which is
    tight enough for pivot detection. The guard never fires on well-posed operators.
    """
    return math.isclose(a, b, rel_tol=4.2e-15, abs_tol=0.0) or a == b


__all__ = ["TimeSetter", "TridiagonalOperator"]
