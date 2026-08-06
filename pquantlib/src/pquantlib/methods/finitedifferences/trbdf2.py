"""TRBDF2 — TR-BDF2 evolver for the operator-algebra FD framework.

# C++ parity: ql/methods/finitedifferences/trbdf2.hpp (v1.43) — header-only,
# ``template <class Operator> class TRBDF2``.

See <http://ssrn.com/abstract=1648878> for the scheme itself. One step is a
trapezoidal half-step of length ``alpha*dt`` followed by a BDF2 half-step that
**reuses the trapezoidal implicit operator** — an optimisation that is only
valid for ``alpha = 2 - sqrt(2)``, which is why ``alpha`` is hard-coded rather
than a constructor argument.

This is a *different* class from
:class:`~pquantlib.methods.finitedifferences.schemes.tr_bdf2_scheme.TrBDF2Scheme`.
That one belongs to the modern ``FdmLinearOpComposite`` framework (apply /
apply_direction / solve_splitting); this one belongs to the pre-1.0 framework
that ``FiniteDifferenceModel`` drives, where the operator is a matrix-like
object supporting ``identity`` / ``apply_to`` / ``solve_for`` and the algebra
``+ - *``. The C++ headers likewise live in different directories.

Structural typing instead of a template
---------------------------------------
C++ parameterises on ``Operator``; the equivalent here is the
:class:`TrBdf2Operator` protocol, which declares exactly the member surface
``trbdf2.hpp`` uses. It is spelled as a protocol rather than an import so that
``pquantlib`` core does not depend on ``pquantlib-helpers``, which is where
this repository hosts the retired pre-1.0 FD framework (``TridiagonalOperator``
and friends) — the only conforming operator that currently ships.

Reference semantics
-------------------
C++ holds ``operator_type L_`` **by value**, so ``L_.setTime(t)`` mutates the
evolver's private copy. Python has no value semantics for objects and the
conforming operators expose no ``clone``, so the operator is held by reference
and a time-dependent operator passed here *will* be re-timed in place. Pass a
dedicated operator instance if that matters. The boundary-condition set is
likewise shared, exactly as in C++ (it holds ``shared_ptr``\\ s).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

import numpy as np

from pquantlib.math.array import Array


@runtime_checkable
class TrBdf2Operator(Protocol):
    """The member surface ``TRBDF2<Operator>`` requires of its operator.

    # C++ parity: the ``\\code`` block in the ``trbdf2.hpp`` class comment,
    # which lists ``size`` / ``setTime`` / ``applyTo`` / ``solveFor`` /
    # ``identity`` and the operator algebra. ``isTimeDependent`` is used by
    # ``step`` and belongs to the list too.
    """

    def size(self) -> int:
        """Square dimension of the operator."""
        ...

    def is_time_dependent(self) -> bool:
        """``True`` iff ``set_time`` actually changes the coefficients."""
        ...

    def set_time(self, t: float) -> None:
        """Update time-dependent coefficients to time ``t``."""
        ...

    def identity(self, size: int) -> Self:
        """The ``size``-by-``size`` identity operator.

        # C++ parity: ``static Operator identity(Size)`` — a static member
        # there, an instance method here (the conforming Python operators
        # follow the retired-framework port's convention).
        """
        ...

    def apply_to(self, v: Array) -> Array:
        """``self @ v``. # C++ parity: ``applyTo``."""
        ...

    def solve_for(self, rhs: Array) -> Array:
        """Solve ``self @ x = rhs``. # C++ parity: ``solveFor``."""
        ...

    def add(self, other: Self) -> Self:
        """``self + other``. # C++ parity: ``operator+``."""
        ...

    def subtract(self, other: Self) -> Self:
        """``self - other``. # C++ parity: ``operator-``."""
        ...

    def multiply(self, a: float) -> Self:
        """``a * self``. # C++ parity: ``operator*(Real, const Operator&)``."""
        ...


class TrBdf2BoundaryCondition[Op](Protocol):
    """Boundary condition over a :class:`TrBdf2Operator`.

    # C++ parity: ``BoundaryCondition<Operator>`` in
    # ql/methods/finitedifferences/boundarycondition.hpp, i.e. the element type
    # of ``OperatorTraits<Operator>::bc_set``.
    """

    def set_time(self, t: float) -> None:
        """# C++ parity: ``setTime(Time)``."""
        ...

    def apply_before_applying(self, op: Op) -> None:
        """# C++ parity: ``applyBeforeApplying(operator_type&)``."""
        ...

    def apply_after_applying(self, a: Array) -> None:
        """# C++ parity: ``applyAfterApplying(array_type&)``."""
        ...

    def apply_before_solving(self, op: Op, a: Array) -> None:
        """# C++ parity: ``applyBeforeSolving(operator_type&, array_type&)``."""
        ...

    def apply_after_solving(self, a: Array) -> None:
        """# C++ parity: ``applyAfterSolving(array_type&)``."""
        ...


class TRBDF2[Op: TrBdf2Operator]:
    """TR-BDF2 evolver over the pre-1.0 operator-algebra FD framework.

    # C++ parity: ``template <class Operator> class TRBDF2``.

    Warning:
        The differential operator must be linear for this evolver to work.
    """

    __slots__ = (
        "_alpha",
        "_bcs",
        "_dt",
        "_explicit_bdf2_part_full",
        "_explicit_bdf2_part_mid",
        "_explicit_trapezoidal_part",
        "_i",
        "_implicit_part",
        "_l",
    )

    def __init__(self, op: Op, bcs: Sequence[TrBdf2BoundaryCondition[Op]] = ()) -> None:
        # C++ parity: ``TRBDF2(const operator_type& L, const bc_set& bcs)``.
        # The C++ parameter is named ``L``; renamed to ``op`` here because a
        # single upper-case ``L`` is not a legal-by-lint Python identifier.
        self._l: Op = op
        self._i: Op = op.identity(op.size())
        self._dt: float = 0.0
        self._bcs: tuple[TrBdf2BoundaryCondition[Op], ...] = tuple(bcs)
        self._alpha: float = 2.0 - math.sqrt(2.0)
        # C++ leaves the four cached parts default-constructed (size-0
        # operators) until setStep is called. Seeding them with a zero-length
        # step instead keeps them independent objects, so a boundary condition
        # can never accidentally mutate ``I_`` through an alias.
        self._implicit_part: Op = self._i
        self._explicit_trapezoidal_part: Op = self._i
        self._explicit_bdf2_part_full: Op = self._i
        self._explicit_bdf2_part_mid: Op = self._i
        self.set_step(0.0)

    def set_step(self, dt: float) -> None:
        """Cache the four operator parts for a step of length ``dt``.

        # C++ parity: ``void TRBDF2::setStep(Time dt)``.
        """
        self._dt = dt
        a = self._alpha
        # C++ ``I_ + 0.5*alpha_*dt_*L_`` groups the scalars left to right.
        self._implicit_part = self._i.add(self._l.multiply(0.5 * a * dt))
        self._explicit_trapezoidal_part = self._i.subtract(self._l.multiply(0.5 * a * dt))
        self._explicit_bdf2_part_full = self._i.multiply(-(1.0 - a) * (1.0 - a) / (a * (2.0 - a)))
        self._explicit_bdf2_part_mid = self._i.multiply(1.0 / (a * (2.0 - a)))

    def step(self, a: Array, t: float) -> Array:
        """Advance ``a`` by one step ending at time ``t``; return the new array.

        # C++ parity: ``void TRBDF2::step(array_type& a, Time t)``. C++ mutates
        # its reference argument; this port returns the new array, matching the
        # rest of the Python FD package (``MixedScheme``, the Fdm schemes).
        # C++'s ``aInit_`` member is scratch that never outlives a step, so it
        # is a local here.
        """
        a_init: Array = np.array(a, dtype=np.float64, copy=True)

        for bc in self._bcs:
            bc.set_time(t)

        # --- trapezoidal explicit part -------------------------------------
        if self._l.is_time_dependent():
            self._l.set_time(t)
            self._explicit_trapezoidal_part = self._i.subtract(
                self._l.multiply(0.5 * self._alpha * self._dt)
            )
        for bc in self._bcs:
            bc.apply_before_applying(self._explicit_trapezoidal_part)
        cur: Array = self._explicit_trapezoidal_part.apply_to(a)
        for bc in self._bcs:
            bc.apply_after_applying(cur)

        # --- trapezoidal implicit part --------------------------------------
        if self._l.is_time_dependent():
            self._l.set_time(t - self._dt)
            self._implicit_part = self._i.add(self._l.multiply(0.5 * self._alpha * self._dt))
        for bc in self._bcs:
            bc.apply_before_solving(self._implicit_part, cur)
        cur = self._implicit_part.solve_for(cur)
        for bc in self._bcs:
            bc.apply_after_solving(cur)

        # --- BDF2 explicit part ---------------------------------------------
        if self._l.is_time_dependent():
            self._l.set_time(t)
        for bc in self._bcs:
            bc.apply_before_applying(self._explicit_bdf2_part_full)
        b0: Array = self._explicit_bdf2_part_full.apply_to(a_init)
        for bc in self._bcs:
            bc.apply_after_applying(b0)

        for bc in self._bcs:
            bc.apply_before_applying(self._explicit_bdf2_part_mid)
        b1: Array = self._explicit_bdf2_part_mid.apply_to(cur)
        for bc in self._bcs:
            bc.apply_after_applying(b1)
        cur = b0 + b1

        # --- reuse the implicit part (valid only for alpha = 2 - sqrt(2)) ----
        for bc in self._bcs:
            bc.apply_before_solving(self._implicit_part, cur)
        cur = self._implicit_part.solve_for(cur)
        for bc in self._bcs:
            bc.apply_after_solving(cur)

        return cur


__all__ = ["TRBDF2", "TrBdf2BoundaryCondition", "TrBdf2Operator"]
