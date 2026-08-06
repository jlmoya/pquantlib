"""FdmSchemeDesc — scheme + theta + mu descriptor.

# C++ parity: ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp
# (``struct FdmSchemeDesc``) + .cpp (v1.42.1).

The C++ ``FdmSchemeDesc`` is a tiny PoD that pairs a scheme tag (an
enum) with two parameters: ``theta`` (the implicit weight, e.g.
0.5 for Crank-Nicolson) and ``mu`` (the second-order correction for
Craig-Sneyd / Hundsdorfer variants — not used by the schemes ported
in L5-D).

All ten C++ static factories are exposed as classmethods, with C++'s exact
constants:

* ``douglas`` — (0.5, 0.0); same as Crank-Nicolson in one dimension.
* ``crank_nicolson`` — (0.5, 0.0).
* ``implicit_euler`` / ``explicit_euler`` — (0.0, 0.0).
* ``craig_sneyd`` — (0.5, 0.5).
* ``modified_craig_sneyd`` — (1/3, 1/3).
* ``hundsdorfer`` — (0.5 + sqrt(3)/6, 0.5).
* ``modified_hundsdorfer`` — (1 - sqrt(2)/2, 0.5), tagged ``HundsdorferType``.
* ``method_of_lines(eps=0.001, rel_init_step_size=0.01)`` — the two arguments
  ride in the ``theta`` / ``mu`` slots.
* ``tr_bdf2`` — (2 - sqrt(2), 1e-8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


class FdmSchemeType(IntEnum):
    """Tag for ``FdmSchemeDesc``.

    # C++ parity: ``FdmSchemeDesc::FdmSchemeType`` declaration order.
    """

    HundsdorferType = 0
    DouglasType = 1
    CraigSneydType = 2
    ModifiedCraigSneydType = 3
    ImplicitEulerType = 4
    ExplicitEulerType = 5
    MethodOfLinesType = 6
    TrBDF2Type = 7
    CrankNicolsonType = 8


@dataclass(frozen=True, slots=True)
class FdmSchemeDesc:
    """Scheme descriptor: type tag + theta + mu.

    # C++ parity: ``struct FdmSchemeDesc``.
    """

    type: FdmSchemeType
    theta: float
    mu: float

    @classmethod
    def crank_nicolson(cls) -> FdmSchemeDesc:
        return cls(FdmSchemeType.CrankNicolsonType, 0.5, 0.0)

    @classmethod
    def douglas(cls) -> FdmSchemeDesc:
        return cls(FdmSchemeType.DouglasType, 0.5, 0.0)

    @classmethod
    def implicit_euler(cls) -> FdmSchemeDesc:
        return cls(FdmSchemeType.ImplicitEulerType, 0.0, 0.0)

    @classmethod
    def explicit_euler(cls) -> FdmSchemeDesc:
        return cls(FdmSchemeType.ExplicitEulerType, 0.0, 0.0)

    @classmethod
    def craig_sneyd(cls) -> FdmSchemeDesc:
        """# C++ parity: ``FdmSchemeDesc::CraigSneyd`` — (0.5, 0.5)."""
        return cls(FdmSchemeType.CraigSneydType, 0.5, 0.5)

    @classmethod
    def modified_craig_sneyd(cls) -> FdmSchemeDesc:
        """# C++ parity: ``FdmSchemeDesc::ModifiedCraigSneyd`` — (1/3, 1/3)."""
        return cls(FdmSchemeType.ModifiedCraigSneydType, 1.0 / 3.0, 1.0 / 3.0)

    @classmethod
    def hundsdorfer(cls) -> FdmSchemeDesc:
        """# C++ parity: ``FdmSchemeDesc::Hundsdorfer`` — (0.5 + sqrt(3)/6, 0.5)."""
        return cls(FdmSchemeType.HundsdorferType, 0.5 + math.sqrt(3.0) / 6.0, 0.5)

    @classmethod
    def modified_hundsdorfer(cls) -> FdmSchemeDesc:
        """# C++ parity: ``FdmSchemeDesc::ModifiedHundsdorfer`` — (1 - sqrt(2)/2, 0.5).

        Note C++ tags this ``HundsdorferType``, not a type of its own.
        """
        return cls(FdmSchemeType.HundsdorferType, 1.0 - math.sqrt(2.0) / 2.0, 0.5)

    @classmethod
    def method_of_lines(
        cls, eps: float = 0.001, rel_init_step_size: float = 0.01
    ) -> FdmSchemeDesc:
        """# C++ parity: ``FdmSchemeDesc::MethodOfLines(eps, relInitStepSize)``."""
        return cls(FdmSchemeType.MethodOfLinesType, eps, rel_init_step_size)

    @classmethod
    def tr_bdf2(cls) -> FdmSchemeDesc:
        """# C++ parity: ``FdmSchemeDesc::TrBDF2`` — (2 - sqrt(2), 1e-8)."""
        return cls(FdmSchemeType.TrBDF2Type, 2.0 - math.sqrt(2.0), 1e-8)


__all__ = ["FdmSchemeDesc", "FdmSchemeType"]
