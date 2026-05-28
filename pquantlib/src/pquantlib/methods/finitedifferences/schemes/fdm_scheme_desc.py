"""FdmSchemeDesc — scheme + theta + mu descriptor.

# C++ parity: ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp
# (``struct FdmSchemeDesc``) + .cpp (v1.42.1).

The C++ ``FdmSchemeDesc`` is a tiny PoD that pairs a scheme tag (an
enum) with two parameters: ``theta`` (the implicit weight, e.g.
0.5 for Crank-Nicolson) and ``mu`` (the second-order correction for
Craig-Sneyd / Hundsdorfer variants — not used by the schemes ported
in L5-D).

Defaults exposed (each via a classmethod factory):

* ``CrankNicolson`` — ``theta = 0.5``, ``mu = 0.0``.
* ``Douglas`` — same as CrankNicolson in 1 dimension.
* ``ImplicitEuler`` — ``theta = 0.0``, ``mu = 0.0``.
* ``ExplicitEuler`` — ``theta = 0.0``, ``mu = 0.0``.

Multi-direction descriptors (``CraigSneyd``, ``Hundsdorfer``,
``ModifiedCraigSneyd``, ``MethodOfLines``, ``TrBDF2``) are listed
in the enum but not yet implemented; calling
``FdmBackwardSolver.rollback`` with one of them raises.
"""

from __future__ import annotations

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


__all__ = ["FdmSchemeDesc", "FdmSchemeType"]
