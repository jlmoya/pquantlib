"""ModTripleBandLinearOp — triple-band operator with mutable band access.

# C++ parity: ql/methods/finitedifferences/operators/modtriplebandlinearop.hpp
# (v1.43).

``TripleBandLinearOp`` keeps its three bands private; the schemes only
ever rebuild them wholesale through ``axpyb`` / ``mult`` / ``add``.
``ModTripleBandLinearOp`` exposes per-entry read **and write** access to
``lower`` / ``diag`` / ``upper`` — C++ does this with reference-returning
overloads (``Real& lower(Size i)``), which Python spells as an explicit
setter pair.

Used by ``FdmSquareRootFwdOp`` and ``FdmHestonFwdOp``, which patch
individual boundary rows after building the operator with the standard
derivative operators.
"""

from __future__ import annotations

from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.triple_band_linear_op import (
    TripleBandLinearOp,
)


class ModTripleBandLinearOp(TripleBandLinearOp):
    """Triple-band operator with mutable per-entry band access.

    # C++ parity: ``class ModTripleBandLinearOp : public TripleBandLinearOp``.
    """

    def __init__(
        self,
        direction: int | None = None,
        mesher: FdmMesher | None = None,
        *,
        op: TripleBandLinearOp | None = None,
    ) -> None:
        """Build from ``(direction, mesher)`` or copy-construct from ``op``.

        # C++ parity: the two constructors —
        # ``ModTripleBandLinearOp(Size, shared_ptr<FdmMesher>)`` and the
        # explicit ``ModTripleBandLinearOp(const TripleBandLinearOp&)``.
        """
        if op is not None:
            # Copy-construct: C++ copies every band and index array.
            super().__init__(op._direction, op._mesher)
            self._lower = op._lower.copy()
            self._diag = op._diag.copy()
            self._upper = op._upper.copy()
            return
        if direction is None or mesher is None:
            raise ValueError("either (direction, mesher) or op= must be given")
        super().__init__(direction, mesher)

    def lower(self, i: int) -> float:
        """# C++ parity: ``Real lower(Size i) const``."""
        return float(self._lower[i])

    def set_lower(self, i: int, value: float) -> None:
        """# C++ parity: ``Real& lower(Size i)`` used as an lvalue."""
        self._lower[i] = value

    def diag(self, i: int) -> float:
        """# C++ parity: ``Real diag(Size i) const``."""
        return float(self._diag[i])

    def set_diag(self, i: int, value: float) -> None:
        """# C++ parity: ``Real& diag(Size i)`` used as an lvalue."""
        self._diag[i] = value

    def upper(self, i: int) -> float:
        """# C++ parity: ``Real upper(Size i) const``."""
        return float(self._upper[i])

    def set_upper(self, i: int, value: float) -> None:
        """# C++ parity: ``Real& upper(Size i)`` used as an lvalue."""
        self._upper[i] = value


__all__ = ["ModTripleBandLinearOp"]
