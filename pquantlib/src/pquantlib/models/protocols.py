"""Cross-cluster runtime-checkable Protocols for models.

# C++ parity: none — this module is the pquantlib structural-typing
# surface that L4-B/C/D/E can target without dragging the full
# CalibratedModel / BlackCalibrationHelper class trees as imports.

The intent is to keep the L4 sub-clusters loosely coupled: a concrete
SwaptionHelper in L4-C only needs to know about
``CalibrationHelperProtocol``, not the full ``CalibrationHelper`` ABC;
likewise a Vasicek implementation in L4-B types its yield-curve hooks
against ``ShortRateModelProtocol`` without depending on the eventual
``ShortRateModel`` class (which itself only crystalizes once concrete
short-rate-model needs are pinned down).

Each Protocol below is ``runtime_checkable`` so ``isinstance(x,
Protocol)`` works for assertions in tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from pquantlib.math.optimization.constraint import Constraint
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.optimization_method import OptimizationMethod
    from pquantlib.models.protocols import CalibrationHelperProtocol  # self-ref for calibrate signature


@runtime_checkable
class ModelProtocol(Protocol):
    """Structural interface for anything model-shaped.

    # C++ parity: none — captures the surface that
    # ``CalibratedModel`` exposes for typing purposes.

    Subclasses of ``pquantlib.models.model.CalibratedModel`` satisfy
    this Protocol automatically.
    """

    def params(self) -> npt.NDArray[np.float64]:
        """Return the flat free-parameter vector."""
        ...

    def set_params(self, params: npt.NDArray[np.float64]) -> None:
        """Set every free parameter at once; notify observers."""
        ...

    def calibrate(
        self,
        instruments: list[CalibrationHelperProtocol],
        method: OptimizationMethod,
        end_criteria: EndCriteria,
        constraint: Constraint | None = None,
        weights: list[float] | None = None,
        fix_parameters: list[bool] | None = None,
    ) -> None:
        """Calibrate to the given instruments via ``method``."""
        ...


@runtime_checkable
class CalibrationHelperProtocol(Protocol):
    """Structural interface for any calibration-helper instrument.

    # C++ parity: ``ql/models/calibrationhelper.hpp`` ``CalibrationHelper``
    # + ``BlackCalibrationHelper`` surface.

    Concrete subclasses of ``BlackCalibrationHelper`` satisfy this
    Protocol automatically.
    """

    def market_value(self) -> float:
        """Return the market price of the instrument."""
        ...

    def model_value(self) -> float:
        """Return the model price of the instrument."""
        ...

    def calibration_error(self) -> float:
        """Return the calibration error (market vs model in the chosen norm)."""
        ...


@runtime_checkable
class ShortRateModelProtocol(Protocol):
    """Structural interface for short-rate models.

    # C++ parity: ``ql/models/shortrate/onefactormodel.hpp``
    # ``OneFactorModel`` + ``AffineModel`` surfaces, merged.

    A short-rate model exposes (at minimum) the analytic discount-bond
    pricing surface; concrete classes (Vasicek, HullWhite, BlackKarasinski)
    in L4-B will satisfy this Protocol.
    """

    def discount(self, t: float) -> float:
        """Implied discount factor at time ``t``."""
        ...

    def discount_bond(self, now: float, maturity: float, x: float) -> float:
        """Discount-bond price ``P(now, maturity)`` conditional on factor ``x``."""
        ...

    def discount_bond_option(
        self,
        option_type: int,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """Closed-form discount-bond option price.

        ``option_type`` matches the C++ ``Option::Type`` enum
        (Call=+1, Put=-1).
        """
        ...
