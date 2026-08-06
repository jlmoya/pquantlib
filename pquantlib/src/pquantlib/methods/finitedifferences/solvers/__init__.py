"""FD solver-desc DTO, backward time-stepping solver, and the solver family.

# C++ parity: ql/methods/finitedifferences/solvers/ (v1.43).

Three layers live here:

* ``FdmSolverDesc`` — the configuration bundle every solver consumes.
* ``FdmBackwardSolver`` — the time-stepping engine, dispatching on
  ``FdmSchemeDesc``.
* the solvers themselves — four rank-specific ones (``Fdm1DimSolver``,
  ``Fdm2DimSolver``, ``Fdm3DimSolver``, ``FdmNdimSolver``) that own the
  rollback plus the interpolation of its result, and nine model-specific
  wrappers that build the right operator and translate between market
  coordinates (spot, variance, short rate) and mesh coordinates.
"""

from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import (
    NULL_REAL,
    Fdm1DimSolver,
    snapshot_time,
)
from pquantlib.methods.finitedifferences.solvers.fdm_2d_black_scholes_solver import (
    Fdm2dBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_3dim_solver import Fdm3DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_backward_solver import FdmBackwardSolver
from pquantlib.methods.finitedifferences.solvers.fdm_bates_solver import FdmBatesSolver
from pquantlib.methods.finitedifferences.solvers.fdm_black_scholes_solver import (
    FdmBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_cir_solver import FdmCIRSolver
from pquantlib.methods.finitedifferences.solvers.fdm_g2_solver import FdmG2Solver
from pquantlib.methods.finitedifferences.solvers.fdm_heston_hull_white_solver import (
    FdmHestonHullWhiteSolver,
    HullWhiteProcessLike,
)
from pquantlib.methods.finitedifferences.solvers.fdm_heston_solver import FdmHestonSolver
from pquantlib.methods.finitedifferences.solvers.fdm_hull_white_solver import FdmHullWhiteSolver
from pquantlib.methods.finitedifferences.solvers.fdm_ndim_solver import FdmNdimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_simple_2d_bs_solver import (
    FdmSimple2dBSSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import (
    FdmSolverDesc,
    InnerValueCalculator,
)

__all__ = [
    "NULL_REAL",
    "Fdm1DimSolver",
    "Fdm2DimSolver",
    "Fdm2dBlackScholesSolver",
    "Fdm3DimSolver",
    "FdmBackwardSolver",
    "FdmBatesSolver",
    "FdmBlackScholesSolver",
    "FdmCIRSolver",
    "FdmG2Solver",
    "FdmHestonHullWhiteSolver",
    "FdmHestonSolver",
    "FdmHullWhiteSolver",
    "FdmNdimSolver",
    "FdmSimple2dBSSolver",
    "FdmSolverDesc",
    "HullWhiteProcessLike",
    "InnerValueCalculator",
    "snapshot_time",
]
