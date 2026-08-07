"""FdBatesVanillaEngine — partial-integro FD Bates vanilla engine.

# C++ parity: ql/pricingengines/vanilla/fdbatesvanillaengine.{hpp,cpp} (v1.43)
# — ``class FdBatesVanillaEngine : public GenericModelEngine<BatesModel,
#    VanillaOption::arguments, VanillaOption::results>``.

The engine owns no mesher of its own: it constructs a helper
:class:`~pquantlib.pricingengines.vanilla.fd_heston_vanilla_engine.FdHestonVanillaEngine`
on the *same* Bates model (``BatesModel`` derives from ``HestonModel``), copies
its own arguments into that helper, and takes the helper's ``FdmSolverDesc``.
The Heston grid is therefore identical; only the operator differs, because
:class:`FdmBatesSolver` adds the jump integral term.

``getSolverDesc(2.0)`` — the argument is ignored by C++ (see
``FdHestonVanillaEngine.get_solver_desc``), so the equity mesh is exactly the
one the plain Heston engine would build.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_bates_solver import FdmBatesSolver
from pquantlib.models.equity.bates_model import BatesModel
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.vanilla.fd_heston_vanilla_engine import FdHestonVanillaEngine


class FdBatesVanillaEngine(
    GenericModelEngine[BatesModel, OptionArguments, OneAssetOptionResults]
):
    """Partial integro finite-differences Bates vanilla option engine.

    # C++ parity: ``class FdBatesVanillaEngine``.
    """

    def __init__(
        self,
        model: BatesModel,
        dividends: Sequence[Dividend] | None = None,
        t_grid: int = 100,
        x_grid: int = 100,
        v_grid: int = 50,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults(), model)
        self._dividends: list[Dividend] = list(dividends) if dividends else []
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._v_grid: int = v_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )

    def calculate(self) -> None:
        """# C++ parity: ``FdBatesVanillaEngine::calculate``
        # (fdbatesvanillaengine.cpp:57-83).
        """
        model = self.model()
        qassert.require(model is not None, "no model specified")
        assert model is not None

        # C++ parity: the helper engine is built on ``model_.currentLink()``,
        # i.e. the very same BatesModel, viewed as its HestonModel base.
        helper_engine = FdHestonVanillaEngine(
            model,
            self._dividends,
            None,
            self._t_grid,
            self._x_grid,
            self._v_grid,
            self._damping_steps,
            self._scheme_desc,
        )
        # C++ parity: ``*dynamic_cast<VanillaOption::arguments*>(
        #                  helperEngine.getArguments()) = arguments_;``
        helper_args = helper_engine.get_arguments()
        helper_args.payoff = self._arguments.payoff
        helper_args.exercise = self._arguments.exercise

        solver_desc = helper_engine.get_solver_desc(2.0)

        process = model.process()
        solver = FdmBatesSolver(process, solver_desc, self._scheme_desc)

        v0 = process.v0
        spot = process.s0().value()

        results = self._results
        results.value = solver.value_at(spot, v0)
        results.delta = solver.delta_at(spot, v0)
        results.gamma = solver.gamma_at(spot, v0)
        results.theta = solver.theta_at(spot, v0)


__all__ = ["FdBatesVanillaEngine"]
