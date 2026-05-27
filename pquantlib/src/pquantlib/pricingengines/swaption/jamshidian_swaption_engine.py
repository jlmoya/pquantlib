"""Jamshidian decomposition swaption engine.

# C++ parity: ql/pricingengines/swaption/jamshidianswaptionengine.{hpp,cpp}
# (v1.42.1).

Jamshidian's decomposition prices a European swaption analytically
under any one-factor affine short-rate model whose ``discount_bond``
function is monotonic in the state variable (Vasicek, HullWhite,
ExtendedCIR). The technique reduces the swaption price to a portfolio
of zero-coupon bond options.

Algorithm (matches C++ jamshidianswaptionengine.cpp:57-128):

1. Solve for the critical short-rate ``r*`` such that the swap value
   at expiry equals zero (Brent solver over the model's discount-bond
   function).
2. For each fixed coupon, compute its strike = ``discountBond(maturity,
   payTime, r*) / discountBond(maturity, valueTime, r*)``.
3. Sum the discount-bond option values across coupons, weighted by
   the coupon amounts (with the nominal added to the last coupon).
4. The option type is ``Put`` for a Payer swaption, ``Call`` for a
   Receiver.

Divergences from C++:

- C++ ``GenericModelEngine<OneFactorAffineModel, ...>`` declares the
  model type explicitly. PQuantLib uses structural typing — the engine
  accepts any object satisfying the inline ``OneFactorAffineModelLike``
  Protocol surface (``discount_bond`` + ``discount_bond_option`` with
  the 5-arg signature).
- C++ uses an inner ``rStarFinder`` class. Python uses a closure.
- ``ParYieldCurve`` cash settlement is rejected (same as C++).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


@runtime_checkable
class OneFactorAffineModelLike(Protocol):
    """Structural surface needed by JamshidianSwaptionEngine.

    Wider than ``ShortRateModelProtocol`` (it requires the 5-arg
    ``discount_bond_option`` for Jamshidian's per-coupon expansion).
    The L4-B concrete classes (Vasicek / HullWhite / ExtendedCIR) will
    satisfy this Protocol when they land.
    """

    def discount(self, t: float) -> float: ...

    def discount_bond(self, now: float, maturity: float, x: float) -> float: ...

    def discount_bond_option(
        self,
        option_type: int,
        strike: float,
        maturity: float,
        value_time: float,
        bond_maturity: float,
    ) -> float:
        """5-arg discount-bond option used in Jamshidian's decomposition."""
        ...


@runtime_checkable
class TermStructureConsistentModelLike(Protocol):
    """Models that hold their own yield term structure handle.

    # C++ parity: ``TermStructureConsistentModel`` in model.hpp.
    """

    @property
    def term_structure(self) -> YieldTermStructureProtocol: ...


class JamshidianSwaptionEngine(GenericEngine[SwaptionArguments, SwaptionResults]):
    """Analytic European swaption via Jamshidian's decomposition.

    # C++ parity: ``class JamshidianSwaptionEngine`` in
    # jamshidianswaptionengine.hpp:44-65 (v1.42.1).
    """

    def __init__(
        self,
        model: OneFactorAffineModelLike,
        term_structure: YieldTermStructureProtocol | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._model: OneFactorAffineModelLike = model
        self._term_structure: YieldTermStructureProtocol | None = term_structure
        self._day_counter: DayCounter | None = day_counter

    def calculate(self) -> None:  # noqa: PLR0915 (faithful C++ port)
        # # C++ parity: jamshidianswaptionengine.cpp:57-128 (v1.42.1).
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.settlement_method != SettlementMethod.ParYieldCurve,
            "cash settled (ParYieldCurve) swaptions not priced with JamshidianSwaptionEngine",
        )
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type().name == "European",
            "cannot use the Jamshidian decomposition on exotic swaptions",
        )
        qassert.require(args.swap is not None, "swap not set")
        swap = args.swap
        assert swap is not None
        qassert.require(
            swap.spread() == 0.0,
            f"non zero spread ({swap.spread()}) not allowed",
        )
        qassert.require(
            args.nominal is not None,
            "non-constant nominals are not supported yet",
        )
        assert args.nominal is not None

        # Resolve the reference date / day-counter via the model
        # (TermStructureConsistentModel) or the engine's term_structure.
        ref_date = None
        dc: DayCounter | None = None
        if isinstance(self._model, TermStructureConsistentModelLike):
            ts = self._model.term_structure
            ref_date = ts.reference_date()
            dc = ts.day_counter()
        if ref_date is None:
            qassert.require(
                self._term_structure is not None,
                "no term_structure available; pass one explicitly or use a TermStructureConsistentModel",
            )
            assert self._term_structure is not None
            ref_date = self._term_structure.reference_date()
            dc = self._term_structure.day_counter()
        if self._day_counter is not None:
            dc = self._day_counter
        assert dc is not None

        # Build the amounts vector (last entry gets +nominal — bond-repayment).
        amounts: list[float] = list(args.fixed_coupons)
        amounts[-1] += args.nominal

        # Year-fractions. Fixed swap dates are typed as ``list[object]``
        # in SwapArguments (carrier polymorphism); narrow back to Date.
        first_reset = args.fixed_reset_dates[0]
        assert isinstance(first_reset, Date)
        maturity = dc.year_fraction(ref_date, args.exercise.date(0))
        value_time = dc.year_fraction(ref_date, first_reset)
        fixed_pay_times: list[float] = []
        for d in args.fixed_pay_dates:
            assert isinstance(d, Date)
            fixed_pay_times.append(dc.year_fraction(ref_date, d))

        # The "rStarFinder" — find the critical short-rate r* such
        # that the swap value at exercise equals the strike (=nominal).
        strike_value = args.nominal

        def r_star(x: float) -> float:
            # # C++ parity: jamshidianswaptionengine.cpp:38-48.
            value = strike_value
            b = self._model.discount_bond(maturity, value_time, x)
            for i, t in enumerate(fixed_pay_times):
                db = self._model.discount_bond(maturity, t, x) / b
                value -= amounts[i] * db
            return value

        solver = Brent()
        min_strike = -10.0
        max_strike = 10.0
        solver.set_max_evaluations(10000)
        solver.set_lower_bound(min_strike)
        solver.set_upper_bound(max_strike)
        r_star_value = solver.solve(r_star, 1e-8, 0.05, min_strike, max_strike)

        # Sign for the discount-bond option: Put for Payer, Call for Receiver.
        w = OptionType.Put if args.swap_type == SwapType.Payer else OptionType.Call

        b = self._model.discount_bond(maturity, value_time, r_star_value)
        value = 0.0
        for i, t in enumerate(fixed_pay_times):
            strike = self._model.discount_bond(maturity, t, r_star_value) / b
            dbo_value = self._model.discount_bond_option(
                int(w), strike, maturity, value_time, t
            )
            value += amounts[i] * dbo_value

        results.value = value


__all__ = [
    "JamshidianSwaptionEngine",
    "OneFactorAffineModelLike",
    "TermStructureConsistentModelLike",
]
