"""BinomialConvertibleEngine — Tsiveriotis-Fernandes convertible engine.

# C++ parity: ql/pricingengines/bond/binomialconvertibleengine.hpp (v1.42.1)
#             — ``template <class T> class BinomialConvertibleEngine``.

Prices a :class:`ConvertibleBond` on a binomial
:class:`~pquantlib.methods.lattices.tsiveriotis_fernandes_lattice.TsiveriotisFernandesLattice`
built from a flattened (constant-coefficient) Black-Scholes process. The
process's risk-free / dividend / vol curves are sampled at the bond maturity
and re-wrapped as flat curves so the tree has constant up/down/probability
coefficients (matching the C++ engine exactly).

# C++ parity divergence — template parameter:
# C++ templates the engine on the tree type ``T`` (e.g. ``CoxRossRubinstein``).
# Python passes the tree *class* as the first ctor argument
# (``BinomialConvertibleEngine(CoxRossRubinstein, process, steps, spread)``).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from pquantlib import qassert
from pquantlib.instruments.bonds.convertible_bonds import (
    ConvertibleBondArguments,
    ConvertibleBondResults,
)
from pquantlib.methods.lattices.tsiveriotis_fernandes_lattice import (
    TsiveriotisFernandesLattice,
)
from pquantlib.pricingengines.bond.discretized_convertible import (
    DiscretizedConvertible,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.cashflows.dividend import Dividend
    from pquantlib.methods.lattices.binomial_tree import BinomialTree
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
    from pquantlib.quotes.quote import Quote


class ConvertibleTreeBuilder(Protocol):
    """Structural type for a binomial-tree class usable by the engine.

    The concrete tree classes (CRR / JarrowRudd / Tian / LeisenReimer) all
    expose a ``(process, end, steps, strike)`` ctor, so their *class* objects
    satisfy this protocol. C++ templates the engine on the tree type ``T``;
    this protocol is the Python structural equivalent.
    """

    def __call__(
        self,
        process: StochasticProcess1D,
        end: float,
        steps: int,
        strike: float,
    ) -> BinomialTree: ...


class BinomialConvertibleEngine(
    GenericEngine[ConvertibleBondArguments, ConvertibleBondResults]
):
    """Binomial Tsiveriotis-Fernandes engine for convertible bonds.

    # C++ parity: ``class BinomialConvertibleEngine<T>``.
    """

    def __init__(
        self,
        tree_class: ConvertibleTreeBuilder,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        credit_spread: Quote,
        dividends: Sequence[Dividend] | None = None,
    ) -> None:
        super().__init__(ConvertibleBondArguments(), ConvertibleBondResults())
        qassert.require(
            time_steps > 0, f"timeSteps must be positive, {time_steps} not allowed"
        )
        self._tree_class: ConvertibleTreeBuilder = tree_class
        self._process: GeneralizedBlackScholesProcess = process
        self._time_steps: int = int(time_steps)
        self._credit_spread: Quote = credit_spread
        self._dividends: list[Dividend] = list(dividends) if dividends else []
        process.register_with(self)
        credit_spread.register_with(self)

    def credit_spread(self) -> Quote:
        return self._credit_spread

    def dividends(self) -> list[Dividend]:
        return list(self._dividends)

    def calculate(self) -> None:
        # C++ parity: binomialconvertibleengine.hpp:75-131.
        results = self._results
        results.reset()
        args = self._arguments

        process = self._process
        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()
        volcal = process.black_volatility().calendar()

        s0 = process.x0()
        qassert.require(s0 > 0.0, "negative or null underlying")
        exercise = args.exercise
        assert exercise is not None
        maturity_date = exercise.last_date()
        v = process.black_volatility().black_vol(maturity_date, s0)
        risk_free_rate = (
            process.risk_free_rate()
            .zero_rate(
                maturity_date,
                Compounding.Continuous,
                Frequency.NoFrequency,
                result_day_counter=rfdc,
            )
            .rate()
        )
        q = (
            process.dividend_yield()
            .zero_rate(
                maturity_date,
                Compounding.Continuous,
                Frequency.NoFrequency,
                result_day_counter=divdc,
            )
            .rate()
        )
        reference_date = process.risk_free_rate().reference_date()

        # subtract dividends
        for dividend in self._dividends:
            if dividend.date() >= reference_date:
                s0 -= dividend.amount() * process.risk_free_rate().discount(
                    dividend.date()
                )
        qassert.require(s0 > 0.0, "negative value after subtracting dividends")

        # binomial trees with constant coefficient
        underlying: Quote = SimpleQuote(s0)
        flat_risk_free = FlatForward.from_rate(reference_date, risk_free_rate, rfdc)
        flat_dividends = FlatForward.from_rate(reference_date, q, divdc)
        flat_vol = BlackConstantVol(
            reference_date=reference_date,
            calendar=volcal,
            volatility=v,
            day_counter=voldc,
        )

        maturity = rfdc.year_fraction(args.settlement_date, maturity_date)
        assert args.redemption is not None
        assert args.conversion_ratio is not None
        strike = args.redemption / args.conversion_ratio

        bs = GeneralizedBlackScholesProcess(
            x0=underlying,
            dividend_ts=flat_dividends,
            risk_free_ts=flat_risk_free,
            black_vol_ts=flat_vol,
        )
        tree = self._tree_class(bs, maturity, self._time_steps, strike)

        credit_spread = self._credit_spread.value()

        lattice = TsiveriotisFernandesLattice(
            tree,
            risk_free_rate,
            maturity,
            self._time_steps,
            credit_spread,
            v,
            q,
        )

        convertible = DiscretizedConvertible(
            args,
            bs,
            self._dividends,
            self._credit_spread,
            TimeGrid.regular(end=maturity, steps=self._time_steps),
        )

        convertible.initialize(lattice, maturity)
        convertible.rollback(0.0)
        value = convertible.present_value()
        results.value = value
        results.settlement_value = value
        qassert.require(
            value < sys.float_info.max, "floating-point overflow on tree grid"
        )


__all__ = ["BinomialConvertibleEngine"]
