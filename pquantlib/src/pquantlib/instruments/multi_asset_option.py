"""MultiAssetOption — base class for options on multiple assets.

# C++ parity: ql/instruments/multiassetoption.{hpp,cpp} (v1.43).

C++ defines ``MultiAssetOption`` as an ``Option`` subclass whose
``results`` carrier is ``Instrument::results`` + ``Greeks``, and which
exposes ``delta`` / ``gamma`` / ``theta`` / ``vega`` / ``rho`` /
``dividendRho`` accessors that raise when the engine left the field
unset.  Each concrete multi-asset exotic option (Himalaya / Everest /
Pagoda / TwoAssetCorrelation / Margrabe / ...) subclasses
``MultiAssetOption`` and adds the option-specific arguments.

The Python port keeps that shape.  Engines that compute NPV only
simply leave the Greek fields at ``None``, and the accessors then
raise ``LibraryException`` — same observable contract as C++'s
``Null<Real>`` check.

Deliberate divergence: ``fetch_results`` does *not* hard-require a
Greeks-carrying results object the way C++'s
``QL_ENSURE(results != nullptr, "no greeks returned from pricing
engine")`` does.  Two engines in this port
(``DynProgVPPIntrinsicValueEngine``, ``FdSimpleKlugeExtOUVPPEngine``)
are declared over bare ``InstrumentResults`` even though their C++
counterparts use ``MultiAssetOption::results``; a hard check here would
break them.  When those two are re-typed, this can become a
``qassert.require``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.instrument import Instrument
from pquantlib.option import Greeks, Option, OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class MultiAssetOptionResults(Greeks):
    """Results carrier for a multi-asset option.

    # C++ parity: ``MultiAssetOption::results : Instrument::results,
    # public Greeks``. ``Greeks`` already derives from
    # ``InstrumentResults`` in this port, so a single base reproduces
    # the C++ diamond.
    """


class MultiAssetOption(Option):
    """Abstract base for multi-asset options.

    # C++ parity: ``class MultiAssetOption : public Option``.

    Subclasses (HimalayaOption / EverestOption / PagodaOption /
    TwoAssetCorrelationOption / MargrabeOption / VanillaVPPOption)
    carry their own ``Arguments`` subclass and override
    ``setup_arguments`` to inject the option-specific parameters.
    """

    def __init__(self, payoff: Payoff, exercise: Exercise) -> None:
        super().__init__(payoff, exercise)
        self._delta: float | None = None
        self._gamma: float | None = None
        self._theta: float | None = None
        self._vega: float | None = None
        self._rho: float | None = None
        self._dividend_rho: float | None = None

    def is_expired(self) -> bool:
        """Always returns ``False`` — defers to engine.

        # C++ parity: ``MultiAssetOption::isExpired`` uses
        # ``Settings::evaluationDate`` (Phase 1 carve-out — pquantlib's
        # ``Settings.evaluation_date`` engine plumbing is not used here).
        """
        return False

    # --- Greek accessors --------------------------------------------------

    def delta(self) -> float:
        """# C++ parity: ``MultiAssetOption::delta``."""
        self.calculate()
        qassert.require(self._delta is not None, "delta not provided")
        assert self._delta is not None
        return self._delta

    def gamma(self) -> float:
        """# C++ parity: ``MultiAssetOption::gamma``."""
        self.calculate()
        qassert.require(self._gamma is not None, "gamma not provided")
        assert self._gamma is not None
        return self._gamma

    def theta(self) -> float:
        """# C++ parity: ``MultiAssetOption::theta``."""
        self.calculate()
        qassert.require(self._theta is not None, "theta not provided")
        assert self._theta is not None
        return self._theta

    def vega(self) -> float:
        """# C++ parity: ``MultiAssetOption::vega``."""
        self.calculate()
        qassert.require(self._vega is not None, "vega not provided")
        assert self._vega is not None
        return self._vega

    def rho(self) -> float:
        """# C++ parity: ``MultiAssetOption::rho``."""
        self.calculate()
        qassert.require(self._rho is not None, "rho not provided")
        assert self._rho is not None
        return self._rho

    def dividend_rho(self) -> float:
        """# C++ parity: ``MultiAssetOption::dividendRho``."""
        self.calculate()
        qassert.require(self._dividend_rho is not None, "dividend rho not provided")
        assert self._dividend_rho is not None
        return self._dividend_rho

    # --- engine plumbing --------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy payoff + exercise into the engine arguments.

        # C++ parity: ``MultiAssetOption::setupArguments`` —
        # ``Option::setupArguments`` does the work.
        """
        qassert.require(
            isinstance(args, OptionArguments),
            "wrong argument type (expected OptionArguments)",
        )
        Option.setup_arguments(self, args)

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull value + Greeks out of the engine results.

        # C++ parity: ``MultiAssetOption::fetchResults`` (see the
        # module docstring for the ``QL_ENSURE`` divergence).
        """
        Instrument.fetch_results(self, results)
        if isinstance(results, Greeks):
            self._delta = results.delta
            self._gamma = results.gamma
            self._theta = results.theta
            self._vega = results.vega
            self._rho = results.rho
            self._dividend_rho = results.dividend_rho
        else:
            self._delta = None
            self._gamma = None
            self._theta = None
            self._vega = None
            self._rho = None
            self._dividend_rho = None

    def setup_expired(self) -> None:
        """Zero NPV and every Greek.

        # C++ parity: ``MultiAssetOption::setupExpired``.
        """
        super().setup_expired()
        self._delta = 0.0
        self._gamma = 0.0
        self._theta = 0.0
        self._vega = 0.0
        self._rho = 0.0
        self._dividend_rho = 0.0


__all__ = ["MultiAssetOption", "MultiAssetOptionResults"]
