"""MultiAssetOption — base class for options on multiple assets.

# C++ parity: ql/instruments/multiassetoption.{hpp,cpp} (v1.42.1).

C++ defines ``MultiAssetOption`` as a ``Option`` subclass that
maintains a ``Greeks`` mixin (``delta`` / ``gamma`` / ``theta`` /
``vega`` / ``rho`` / ``dividend_rho``).  Each concrete multi-asset
exotic option (Himalaya / Everest / Pagoda / TwoAssetCorrelation /
...) subclasses ``MultiAssetOption`` and adds the option-specific
arguments.

The Python port keeps the same shape (``Option`` subclass) but
defers Greek population — engines in this cluster compute NPV
only.  Greeks for these exotic options aren't computed analytically
in C++ either (the analytic engines fill ``value`` and a couple of
Greeks; the MC engines compute ``value`` only).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class MultiAssetOptionResults(InstrumentResults):
    """Results carrier for a multi-asset option.

    # C++ parity: ``MultiAssetOption::results : Instrument::results +
    # Greeks``. Python carries ``value`` (via base) only; Greeks for
    # this cluster are not populated.
    """


class MultiAssetOption(Option):
    """Abstract base for European multi-asset options.

    # C++ parity: ``class MultiAssetOption : public Option``.

    Subclasses (HimalayaOption / EverestOption / PagodaOption /
    TwoAssetCorrelationOption) carry their own ``Arguments`` subclass
    and override ``setup_arguments`` to inject the option-specific
    parameters.
    """

    def __init__(self, payoff: Payoff, exercise: Exercise) -> None:
        super().__init__(payoff, exercise)

    def is_expired(self) -> bool:
        """Always returns ``False`` — defers to engine.

        # C++ parity: ``MultiAssetOption::isExpired`` uses
        # ``Settings::evaluationDate`` (Phase 1 carve-out — pquantlib's
        # ``Settings.evaluation_date`` engine plumbing is not used here).
        """
        return False

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
        """Pull value out of the engine results.

        # C++ parity: ``MultiAssetOption::fetchResults`` reads value +
        # Greeks; the Python port reads only ``value`` (Greeks deferred).
        """
        Instrument.fetch_results(self, results)


__all__ = ["MultiAssetOption", "MultiAssetOptionResults"]
