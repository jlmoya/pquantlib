"""QuantoVanillaOption + the quanto results carrier.

# C++ parity: ql/instruments/quantovanillaoption.{hpp,cpp} (v1.43).

A quanto option is written on a foreign-currency asset but pays out in
domestic currency at a fixed exchange rate. Beyond the usual greeks it
therefore has three sensitivities of its own, all filled by
:class:`~pquantlib.pricingengines.quanto.quanto_engine.QuantoEngine`:

* ``qvega``   — sensitivity to the exchange-rate volatility,
* ``qrho``    — sensitivity to the foreign risk-free rate,
* ``qlambda`` — sensitivity to the underlying/exchange-rate correlation.

C++ parameterises the results carrier over the base results type::

    template<class ResultsType>
    class QuantoOptionResults : public ResultsType { ... };

and instantiates it three times — over ``OneAssetOption::results`` for
``QuantoVanillaOption``, over ``ForwardVanillaOption::results`` for
``QuantoForwardVanillaOption`` and over ``BarrierOption::results`` for
``QuantoBarrierOption``. Python cannot inherit from a type parameter, but
it does not need to: all three of those C++ typedefs resolve to the *same*
type, ``OneAssetOption::results`` (``ForwardVanillaOption::results`` is a
typedef for ``OneAssetOption::results``, forwardvanillaoption.hpp:50, and
``BarrierOption`` declares no results class of its own,
barrieroption.hpp:43). So the single concrete class below covers every
instantiation the library actually uses, and the C++ name is kept.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption, OneAssetOptionResults
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineResults


class QuantoOptionResults(OneAssetOptionResults):
    """Option results plus the three quanto greeks.

    # C++ parity: ``QuantoOptionResults<ResultsType>`` — see the module
    # docstring for why a single concrete class is enough.
    """

    def __init__(self) -> None:
        super().__init__()
        self.qvega: float | None = None
        self.qrho: float | None = None
        self.qlambda: float | None = None
        # C++ parity: ``QuantoOptionResults() { reset(); }``
        # (quantovanillaoption.hpp:35).
        self.reset()

    def reset(self) -> None:
        """Null out the base results and the three quanto greeks.

        # C++ parity: ``QuantoOptionResults::reset``
        # (quantovanillaoption.hpp:36-39) — the quanto greeks go to
        # ``Null<Real>()``, which this port spells ``None``. Note this is a
        # *different* sentinel from the 0.0 that ``setup_expired`` writes.
        """
        super().reset()
        self.qvega = None
        self.qrho = None
        self.qlambda = None


class QuantoVanillaOption(OneAssetOption):
    """Quanto version of a vanilla option.

    # C++ parity: ``class QuantoVanillaOption : public OneAssetOption``.
    """

    def __init__(self, payoff: StrikedTypePayoff, exercise: Exercise) -> None:
        super().__init__(payoff, exercise)
        self._qvega: float | None = None
        self._qrho: float | None = None
        self._qlambda: float | None = None

    # --- quanto greeks --------------------------------------------------------

    def qvega(self) -> float:
        """Sensitivity to the exchange-rate volatility."""
        self.calculate()
        qassert.require(self._qvega is not None, "exchange rate vega calculation failed")
        assert self._qvega is not None
        return self._qvega

    def qrho(self) -> float:
        """Sensitivity to the foreign risk-free rate."""
        self.calculate()
        qassert.require(self._qrho is not None, "foreign interest rate rho calculation failed")
        assert self._qrho is not None
        return self._qrho

    def qlambda(self) -> float:
        """Sensitivity to the underlying/exchange-rate correlation."""
        self.calculate()
        qassert.require(self._qlambda is not None, "quanto correlation sensitivity calculation failed")
        assert self._qlambda is not None
        return self._qlambda

    # --- Instrument plumbing --------------------------------------------------

    def is_expired(self) -> bool:
        """Return ``False``.

        # C++ parity: ``OneAssetOption::isExpired`` defers to
        # ``Settings::evaluationDate``. Until evaluation_date is wired into
        # pquantlib (deferred per L1 carve-out) this returns ``False`` so
        # the engine always runs, matching ``VanillaOption.is_expired``.
        # Tests that need an expired option subclass and override.
        """
        return False

    def fetch_results(self, results: PricingEngineResults) -> None:
        """# C++ parity: ``QuantoVanillaOption::fetchResults``."""
        super().fetch_results(results)
        qassert.require(
            isinstance(results, QuantoOptionResults),
            "no quanto results returned from pricing engine",
        )
        assert isinstance(results, QuantoOptionResults)
        self._qrho = results.qrho
        self._qvega = results.qvega
        self._qlambda = results.qlambda

    def setup_expired(self) -> None:
        """# C++ parity: ``QuantoVanillaOption::setupExpired`` — 0.0, not null."""
        super().setup_expired()
        self._qvega = 0.0
        self._qrho = 0.0
        self._qlambda = 0.0


__all__ = ["QuantoOptionResults", "QuantoVanillaOption"]
