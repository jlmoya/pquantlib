"""BaseCorrelationLossModel — base-correlation tranche loss model.

# C++ parity: ql/experimental/credit/basecorrelationlossmodel.hpp:91-290 (v1.43).

Interpolation is performed by portfolio (live) amount percentage. The tranche
[A, D] expected loss is the difference of two *equity* tranche losses, each
priced by the base model at the correlation the surface quotes for that
attachment level::

    ETL([A, D]) = ETL_base([0, D] @ rho(D)) - ETL_base([0, A] @ rho(A))

For background see chapters 19-21 of "Modelling single name and multi-name
credit derivatives", Dominic O'Kane, Wiley Finance 2008, and the JP Morgan /
Bear Stearns / Lehman / Nomura primers the C++ header lists.

Cross-validated against the ``bclm_*`` keys of
``migration-harness/references/v143/experimental/creditloss.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.credit.base_correlation_structure import (
    BaseCorrelationTermStructure,
)
from pquantlib.experimental.credit.gaussian_lhp_loss_model import GaussianLHPLossModel
from pquantlib.experimental.credit.saddlepoint_loss_model import (
    remaining_probabilities,
)
from pquantlib.quotes.simple_quote import SimpleQuote

if TYPE_CHECKING:
    from pquantlib.experimental.credit.basket import Basket
    from pquantlib.time.date import Date


class BaseCorrelationLossModel:
    """Base-correlation loss model over a scalar-correlation base model.

    # C++ parity: ``template <class BaseModel_T, class Corr2DInt_T>
    # class BaseCorrelationLossModel`` (basecorrelationlossmodel.hpp:91-158,
    # definitions :162-178 and the ``setupModels`` specialisations :191-281).

    The two C++ template parameters are the base loss model and the surface's
    2-D interpolator. The interpolator is already baked into the
    :class:`BaseCorrelationTermStructure` handed in; the base model is the
    Gaussian LHP one, i.e. this class is the C++ ``GaussianLHPFlatBCLM``
    typedef (basecorrelationlossmodel.hpp:286-290). The other three
    ``setupModels`` specialisations (Gaussian/T binomial and
    ``IHGaussPoolLossModel``, hpp:206-281) are not instantiated here.

    The C++ criticism applies verbatim: "This model is not as generic as it
    could be. In principle a default loss model dependent on a single factor
    correlation parameter is the only restriction on the base loss model(s)
    type."

    Args:
        correl_ts: the base-correlation surface.
        recoveries: one recovery rate per name.
    """

    __slots__ = (
        # ``Observable.register_with`` stores observers in a WeakSet, so a
        # __slots__ class must opt into weak references explicitly.
        "__weakref__",
        "_attach_ratio",
        "_basket",
        "_basket_attach",
        "_basket_detach",
        "_correl_ts",
        "_detach_ratio",
        "_local_correlation_attach",
        "_local_correlation_detach",
        "_recoveries",
        "_remaining_notional",
        "_scalar_correl_model_attach",
        "_scalar_correl_model_detach",
    )

    def __init__(
        self,
        correl_ts: BaseCorrelationTermStructure,
        recoveries: Sequence[float],
    ) -> None:
        # C++ parity: basecorrelationlossmodel.hpp:97-105.
        self._local_correlation_attach = SimpleQuote(0.0)
        self._local_correlation_detach = SimpleQuote(0.0)
        self._recoveries = list(recoveries)
        self._correl_ts = correl_ts
        self._basket: Basket | None = None
        self._basket_attach: Basket | None = None
        self._basket_detach: Basket | None = None
        self._scalar_correl_model_attach: GaussianLHPLossModel | None = None
        self._scalar_correl_model_detach: GaussianLHPLossModel | None = None
        self._attach_ratio: float = 0.0
        self._detach_ratio: float = 0.0
        self._remaining_notional: float = 0.0
        correl_ts.register_with(self)

    # ---- basket wiring ---------------------------------------------------

    def set_basket(self, basket: object) -> None:
        """# C++ parity: ``DefaultLossModel::setBasket`` + ``resetModel``."""
        from pquantlib.experimental.credit.basket import Basket as _Basket  # noqa: PLC0415

        qassert.require(isinstance(basket, _Basket), "not a Basket")
        assert isinstance(basket, _Basket)
        self._basket = basket
        self.reset_model()

    def basket(self) -> Basket:
        qassert.require(self._basket is not None, "No portfolio basket set.")
        assert self._basket is not None
        return self._basket

    def update(self) -> None:
        """React to a surface notification (quotes or reference date).

        # C++ parity: basecorrelationlossmodel.hpp:109-114.
        """
        self.setup_models()
        if self._basket is not None:
            self._basket.notify_observers()

    def reset_model(self) -> None:
        """Rebuild the two equity sub-baskets and their base models.

        # C++ parity: basecorrelationlossmodel.hpp:117-129.
        """
        from pquantlib.experimental.credit.basket import Basket as _Basket  # noqa: PLC0415

        basket = self.basket()
        self._remaining_notional = basket.remaining_notional()
        self._attach_ratio = (
            basket.attachment_amount() / self._remaining_notional
        )
        self._detach_ratio = (
            basket.detachment_amount() / self._remaining_notional
        )

        # # C++ parity divergence: C++ passes ``remainingNames()`` /
        # ``remainingNotionals()``; the PQuantLib ``Basket`` has no
        # defaulted-name accounting, so those are ``names()`` / ``notionals()``.
        self._basket_attach = _Basket(
            basket.ref_date(),
            basket.names(),
            basket.notionals(),
            basket.pool(),
            0.0,
            self._attach_ratio,
            basket.claim(),
        )
        self._basket_detach = _Basket(
            basket.ref_date(),
            basket.names(),
            basket.notionals(),
            basket.pool(),
            0.0,
            self._detach_ratio,
            basket.claim(),
        )
        self.setup_models()

    def setup_models(self) -> None:
        """Build the attach/detach base models on the local correlation quotes.

        # C++ parity: the ``BaseCorrelationLossModel<GaussianLHPLossModel,
        # BilinearInterpolation>::setupModels`` specialisation
        # (basecorrelationlossmodel.hpp:191-204), which is the one the
        # ``GaussianLHPFlatBCLM`` typedef instantiates.

        # C++ parity divergence: C++ constructs each ``GaussianLHPLossModel``
        # on a ``Handle<Quote>`` and assigns it to the sub-basket, so a later
        # ``setValue`` on the quote propagates through the observer graph. The
        # PQuantLib ``GaussianLHPLossModel`` takes a plain correlation and
        # exposes the basket-agnostic ``expectedTrancheLossImpl`` signature, so
        # the model is rebuilt from the quote value at each use in
        # :meth:`expected_tranche_loss` — the same value, without the observer
        # round trip.
        """
        if self._basket_attach is None:
            return
        self._scalar_correl_model_attach = self._make_base_model(
            self._local_correlation_attach.value()
        )
        self._scalar_correl_model_detach = self._make_base_model(
            self._local_correlation_detach.value()
        )

    def _make_base_model(self, correlation: float) -> GaussianLHPLossModel | None:
        """A base model at ``correlation``, or None while the quote is still 0.

        ``GaussianLHPLossModel`` requires a correlation in (0, 1); the C++
        quotes start life at 0.0 (basecorrelationlossmodel.hpp:100-101) and are
        only overwritten inside ``expectedTrancheLoss``, so at construction
        time there is no usable model yet. C++ builds one anyway because its
        constructor does not validate.
        """
        if not (0.0 < correlation < 1.0):
            return None
        # The average recovery is a per-date quantity in this model, so the
        # value handed to the constructor is irrelevant and overwritten at use.
        return GaussianLHPLossModel(correlation, self._recoveries[0])

    # ---- statistics ------------------------------------------------------

    def expected_tranche_loss(self, d: Date) -> float:
        """Expected loss on the live part of the tranche at ``d``.

        # C++ parity: basecorrelationlossmodel.hpp:162-178.

        "Most of the statistics are not implemented, not impossible but the
        model is intended for pricing rather than ptfolio risk management."
        """
        correl_k1 = self._correl_ts.correlation(d, self._attach_ratio)
        correl_k2 = self._correl_ts.correlation(d, self._detach_ratio)

        # reset correl and call base models which have the different baskets
        # associated. In C++ the ``setValue`` alone suffices because the base
        # model holds the quote by handle; here ``setup_models`` stands in for
        # that observer hop.
        self._local_correlation_attach.set_value(correl_k1)
        self._local_correlation_detach.set_value(correl_k2)
        self.setup_models()
        exp_loss_k1 = self._equity_tranche_loss(
            self._scalar_correl_model_attach, self._attach_ratio, d
        )
        exp_loss_k2 = self._equity_tranche_loss(
            self._scalar_correl_model_detach, self._detach_ratio, d
        )
        return exp_loss_k2 - exp_loss_k1

    def _equity_tranche_loss(
        self, model: GaussianLHPLossModel | None, detach_ratio: float, d: Date
    ) -> float:
        """``basketX_->expectedTrancheLoss(d)`` for the [0, detach_ratio] basket.

        # C++ parity: ``Basket::expectedTrancheLoss`` (basket.cpp:255-259)
        # forwards to ``GaussianLHPLossModel::expectedTrancheLoss(d)``
        # (gaussianlhplossmodel.hpp:97-113), which reduces to
        # ``expectedTrancheLossImpl(remainingNotional(d), averageProb(d),
        # averageRecovery(d), remainingAttachmentAmount()/remainingNotional(d),
        # remainingDetachmentAmount()/remainingNotional(d))``.

        # C++ parity divergence: PQuantLib's ``GaussianLHPLossModel`` exposes
        # exactly ``expectedTrancheLossImpl`` and is not ``Basket``-coupled, so
        # the basket read is performed here instead of inside the base model.
        # The arithmetic is unchanged.
        """
        qassert.require(
            model is not None,
            "Base correlation quote is outside (0, 1); no base model built.",
        )
        assert model is not None
        basket = self.basket()
        probs = remaining_probabilities(basket, d)
        notionals = basket.notionals()
        remaining_full_not = basket.remaining_notional()

        # # C++ parity: ``GaussianLHPLossModel::averageProb``
        # (gaussianlhplossmodel.hpp:158-166) — notional-weighted average.
        average_prob = (
            sum(probs[i] * notionals[i] for i in range(len(probs)))
            / remaining_full_not
        )
        # # C++ parity: ``GaussianLHPLossModel::averageRecovery``
        # (gaussianlhplossmodel.hpp:175-193) — weighted by notional * prob so
        # that the average and the original portfolio share an expected loss.
        denominator = sum(notionals[i] * probs[i] for i in range(len(probs)))
        if denominator == 0.0:
            average_rr = 0.0
        else:
            average_rr = (
                sum(
                    self._recoveries[i] * notionals[i] * probs[i]
                    for i in range(len(probs))
                )
                / denominator
            )

        return model.expected_tranche_loss(
            remaining_full_not, average_prob, average_rr, 0.0, detach_ratio
        )

    # ---- accessors -------------------------------------------------------

    def correlation_term_structure(self) -> BaseCorrelationTermStructure:
        return self._correl_ts

    def recoveries(self) -> list[float]:
        return list(self._recoveries)

    def attach_ratio(self) -> float:
        return self._attach_ratio

    def detach_ratio(self) -> float:
        return self._detach_ratio


# C++ parity: the "Vanilla BC model" typedef at
# basecorrelationlossmodel.hpp:286-290. Python has no template arguments to
# bind; the interpolator lives in the surface and the base model is the class
# default.
GaussianLHPFlatBCLM = BaseCorrelationLossModel


__all__ = ["BaseCorrelationLossModel", "GaussianLHPFlatBCLM"]
