"""Monte-Carlo default latent models.

This module hosts four classes:

* :class:`Root` — the time-inversion target functor.
  # C++ parity: ``namespace detail { class Root; }`` in
  # ql/experimental/credit/randomdefaultlatentmodel.hpp:747-775 (v1.43).
* :class:`RandomLM` — the copula-agnostic Monte-Carlo base plus the whole
  ``DefaultLossModel`` statistics suite computed off the simulation buffer.
  # C++ parity: randomdefaultlatentmodel.hpp:94-730 (v1.43).
* :class:`RandomDefaultLM` — default-only simulation with fixed recoveries.
  # C++ parity: randomdefaultlatentmodel.hpp:805-963 (v1.43).
* :class:`RandomDefaultLatentModel` — a pre-existing one-factor convenience
  sampler. Not a port of a C++ class; see its own docstring.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.math.beta import incomplete_beta_function
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.randomnumbers.mersenne_twister import (
    MersenneTwisterUniformRng,
)
from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.math.statistics.histogram import Histogram
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.experimental.credit.basket import Basket
    from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
    from pquantlib.experimental.credit.default_probability_latent_model import (
        DefaultLatentModel,
    )
    from pquantlib.experimental.math.latent_model import CopulaPolicy
    from pquantlib.termstructures.credit.default_probability_term_structure import (
        DefaultProbabilityTermStructure,
    )


class Root:
    """Target function inverting a default-probability curve for time.

    # C++ parity: ``detail::Root`` (randomdefaultlatentmodel.hpp:749-774).

    ``Root(dts, pd)(t)`` is ``dts.default_probability(ref + int(t) days) - pd``.

    # C++ parity note: the C++ builds ``Period(static_cast<Integer>(t), Days)``,
    # so ``t`` is *truncated to whole days* and the target is a step function.
    # A 1-D solver run on it converges to somewhere inside the step containing
    # the root, which is exactly what the caller then truncates to a day count.
    # Reproduced verbatim, truncation included.
    """

    __slots__ = ("_curve_ref", "_dts", "_pd")

    def __init__(self, dts: DefaultProbabilityTermStructure, pd: float) -> None:
        self._dts = dts
        self._pd = pd
        self._curve_ref: Date = dts.reference_date()

    def __call__(self, t: float) -> float:
        qassert.require(t >= 0.0, "t < 0")
        return (
            self._dts.default_probability(
                self._curve_ref + Period(int(t), TimeUnit.Days), True
            )
            - self._pd
        )


class _FactorSampler:
    """Sobol-driven sampler of all the model's independent factors.

    # C++ parity: ``LatentModel<CP>::FactorSampler<USNG>``
    # (latentmodel.hpp:403-436) at its default ``USNG = SobolRsg``: draw a
    # ``copula.numFactors()``-dimensional Sobol point and push it through
    # ``copula.allFactorCumulInverter``.

    # Port note: the C++ class is a nested template of ``LatentModel`` in
    # ``latentmodel.hpp``; the PQuantLib port of that header does not expose it
    # yet, so the default specialisation is mirrored here for the two credit
    # simulation models that need it.
    """

    __slots__ = ("_copula", "_gen")

    def __init__(self, copula: CopulaPolicy, seed: int = 0) -> None:
        self._copula = copula
        self._gen = SobolRsg(copula.num_factors(), seed)

    def next_sequence(self) -> list[float]:
        """Next full factor draw (systemic factors first, then idiosyncratic)."""
        return self._copula.all_factor_cumul_inverter(
            [float(x) for x in self._gen.next_sequence()]
        )


class DefaultSimEvent(NamedTuple):
    """One simulated default: which name, and how many days from today.

    # C++ parity: ``simEvent<RandomDefaultLM<copulaPolicy, USNG> >``
    # (randomdefaultlatentmodel.hpp:791-800).

    # C++ parity note: C++ stores both fields in 16-bit bitfields, so a name
    # index above 65535 or a day count above 65535 silently wraps. The
    # constructor here reproduces the truncation.
    """

    name_idx: int
    day_from_ref: int

    @classmethod
    def make(cls, name_idx: int, day_from_ref: int) -> DefaultSimEvent:
        """Build one, applying C++'s 16-bit bitfield truncation."""
        return cls(name_idx & 0xFFFF, day_from_ref & 0xFFFF)


class SimEventLike(Protocol):
    """What :class:`RandomLM` needs from whichever ``simEvent`` it stores.

    # C++ parity: the members ``RandomLM`` reads off ``simEvent<derived>``
    # (randomdefaultlatentmodel.hpp:263, 296, 456). C++ gets these through the
    # ``simEvent`` template specialisation named by the CRTP parameter; the
    # two specialisations are unrelated types with different bitfield widths,
    # so the Python port needs a structural bound rather than a base class.
    """

    @property
    def name_idx(self) -> int: ...

    @property
    def day_from_ref(self) -> int: ...


class RandomLM[EventT: SimEventLike](ABC):
    """Base class for latent-model Monte-Carlo simulation.

    # C++ parity: ``template<template <class, class> class derivedRandomLM,
    # class copulaPolicy, class USNG> class RandomLM``
    # (randomdefaultlatentmodel.hpp:94-221) plus its out-of-line statistics
    # (:226-730).

    Generates the factor samples and delegates event specification to the
    derived class' :meth:`_next_sample`; then serves the whole
    ``DefaultLossModel`` statistics suite off the stored simulation buffer.

    # C++ parity divergence: C++ uses CRTP purely to dodge virtual dispatch in
    # the hot loop. Python has no such lever, so the port uses ordinary
    # abstract methods; the arithmetic is unchanged.
    """

    #: Maximum time-inversion horizon, in days (~11 years).
    #: # C++ parity: ``RandomLM::maxHorizon_`` (randomdefaultlatentmodel.hpp:219).
    MAX_HORIZON: int = 4050

    def __init__(
        self,
        num_factors: int,
        num_lm_vars: int,
        copula: CopulaPolicy,
        n_sims: int,
        seed: int,
    ) -> None:
        self._num_factors = num_factors
        self._num_lm_vars = num_lm_vars
        self._copula = copula
        self._n_sims = n_sims
        self._seed = seed
        self._basket: Basket | None = None
        self._sims_buffer: list[list[EventT]] = []
        self._calculated = False

    # ---- basket / lazy-object plumbing ----

    def basket(self) -> Basket:
        """The attached basket, or raise if none was set."""
        qassert.require(self._basket is not None, "No portfolio basket set.")
        assert self._basket is not None
        return self._basket

    def set_basket(self, basket: object) -> None:
        """Receive the owning basket and reset the model.

        # C++ parity: ``DefaultLossModel::setBasket`` (defaultlossmodel.hpp:141)
        # followed by the derived class' ``resetModel``.
        """
        # local import: Basket imports this module, so a module-level import
        # would close a cycle. C++ has no such constraint (basket.hpp is a
        # forward declaration there).
        from pquantlib.experimental.credit.basket import Basket as _Basket  # noqa: PLC0415

        qassert.require(isinstance(basket, _Basket), "not a Basket")
        assert isinstance(basket, _Basket)
        if self._basket is basket:
            return
        self._basket = basket
        self._reset_model()

    def update(self) -> None:
        """Invalidate the simulation buffer.

        # C++ parity: randomdefaultlatentmodel.hpp:108-114.
        """
        self._sims_buffer = []
        self._calculated = False

    def calculate(self) -> None:
        """Run the simulations if they are not current.

        # C++ parity: ``LazyObject::calculate`` around
        # ``RandomLM::performCalculations`` (randomdefaultlatentmodel.hpp:116-132).
        """
        if self._calculated:
            return
        self._sims_buffer = []
        self._init_dates()
        sampler = _FactorSampler(self._copula, self._seed)
        for _ in range(self._n_sims):
            self._next_sample(sampler.next_sequence())
        self._calculated = True

    def n_sims(self) -> int:
        """Number of Monte-Carlo paths."""
        return self._n_sims

    def get_sim(self, i_sim: int) -> list[EventT]:
        """Events of simulation ``i_sim``.

        # C++ parity: randomdefaultlatentmodel.hpp:140-141.
        """
        return self._sims_buffer[i_sim]

    def sims_buffer(self) -> list[list[EventT]]:
        """The whole simulation buffer (after :meth:`calculate`)."""
        return self._sims_buffer

    # ---- derived-class hooks ----

    @abstractmethod
    def _next_sample(self, values: Sequence[float]) -> None:
        """Turn one factor draw into zero or more simulation events."""

    @abstractmethod
    def _init_dates(self) -> None:
        """Pre-compute the maximum-horizon default probabilities."""

    @abstractmethod
    def _reset_model(self) -> None:
        """React to a new basket."""

    @abstractmethod
    def get_event_recovery(self, evt: EventT) -> float:
        """Recovery attached to a simulated event.

        # C++ parity: randomdefaultlatentmodel.hpp:145-150.
        """

    # ---- helpers shared by the statistics ----

    def _today(self) -> Date:
        return ObservableSettings().evaluation_date_or_today()

    def _portfolio_loss(self, events: Sequence[EventT], val: int) -> float:
        """Untranched portfolio loss of one simulation up to ``val`` days."""
        basket = self.basket()
        today = self._today()
        total = 0.0
        for evt in events:
            if val > evt.day_from_ref:
                total += basket.exposure(
                    basket.names()[evt.name_idx],
                    Date(evt.day_from_ref + today.serial_number()),
                ) * (1.0 - self.get_event_recovery(evt))
        return total

    @staticmethod
    def _tranche(loss: float, attach: float, detach: float) -> float:
        return min(max(loss - attach, 0.0), detach - attach)

    def _tranched_losses(self, d: Date) -> list[float]:
        basket = self.basket()
        attach = basket.attachment_amount()
        detach = basket.detachment_amount()
        val = d.serial_number() - self._today().serial_number()
        return [
            self._tranche(self._portfolio_loss(self.get_sim(i), val), attach, detach)
            for i in range(self._n_sims)
        ]

    # ---- statistics (DefaultLossModel interface) ----

    def prob_at_least_n_events(self, n: int, d: Date) -> float:
        """Probability of ``n`` or more defaults by ``d``.

        # C++ parity: randomdefaultlatentmodel.hpp:226-251.
        """
        self.calculate()
        today = self._today()
        qassert.require(d > today, "Date for statistic must be in the future.")
        val = d.serial_number() - today.serial_number()
        if n == 0:
            return 1.0
        counts = 0.0
        for i_sim in range(self._n_sims):
            sim_count = 0
            for evt in self.get_sim(i_sim):
                if val > evt.day_from_ref:
                    sim_count += 1
            if sim_count >= n:
                counts += 1
        return counts / self._n_sims

    def probs_being_nth_event(self, n: int, d: Date) -> list[float]:
        """Per-name probability of being the ``n``-th default by ``d``.

        # C++ parity: randomdefaultlatentmodel.hpp:253-293.
        """
        self.calculate()
        basket = self.basket()
        basket_size = basket.size()
        qassert.require(0 < n <= basket_size, "Impossible number of defaults.")
        today = self._today()
        qassert.require(d > today, "Date for statistic must be in the future.")
        val = d.serial_number() - today.serial_number()

        hits_by_date = [0.0] * basket_size
        for i_sim in range(self._n_sims):
            # C++ parity note: a std::map keyed on dayFromRef, filled with
            # `insert`, so two names defaulting on the SAME day collide and the
            # later one is dropped. Reproduced with setdefault.
            names_defaulting: dict[int, int] = {}
            for evt in self.get_sim(i_sim):
                if val > evt.day_from_ref:
                    names_defaulting.setdefault(evt.day_from_ref, evt.name_idx)
            if len(names_defaulting) >= n:
                ordered = sorted(names_defaulting.items())
                hits_by_date[ordered[n - 1][1]] += 1
        return [x / self._n_sims for x in hits_by_date]

    def default_correlation(self, d: Date, i_name: int, j_name: int) -> float:
        """Pearson correlation of the two simulated default indicators.

        # C++ parity: randomdefaultlatentmodel.hpp:296-334.
        """
        self.calculate()
        today = self._today()
        qassert.require(d > today, "Date for statistic must be in the future.")
        val = d.serial_number() - today.serial_number()

        expected_di_dj = 0.0
        expected_di = 0.0
        expected_dj = 0.0
        for i_sim in range(self._n_sims):
            imatch = 0.0
            jmatch = 0.0
            for evt in self.get_sim(i_sim):
                if val > evt.day_from_ref and evt.name_idx == i_name:
                    imatch = 1.0
                if val > evt.day_from_ref and evt.name_idx == j_name:
                    jmatch = 1.0
            expected_di_dj += imatch * jmatch
            expected_di += imatch
            expected_dj += jmatch
        # C++ parity note: E[1_i 1_j] is divided by (nSims-1) — "unbiased" —
        # while the marginals are divided by nSims. Reproduced verbatim.
        expected_di_dj = expected_di_dj / (self._n_sims - 1)
        expected_di = expected_di / self._n_sims
        expected_dj = expected_dj / self._n_sims
        return (expected_di_dj - expected_di * expected_dj) / math.sqrt(
            expected_di * expected_dj * (1.0 - expected_di) * (1.0 - expected_dj)
        )

    def expected_tranche_loss(self, d: Date) -> float:
        """Expected tranche loss at ``d``.

        # C++ parity: randomdefaultlatentmodel.hpp:337-341.
        """
        return self.expected_tranche_loss_interval(d, 0.95)[0]

    def expected_tranche_loss_interval(
        self, d: Date, confidence_perc: float
    ) -> tuple[float, float]:
        """Expected tranche loss and its confidence half-width.

        # C++ parity: randomdefaultlatentmodel.hpp:344-380.
        """
        self.calculate()
        loss_stats = GeneralStatistics()
        for x in self._tranched_losses(d):
            loss_stats.add(x)
        return (
            loss_stats.mean(),
            loss_stats.error_estimate()
            * InverseCumulativeNormal.standard_value(
                0.5 * (1.0 + confidence_perc)
            ),
        )

    def loss_distribution(self, d: Date) -> dict[float, float]:
        """Cumulative tranche-loss distribution at ``d``.

        # C++ parity: randomdefaultlatentmodel.hpp:383-397.
        """
        hist = self.compute_histogram(d)
        distrib: dict[float, float] = {}
        suma = hist.frequency(0)
        distrib[0.0] = suma
        for i in range(1, hist.bins()):
            suma += hist.frequency(i)
            # C++ inserts into a std::map, so an already-present key is kept.
            distrib.setdefault(hist.breaks()[i - 1], suma)
        return distrib

    def compute_histogram(self, d: Date) -> Histogram:
        """Histogram of the simulated tranche losses at ``d``.

        # C++ parity: randomdefaultlatentmodel.hpp:400-439.
        """
        today = self._today()
        qassert.require(
            d >= today, "Requested percentile date must lie after computation date."
        )
        self.calculate()
        data = self._tranched_losses(d)
        n_pts = min(len(data), 150)
        return Histogram(data, breaks=n_pts)

    def expected_shortfall(self, d: Date, percent: float) -> float:
        """Expected shortfall of the tranche loss at ``d``.

        # C++ parity: randomdefaultlatentmodel.hpp:442-521.
        """
        today = self._today()
        qassert.require(
            d >= today, "Requested percentile date must lie after computation date."
        )
        self.calculate()
        val = d.serial_number() - today.serial_number()
        if val <= 0:
            return 0.0
        losses = sorted(self._tranched_losses(d))
        posit = math.ceil(percent * self._n_sims)
        posit = posit if posit >= 0.0 else 0.0
        position = int(posit)
        perctl_inf = losses[position]
        prob_over_q = float(len(losses) - position) / float(self._n_sims)
        return (
            perctl_inf * (1.0 - percent - prob_over_q)
            + sum(losses[position:]) / self._n_sims
        ) / (1.0 - percent)

    def percentile(self, d: Date, percentile: float) -> float:
        """VaR of the tranche loss at ``d``.

        # C++ parity: randomdefaultlatentmodel.hpp:524-528.
        """
        return self.percentile_and_interval(d, percentile)[0]

    def percentile_and_interval(
        self, d: Date, percentile: float
    ) -> tuple[float, float, float]:
        """VaR plus the 95%-confidence interval around it.

        # C++ parity: randomdefaultlatentmodel.hpp:537-617, following
        # Appendix-A of Pritsker (1996).
        """
        qassert.require(0.0 <= percentile <= 1.0, "Incorrect percentile")
        self.calculate()
        rank_losses = sorted(self._tranched_losses(d))
        quantile_position = math.floor(self._n_sims * percentile)
        quantile_value = rank_losses[quantile_position]

        conf_interval = 0.95
        r = quantile_position - 1
        s = quantile_position + 1
        r_locked = False
        s_locked = False
        for _delta in range(1, quantile_position):
            cached = incomplete_beta_function(
                float(s), float(self._n_sims + 1 - s), percentile, 1.0e-8, 500
            )
            p_minus = (
                incomplete_beta_function(
                    float(r + 1), float(self._n_sims - r), percentile, 1.0e-8, 500
                )
                - cached
            )
            p_plus = (
                incomplete_beta_function(
                    float(r), float(self._n_sims - r + 1), percentile, 1.0e-8, 500
                )
                - cached
            )
            if p_minus > conf_interval and not r_locked:
                r_locked = True
            if p_plus >= conf_interval and not s_locked:
                s_locked = True
            if r_locked and s_locked:
                break
            r -= 1
            s += 1
            s = min(self._n_sims - 1, s)
        return quantile_value, rank_losses[r], rank_losses[s]

    def split_var_level(self, date: Date, loss: float) -> list[float]:
        """Distribute a VaR amount over the counterparties.

        # C++ parity: randomdefaultlatentmodel.hpp:620-629.
        """
        var_levels = self.split_var_and_error(date, loss, 0.95)[0]
        return [x * loss for x in var_levels]

    def split_var_and_error(
        self, date: Date, loss: float, conf_interval: float
    ) -> list[list[float]]:
        """VaR split plus its confidence band.

        # C++ parity: randomdefaultlatentmodel.hpp:633-730.

        # C++ parity note: the C++ declares ``split`` OUTSIDE the simulation
        # loop and only ``assign``s it inside the ``portfSimLoss > loss``
        # branch, so a simulation below the level re-uses the previous
        # simulation's split. The statistics are likewise only updated inside
        # that branch. Both quirks are reproduced.
        """
        self.calculate()
        basket = self.basket()
        attach = basket.attachment_amount()
        detach = basket.detachment_amount()
        num_live_names = basket.remaining_size()
        today = self._today()
        val = date.serial_number() - today.serial_number()

        split = [0.0] * num_live_names
        split_stats = [GeneralStatistics() for _ in range(num_live_names)]

        for i_sim in range(self._n_sims):
            events = self.get_sim(i_sim)
            portf_sim_loss = 0.0
            split_events_buffer: list[EventT] = []
            for evt in events:
                if val > evt.day_from_ref:
                    portf_sim_loss += basket.exposure(
                        basket.names()[evt.name_idx],
                        Date(evt.day_from_ref + today.serial_number()),
                    ) * (1.0 - self.get_event_recovery(evt))
                    split_events_buffer.append(evt)
            portf_sim_loss = self._tranche(portf_sim_loss, attach, detach)

            ptfl_cumul_loss = 0.0
            if portf_sim_loss > loss:
                split_events_buffer.sort(key=lambda e: e.day_from_ref)
                split = [0.0] * num_live_names
                for evt in split_events_buffer:
                    loss_name = basket.exposure(
                        basket.names()[evt.name_idx],
                        Date(evt.day_from_ref + today.serial_number()),
                    ) * (1.0 - self.get_event_recovery(evt))
                    tranched_before = self._tranche(ptfl_cumul_loss, attach, detach)
                    ptfl_cumul_loss += loss_name
                    tranched_after = self._tranche(ptfl_cumul_loss, attach, detach)
                    split[evt.name_idx] += tranched_after - tranched_before
                denom = self._tranche(ptfl_cumul_loss, attach, detach)
                for i_name in range(num_live_names):
                    split_stats[i_name].add(split[i_name] / denom)

        confid_factor = InverseCumulativeNormal()(0.5 + conf_interval / 2.0)
        means: list[float] = []
        range_up: list[float] = []
        range_down: list[float] = []
        for i_name in range(num_live_names):
            means.append(split_stats[i_name].mean())
            error = confid_factor * split_stats[i_name].error_estimate()
            range_down.append(means[-1] - error)
            range_up.append(means[-1] + error)
        return [means, range_down, range_up]


class RandomDefaultLM(RandomLM[DefaultSimEvent]):
    """Default-only latent-model simulation with fixed recovery amounts.

    # C++ parity: ``template<class copulaPolicy, class USNG> class RandomDefaultLM``
    # (randomdefaultlatentmodel.hpp:805-963).

    :param model: the underlying ``DefaultLatentModel``. If it is a
        :class:`~pquantlib.experimental.credit.constant_loss_latent_model.ConstantLossLatentmodel`
        and ``recoveries`` is omitted, its recovery vector is used — the C++
        second constructor.
    :param recoveries: per-name recoveries; empty means all-zero, as in C++.
    :param n_sims: number of Monte-Carlo paths.
    :param accuracy: accuracy of the Brent time inversion.
    :param seed: Sobol seed.
    """

    def __init__(
        self,
        model: DefaultLatentModel,
        recoveries: Sequence[float] | None = None,
        n_sims: int = 0,
        accuracy: float = 1.0e-6,
        seed: int = 2863311530,
    ) -> None:
        super().__init__(
            model.num_factors(), model.size(), model.copula(), n_sims, seed
        )
        if recoveries is None:
            # C++ parity: the second constructor overload
            # (randomdefaultlatentmodel.hpp:832-847) takes a
            # ``ConstantLossLatentmodel`` and seeds ``recoveries_`` from
            # ``model->recoveries()``. Python collapses the two overloads into
            # one, so the recovery-carrying model is recognised structurally;
            # an ``isinstance`` check would close an import cycle.
            getter = cast(
                "Callable[[], Sequence[float]] | None",
                getattr(model, "recoveries", None),
            )
            recoveries = getter() if getter is not None else []
        self._model: DefaultLatentModel = model
        # C++ parity: an empty recovery vector means zero recovery everywhere
        # (randomdefaultlatentmodel.hpp:826).
        self._recoveries: list[float] = (
            list(recoveries) if len(recoveries) > 0 else [0.0] * model.size()
        )
        self._accuracy = accuracy
        self._horizon_default_ps: list[float] = []

    def model(self) -> DefaultLatentModel:
        """The underlying default latent model."""
        return self._model

    def recoveries(self) -> list[float]:
        """The per-name recovery vector."""
        return self._recoveries

    # ---- RandomLM hooks ----

    def _init_dates(self) -> None:
        """# C++ parity: randomdefaultlatentmodel.hpp:863-876."""
        basket = self.basket()
        today = self._today()
        max_horizon_date = today + Period(self.MAX_HORIZON, TimeUnit.Days)
        pool = basket.pool()
        self._horizon_default_ps = [
            pool.get(pool.names()[i])
            .default_probability(basket.default_keys()[i])
            .default_probability(max_horizon_date, True)
            for i in range(basket.size())
        ]

    def get_event_recovery(self, evt: DefaultSimEvent) -> float:
        """# C++ parity: randomdefaultlatentmodel.hpp:877-879."""
        return self._recoveries[evt.name_idx]

    def expected_recovery(
        self, d: Date, i_name: int, key: DefaultProbKey
    ) -> float:
        """# C++ parity: randomdefaultlatentmodel.hpp:880-883 — deterministic."""
        del d, key
        return self._recoveries[i_name]

    def latent_var_value(self, factors_sample: Sequence[float], i_var: int) -> float:
        """# C++ parity: randomdefaultlatentmodel.hpp:885-888."""
        return self._model.latent_var_value(factors_sample, i_var)

    def basket_size(self) -> int:
        """# C++ parity: randomdefaultlatentmodel.hpp:891."""
        return self._model.size()

    def _reset_model(self) -> None:
        """# C++ parity: randomdefaultlatentmodel.hpp:893-907."""
        basket = self.basket()
        self._model.reset_basket(basket)
        qassert.require(
            basket.size() == self._model.size(),
            "Incompatible basket and model sizes.",
        )
        qassert.require(
            len(self._recoveries) == basket.size(),
            "Incompatible basket and recovery sizes.",
        )
        self.update()

    def _next_sample(self, values: Sequence[float]) -> None:
        """# C++ parity: randomdefaultlatentmodel.hpp:919-963."""
        basket = self.basket()
        pool = basket.pool()
        self._sims_buffer.append([])
        for i_name in range(self._model.size()):
            latent_var_sample = self._model.latent_var_value(values, i_name)
            sim_default_prob = self._model.cumulative_y(latent_var_sample, i_name)
            if self._horizon_default_ps[i_name] >= sim_default_prob:
                dfts = pool.get(pool.names()[i_name]).default_probability(
                    basket.default_keys()[i_name]
                )
                date_stride = int(
                    Brent().solve(Root(dfts, sim_default_prob), self._accuracy, 0.0, 1.0)
                )
                self._sims_buffer[-1].append(
                    DefaultSimEvent.make(i_name, date_stride)
                )


# ---------------------------------------------------------------------------
# Pre-existing one-factor convenience sampler (not a C++ class).
# ---------------------------------------------------------------------------


class RandomDefaultLatentModel:
    """Monte-Carlo default-event sampler.

    .. warning::

       This class is **not** a port of a C++ class. It is a one-factor
       convenience sampler over
       :class:`~pquantlib.experimental.credit.one_factor_copula.OneFactorCopula`
       driven by ``MersenneTwisterUniformRng``, taking the per-name
       unconditional default probabilities as an explicit argument.

       The faithful port of C++ ``RandomDefaultLM<copulaPolicy, USNG>`` is
       :class:`RandomDefaultLM`, above: Sobol-driven, ``Basket``-coupled, and
       simulating default *times* rather than a default indicator.

    Each draw selects:

      1. A common factor draw ``m`` from ``F_Y``.
      2. Per-name idiosyncratic draws ``z_i`` from ``F_Z``.
      3. Builds ``y_i = sqrt(rho) m + sqrt(1 - rho) z_i``.
      4. Marks name ``i`` defaulted iff ``y_i < F_Y^-1(p_i)``.
    """

    __slots__ = ("_copula", "_n", "_seed")

    def __init__(
        self,
        copula: OneFactorCopula,
        pool_size: int,
        seed: int = 42,
    ) -> None:
        qassert.require(pool_size > 0, f"pool_size must be > 0, got {pool_size}")
        self._copula = copula
        self._n = pool_size
        self._seed = seed

    def copula(self) -> OneFactorCopula:
        return self._copula

    def pool_size(self) -> int:
        return self._n

    def simulate_default_counts(
        self,
        probs: Sequence[float],
        n_paths: int,
    ) -> list[int]:
        """Return a list of ``n_paths`` default counts (0..pool_size).

        For each path: draw common factor, draw n idiosyncratic factors,
        count the number of names where Y_i < F_Y^-1(p_i).
        """
        qassert.require(
            len(probs) == self._n,
            f"probs size {len(probs)} != pool_size {self._n}",
        )
        self._copula.calculate()
        rng = MersenneTwisterUniformRng(self._seed)
        # Pre-compute the per-name default thresholds.
        thresholds = [self._copula.inverse_cumulative_y(p) for p in probs]
        rho = self._copula.correlation()
        sqrt_rho = float(np.sqrt(rho))
        sqrt_1mr = float(np.sqrt(1.0 - rho))
        counts: list[int] = [0] * n_paths
        for path in range(n_paths):
            # Draw uniform for M -> inverse_cumulative_y -> m.
            u_m = rng.next().value
            m = self._copula.inverse_cumulative_y(u_m)
            count = 0
            for i in range(self._n):
                u_z = rng.next().value
                z_i = self._copula.inverse_cumulative_y(u_z)
                y_i = sqrt_rho * m + sqrt_1mr * z_i
                if y_i < thresholds[i]:
                    count += 1
            counts[path] = count
        return counts

    def prob_at_least_n_events_mc(
        self,
        n: int,
        probs: Sequence[float],
        n_paths: int,
    ) -> float:
        """MC estimate of P(>= n defaults)."""
        counts = self.simulate_default_counts(probs, n_paths)
        return sum(1 for c in counts if c >= n) / float(n_paths)


__all__ = [
    "DefaultSimEvent",
    "RandomDefaultLM",
    "RandomDefaultLatentModel",
    "RandomLM",
    "Root",
]
