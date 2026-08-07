"""Monte-Carlo spot-recovery loss latent models.

This module hosts three classes:

* :class:`RandomLossLM` — the faithful port of the C++ class template
  ``RandomLossLM<copulaPolicy, USNG>``.
  # C++ parity: ql/experimental/credit/randomlosslatentmodel.hpp:70-229 (v1.43).
* :class:`LossSimEvent` — its simulation-event trait, bit-packing included.
  # C++ parity: randomlosslatentmodel.hpp:36-66 (v1.43).
* :class:`RandomLossLatentModel` — a pre-existing one-factor convenience
  sampler. Not a port of a C++ class; see its own docstring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.experimental.credit.random_default_latent_model import (
    RandomLM,
    Root,
)
from pquantlib.math.randomnumbers.mersenne_twister import (
    MersenneTwisterUniformRng,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.experimental.credit.spot_loss_latent_model import (
        SpotRecoveryLatentModel,
    )

#: Recovery-rate quantisation step of the packed simulation event.
#: # C++ parity: ``simEvent<RandomLossLM<C, G> >::rrGranular``
#: (randomlosslatentmodel.hpp:58, 63-64) — ``1./256.``, i.e. 2^-8.
RR_GRANULAR: float = 1.0 / 256.0


class LossSimEvent:
    """One simulated default with its (quantised) realised recovery.

    # C++ parity: ``simEvent<RandomLossLM<copulaPolicy, USNG> >``
    # (randomlosslatentmodel.hpp:36-59).

    # C++ parity note: C++ packs the event into bitfields —
    # ``nameIdx : 12``, ``dayFromRef : 12``, ``compactRR : 8`` — and stores
    # ``compactRR = std::lround(r / rrGranular)``. Three consequences are
    # reproduced verbatim:
    #
    #  * a name index above 4095 wraps;
    #  * a default beyond 4095 days (~11.2 years) wraps, even though
    #    ``RandomLM::maxHorizon_`` is 4050 days, so the field is only just
    #    wide enough;
    #  * a recovery of exactly 1.0 rounds to 256, which does NOT fit in 8
    #    bits and wraps to 0 — ``recovery()`` then returns 0.0, the opposite
    #    of what was sampled. Recoveries below ``rrGranular/2`` likewise
    #    collapse to 0.
    """

    __slots__ = ("_compact_rr", "day_from_ref", "name_idx")

    def __init__(self, name_idx: int, day_from_ref: int, r: float) -> None:
        self.name_idx: int = name_idx & 0xFFF
        self.day_from_ref: int = day_from_ref & 0xFFF
        # C++ std::lround: round half away from zero, then truncate to 8 bits.
        self._compact_rr: int = math.floor(r / RR_GRANULAR + 0.5) & 0xFF

    def recovery(self) -> float:
        """The de-quantised recovery.

        # C++ parity: randomlosslatentmodel.hpp:51-57.
        """
        return RR_GRANULAR * self._compact_rr

    def __repr__(self) -> str:
        return (
            f"LossSimEvent(name_idx={self.name_idx}, "
            f"day_from_ref={self.day_from_ref}, recovery={self.recovery()})"
        )


class RandomLossLM(RandomLM[LossSimEvent]):
    """Random spot-recovery-rate loss model simulation for an arbitrary copula.

    # C++ parity: ``template<class copulaPolicy, class USNG> class RandomLossLM``
    # (randomlosslatentmodel.hpp:70-229).

    :param copula: the :class:`SpotRecoveryLatentModel` driving both the
        default and the recovery latent variables (``2N`` variables).
    :param n_sims: number of Monte-Carlo paths.
    :param accuracy: accuracy of the Brent time inversion.
    :param seed: Sobol seed.
    """

    def __init__(
        self,
        copula: SpotRecoveryLatentModel,
        n_sims: int = 0,
        accuracy: float = 1.0e-6,
        seed: int = 2863311530,
    ) -> None:
        super().__init__(
            copula.num_factors(), copula.size(), copula.copula(), n_sims, seed
        )
        self._spot_model = copula
        self._accuracy = accuracy
        self._horizon_default_ps: list[float] = []

    def spot_model(self) -> SpotRecoveryLatentModel:
        """The underlying spot-recovery latent model."""
        return self._spot_model

    # ---- RandomLM hooks ----

    def _init_dates(self) -> None:
        """# C++ parity: randomlosslatentmodel.hpp:107-120."""
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

    def get_event_recovery(self, evt: object) -> float:
        """# C++ parity: randomlosslatentmodel.hpp:121-123."""
        assert isinstance(evt, LossSimEvent)
        return evt.recovery()

    def latent_var_value(self, factors_sample: Sequence[float], i_var: int) -> float:
        """# C++ parity: randomlosslatentmodel.hpp:125-128."""
        return self._spot_model.latent_var_value(factors_sample, i_var)

    def basket_size(self) -> int:
        """# C++ parity: randomlosslatentmodel.hpp:129."""
        return self.basket().size()

    def conditional_recovery(
        self, latent_var_sample: float, i_name: int, d: Date
    ) -> float:
        """# C++ parity: randomlosslatentmodel.hpp:131-132 — declared but never
        # defined in v1.43 (a tree-wide grep finds no
        # ``RandomLossLM<C, URNG>::conditionalRecovery`` definition), so any
        # C++ caller fails to link. ``nextSample`` calls the *copula's*
        # ``conditionalRecovery``, which is what this forwards to.
        """
        return self._spot_model.conditional_recovery(latent_var_sample, i_name, d)

    def _reset_model(self) -> None:
        """# C++ parity: randomlosslatentmodel.hpp:134-146."""
        basket = self.basket()
        self._spot_model.reset_basket(basket)
        qassert.require(
            2 * basket.size() == self._spot_model.size(),
            "Incompatible basket and model sizes.",
        )
        self.update()

    def _next_sample(self, values: Sequence[float]) -> None:
        """# C++ parity: randomlosslatentmodel.hpp:156-229."""
        basket = self.basket()
        pool = basket.pool()
        self._sims_buffer.append([])
        today = self._today()
        # C++ parity note: "half the model is defaults, the other half are RRs"
        # — the sample vector is longer than the default half needs, and the
        # trailing idiosyncratic values are only read once a default happens.
        for i_name in range(self._spot_model.size() // 2):
            latent_var_sample = self._spot_model.latent_var_value(values, i_name)
            sim_default_prob = self._spot_model.cumulative_y(
                latent_var_sample, i_name
            )
            if self._horizon_default_ps[i_name] >= sim_default_prob:
                dfts = pool.get(pool.names()[i_name]).default_probability(
                    basket.default_keys()[i_name]
                )
                date_stride = int(
                    Brent().solve(
                        Root(dfts, sim_default_prob), self._accuracy, 0.0, 1.0
                    )
                )
                # C++ parity: the event date is clamped up to the curve
                # reference date so a negative-time probability is never
                # requested (randomlosslatentmodel.hpp:198-209).
                event_date = max(
                    today + Period(int(date_stride), TimeUnit.Days),
                    dfts.reference_date(),
                )
                latent_rr_var_sample = self._spot_model.latent_rr_var_value(
                    values, i_name
                )
                recovery = self._spot_model.conditional_recovery(
                    latent_rr_var_sample, i_name, event_date
                )
                self._sims_buffer[-1].append(
                    LossSimEvent(i_name, date_stride, recovery)
                )


# ---------------------------------------------------------------------------
# Pre-existing one-factor convenience sampler (not a C++ class).
# ---------------------------------------------------------------------------


class _ConditionalRecoveryDraw(Protocol):
    """Callable returning (random_recovery_value) given (i_name, m, u_rr) ∈ [0,1]."""

    def __call__(self, i_name: int, m: float, u_rr: float) -> float: ...


class RandomLossLatentModel:
    """Monte-Carlo loss-event sampler — extends RandomDefault with LGD draws.

    .. warning::

       This class is **not** a port of a C++ class. It is a one-factor
       convenience sampler over
       :class:`~pquantlib.experimental.credit.one_factor_copula.OneFactorCopula`
       driven by ``MersenneTwisterUniformRng``, with a caller-supplied
       recovery draw.

       The faithful port of C++ ``RandomLossLM<copulaPolicy, USNG>`` is
       :class:`RandomLossLM`, above: Sobol-driven, ``Basket``-coupled, and
       drawing its recoveries from a
       :class:`~pquantlib.experimental.credit.spot_loss_latent_model.SpotRecoveryLatentModel`.

    The recovery_draw callable should map (i_name, m, u_rr ∈ [0,1]) to a
    recovery value in [0, 1]. The model passes it the uniform draw used
    for that path's RR variable so the caller can pick its own
    distribution (constant ignores u_rr; spot uses
    ``CumulativeNormalDistribution`` with the model A noise; users may
    provide arbitrary functionals).

    For convenience the module ships ``constant_recovery_draw`` which
    closes over a per-name recovery vector and ignores u_rr.
    """

    __slots__ = ("_copula", "_n", "_notionals", "_recovery_draw", "_seed")

    def __init__(
        self,
        copula: OneFactorCopula,
        notionals: Sequence[float],
        recovery_draw: _ConditionalRecoveryDraw,
        seed: int = 42,
    ) -> None:
        qassert.require(
            len(notionals) > 0,
            "notionals must be non-empty",
        )
        self._copula = copula
        self._notionals = list(notionals)
        self._recovery_draw = recovery_draw
        self._n = len(notionals)
        self._seed = seed

    def copula(self) -> OneFactorCopula:
        return self._copula

    def pool_size(self) -> int:
        return self._n

    def notionals(self) -> list[float]:
        return list(self._notionals)

    def simulate_loss_distribution(
        self,
        probs: Sequence[float],
        n_paths: int,
    ) -> list[float]:
        """Return ``n_paths`` total-loss values, one per path."""
        qassert.require(
            len(probs) == self._n,
            f"probs size {len(probs)} != pool_size {self._n}",
        )
        self._copula.calculate()
        rng = MersenneTwisterUniformRng(self._seed)
        thresholds = [self._copula.inverse_cumulative_y(p) for p in probs]
        rho = self._copula.correlation()
        sqrt_rho = float(np.sqrt(rho))
        sqrt_1mr = float(np.sqrt(1.0 - rho))
        losses: list[float] = [0.0] * n_paths
        for path in range(n_paths):
            u_m = rng.next().value
            m = self._copula.inverse_cumulative_y(u_m)
            total = 0.0
            for i in range(self._n):
                u_z = rng.next().value
                z_i = self._copula.inverse_cumulative_y(u_z)
                y_i = sqrt_rho * m + sqrt_1mr * z_i
                if y_i < thresholds[i]:
                    u_rr = rng.next().value
                    rr_draw = self._recovery_draw(i, m, u_rr)
                    total += self._notionals[i] * (1.0 - rr_draw)
                else:
                    # advance RR draw to keep RNG stream deterministic
                    _ = rng.next()
            losses[path] = total
        return losses

    def expected_loss_mc(
        self,
        probs: Sequence[float],
        n_paths: int,
    ) -> float:
        """MC estimate of expected total loss."""
        losses = self.simulate_loss_distribution(probs, n_paths)
        return sum(losses) / float(n_paths)


def constant_recovery_draw(
    recoveries: Sequence[float],
) -> _ConditionalRecoveryDraw:
    """Build a constant-recovery draw closure for ``RandomLossLatentModel``.

    The closure ignores ``m`` and ``u_rr`` and returns the stored per-name
    recovery — i.e. the deterministic ``ConstantLossLatentModel`` behavior.
    """
    rs = list(recoveries)

    def _draw(i_name: int, m: float, u_rr: float) -> float:
        _ = m  # ignored — constant model
        _ = u_rr
        return rs[i_name]

    return _draw
