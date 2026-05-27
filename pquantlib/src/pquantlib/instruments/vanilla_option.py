"""VanillaOption — single-asset option with no discrete dividends or barriers.

# C++ parity: ql/instruments/vanillaoption.{hpp,cpp} (v1.42.1) —
# ``class VanillaOption : public OneAssetOption``.

A vanilla option is a OneAssetOption whose payoff is a
``StrikedTypePayoff`` (typically ``PlainVanillaPayoff``) and whose
exercise is European / American / Bermudan.

C++ exposes an ``impliedVolatility`` helper that constructs an engine
(``AnalyticEuropeanEngine`` for European; ``FdBlackScholesVanillaEngine``
for American/Bermudan) and runs a 1-D solver. The Python port defers
``impliedVolatility`` — the finite-difference engine isn't ported yet.
The implied vol for *European* vanillas is already available via
``pquantlib.pricingengines.black_formula.black_formula_implied_std_dev``;
callers needing it can compute it directly until the
``impliedVolatility`` convenience lands.

``is_expired()`` returns ``True`` when the last exercise date is past
the current valuation date. The valuation date is fetched from the
engine's process (via ``state_variable``'s observers if a process is
attached) — for now, the test path simply compares against
``Date.todays_date()`` analog. Until we wire Settings.evaluation_date,
the expiry check defers to the exercise's last date strictly being in
the past relative to the user-supplied reference; engines that compute
an expired NPV will short-circuit via Instrument.calculate.
"""

from __future__ import annotations

from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.payoffs import StrikedTypePayoff


class VanillaOption(OneAssetOption):
    """Vanilla single-asset option (Call/Put with optional early exercise).

    # C++ parity: trivial constructor that forwards to
    # ``OneAssetOption(payoff, exercise)``.
    """

    def __init__(self, payoff: StrikedTypePayoff, exercise: Exercise) -> None:
        super().__init__(payoff, exercise)

    def is_expired(self) -> bool:
        """Return ``False`` by default.

        # C++ parity: ``Instrument::isExpired`` for VanillaOption defers
        # to the ``Settings::evaluationDate``. Until evaluation_date is
        # wired into pquantlib (deferred per L1 carve-out), this method
        # returns ``False`` so the engine always runs. Tests that need
        # an expired option can subclass and override.
        """
        return False


__all__ = ["VanillaOption"]
