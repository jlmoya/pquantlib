"""HolderExtensibleOption — option holder pays a premium to extend maturity.

# C++ parity: ql/instruments/holderextensibleoption.{hpp,cpp} (v1.42.1).

A holder-extensible option can be:

* Exercised at the original expiry ``t1`` (using the original strike
  ``X1`` from the underlying ``PlainVanillaPayoff``), or
* Extended by the **holder** by paying a premium ``A`` to the writer.
  The new expiry is ``T2`` and the new strike is ``X2``.

The instrument carries the OneAssetOption base (original strike +
expiry) plus three extra fields (``premium``, ``second_expiry_date``,
``second_strike``) propagated into the engine arguments.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class HolderExtensibleOptionArguments(OptionArguments):
    """Engine arguments for holder-extensible options.

    # C++ parity: ``HolderExtensibleOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.premium: float | None = None
        self.second_expiry_date: Date = Date()
        self.second_strike: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.premium is not None, "no premium given")
        qassert.require(
            self.second_expiry_date != Date(), "no second expiry date given"
        )
        qassert.require(self.second_strike is not None, "no second strike given")


class HolderExtensibleOption(OneAssetOption):
    """Holder-extensible option (Haug 2007).

    # C++ parity: ``HolderExtensibleOption``.
    """

    def __init__(
        self,
        option_type: OptionType,
        premium: float,
        second_expiry_date: Date,
        second_strike: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._premium: float = premium
        self._second_expiry_date: Date = second_expiry_date
        self._second_strike: float = second_strike

    def premium(self) -> float:
        return self._premium

    def second_expiry_date(self) -> Date:
        return self._second_expiry_date

    def second_strike(self) -> float:
        return self._second_strike

    def is_expired(self) -> bool:
        """Return ``False`` — defers to engine.

        # C++ parity: ``OneAssetOption::isExpired`` defers to
        # ``Settings.evaluationDate``. Until Settings is wired,
        # we return False (same as VanillaOption.is_expired).
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Propagate extended-expiry data into the engine arguments.

        # C++ parity: ``HolderExtensibleOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, HolderExtensibleOptionArguments),
            "wrong argument type",
        )
        assert isinstance(args, HolderExtensibleOptionArguments)
        args.premium = self._premium
        args.second_expiry_date = self._second_expiry_date
        args.second_strike = self._second_strike


__all__ = ["HolderExtensibleOption", "HolderExtensibleOptionArguments"]
