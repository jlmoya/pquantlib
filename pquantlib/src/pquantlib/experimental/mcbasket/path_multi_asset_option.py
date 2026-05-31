"""PathMultiAssetOption — instrument priced over multi-asset paths.

# C++ parity: ql/experimental/mcbasket/pathmultiassetoption.{hpp,cpp}
#             (v1.42.1).

A ``PathMultiAssetOption`` is an ``Instrument`` whose payoff is a
:class:`~pquantlib.experimental.mcbasket.path_payoff.PathPayoff`
evaluated over the full sampled multi-asset path at a set of fixing
dates. C++ makes ``pathPayoff()`` / ``fixingDates()`` pure-virtual
(the library ships no concrete); the Python port makes the class
concrete by taking the payoff + fixing dates in the constructor, which
is what callers need.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.mcbasket.path_payoff import PathPayoff
from pquantlib.instruments.instrument import Instrument
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class PathMultiAssetOptionArguments(OptionArguments):
    """Engine arguments for a path multi-asset option.

    # C++ parity: ``PathMultiAssetOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.path_payoff: PathPayoff | None = None
        self.fixing_dates: list[Date] = []

    def validate(self) -> None:
        # Note: does NOT call OptionArguments.validate (no scalar payoff/exercise).
        qassert.require(self.path_payoff is not None, "no payoff given")
        qassert.require(len(self.fixing_dates) != 0, "no dates given")


class PathMultiAssetOption(Instrument):
    """Path-dependent option on multiple assets.

    # C++ parity: ``PathMultiAssetOption``.

    Args:
        path_payoff: the multi-asset path payoff.
        fixing_dates: the observation/fixing dates.
    """

    def __init__(self, path_payoff: PathPayoff, fixing_dates: list[Date]) -> None:
        super().__init__()
        self._path_payoff: PathPayoff = path_payoff
        self._fixing_dates: list[Date] = list(fixing_dates)

    def path_payoff(self) -> PathPayoff:
        return self._path_payoff

    def fixing_dates(self) -> list[Date]:
        return list(self._fixing_dates)

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        qassert.require(
            isinstance(args, PathMultiAssetOptionArguments), "wrong argument type"
        )
        assert isinstance(args, PathMultiAssetOptionArguments)
        args.path_payoff = self._path_payoff
        args.fixing_dates = list(self._fixing_dates)


__all__ = ["PathMultiAssetOption", "PathMultiAssetOptionArguments"]
