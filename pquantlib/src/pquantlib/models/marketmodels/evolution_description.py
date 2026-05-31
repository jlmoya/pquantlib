"""EvolutionDescription — the rate-time / evolution-time grid for BGM.

# C++ parity: ql/models/marketmodels/evolutiondescription.{hpp,cpp} (v1.42.1).

``EvolutionDescription`` is the foundational tuple of:

- ``rate_times`` — the ``n + 1`` payment/reset times defining ``n`` forward
  rates,
- ``evolution_times`` — the times at which the rates must be known (default:
  ``rate_times`` minus the last element),
- ``relevance_rates`` — the ``(first, last)`` index pair giving the relevant
  rate range for each evolution step (default: ``(0, n)``),

plus convenience ``rate_taus`` and ``first_alive_rate`` bookkeeping.

The numeraire-measure free functions (``check_compatibility``,
``is_in_terminal_measure``, ``terminal_measure``, ``money_market_measure``,
``money_market_plus_measure``, etc.) live here too, matching the C++ TU.

Divergences from C++:

- ``effectiveStopTimes()`` / ``effStopTime_`` is commented out in the
  v1.42.1 source (the ``Matrix`` of ``min(evolutionTime[j], rateTime[i])``);
  it is therefore **not** ported. No public ``effective_stop_time`` accessor
  exists in v1.42.1.
- ``relevance_rates`` is a list of ``(int, int)`` tuples rather than a
  ``std::vector<std::pair<Size,Size>>``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.utilities import (
    check_increasing_times,
    check_increasing_times_and_calculate_taus,
)


class EvolutionDescription:
    """Market-model evolution description: rate/evolution time grid.

    # C++ parity: evolutiondescription.hpp EvolutionDescription.
    """

    def __init__(
        self,
        rate_times: list[float],
        evolution_times: list[float] | None = None,
        relevance_rates: list[tuple[int, int]] | None = None,
    ) -> None:
        self._number_of_rates = 0 if not rate_times else len(rate_times) - 1
        self._rate_times = list(rate_times)
        # default evolution times = rate times minus the last
        if (evolution_times is None or len(evolution_times) == 0) and rate_times:
            self._evolution_times = list(rate_times[:-1])
        else:
            self._evolution_times = list(evolution_times or [])
        self._relevance_rates = list(relevance_rates or [])

        self._rate_taus = check_increasing_times_and_calculate_taus(self._rate_times)
        check_increasing_times(self._evolution_times)
        number_of_steps = len(self._evolution_times)

        qassert.require(
            self._evolution_times[-1] <= rate_times[len(rate_times) - 2],
            f"The last evolution time ({self._evolution_times[-1]}) is past the last "
            f"fixing time ({rate_times[self._number_of_rates - 2]})",
        )

        if not self._relevance_rates:
            self._relevance_rates = [(0, self._number_of_rates)] * number_of_steps
        else:
            qassert.require(
                len(self._relevance_rates) == number_of_steps,
                "relevanceRates / evolutionTimes mismatch",
            )

        self._first_alive_rate = [0] * number_of_steps
        current_evolution_time = 0.0
        first_alive_rate = 0
        for j in range(number_of_steps):
            while self._rate_times[first_alive_rate] <= current_evolution_time:
                first_alive_rate += 1
            self._first_alive_rate[j] = first_alive_rate
            current_evolution_time = self._evolution_times[j]

    def rate_times(self) -> list[float]:
        """The ``n + 1`` rate (payment/reset) times."""
        return self._rate_times

    def rate_taus(self) -> list[float]:
        """The ``n`` consecutive rate-time differences."""
        return self._rate_taus

    def evolution_times(self) -> list[float]:
        """The times at which rates must be known."""
        return self._evolution_times

    def first_alive_rate(self) -> list[int]:
        """Per-step index of the first not-yet-expired rate."""
        return self._first_alive_rate

    def relevance_rates(self) -> list[tuple[int, int]]:
        """Per-step ``(first, last)`` relevant-rate index range."""
        return self._relevance_rates

    def number_of_rates(self) -> int:
        """Number of forward rates ``n``."""
        return self._number_of_rates

    def number_of_steps(self) -> int:
        """Number of evolution steps."""
        return len(self._evolution_times)


# --- Numeraire functions (free functions; C++ parity: same TU) ---------------


def check_compatibility(evolution: EvolutionDescription, numeraires: list[int]) -> None:
    """Require one (un-expired) numeraire per evolution time.

    # C++ parity: evolutiondescription.cpp checkCompatibility.
    """
    evolution_times = evolution.evolution_times()
    n = len(evolution_times)
    qassert.require(
        len(numeraires) == n,
        f"Size mismatch between numeraires ({len(numeraires)}) and evolution times ({n})",
    )
    rate_times = evolution.rate_times()
    for i in range(n - 1):
        qassert.require(
            rate_times[numeraires[i]] >= evolution_times[i],
            f"step {i + 1}, evolution time {evolution_times[i]}: the numeraire "
            f"({numeraires[i]}), corresponding to rate time "
            f"{rate_times[numeraires[i]]}, is expired",
        )


def is_in_terminal_measure(evolution: EvolutionDescription, numeraires: list[int]) -> bool:
    """True iff every numeraire is the terminal (last) bond.

    # C++ parity: evolutiondescription.cpp isInTerminalMeasure.
    """
    rate_times = evolution.rate_times()
    return min(numeraires) == len(rate_times) - 1


def is_in_money_market_plus_measure(
    evolution: EvolutionDescription,
    numeraires: list[int],
    offset: int = 1,
) -> bool:
    """True iff numeraires match the offset money-market-plus measure.

    # C++ parity: evolutiondescription.cpp isInMoneyMarketPlusMeasure.
    """
    res = True
    rate_times = evolution.rate_times()
    max_numeraire = len(rate_times) - 1
    qassert.require(
        offset <= max_numeraire,
        f"offset ({offset}) is greater than the max allowed value for numeraire "
        f"({max_numeraire})",
    )
    evolution_times = evolution.evolution_times()
    j = 0
    for i in range(len(evolution_times)):
        while rate_times[j] < evolution_times[i]:
            j += 1
        res = (numeraires[i] == min(j + offset, max_numeraire)) and res
    return res


def is_in_money_market_measure(
    evolution: EvolutionDescription, numeraires: list[int]
) -> bool:
    """True iff numeraires match the money-market measure (offset 0).

    # C++ parity: evolutiondescription.cpp isInMoneyMarketMeasure.
    """
    return is_in_money_market_plus_measure(evolution, numeraires, 0)


def terminal_measure(evolution: EvolutionDescription) -> list[int]:
    """Terminal measure: the last bond as numeraire for every step.

    # C++ parity: evolutiondescription.cpp terminalMeasure.
    """
    return [len(evolution.rate_times()) - 1] * len(evolution.evolution_times())


def money_market_plus_measure(
    evolution: EvolutionDescription, offset: int = 1
) -> list[int]:
    """Offset discretely-compounded money-market-account measure.

    # C++ parity: evolutiondescription.cpp moneyMarketPlusMeasure.
    """
    rate_times = evolution.rate_times()
    max_numeraire = len(rate_times) - 1
    qassert.require(
        offset <= max_numeraire,
        f"offset ({offset}) is greater than the max allowed value for numeraire "
        f"({max_numeraire})",
    )
    evolution_times = evolution.evolution_times()
    n = len(evolution_times)
    numeraires = [0] * n
    j = 0
    for i in range(n):
        while rate_times[j] < evolution_times[i]:
            j += 1
        numeraires[i] = min(j + offset, max_numeraire)
    return numeraires


def money_market_measure(evolution: EvolutionDescription) -> list[int]:
    """Discretely-compounded money-market-account measure (offset 0).

    # C++ parity: evolutiondescription.cpp moneyMarketMeasure.
    """
    return money_market_plus_measure(evolution, 0)
