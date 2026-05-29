"""RecoveryRateQuote — Quote subclass storing a recovery rate + seniority.

# C++ parity: ql/experimental/credit/recoveryratequote.{hpp,cpp} (v1.42.1).

Behaves as an Observable Quote. The C++ class adds a small static helper
for conventional ISDA recoveries indexed by seniority; we re-expose those
via ``conventional_recovery`` (mirroring the C++ static method).

# C++ parity divergence: C++ ``Null<Real>()`` sentinel maps to Python
# ``None``, same as SimpleQuote. ``setValue(Null<Real>())`` -> reset.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.credit.default_event import ISDA_CONV_RECOVERIES
from pquantlib.experimental.credit.default_type import Seniority
from pquantlib.quotes.quote import Quote


class RecoveryRateQuote(Quote):
    """Quote for a recovery rate plus the seniority it refers to."""

    __slots__ = ("_recovery_rate", "_seniority")

    def __init__(
        self,
        value: float | None = None,
        seniority: Seniority = Seniority.NoSeniority,
    ) -> None:
        super().__init__()
        qassert.require(
            value is None or (0.0 <= value <= 1.0),
            "Recovery value must be a fractional unit.",
        )
        self._recovery_rate: float | None = value
        self._seniority: Seniority = seniority

    @staticmethod
    def conventional_recovery(seniority: Seniority) -> float:
        """Return the ISDA conventional recovery rate for a seniority.

        # C++ parity: RecoveryRateQuote::conventionalRecovery at
        # recoveryratequote.hpp:36-38.
        """
        return ISDA_CONV_RECOVERIES[seniority]

    def value(self) -> float:
        qassert.require(self.is_valid(), "invalid Recovery Quote")
        assert self._recovery_rate is not None
        return self._recovery_rate

    def seniority(self) -> Seniority:
        return self._seniority

    def is_valid(self) -> bool:
        # Mirrors C++ recoveryRate != Null<Real>() check at
        # recoveryratequote.hpp:79.
        return self._recovery_rate is not None

    def set_value(self, value: float | None = None) -> float:
        """Update the stored value; return the diff from the prior value.

        Notifies observers iff the diff is non-zero. Transitions to/from
        ``None`` always notify (and return 0.0 — see SimpleQuote docstring).
        """
        if self._recovery_rate is None or value is None:
            if self._recovery_rate is value:
                return 0.0
            self._recovery_rate = value
            self.notify_observers()
            return 0.0
        diff = value - self._recovery_rate
        if diff != 0.0:
            self._recovery_rate = value
            self.notify_observers()
        return diff

    def reset(self) -> None:
        """Make the quote invalid + drop the seniority back to NoSeniority.

        # C++ parity: recoveryratequote.cpp:54-57.
        """
        self.set_value(None)
        self._seniority = Seniority.NoSeniority
