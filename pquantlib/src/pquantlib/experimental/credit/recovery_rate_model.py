"""RecoveryRateModel — abstract recovery-rate model.

# C++ parity: ql/experimental/credit/recoveryratemodel.{hpp,cpp} (v1.42.1).

RecoveryRateModel is an Observable abstract base class returning a
forward-looking recovery rate for a given default date + contract key.
The concrete ``ConstantRecoveryModel`` ignores both inputs and returns
the stored quote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import Seniority
from pquantlib.experimental.credit.recovery_rate_quote import RecoveryRateQuote
from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date


class RecoveryRateModel(Observable, ABC):
    """Abstract base for forward-looking recovery-rate models."""

    def __init__(self) -> None:
        super().__init__()

    def recovery_value(
        self,
        default_date: Date,
        default_key: DefaultProbKey | None = None,
    ) -> float | None:
        """Return the expected recovery rate at ``default_date`` for ``default_key``.

        Delegates to the protected ``_recovery_value_impl``. Mirrors C++
        public + protected split (recoveryratemodel.hpp:38-55).
        """
        key = default_key if default_key is not None else DefaultProbKey()
        return self._recovery_value_impl(default_date, key)

    @abstractmethod
    def applies_to_seniority(self, seniority: Seniority) -> bool:
        """True iff this model produces a recovery rate for ``seniority``."""

    @abstractmethod
    def _recovery_value_impl(
        self, default_date: Date, default_key: DefaultProbKey
    ) -> float | None:
        """Override hook — return ``None`` if no rate is available."""


class ConstantRecoveryModel(RecoveryRateModel):
    """Recovery-rate model returning a fixed value (a Quote) independent of inputs.

    # C++ parity: recoveryratemodel.cpp:24-33 — two ctors, one taking
    # a Quote handle and one taking a (Real, Seniority) pair. The Python
    # port keeps both via classmethod factories.
    """

    __slots__ = ("_quote",)

    def __init__(self, quote: RecoveryRateQuote) -> None:
        super().__init__()
        self._quote = quote
        # Register as observer of the quote — when the quote notifies, we re-emit.
        self._quote.register_with(self)

    @classmethod
    def from_rate(
        cls,
        recovery: float,
        seniority: Seniority = Seniority.NoSeniority,
    ) -> ConstantRecoveryModel:
        """Build with a fresh internally-owned RecoveryRateQuote.

        # C++ parity: recoveryratemodel.cpp:30-33.
        """
        return cls(RecoveryRateQuote(recovery, seniority))

    def quote(self) -> RecoveryRateQuote:
        return self._quote

    def update(self) -> None:
        """Observer callback — propagate quote updates downstream."""
        self.notify_observers()

    def applies_to_seniority(self, seniority: Seniority) -> bool:
        # # C++ parity: recoveryratemodel.hpp:69 — "all pass".
        return True

    def _recovery_value_impl(
        self, default_date: Date, default_key: DefaultProbKey
    ) -> float | None:
        return self._quote.value() if self._quote.is_valid() else None
