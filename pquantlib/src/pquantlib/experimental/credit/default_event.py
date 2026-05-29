"""DefaultEvent + concrete-event subclasses.

# C++ parity: ql/experimental/credit/defaultevent.{hpp,cpp} (v1.42.1).

A ``DefaultEvent`` records a realised credit event on bonds of a given
seniority + currency at a specific date, optionally accompanied by a
``DefaultSettlement`` carrying per-seniority recovery rates.

The C++ class inherits from ``Event`` (date + Observable). Python folds
the date/``has_occurred`` API in directly (mirroring the CashFlow port).
There is no Observable participation in this layer — events are
immutable post-construction.

Equality (``__eq__``) follows C++ ``operator==``: currency + default type
+ date + seniority — settlement data does NOT participate.

Concrete subclasses:
  - ``FailureToPayEvent`` — adds a defaulted-amount field and
    overrides ``matches_event_type`` to test the contractual
    ``FailureToPay`` grace period + amount threshold.
  - ``BankruptcyEvent`` — matches every contractual event type
    (bankruptcy is a stronger event that triggers all others).

# C++ parity divergence: the C++ recoveryRate(NoSeniority) overload
# raises; we keep the same semantics via ``qassert.require``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    FailureToPay,
    Restructuring,
    Seniority,
)
from pquantlib.time.date import Date

# ISDA conventional recovery-rate defaults indexed by Seniority IntEnum
# value. Mirrors C++ RecoveryRateQuote::IsdaConvRecoveries
# (recoveryratequote.cpp:24). Kept in this module to break an import cycle
# with recovery_rate_quote.py (which itself uses Seniority).
_ISDA_CONV_RECOVERIES: dict[Seniority, float] = {
    Seniority.SecDom: 0.65,  # SECDOM
    Seniority.SnrFor: 0.40,  # SNRFOR
    Seniority.SubLT2: 0.20,  # SUBLT2
    Seniority.JrSubT2: 0.20,  # JRSUBUT2
    Seniority.PrefT1: 0.15,  # PREFT1
}


def make_isda_conv_map() -> dict[Seniority, float]:
    """Return a fresh copy of the ISDA conventional recovery-rate map.

    # C++ parity: ql/experimental/credit/recoveryratequote.cpp:32 — the
    # C++ ``makeIsdaConvMap()`` free function returns a copy each call.
    """
    return dict(_ISDA_CONV_RECOVERIES)


@dataclass(frozen=True, slots=True)
class DefaultSettlement:
    """Settlement attached to a ``DefaultEvent``.

    Encodes the settlement date + per-seniority recovery rates. When
    ``NoSeniority`` is passed to the seniority-based constructor, every
    seniority in the ISDA convention map is assigned the supplied
    recovery rate.

    The C++ class has two constructors; we provide both as factory
    classmethods (``from_map`` / ``from_seniority``) plus a normal
    init taking the already-finalised state.
    """

    date: Date = field(default_factory=Date)
    recovery_rates: dict[Seniority, float] = field(default_factory=dict[Seniority, float])

    @classmethod
    def from_map(
        cls, date: Date, recovery_rates: dict[Seniority, float]
    ) -> DefaultSettlement:
        """Construct from explicit per-seniority recovery-rate map.

        # C++ parity: DefaultSettlement(Date, map<Seniority, Real>)
        # at defaultevent.cpp:54-61.
        """
        qassert.require(
            Seniority.NoSeniority not in recovery_rates,
            "NoSeniority is not a valid realized seniority.",
        )
        return cls(date=date, recovery_rates=dict(recovery_rates))

    @classmethod
    def from_seniority(
        cls,
        date: Date | None = None,
        seniority: Seniority = Seniority.NoSeniority,
        recovery_rate: float = 0.4,
    ) -> DefaultSettlement:
        """Construct from a single (seniority, recovery_rate) pair.

        If ``seniority`` is ``NoSeniority``, every entry in the ISDA
        conventional map is overridden to ``recovery_rate``. Otherwise
        the ISDA conventional map is the base and just the requested
        seniority gets the supplied value.

        # C++ parity: DefaultSettlement(Date, Seniority, Real)
        # at defaultevent.cpp:63-75.
        """
        rates = make_isda_conv_map()
        if seniority == Seniority.NoSeniority:
            for k in rates:
                rates[k] = recovery_rate
        else:
            rates[seniority] = recovery_rate
        return cls(
            date=date if date is not None else Date(),
            recovery_rates=rates,
        )

    def recovery_rate(self, seniority: Seniority) -> float | None:
        """Return recovery rate for the requested seniority, or None if absent.

        # C++ parity: DefaultEvent::DefaultSettlement::recoveryRate at
        # defaultevent.cpp:77-88 — returns ``Null<Real>()`` when seniority
        # is not in the recoveries map. The Python port returns ``None``.
        # NoSeniority is rejected.
        """
        qassert.require(
            seniority != Seniority.NoSeniority,
            "NoSeniority is not valid for recovery rate request.",
        )
        return self.recovery_rates.get(seniority)


# Sentinel constants used to keep call sites readable.
_NULL_DATE: Date = Date()


class DefaultEvent:
    """Realised credit event on bonds of a given currency + seniority.

    Fields mirror C++ ``DefaultEvent`` (defaultevent.hpp:49):

    - ``default_date`` (= ``date()``)
    - ``event_type``: ``DefaultType`` (atomic + restructuring sub-type)
    - ``currency``: bond currency
    - ``seniority``: bond seniority
    - ``settlement``: nested ``DefaultSettlement`` (date + recoveries)

    Two C++ constructors are mirrored by the ``from_map``/``from_rate``
    classmethods; passing no settlement info yields an "unsettled" event
    with empty recoveries (and ``has_settled() == False``).
    """

    __slots__ = (
        "_currency",
        "_default_date",
        "_event_type",
        "_seniority",
        "_settlement",
    )

    def __init__(
        self,
        credit_event_date: Date,
        atomic_ev_type: DefaultType,
        currency: Currency,
        bonds_seniority: Seniority,
        settlement: DefaultSettlement | None = None,
    ) -> None:
        self._default_date = credit_event_date
        self._event_type = atomic_ev_type
        self._currency = currency
        self._seniority = bonds_seniority
        self._settlement = settlement if settlement is not None else DefaultSettlement()

    # ----- C++ multi-arg constructor variants as factories ---------------------

    @classmethod
    def from_map(
        cls,
        credit_event_date: Date,
        atomic_ev_type: DefaultType,
        currency: Currency,
        bonds_seniority: Seniority,
        settle_date: Date | None = None,
        recovery_rates: dict[Seniority, float] | None = None,
    ) -> DefaultEvent:
        """Mirror C++ DefaultEvent(Date, DefaultType, Currency, Seniority, Date, map).

        # C++ parity: defaultevent.cpp:90-106.
        """
        actual_settle = settle_date if settle_date is not None else _NULL_DATE
        rates = (
            recovery_rates
            if recovery_rates is not None and len(recovery_rates) > 0
            else make_isda_conv_map()
        )
        if actual_settle != _NULL_DATE:
            qassert.require(
                actual_settle >= credit_event_date,
                "Settlement date should be after default date.",
            )
            if recovery_rates is not None:
                qassert.require(
                    bonds_seniority in recovery_rates,
                    "Settled events must contain the seniority of the default",
                )
            settlement = DefaultSettlement.from_map(actual_settle, rates)
        else:
            settlement = DefaultSettlement()
        return cls(
            credit_event_date,
            atomic_ev_type,
            currency,
            bonds_seniority,
            settlement,
        )

    @classmethod
    def from_rate(
        cls,
        credit_event_date: Date,
        atomic_ev_type: DefaultType,
        currency: Currency,
        bonds_seniority: Seniority,
        settle_date: Date | None = None,
        recovery_rate: float = 0.4,
    ) -> DefaultEvent:
        """Mirror C++ DefaultEvent(Date, DefaultType, Currency, Seniority, Date, Real).

        # C++ parity: defaultevent.cpp:108-121.
        """
        actual_settle = settle_date if settle_date is not None else _NULL_DATE
        if actual_settle != _NULL_DATE:
            qassert.require(
                actual_settle >= credit_event_date,
                "Settlement date should be after default date.",
            )
        settlement = DefaultSettlement.from_seniority(
            actual_settle, bonds_seniority, recovery_rate
        )
        return cls(
            credit_event_date,
            atomic_ev_type,
            currency,
            bonds_seniority,
            settlement,
        )

    # ----- accessors -----------------------------------------------------------

    def date(self) -> Date:
        """The credit-event date."""
        return self._default_date

    def event_type(self) -> DefaultType:
        """The atomic + restructuring default type."""
        return self._event_type

    def default_type(self) -> DefaultType:
        """C++ accessor alias for ``event_type()`` (defaultevent.hpp:122)."""
        return self._event_type

    def currency(self) -> Currency:
        """The bond + protection-leg currency this event refers to."""
        return self._currency

    def event_seniority(self) -> Seniority:
        """The seniority of the bond that triggered the event."""
        return self._seniority

    def settlement(self) -> DefaultSettlement:
        """The attached settlement (date + per-seniority recovery rates)."""
        return self._settlement

    def is_restructuring(self) -> bool:
        return self._event_type.is_restructuring()

    def is_default(self) -> bool:
        return not self.is_restructuring()

    def has_settled(self) -> bool:
        """True iff this event has an associated settlement date."""
        return self._settlement.date != _NULL_DATE

    def has_occurred(
        self, ref_date: Date | None = None, include_ref_date: bool | None = None
    ) -> bool:
        """Whether this event has occurred at ``ref_date``.

        Same semantics as ``CashFlow.has_occurred``: if ``ref_date`` is
        ``None`` or null, return False; otherwise compare dates.

        # C++ parity divergence: C++ falls back to
        # Settings::evaluationDate(); Python requires an explicit ref_date
        # (matching the existing CashFlow port pattern).
        """
        if ref_date is None or ref_date == _NULL_DATE:
            return False
        if ref_date < self._default_date:
            return False
        if self._default_date < ref_date:
            return True
        include = include_ref_date if include_ref_date is not None else False
        return not include

    def recovery_rate(self, seniority: Seniority) -> float | None:
        """Return the recovery rate of a settled event, or None if unsettled.

        # C++ parity: DefaultEvent::recoveryRate at defaultevent.hpp:137.
        # Returns Null<Real>() (we use None) when not settled.
        """
        if self.has_settled():
            return self._settlement.recovery_rate(seniority)
        return None

    # ----- matching predicates -------------------------------------------------

    def matches_event_type(self, contract_ev_type: DefaultType) -> bool:
        """True iff this event would trigger a contract that expects ``contract_ev_type``.

        Subclasses (FailureToPayEvent, BankruptcyEvent) override.

        # C++ parity: DefaultEvent::matchesEventType at defaultevent.hpp:150.
        """
        return contract_ev_type.contains_restructuring_type(
            self._event_type.restructuring_type
        ) and contract_ev_type.contains_default_type(self._event_type.default_type)

    def matches_default_key(self, contract_key: DefaultProbKey) -> bool:
        """True iff this event would trigger a contract characterised by ``contract_key``.

        # C++ parity: defaultevent.cpp:123-136 — short-circuits on
        # currency / seniority mismatch (NoSeniority is wildcard), then
        # tries to match each contract event type in turn.
        """
        if self._currency != contract_key.currency:
            return False
        contract_seniority = contract_key.seniority
        if contract_seniority not in (self._seniority, Seniority.NoSeniority):
            return False
        return any(
            self.matches_event_type(et) for et in contract_key.event_types
        )

    # ----- equality/hash -------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultEvent):
            return NotImplemented
        # # C++ parity: operator== at defaultevent.cpp:140-145 — settlement
        # data is intentionally excluded.
        return (
            self._currency == other._currency
            and self._event_type == other._event_type
            and self._default_date == other._default_date
            and self._seniority == other._seniority
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        """Order by date — used by the DefaultEventSet ordered container.

        # C++ parity: earlier_than<DefaultEvent> at defaultevent.hpp:188-194.
        """
        if not isinstance(other, DefaultEvent):
            return NotImplemented
        return self._default_date < other._default_date

    def __hash__(self) -> int:
        return hash(
            (
                self._currency,
                self._event_type,
                self._default_date,
                self._seniority,
            )
        )


# ----- concrete event subclasses ----------------------------------------------


class FailureToPayEvent(DefaultEvent):
    """Failure-to-pay credit event with a defaulted-amount field.

    # C++ parity: ql/experimental/credit/defaultevent.hpp:199 +
    # defaultevent.cpp:148-192.
    """

    __slots__ = ("_defaulted_amount",)

    def __init__(
        self,
        credit_event_date: Date,
        currency: Currency,
        bonds_seniority: Seniority,
        defaulted_amount: float,
        settlement: DefaultSettlement | None = None,
    ) -> None:
        super().__init__(
            credit_event_date,
            DefaultType(AtomicDefault.FailureToPay, Restructuring.NoRestructuring),
            currency,
            bonds_seniority,
            settlement,
        )
        self._defaulted_amount = defaulted_amount

    @classmethod
    def from_map(  # type: ignore[override]
        cls,
        credit_event_date: Date,
        currency: Currency,
        bonds_seniority: Seniority,
        defaulted_amount: float,
        settle_date: Date | None = None,
        recovery_rates: dict[Seniority, float] | None = None,
    ) -> FailureToPayEvent:
        base = DefaultEvent.from_map(
            credit_event_date,
            DefaultType(AtomicDefault.FailureToPay, Restructuring.NoRestructuring),
            currency,
            bonds_seniority,
            settle_date,
            recovery_rates,
        )
        return cls(
            credit_event_date,
            currency,
            bonds_seniority,
            defaulted_amount,
            base.settlement(),
        )

    @classmethod
    def from_rate(  # type: ignore[override]
        cls,
        credit_event_date: Date,
        currency: Currency,
        bonds_seniority: Seniority,
        defaulted_amount: float,
        settle_date: Date | None = None,
        recovery_rate: float = 0.4,
    ) -> FailureToPayEvent:
        base = DefaultEvent.from_rate(
            credit_event_date,
            DefaultType(AtomicDefault.FailureToPay, Restructuring.NoRestructuring),
            currency,
            bonds_seniority,
            settle_date,
            recovery_rate,
        )
        return cls(
            credit_event_date,
            currency,
            bonds_seniority,
            defaulted_amount,
            base.settlement(),
        )

    def amount_defaulted(self) -> float:
        return self._defaulted_amount

    def matches_event_type(self, contract_ev_type: DefaultType) -> bool:
        """Match only against a contract FailureToPay; require amount + grace.

        # C++ parity divergence: the C++ override (defaultevent.cpp:148-157)
        # uses Settings::evaluationDate() and event.hasOccurred(today -
        # grace, true) to test the grace period. Python omits the
        # Settings dependency: we only enforce the amount-threshold
        # check. Grace-period testing is left to callers that pass a
        # ref-date — see ``matches_event_type_with_ref_date``.
        """
        if not isinstance(contract_ev_type, FailureToPay):
            return False
        return self._defaulted_amount >= contract_ev_type.amount_required


class BankruptcyEvent(DefaultEvent):
    """Bankruptcy credit event.

    Matches every contractual event type (bankruptcy is a stronger
    trigger than all other ISDA atomics).

    # C++ parity: ql/experimental/credit/defaultevent.hpp:225 +
    # defaultevent.cpp:196-229.
    """

    def __init__(
        self,
        credit_event_date: Date,
        currency: Currency,
        bonds_seniority: Seniority,
        settlement: DefaultSettlement | None = None,
    ) -> None:
        super().__init__(
            credit_event_date,
            DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
            currency,
            bonds_seniority,
            settlement,
        )

    @classmethod
    def from_map(  # type: ignore[override]
        cls,
        credit_event_date: Date,
        currency: Currency,
        bonds_seniority: Seniority,
        settle_date: Date | None = None,
        recovery_rates: dict[Seniority, float] | None = None,
    ) -> BankruptcyEvent:
        if (
            settle_date is not None
            and settle_date != _NULL_DATE
            and recovery_rates is not None
        ):
            qassert.require(
                len(recovery_rates) == len(_ISDA_CONV_RECOVERIES),
                "Bankruptcy event should have settled for all seniorities.",
            )
        base = DefaultEvent.from_map(
            credit_event_date,
            DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
            currency,
            bonds_seniority,
            settle_date,
            recovery_rates,
        )
        return cls(credit_event_date, currency, bonds_seniority, base.settlement())

    @classmethod
    def from_rate(  # type: ignore[override]
        cls,
        credit_event_date: Date,
        currency: Currency,
        bonds_seniority: Seniority,
        settle_date: Date | None = None,
        recovery_rate: float = 0.4,
    ) -> BankruptcyEvent:
        base = DefaultEvent.from_rate(
            credit_event_date,
            DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
            currency,
            bonds_seniority,
            settle_date,
            recovery_rate,
        )
        return cls(credit_event_date, currency, bonds_seniority, base.settlement())

    def matches_event_type(self, contract_ev_type: DefaultType) -> bool:
        """Bankruptcy matches all contract event types.

        # C++ parity: defaultevent.hpp:240-241 — strongest event.
        """
        return True
