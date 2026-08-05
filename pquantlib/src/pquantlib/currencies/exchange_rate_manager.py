"""Exchange-rate repository.

# C++ parity: ql/currencies/exchangeratemanager.hpp + .cpp (v1.43).

A singleton store of date-ranged exchange rates, seeded at construction with
the fixed euro-conversion rates and the handful of redenominations
(TRY/TRL, RON/ROL, PEN/PEI/PEH).

Rates are keyed by an order-independent hash of the two ISO numeric codes, so
one entry serves both directions; it is :meth:`ExchangeRate.exchange` that
knows which way round to apply it. :meth:`ExchangeRateManager.lookup` prefers a
direct rate, then routes through the source's or the target's triangulation
currency, and only then walks the whole store looking for a chain.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib import qassert
from pquantlib.currencies.america import PEHCurrency, PEICurrency, PENCurrency
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.europe import (
    ATSCurrency,
    BEFCurrency,
    DEMCurrency,
    ESPCurrency,
    EURCurrency,
    FIMCurrency,
    FRFCurrency,
    GRDCurrency,
    IEPCurrency,
    ITLCurrency,
    LUFCurrency,
    NLGCurrency,
    PTECurrency,
    ROLCurrency,
    RONCurrency,
    TRLCurrency,
    TRYCurrency,
)
from pquantlib.currencies.exchange_rate import ExchangeRate, ExchangeRateType
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.singleton import Singleton
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@dataclass(frozen=True, slots=True)
class Entry:
    """One stored rate together with the dates it is valid between.

    # C++ parity: the nested ``ExchangeRateManager::Entry`` struct.
    """

    rate: ExchangeRate
    start_date: Date
    end_date: Date

    def valid_at(self, date: Date) -> bool:
        """Mirrors the C++ ``valid_at`` predicate — inclusive on both ends."""
        return self.start_date <= date <= self.end_date


class ExchangeRateManager(Singleton):
    """Repository of exchange rates, with direct / triangulated / chained lookup."""

    # Nested-class alias for the C++ idiom ``ExchangeRateManager.Entry``.
    Entry = Entry

    def __init__(self) -> None:
        # The Singleton metaclass caches the instance and only runs __init__
        # on first construction, so this is not re-entered.
        super().__init__()
        self._data: dict[int, list[Entry]] = {}
        self._add_known_rates()

    @classmethod
    def instance(cls) -> ExchangeRateManager:
        """Return the singleton instance (parity with C++ ``::instance()``)."""
        return cls()

    # ---- public API ----

    def add(
        self,
        rate: ExchangeRate,
        start_date: Date | None = None,
        end_date: Date | None = None,
    ) -> None:
        """Add a rate, valid between the given dates (defaults: the whole Date range).

        If two rates are given between the same currencies with overlapping
        date ranges, the one added last takes precedence during lookup.
        """
        start = start_date if start_date is not None else Date.min_date()
        end = end_date if end_date is not None else Date.max_date()
        key = self._hash(rate.source, rate.target)
        # C++ uses emplace_front, so the newest entry is found first.
        self._data.setdefault(key, []).insert(0, Entry(rate, start, end))

    def lookup(
        self,
        source: Currency,
        target: Currency,
        date: Date | None = None,
        rate_type: ExchangeRateType = ExchangeRateType.Derived,
    ) -> ExchangeRate:
        """Look up the rate between two currencies at a date.

        With ``Direct`` only directly-stored rates are returned; with
        ``Derived`` (the default) triangulated and chained rates are allowed,
        direct ones still being preferred.

        # C++ parity: the C++ default ``date = Date()`` is the null date,
        # which the body replaces with the global evaluation date; Python
        # spells that ``None``.
        """
        if source == target:
            return ExchangeRate(source, target, 1.0)

        as_of: Date = date if date is not None else ObservableSettings().evaluation_date_or_today()

        if rate_type == ExchangeRateType.Direct:
            return self._direct_lookup(source, target, as_of)

        source_link = source.triangulation_currency
        if source_link is not None and not source_link.empty():
            if source_link == target:
                return self._direct_lookup(source, source_link, as_of)
            return ExchangeRate.chain(
                self._direct_lookup(source, source_link, as_of),
                self.lookup(source_link, target, as_of),
            )

        target_link = target.triangulation_currency
        if target_link is not None and not target_link.empty():
            if source == target_link:
                return self._direct_lookup(target_link, target, as_of)
            return ExchangeRate.chain(
                self.lookup(source, target_link, as_of),
                self._direct_lookup(target_link, target, as_of),
            )

        return self._smart_lookup(source, target, as_of, [])

    def clear(self) -> None:
        """Drop every added rate and re-seed the known ones."""
        self._data.clear()
        self._add_known_rates()

    # ---- internals ----

    @staticmethod
    def _hash(c1: Currency, c2: Currency) -> int:
        """Order-independent key built from the two ISO numeric codes."""
        return min(c1.numeric_code, c2.numeric_code) * 1000 + max(c1.numeric_code, c2.numeric_code)

    @staticmethod
    def _hashes(key: int, c: Currency) -> bool:
        """True if ``c`` is one of the two currencies a key was built from."""
        return c.numeric_code in (key % 1000, key // 1000)

    def _fetch(self, source: Currency, target: Currency, date: Date) -> ExchangeRate | None:
        for entry in self._data.get(self._hash(source, target), ()):
            if entry.valid_at(date):
                return entry.rate
        return None

    def _direct_lookup(self, source: Currency, target: Currency, date: Date) -> ExchangeRate:
        rate = self._fetch(source, target, date)
        if rate is None:
            qassert.fail(f"no direct conversion available from {source.code} to {target.code} for {date}")
        return rate

    def _smart_lookup(
        self,
        source: Currency,
        target: Currency,
        date: Date,
        forbidden: list[int],
    ) -> ExchangeRate:
        # Direct rates are preferred.
        direct = self._fetch(source, target, date)
        if direct is not None:
            return direct

        # Otherwise walk the store. The source currency is forbidden to
        # subsequent lookups so the walk cannot cycle.
        forbidden = [*forbidden, source.numeric_code]
        # C++ iterates a std::map, i.e. in ascending key order; Python dicts
        # keep insertion order, so sort explicitly to pick the same chain when
        # more than one is possible (C++ leaves that choice unspecified, but
        # reproducing it is what makes the reference meaningful).
        for key in sorted(self._data):
            entries = self._data[key]
            if not entries or not self._hashes(key, source):
                continue
            # ...whose other currency is not forbidden...
            entry = entries[0]
            other = entry.rate.target if source == entry.rate.source else entry.rate.source
            if other.numeric_code in forbidden:
                continue
            # ...and which carries information for the requested date.
            head = self._fetch(source, other, date)
            if head is None:
                continue
            try:
                tail = self._smart_lookup(other, target, date, forbidden)
            except LibraryException:
                # No route to the target from here; discard this rate.
                continue
            return ExchangeRate.chain(head, tail)

        qassert.fail(f"no conversion available from {source.code} to {target.code} for {date}")

    def _add_known_rates(self) -> None:
        max_date = Date.max_date()
        # Currencies obsoleted by the euro, at their fixed conversion rates.
        euro_legacy: tuple[tuple[Currency, float, Date], ...] = (
            (ATSCurrency(), 13.7603, Date.from_ymd(1, Month.January, 1999)),
            (BEFCurrency(), 40.3399, Date.from_ymd(1, Month.January, 1999)),
            (DEMCurrency(), 1.95583, Date.from_ymd(1, Month.January, 1999)),
            (ESPCurrency(), 166.386, Date.from_ymd(1, Month.January, 1999)),
            (FIMCurrency(), 5.94573, Date.from_ymd(1, Month.January, 1999)),
            (FRFCurrency(), 6.55957, Date.from_ymd(1, Month.January, 1999)),
            (GRDCurrency(), 340.750, Date.from_ymd(1, Month.January, 2001)),
            (IEPCurrency(), 0.787564, Date.from_ymd(1, Month.January, 1999)),
            (ITLCurrency(), 1936.27, Date.from_ymd(1, Month.January, 1999)),
            (LUFCurrency(), 40.3399, Date.from_ymd(1, Month.January, 1999)),
            (NLGCurrency(), 2.20371, Date.from_ymd(1, Month.January, 1999)),
            (PTECurrency(), 200.482, Date.from_ymd(1, Month.January, 1999)),
        )
        eur = EURCurrency()
        for currency, rate, start in euro_legacy:
            self.add(ExchangeRate(eur, currency, rate), start, max_date)

        # Other obsoleted currencies (redenominations).
        redenominations: tuple[tuple[Currency, Currency, float, Date], ...] = (
            (TRYCurrency(), TRLCurrency(), 1000000.0, Date.from_ymd(1, Month.January, 2005)),
            (RONCurrency(), ROLCurrency(), 10000.0, Date.from_ymd(1, Month.July, 2005)),
            (PENCurrency(), PEICurrency(), 1000000.0, Date.from_ymd(1, Month.July, 1991)),
            (PEICurrency(), PEHCurrency(), 1000.0, Date.from_ymd(1, Month.February, 1985)),
        )
        for new, old, rate, start in redenominations:
            self.add(ExchangeRate(new, old, rate), start, max_date)
