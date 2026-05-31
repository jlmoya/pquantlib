"""ExchangeContract — a futures exchange contract specification.

# C++ parity: ql/experimental/commodities/exchangecontract.hpp (v1.42.1).

A value type describing one exchange-traded futures contract: its code, its
expiration date, and the start/end dates of the underlying delivery period.
The C++ ``ExchangeContracts`` typedef (``map<Date, ExchangeContract>``) maps
to a plain ``dict[Date, ExchangeContract]`` on the Python side.
"""

from __future__ import annotations

from pquantlib.time.date import Date


class ExchangeContract:
    """A futures exchange contract spec (code + expiration + underlying dates)."""

    def __init__(
        self,
        code: str = "",
        expiration_date: Date | None = None,
        underlying_start_date: Date | None = None,
        underlying_end_date: Date | None = None,
    ) -> None:
        self._code: str = code
        self._expiration_date: Date = (
            expiration_date if expiration_date is not None else Date()
        )
        self._underlying_start_date: Date = (
            underlying_start_date if underlying_start_date is not None else Date()
        )
        self._underlying_end_date: Date = (
            underlying_end_date if underlying_end_date is not None else Date()
        )

    @property
    def code(self) -> str:
        return self._code

    @property
    def expiration_date(self) -> Date:
        return self._expiration_date

    @property
    def underlying_start_date(self) -> Date:
        return self._underlying_start_date

    @property
    def underlying_end_date(self) -> Date:
        return self._underlying_end_date

    def __repr__(self) -> str:
        return f"ExchangeContract({self._code!r}, expires {self._expiration_date})"


# C++ parity: ``typedef std::map<Date, ExchangeContract> ExchangeContracts;``
ExchangeContracts = dict[Date, ExchangeContract]
