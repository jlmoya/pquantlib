"""TimeBasket — a distribution of cash amounts over a number of dates.

# C++ parity: ql/cashflows/timebasket.{hpp,cpp} (v1.43).

C++ ``TimeBasket`` *privately* derives from ``std::map<Date, Real>`` and
re-exports part of the map surface with ``using`` declarations
(``size``, ``operator[]``, ``begin``/``end``, ``rbegin``/``rend``), so the
map semantics are part of the class contract, not an implementation detail:

- **Entries iterate in ascending date order**, whatever the insertion order
  (``std::map``, not a Python ``dict``). :meth:`items` / :meth:`keys` /
  :meth:`values` / ``iter()`` all sort; :func:`reversed` gives the
  ``rbegin``/``rend`` direction.
- **``operator[]`` on a missing key default-constructs ``0.0`` and inserts
  it**, so a plain read grows the basket. :meth:`__getitem__` reproduces
  that faithfully — ``operator+=``, ``operator-=`` and :meth:`rebin` are all
  built on top of it, and a non-mutating read would silently change their
  results (``a += b`` must leave ``b``'s dates present in ``a`` with value
  ``0.0`` even after ``a -= b``). Use :meth:`has_date` for a non-mutating
  membership test.
- **Construction from parallel vectors assigns, it does not accumulate**, so
  a repeated date keeps the last value (timebasket.cpp:27-34).

:meth:`rebin` redistributes every entry over a new set of bucket dates
(timebasket.cpp:36-71): an entry that falls strictly between two buckets is
split linearly by day count between them; an entry on a bucket, before the
first bucket or after the last one goes wholly to a single bucket.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class TimeBasket:
    """Distribution of ``Real`` amounts over a number of ``Date``\\ s."""

    def __init__(
        self,
        dates: Sequence[Date] | None = None,
        values: Sequence[float] | None = None,
    ) -> None:
        """Build a basket from parallel ``dates`` / ``values`` sequences.

        # C++ parity: timebasket.cpp:27-34 — ``QL_REQUIRE`` on equal sizes,
        # then ``self[dates[i]] = values[i]`` (assignment, so a repeated date
        # keeps the last value). No arguments gives the empty basket
        # (``TimeBasket() = default``).
        """
        self._data: dict[Date, float] = {}
        if dates is None and values is None:
            return
        dates = () if dates is None else dates
        values = () if values is None else values
        qassert.require(
            len(dates) == len(values),
            "number of dates differs from number of values",
        )
        for d, v in zip(dates, values, strict=True):
            self._data[d] = float(v)

    # --- map interface --------------------------------------------------

    def size(self) -> int:
        """Number of entries. # C++ parity: ``using super::size``."""
        return len(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, date: Date) -> float:
        """Amount at ``date``, **inserting 0.0 when absent**.

        # C++ parity: ``using super::operator[]`` — ``std::map::operator[]``
        # value-initialises and inserts a missing key. The insert is
        # observable through :meth:`size` and iteration; it is what makes
        # ``operator+=`` / ``operator-=`` / :meth:`rebin` work.
        """
        if date not in self._data:
            self._data[date] = 0.0
        return self._data[date]

    def __setitem__(self, date: Date, value: float) -> None:
        """# C++ parity: ``basket[date] = value`` via ``operator[]``."""
        self._data[date] = float(value)

    def has_date(self, date: Date) -> bool:
        """Non-mutating membership test. # C++ parity: timebasket.hpp:73-76."""
        return date in self._data

    def __contains__(self, date: Date) -> bool:
        return date in self._data

    def keys(self) -> list[Date]:
        """Dates in ascending order (``std::map`` key order)."""
        return sorted(self._data)

    def values(self) -> list[float]:
        """Amounts ordered by ascending date."""
        return [self._data[d] for d in sorted(self._data)]

    def items(self) -> list[tuple[Date, float]]:
        """``(date, amount)`` pairs in ascending date order.

        # C++ parity: ``begin()`` / ``end()``.
        """
        return [(d, self._data[d]) for d in sorted(self._data)]

    def __iter__(self) -> Iterator[Date]:
        return iter(sorted(self._data))

    def __reversed__(self) -> Iterator[Date]:
        """# C++ parity: ``rbegin()`` / ``rend()`` — descending date order."""
        return iter(sorted(self._data, reverse=True))

    # --- algebra ---------------------------------------------------------

    def __iadd__(self, other: TimeBasket) -> TimeBasket:
        """Add ``other`` entry-wise. # C++ parity: timebasket.hpp:78-83."""
        for d, v in other.items():
            self[d] = self[d] + v
        return self

    def __isub__(self, other: TimeBasket) -> TimeBasket:
        """Subtract ``other`` entry-wise. # C++ parity: timebasket.hpp:85-90."""
        for d, v in other.items():
            self[d] = self[d] - v
        return self

    # --- other methods ---------------------------------------------------

    def rebin(self, buckets: Sequence[Date]) -> TimeBasket:
        """Redistribute the entries over ``buckets``.

        # C++ parity: timebasket.cpp:36-71. ``buckets`` need not be sorted —
        # a sorted copy is taken, exactly as C++ does. Every bucket appears
        # in the result, with 0.0 if nothing was redistributed onto it.

        For an entry ``(date, value)``, ``pDate`` is the first bucket at or
        after ``date`` (or the last bucket when ``date`` is past all of them)
        and ``nDate`` the bucket before it (null when ``pDate`` is the first
        bucket, or when ``date`` is past all buckets). With a null ``nDate``,
        or with ``date`` exactly on ``pDate``, the whole value goes to
        ``pDate``; otherwise it is split linearly by day count.
        """
        qassert.require(len(buckets) > 0, "empty bucket structure")

        sbuckets = sorted(buckets)
        n_buckets = len(sbuckets)
        null_date = Date()

        result = TimeBasket()
        for bucket in sbuckets:
            result[bucket] = 0.0

        for date, value in self.items():
            # ``bi`` mirrors ``std::lower_bound``: index of the first bucket
            # not less than ``date`` (``n_buckets`` == C++ ``end()``).
            bi = bisect.bisect_left(sbuckets, date)
            p_date = sbuckets[-1] if bi == n_buckets else sbuckets[bi]
            n_date = sbuckets[bi - 1] if 0 < bi < n_buckets else null_date

            if p_date == date or n_date == null_date:
                result[p_date] = result[p_date] + value
            else:
                p_days = float(p_date - date)
                n_days = float(date - n_date)
                t_days = float(p_date - n_date)
                result[p_date] = result[p_date] + value * (n_days / t_days)
                result[n_date] = result[n_date] + value * (p_days / t_days)

        return result

    # --- diagnostics -----------------------------------------------------

    def __repr__(self) -> str:
        body = ", ".join(f"{d}: {v!r}" for d, v in self.items())
        return f"TimeBasket({{{body}}})"


__all__ = ["TimeBasket"]
