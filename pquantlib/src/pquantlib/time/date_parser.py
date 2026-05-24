"""Date string parsing.

# C++ parity: ql/utilities/dataparsers.hpp ``class DateParser`` (v1.42.1).

The C++ static class is flattened to a module of free functions:

- ``parse_iso(s)``: parses ``"YYYY-MM-DD"`` exactly as the C++ implementation
  (validates the 10-character shape and the dashes at positions 4 and 7).
- ``parse_formatted(s, fmt)``: deliberate divergence from C++. The C++
  version delegates to boost::date_time's locale-driven facets with
  boost-style format strings; this port uses Python's ``datetime.strptime``
  format codes. Common codes (``%Y``, ``%m``, ``%d``) work identically.
  Code-by-code mappings between the two are documented at
  https://www.boost.org/doc/libs/release/doc/html/date_time/date_time_io.html
  vs the Python ``strptime`` docs.
"""

from __future__ import annotations

from datetime import datetime

from pquantlib import qassert
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def parse_iso(s: str) -> Date:
    """Parse ``YYYY-MM-DD`` (10 chars, dashes at positions 4 and 7)."""
    qassert.require(
        len(s) == 10 and s[4] == "-" and s[7] == "-",
        "invalid format",
    )
    try:
        year = int(s[0:4])
        month = int(s[5:7])
        day = int(s[8:10])
    except ValueError as exc:
        qassert.fail(f"invalid format: {exc}")
    return Date.from_ymd(day, Month(month), year)


def parse_formatted(s: str, fmt: str) -> Date:
    """Parse ``s`` using Python ``strptime`` format codes.

    Divergence from C++ ``DateParser::parseFormatted`` is documented in this
    module's docstring.
    """
    try:
        dt = datetime.strptime(s, fmt)
    except ValueError as exc:
        qassert.fail(f"unable to parse '{s}' with format '{fmt}': {exc}")
    return Date.from_ymd(dt.day, Month(dt.month), dt.year)
