"""IMM / ASX / ECB must default their reference date to the evaluation date.

C++ resolves an unset reference date to ``Settings::instance().evaluationDate()``
in every one of these entry points:

* ``IMM::date``      — imm.cpp:130-132
* ``IMM::nextDate``  — imm.cpp:165-167
* ``ASX::date``      — asx.cpp:95-97
* ``ASX::nextDate``  — asx.cpp:119-121
* ``ECB::date``      — ecb.cpp:192-194
* ``ECB::nextDate``  — ecb.cpp:226-228 and :238-240

This port defaulted all seven to ``Date.todays_date()``, i.e. the machine's wall
clock. That is invisible while the wall clock happens to sit in the same decade
as the pinned evaluation date, and silently wrong otherwise: a two-character IMM
code carries no century, so ``"U6"`` resolves to 2016, 2026 or 2036 depending on
when the process runs.

These tests pin an evaluation date far from any plausible wall clock, so they
fail if the default ever reverts to "today".
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time import asx, ecb, imm
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# Deliberately far from any wall clock this code will run under, in both
# directions, so neither a past nor a future "today" can accidentally agree.
_PAST = Date.from_ymd(15, Month.June, 1996)
_FUTURE = Date.from_ymd(15, Month.June, 2046)


@pytest.fixture
def _restore_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    saved = ObservableSettings().evaluation_date
    yield
    ObservableSettings().evaluation_date = saved


@pytest.mark.usefixtures("_restore_evaluation_date")
@pytest.mark.parametrize("pinned", [_PAST, _FUTURE])
def test_imm_date_defaults_to_the_evaluation_date(pinned: Date) -> None:
    ObservableSettings().evaluation_date = pinned
    assert imm.date("U6") == imm.date("U6", reference_date=pinned)


@pytest.mark.usefixtures("_restore_evaluation_date")
@pytest.mark.parametrize("pinned", [_PAST, _FUTURE])
def test_imm_next_date_defaults_to_the_evaluation_date(pinned: Date) -> None:
    ObservableSettings().evaluation_date = pinned
    assert imm.next_date() == imm.next_date(pinned)


@pytest.mark.usefixtures("_restore_evaluation_date")
@pytest.mark.parametrize("pinned", [_PAST, _FUTURE])
def test_asx_next_date_defaults_to_the_evaluation_date(pinned: Date) -> None:
    ObservableSettings().evaluation_date = pinned
    assert asx.next_date() == asx.next_date(pinned)


# ECB's known-date table is finite (200 dates, 2005-01-19 .. 2024-12-18) and
# C++ raises past its end (ecb.cpp:233-234), so the ECB cases must sit inside
# that window rather than reuse the far-future pin above. Two dates ten years
# apart still discriminate against any plausible wall clock.
_ECB_EARLY = Date.from_ymd(15, Month.June, 2006)
_ECB_LATE = Date.from_ymd(15, Month.June, 2016)


@pytest.mark.usefixtures("_restore_evaluation_date")
@pytest.mark.parametrize("pinned", [_ECB_EARLY, _ECB_LATE])
def test_ecb_next_date_defaults_to_the_evaluation_date(pinned: Date) -> None:
    ObservableSettings().evaluation_date = pinned
    assert ecb.next_date() == ecb.next_date(pinned)


@pytest.mark.usefixtures("_restore_evaluation_date")
def test_ecb_raises_past_the_known_table_as_cpp_does() -> None:
    """Past the last known ECB date C++ raises rather than extrapolating.

    ecb.cpp:233-234. Pinned so the evaluation-date fix cannot be "fixed" later
    by inventing dates beyond the table.
    """
    ObservableSettings().evaluation_date = _FUTURE
    with pytest.raises(LibraryException, match="are unknown"):
        ecb.next_date()


@pytest.mark.usefixtures("_restore_evaluation_date")
def test_the_two_pinned_dates_actually_discriminate() -> None:
    """Guard the guard.

    If the two evaluation dates produced the same answer, every test above
    would pass against a wall-clock default and prove nothing. This asserts
    the market discriminates, so the tests have teeth.
    """
    ObservableSettings().evaluation_date = _PAST
    past_imm, past_asx = imm.next_date(), asx.next_date()
    ObservableSettings().evaluation_date = _ECB_EARLY
    early_ecb = ecb.next_date()

    ObservableSettings().evaluation_date = _FUTURE
    assert imm.next_date() != past_imm
    assert asx.next_date() != past_asx
    ObservableSettings().evaluation_date = _ECB_LATE
    assert ecb.next_date() != early_ecb


@pytest.mark.usefixtures("_restore_evaluation_date")
def test_unset_evaluation_date_still_falls_back_to_today() -> None:
    """``None`` means "use today", and that behaviour is unchanged.

    ``evaluation_date_or_today`` is the port's spelling of C++ reading a
    Settings singleton that was never assigned. The fix redirects the default
    through Settings; it does not remove the fallback.
    """
    ObservableSettings().evaluation_date = None
    assert imm.next_date() == imm.next_date(Date.todays_date())
