"""Schedule — payment / coupon date list.

# C++ parity: ql/time/schedule.hpp + ql/time/schedule.cpp (v1.42.1).

The C++ class has two constructors:

1. ``Schedule(dates, calendar, ...)`` — wraps a caller-supplied date list
   (no plausibility check) plus optional generation metadata.
2. ``Schedule(effective, termination, tenor, calendar, conv, term_conv,
   rule, end_of_month, first, next_to_last)`` — rule-based generator
   producing the date list.

Python design notes:
- Schedule is a regular class with ``__init__`` (not a ``@dataclass``)
  because the rule-based constructor mutates internal state during
  generation. Once constructed it is immutable from the caller's point
  of view (``dates`` is exposed as a ``tuple``, ``is_regular`` as a
  ``tuple``).
- The rule-based path uses ``Schedule.from_rule(...)`` classmethod to
  avoid the C++ ctor-overload-by-positional-args ambiguity.
- The Settings.evaluation_date fallback in the C++ backward-rule path
  (when effective_date is null) is NOT ported here — Python callers
  must pass an explicit effective_date. Documented divergence.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.time import imm
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib.time.weekday import Weekday

_OLD_CDS_STUB_DAYS: int = 30
_QUARTER_MONTHS: int = 3

# Module-level null Date used as default for optional Date arguments
# (avoids ruff B008 by not constructing per call). Date is frozen+slots
# so this is safe to share.
_NULL_DATE: Date = Date()


def _next_twentieth(d: Date, rule: DateGeneration) -> Date:
    """Mirrors C++ anonymous-namespace ``nextTwentieth``."""
    result = Date.from_ymd(20, d.month(), d.year())
    if result < d:
        result = result + Period(1, TimeUnit.Months)
    if rule in (
        DateGeneration.TwentiethIMM,
        DateGeneration.OldCDS,
        DateGeneration.CDS,
        DateGeneration.CDS2015,
    ):
        m = int(result.month())
        if m % _QUARTER_MONTHS != 0:
            skip = _QUARTER_MONTHS - m % _QUARTER_MONTHS
            result = result + Period(skip, TimeUnit.Months)
    return result


def previous_twentieth(d: Date, rule: DateGeneration) -> Date:
    """Mirrors C++ free function ``previousTwentieth``."""
    result = Date.from_ymd(20, d.month(), d.year())
    if result > d:
        result = result - Period(1, TimeUnit.Months)
    if rule in (
        DateGeneration.TwentiethIMM,
        DateGeneration.OldCDS,
        DateGeneration.CDS,
        DateGeneration.CDS2015,
    ):
        m = int(result.month())
        if m % _QUARTER_MONTHS != 0:
            skip = m % _QUARTER_MONTHS
            result = result - Period(skip, TimeUnit.Months)
    return result


def allows_end_of_month(tenor: Period) -> bool:
    """Mirrors C++ free function ``allowsEndOfMonth``."""
    if tenor.units not in (TimeUnit.Months, TimeUnit.Years):
        return False
    return tenor >= Period(1, TimeUnit.Months)


class Schedule:
    """Payment / coupon schedule (sequence of dates)."""

    def __init__(
        self,
        dates: Sequence[Date],
        *,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
        termination_date_convention: BusinessDayConvention | None = None,
        tenor: Period | None = None,
        rule: DateGeneration | None = None,
        end_of_month: bool | None = None,
        is_regular: Sequence[bool] = (),
    ) -> None:
        """Date-list constructor.

        Mirrors C++ ``Schedule(const std::vector<Date>&, Calendar, BDC,
        optional<BDC>, optional<Period>, optional<Rule>, optional<bool>,
        std::vector<bool>)``.
        """
        if tenor is not None and not allows_end_of_month(tenor):
            self._end_of_month: bool | None = False
        else:
            self._end_of_month = end_of_month
        qassert.require(
            len(is_regular) == 0 or (len(dates) > 0 and len(is_regular) == len(dates) - 1),
            f"isRegular size ({len(is_regular)}) must be zero or equal to "
            f"the number of dates minus 1 ({len(dates) - 1 if dates else 0})",
        )
        self._tenor: Period | None = tenor
        self._calendar: Calendar = calendar if calendar is not None else NullCalendar()
        self._convention: BusinessDayConvention = convention
        self._termination_date_convention: BusinessDayConvention | None = termination_date_convention
        self._rule: DateGeneration | None = rule
        self._first_date: Date = Date()
        self._next_to_last_date: Date = Date()
        self._dates: tuple[Date, ...] = tuple(dates)
        self._is_regular: tuple[bool, ...] = tuple(is_regular)

    # --- factory: rule-based constructor ----------------------------------

    @classmethod
    def from_rule(
        cls,
        effective_date: Date,
        termination_date: Date,
        tenor: Period,
        calendar: Calendar,
        convention: BusinessDayConvention,
        termination_date_convention: BusinessDayConvention,
        rule: DateGeneration,
        end_of_month: bool,
        first_date: Date = _NULL_DATE,
        next_to_last_date: Date = _NULL_DATE,
    ) -> Schedule:
        """Mirrors C++ rule-based ``Schedule(eff, term, tenor, cal, conv, ...)``."""
        qassert.require(termination_date != Date(), "null termination date")
        qassert.require(
            effective_date != Date(),
            "null effective date (Settings.evaluation_date fallback not ported — "
            "pass an explicit effective_date)",
        )
        qassert.require(
            effective_date < termination_date,
            f"effective date ({effective_date}) later than or equal to termination date ({termination_date})",
        )

        # Normalize first / next_to_last (drop if equal to bracketing date).
        first = Date() if first_date == effective_date else first_date
        next_to_last = Date() if next_to_last_date == termination_date else next_to_last_date

        # Zero-length tenor forces Zero rule.
        if tenor.length == 0:
            rule = DateGeneration.Zero
        else:
            qassert.require(tenor.length > 0, f"non positive tenor ({tenor}) not allowed")

        cls._validate_anchor("first date", first, rule, effective_date, termination_date)
        cls._validate_anchor("next to last date", next_to_last, rule, effective_date, termination_date)

        eom_flag = end_of_month if allows_end_of_month(tenor) else False

        dates: list[Date] = []
        is_regular: list[bool] = []
        null_cal = NullCalendar()
        seed = effective_date  # default seed; overwritten below where relevant

        if rule == DateGeneration.Zero:
            dates.extend([effective_date, termination_date])
            is_regular.append(True)
            tenor = Period(0, TimeUnit.Years)
        elif rule == DateGeneration.Backward:
            seed = cls._gen_backward(
                dates,
                is_regular,
                effective_date,
                termination_date,
                tenor,
                calendar,
                convention,
                eom_flag,
                first,
                next_to_last,
                null_cal,
            )
        else:
            # Forward, Twentieth, TwentiethIMM, ThirdWednesday,
            # ThirdWednesdayInclusive, OldCDS, CDS, CDS2015 all share the
            # forward-walking generator with rule-specific stubs.
            if rule in (
                DateGeneration.Twentieth,
                DateGeneration.TwentiethIMM,
                DateGeneration.ThirdWednesday,
                DateGeneration.ThirdWednesdayInclusive,
                DateGeneration.OldCDS,
                DateGeneration.CDS,
                DateGeneration.CDS2015,
            ):
                qassert.require(
                    not eom_flag,
                    f"endOfMonth convention incompatible with {rule.name} date generation rule",
                )
            seed = cls._gen_forward(
                dates,
                is_regular,
                effective_date,
                termination_date,
                tenor,
                calendar,
                convention,
                termination_date_convention,
                eom_flag,
                first,
                next_to_last,
                rule,
                null_cal,
            )

        cls._post_generation_adjustments(
            dates, calendar, convention, termination_date_convention, rule, seed, eom_flag
        )
        dates_t, is_regular_t = cls._final_safety_dedup(dates, is_regular)

        qassert.require(
            len(dates_t) > 1,
            f"degenerate single date schedule (seed: {seed}, "
            f"effective: {effective_date}, termination: {termination_date}, rule: {rule.name})",
        )

        s = cls(
            dates_t,
            calendar=calendar,
            convention=convention,
            termination_date_convention=termination_date_convention,
            tenor=tenor,
            rule=rule,
            end_of_month=eom_flag,
            is_regular=is_regular_t,
        )
        s._first_date = first
        s._next_to_last_date = next_to_last
        return s

    # --- validation helpers (called from from_rule) ------------------------

    @staticmethod
    def _validate_anchor(
        label: str,
        anchor: Date,
        rule: DateGeneration,
        effective_date: Date,
        termination_date: Date,
    ) -> None:
        if anchor == Date():
            return
        if rule in (DateGeneration.Backward, DateGeneration.Forward):
            if label == "first date":
                qassert.require(
                    anchor > effective_date and anchor <= termination_date,
                    f"first date ({anchor}) out of effective-termination "
                    f"date range ({effective_date}, {termination_date}]",
                )
            else:  # next to last date
                qassert.require(
                    anchor >= effective_date and anchor < termination_date,
                    f"next to last date ({anchor}) out of effective-termination "
                    f"date range [{effective_date}, {termination_date})",
                )
            return
        if rule == DateGeneration.ThirdWednesday:
            qassert.require(
                imm.is_imm_date(anchor, main_cycle=False), f"{label} ({anchor}) is not an IMM date"
            )
            return
        qassert.fail(f"{label} incompatible with {rule.name} date generation rule")

    # --- generation: Backward ---------------------------------------------

    @staticmethod
    def _gen_backward(
        dates: list[Date],
        is_regular: list[bool],
        effective_date: Date,
        termination_date: Date,
        tenor: Period,
        calendar: Calendar,
        convention: BusinessDayConvention,
        eom_flag: bool,
        first: Date,
        next_to_last: Date,
        null_cal: NullCalendar,
    ) -> Date:
        """Mirrors the Backward branch of the C++ Schedule constructor."""
        dates.append(termination_date)
        seed = termination_date

        if next_to_last != Date():
            dates.append(next_to_last)
            temp = null_cal.advance_period(seed, -tenor, convention, eom_flag)
            is_regular.append(temp == next_to_last)
            seed = next_to_last

        exit_date = first if first != Date() else effective_date
        periods = 1
        while True:
            temp = null_cal.advance_period(seed, -periods * tenor, convention, eom_flag)
            if temp < exit_date:
                if first != Date() and (
                    calendar.adjust(dates[-1], convention) != calendar.adjust(first, convention)
                ):
                    dates.append(first)
                    is_regular.append(
                        null_cal.advance_period(dates[-2], -tenor, convention, eom_flag) == first
                    )
                break
            if calendar.adjust(dates[-1], convention) != calendar.adjust(temp, convention):
                dates.append(temp)
                is_regular.append(True)
            periods += 1

        if calendar.adjust(dates[-1], convention) != calendar.adjust(effective_date, convention):
            dates.append(effective_date)
            is_regular.append(
                null_cal.advance_period(dates[-2], -tenor, convention, eom_flag) == effective_date
            )

        dates.reverse()
        is_regular.reverse()
        return seed

    # --- generation: Forward / Twentieth / ThirdWednesday / CDS family ----

    @staticmethod
    def _gen_forward(
        dates: list[Date],
        is_regular: list[bool],
        effective_date: Date,
        termination_date: Date,
        tenor: Period,
        calendar: Calendar,
        convention: BusinessDayConvention,
        termination_date_convention: BusinessDayConvention,
        eom_flag: bool,
        first: Date,
        next_to_last: Date,
        rule: DateGeneration,
        null_cal: NullCalendar,
    ) -> Date:
        # CDS / CDS2015 may need a leading prev-twentieth + stub.
        if rule in (DateGeneration.CDS, DateGeneration.CDS2015):
            prev20 = previous_twentieth(effective_date, rule)
            if calendar.adjust(prev20, convention) > effective_date:
                dates.append(prev20 - Period(_QUARTER_MONTHS, TimeUnit.Months))
                is_regular.append(True)
            dates.append(prev20)
        else:
            dates.append(effective_date)

        seed = dates[-1]

        if first != Date():
            dates.append(first)
            temp = null_cal.advance_period(seed, tenor, convention, eom_flag)
            is_regular.append(temp == first)
            seed = first
        elif rule in (
            DateGeneration.Twentieth,
            DateGeneration.TwentiethIMM,
            DateGeneration.OldCDS,
            DateGeneration.CDS,
            DateGeneration.CDS2015,
        ):
            next20 = _next_twentieth(effective_date, rule)
            if rule == DateGeneration.OldCDS and next20 - effective_date < _OLD_CDS_STUB_DAYS:
                next20 = _next_twentieth(next20 + 1, rule)
            if next20 != effective_date:
                dates.append(next20)
                is_regular.append(rule in (DateGeneration.CDS, DateGeneration.CDS2015))
                seed = next20

        exit_date = next_to_last if next_to_last != Date() else termination_date
        periods = 1
        while True:
            temp = null_cal.advance_period(seed, periods * tenor, convention, eom_flag)
            if temp > exit_date:
                if next_to_last != Date() and (
                    calendar.adjust(dates[-1], convention) != calendar.adjust(next_to_last, convention)
                ):
                    dates.append(next_to_last)
                    is_regular.append(
                        null_cal.advance_period(dates[-2], tenor, convention, eom_flag) == next_to_last
                    )
                break
            if calendar.adjust(dates[-1], convention) != calendar.adjust(temp, convention):
                dates.append(temp)
                is_regular.append(True)
            periods += 1

        if calendar.adjust(dates[-1], termination_date_convention) != calendar.adjust(
            termination_date, termination_date_convention
        ):
            if rule in (
                DateGeneration.Twentieth,
                DateGeneration.TwentiethIMM,
                DateGeneration.OldCDS,
                DateGeneration.CDS,
                DateGeneration.CDS2015,
            ):
                dates.append(_next_twentieth(termination_date, rule))
                is_regular.append(True)
            else:
                dates.append(termination_date)
                is_regular.append(False)
        return seed

    # --- post-generation pass (ThirdWednesday remap + final adjust) -------

    @staticmethod
    def _post_generation_adjustments(
        dates: list[Date],
        calendar: Calendar,
        convention: BusinessDayConvention,
        termination_date_convention: BusinessDayConvention,
        rule: DateGeneration,
        seed: Date,
        eom_flag: bool,
    ) -> None:
        if rule == DateGeneration.ThirdWednesday:
            for i in range(1, len(dates) - 1):
                dates[i] = Date.nth_weekday(3, Weekday.Wednesday, dates[i].month(), dates[i].year())
        elif rule == DateGeneration.ThirdWednesdayInclusive:
            for i in range(len(dates)):
                dates[i] = Date.nth_weekday(3, Weekday.Wednesday, dates[i].month(), dates[i].year())

        # First date adjusted unless OldCDS.
        if convention != BusinessDayConvention.Unadjusted and rule != DateGeneration.OldCDS:
            dates[0] = calendar.adjust(dates[0], convention)

        # Termination date is NOT adjusted unless explicitly non-Unadjusted
        # AND rule is not CDS / CDS2015 (per ISDA).
        if termination_date_convention != BusinessDayConvention.Unadjusted and rule not in (
            DateGeneration.CDS,
            DateGeneration.CDS2015,
        ):
            dates[-1] = calendar.adjust(dates[-1], termination_date_convention)

        # End-of-month adjustment for inner dates.
        if eom_flag and calendar.is_end_of_month(seed):
            for i in range(1, len(dates) - 1):
                dates[i] = calendar.adjust(Date.end_of_month(dates[i]), convention)
        else:
            for i in range(1, len(dates) - 1):
                dates[i] = calendar.adjust(dates[i], convention)

    @staticmethod
    def _final_safety_dedup(
        dates: list[Date], is_regular: list[bool]
    ) -> tuple[tuple[Date, ...], tuple[bool, ...]]:
        # Drop trailing near-duplicate (penultimate >= last).
        if len(dates) >= 2 and dates[-2] >= dates[-1]:
            if len(is_regular) >= 2:
                is_regular[-2] = dates[-2] == dates[-1]
            dates[-2] = dates[-1]
            dates.pop()
            is_regular.pop()
        # Drop leading near-duplicate (second <= first).
        if len(dates) >= 2 and dates[1] <= dates[0]:
            is_regular[1] = dates[1] == dates[0]
            dates[1] = dates[0]
            dates.pop(0)
            is_regular.pop(0)
        return tuple(dates), tuple(is_regular)

    # --- element access ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._dates)

    def size(self) -> int:
        return len(self._dates)

    def __getitem__(self, i: int) -> Date:
        return self._dates[i]

    def at(self, i: int) -> Date:
        return self._dates[i]

    def date(self, i: int) -> Date:
        return self._dates[i]

    @property
    def dates(self) -> tuple[Date, ...]:
        return self._dates

    def empty(self) -> bool:
        return len(self._dates) == 0

    def front(self) -> Date:
        qassert.require(len(self._dates) > 0, "no front date for empty schedule")
        return self._dates[0]

    def back(self) -> Date:
        qassert.require(len(self._dates) > 0, "no back date for empty schedule")
        return self._dates[-1]

    # --- inspectors --------------------------------------------------------

    @property
    def calendar(self) -> Calendar:
        return self._calendar

    @property
    def start_date(self) -> Date:
        qassert.require(len(self._dates) > 0, "empty Schedule: no start date")
        return self._dates[0]

    @property
    def end_date(self) -> Date:
        qassert.require(len(self._dates) > 0, "empty Schedule: no end date")
        return self._dates[-1]

    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return self._convention

    def has_tenor(self) -> bool:
        return self._tenor is not None

    @property
    def tenor(self) -> Period:
        qassert.require(self.has_tenor(), "full interface (tenor) not available")
        assert self._tenor is not None
        return self._tenor

    def has_termination_date_business_day_convention(self) -> bool:
        return self._termination_date_convention is not None

    @property
    def termination_date_business_day_convention(self) -> BusinessDayConvention:
        qassert.require(
            self.has_termination_date_business_day_convention(),
            "full interface (termination date bdc) not available",
        )
        assert self._termination_date_convention is not None
        return self._termination_date_convention

    def has_rule(self) -> bool:
        return self._rule is not None

    @property
    def rule(self) -> DateGeneration:
        qassert.require(self.has_rule(), "full interface (rule) not available")
        assert self._rule is not None
        return self._rule

    def has_end_of_month(self) -> bool:
        return self._end_of_month is not None

    @property
    def end_of_month(self) -> bool:
        qassert.require(self.has_end_of_month(), "full interface (end of month) not available")
        assert self._end_of_month is not None
        return self._end_of_month

    def has_is_regular(self) -> bool:
        return len(self._is_regular) > 0

    @property
    def is_regular(self) -> tuple[bool, ...]:
        qassert.require(self.has_is_regular(), "full interface (isRegular) not available")
        return self._is_regular

    def is_regular_at(self, i: int) -> bool:
        """1-based access matching C++ ``Schedule::isRegular(Size i)``."""
        qassert.require(self.has_is_regular(), "full interface (isRegular) not available")
        qassert.require(
            1 <= i <= len(self._is_regular),
            f"index ({i}) must be in [1, {len(self._is_regular)}]",
        )
        return self._is_regular[i - 1]

    # --- iteration + lookups -----------------------------------------------

    def __iter__(self):
        return iter(self._dates)

    def _lower_bound_index(self, ref: Date) -> int:
        """Index of the first date >= ``ref`` (mirrors C++ std::lower_bound)."""
        lo, hi = 0, len(self._dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._dates[mid] < ref:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def next_date(self, ref: Date) -> Date:
        idx = self._lower_bound_index(ref)
        return self._dates[idx] if idx < len(self._dates) else Date()

    def previous_date(self, ref: Date) -> Date:
        idx = self._lower_bound_index(ref)
        return self._dates[idx - 1] if idx > 0 else Date()

    # --- truncation --------------------------------------------------------

    def after(self, truncation_date: Date) -> Schedule:
        qassert.require(
            truncation_date < self._dates[-1],
            f"truncation date {truncation_date} must be before the last schedule date {self._dates[-1]}",
        )
        dates = list(self._dates)
        is_regular = list(self._is_regular)
        term_conv = self._termination_date_convention
        if truncation_date > dates[0]:
            while dates[0] < truncation_date:
                dates.pop(0)
                if is_regular:
                    is_regular.pop(0)
            if truncation_date != dates[0]:
                dates.insert(0, truncation_date)
                is_regular.insert(0, False)
                term_conv = BusinessDayConvention.Unadjusted
            else:
                term_conv = self._convention
        result = Schedule(
            dates,
            calendar=self._calendar,
            convention=self._convention,
            termination_date_convention=term_conv,
            tenor=self._tenor,
            rule=self._rule,
            end_of_month=self._end_of_month,
            is_regular=is_regular,
        )
        result._first_date = self._first_date if self._first_date > truncation_date else Date()
        result._next_to_last_date = (
            self._next_to_last_date if self._next_to_last_date > truncation_date else Date()
        )
        return result

    def until(self, truncation_date: Date) -> Schedule:
        qassert.require(
            truncation_date > self._dates[0],
            f"truncation date {truncation_date} must be later than schedule first date {self._dates[0]}",
        )
        dates = list(self._dates)
        is_regular = list(self._is_regular)
        term_conv = self._termination_date_convention
        if truncation_date < dates[-1]:
            while dates[-1] > truncation_date:
                dates.pop()
                if is_regular:
                    is_regular.pop()
            if truncation_date != dates[-1]:
                dates.append(truncation_date)
                is_regular.append(False)
                term_conv = BusinessDayConvention.Unadjusted
            else:
                term_conv = self._convention
        result = Schedule(
            dates,
            calendar=self._calendar,
            convention=self._convention,
            termination_date_convention=term_conv,
            tenor=self._tenor,
            rule=self._rule,
            end_of_month=self._end_of_month,
            is_regular=is_regular,
        )
        result._first_date = self._first_date if self._first_date < truncation_date else Date()
        result._next_to_last_date = (
            self._next_to_last_date if self._next_to_last_date < truncation_date else Date()
        )
        return result


# --- MakeSchedule builder --------------------------------------------------


class MakeSchedule:
    """Fluent builder for ``Schedule.from_rule(...)``.

    # C++ parity: ql/time/schedule.hpp ``class MakeSchedule`` (v1.42.1).
    """

    def __init__(self) -> None:
        self._effective_date: Date = Date()
        self._termination_date: Date = Date()
        self._tenor: Period | None = None
        self._calendar: Calendar | None = None
        self._convention: BusinessDayConvention | None = None
        self._termination_date_convention: BusinessDayConvention | None = None
        self._rule: DateGeneration = DateGeneration.Backward
        self._end_of_month: bool = False
        self._first_date: Date = Date()
        self._next_to_last_date: Date = Date()

    def from_date(self, d: Date) -> MakeSchedule:
        self._effective_date = d
        return self

    def to(self, d: Date) -> MakeSchedule:
        self._termination_date = d
        return self

    def with_tenor(self, tenor: Period) -> MakeSchedule:
        self._tenor = tenor
        return self

    def with_frequency(self, f: Frequency) -> MakeSchedule:
        self._tenor = Period.from_frequency(f)
        return self

    def with_calendar(self, c: Calendar) -> MakeSchedule:
        self._calendar = c
        return self

    def with_convention(self, c: BusinessDayConvention) -> MakeSchedule:
        self._convention = c
        return self

    def with_termination_date_convention(self, c: BusinessDayConvention) -> MakeSchedule:
        self._termination_date_convention = c
        return self

    def with_rule(self, r: DateGeneration) -> MakeSchedule:
        self._rule = r
        return self

    def forwards(self) -> MakeSchedule:
        self._rule = DateGeneration.Forward
        return self

    def backwards(self) -> MakeSchedule:
        self._rule = DateGeneration.Backward
        return self

    def with_end_of_month(self, flag: bool = True) -> MakeSchedule:
        self._end_of_month = flag
        return self

    def with_first_date(self, d: Date) -> MakeSchedule:
        self._first_date = d
        return self

    def with_next_to_last_date(self, d: Date) -> MakeSchedule:
        self._next_to_last_date = d
        return self

    def build(self) -> Schedule:
        qassert.require(self._effective_date != Date(), "effective date not provided")
        qassert.require(self._termination_date != Date(), "termination date not provided")
        qassert.require(self._tenor is not None, "tenor/frequency not provided")
        assert self._tenor is not None

        convention = self._convention
        if convention is None:
            convention = (
                BusinessDayConvention.Following
                if self._calendar is not None
                else BusinessDayConvention.Unadjusted
            )

        term_conv = self._termination_date_convention
        if term_conv is None:
            # ISDA default: same as convention.
            term_conv = convention

        calendar = self._calendar if self._calendar is not None else NullCalendar()

        return Schedule.from_rule(
            self._effective_date,
            self._termination_date,
            self._tenor,
            calendar,
            convention,
            term_conv,
            self._rule,
            self._end_of_month,
            self._first_date,
            self._next_to_last_date,
        )

    # Optional sugar so callers can write `Schedule(MakeSchedule().from_date(...).to(...))`
    # idiomatically — matches C++ ``operator Schedule() const`` conversion.
    def __call__(self) -> Schedule:
        return self.build()


__all__ = [
    "MakeSchedule",
    "Schedule",
    "allows_end_of_month",
    "previous_twentieth",
]
