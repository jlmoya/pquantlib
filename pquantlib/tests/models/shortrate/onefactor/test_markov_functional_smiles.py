"""MarkovFunctional's shifted-smile and CustomSmile calibration paths.

Every expected value comes from
``migration-harness/cpp/probes/v143_models_markovfunctional/probe.cpp`` run
against C++ QuantLib v1.43; the reference lives at
``migration-harness/references/v143/models/markovfunctional.json``.

These two paths are the ones ``cluster_w1a`` does not reach:

* a NON-ZERO smile shift, which is what makes
  ``lowerRateBound_ - smileSection_->shift()`` differ from ``lowerRateBound_``
  in ``updateSmiles`` (markovfunctional.cpp:426-437) and in the y-sweep
  (markovfunctional.cpp:558-559, :565-567), and what the ``shift`` argument of
  ``marketSwapRate`` (markovfunctional.cpp:809-823) moves the lower Brent
  bracket by;
* ``ModelSettings::CustomSmile``, i.e. ``MarkovFunctional::CustomSmileSection``
  and ``MarkovFunctional::CustomSmileFactory`` (markovfunctional.hpp:103, :108).

Two tenors are probed, for different reasons:

* **1Y tenor, expiries one year apart** — every swaption's single payment date
  is already a calibration expiry, so C++'s back-fill loop
  (markovfunctional.cpp:169-203) adds nothing and calibrates exactly the three
  input points, same as this port. That makes the comparison confound-free, so
  it is what the shift and CustomSmile branches are asserted against, at LOOSE.
* **10Y tenor** — the back-fill fires hard, turning 3 input expiries into 12
  calibration points. Those sections are NOT asserted value-by-value; they are
  the exhibit for ``test_backfill_gap_is_exactly_this_big``, which measures the
  pre-existing gap rather than hiding it inside a widened tolerance.

The probe sets ``Settings::instance().evaluationDate() = TODAY`` at the top of
``emitConfiguration``; the ``_pinned_evaluation_date`` autouse fixture below
pins the same date (read out of the reference's ``today_serial``) and restores
it in teardown.

``ProbeCustomSmileSection.inverse_digital_call`` is a deliberate mirror of the
probe's closed form rather than a Brent inversion — two independent inversions
would agree only to solver tolerance and would hide a threading error
underneath that noise. What is under test is that the branch is taken and the
arguments reach it, so both sides compute the same exact function.

KNOWN FAILING — read before "fixing" it
---------------------------------------
``test_baseline_1y``, ``test_shifted_lognormal_smile`` and ``test_custom_smile``
currently FAIL at LOOSE, and are deliberately left failing rather than given a
fitted tolerance or an xfail. The measured relative error profile, with the
back-fill confound eliminated (both sides calibrate the same 3 points and reach
the same numeraire date, asserted above):

    t = 0   rel = 0            exact, all three configurations
    t = 3   rel ~ 1e-7 .. 1e-6 the FIRST row calibrated, from the curve alone
    t = 2   rel ~ 1e-6 .. 4e-4
    t = 1   rel ~ 4e-6 .. 9e-4

That is the signature of a small systematic difference seeded in the first
calibrated row and amplified by the backward sweep, not of round-off. Two
causes have already been found and fixed and are NOT it:

* the omitted back-fill loop (markovfunctional.cpp:169-203) — still present as
  a defect, but it is inert at this tenor, which is why this configuration was
  chosen; see ``test_backfill_gap_is_exactly_this_big``;
* the wrong cubic configuration — the port used a natural spline where C++ uses
  ``Spline, monotonic=true, Lagrange/Lagrange`` (markovfunctional.cpp:221-223
  and :478-482). Fixed in ``_mf_cubic``; it moved the numbers by ~4e-14, so it
  was not the driver either.

The prime remaining suspect is the module docstring's claim that
``numpy.polynomial.hermite.hermgauss`` "matches QL to TIGHT" — a 1e-7 residual
in the first calibrated row is what a quadrature difference would look like,
and that claim has not been verified against ``GaussHermiteIntegration``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Final

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.swap.euribor_swap_isda_fix_a import EuriborSwapIsdaFixA
from pquantlib.models.shortrate.onefactor.markov_functional import (
    CustomSmileFactory,
    CustomSmileSection,
    MarkovFunctional,
    MarkovFunctionalSettings,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF: Final[dict[str, Any]] = load_reference("v143/models/markovfunctional")
_TODAY: Final[Date] = Date(int(_REF["today_serial"]))


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


class ProbeCustomSmileSection(CustomSmileSection):
    """Mirror of the probe's ``ProbeCustomSmileSection``."""

    def __init__(self, source: SmileSection, atm: float) -> None:
        super().__init__(
            exercise_date=source.exercise_date(),
            day_counter=source.day_counter(),
            reference_date=source.reference_date(),
            volatility_type=source.volatility_type(),
            shift=source.shift(),
        )
        self._source = source
        self._atm = atm

    def min_strike(self) -> float:
        return self._source.min_strike()

    def max_strike(self) -> float:
        return self._source.max_strike()

    def atm_level(self) -> float:
        return self._atm

    def _volatility_impl(self, strike: float) -> float:
        return self._source.volatility(strike)

    def inverse_digital_call(self, price: float, discount: float = 1.0) -> float:
        # probe.cpp ProbeCustomSmileSection::inverseDigitalCall, verbatim.
        return self._atm * math.exp(-(price / discount - 0.5))


class ProbeCustomSmileFactory(CustomSmileFactory):
    def smile_section(self, source: SmileSection, atm: float) -> CustomSmileSection:
        return ProbeCustomSmileSection(source, atm)


def _build(section: dict[str, Any]) -> MarkovFunctional:
    """Rebuild the probe's ``emitConfiguration`` from the reference's own knobs."""
    custom = bool(section["custom_smile"])
    yts = FlatForward(_TODAY, SimpleQuote(0.03), Actual365Fixed())
    cal = TARGET()
    swaption_expiries = [
        cal.advance_period(_TODAY, Period(n, TimeUnit.Years)) for n in (1, 2, 3)
    ]
    swap_idx = EuriborSwapIsdaFixA(
        Period(int(section["tenor_years"]), TimeUnit.Years), yts, yts
    )
    return MarkovFunctional(
        term_structure=yts,
        reversion=0.01,
        volatility=[0.01, 0.01, 0.01, 0.01],
        smile_step_dates=swaption_expiries,
        swap_indexes=[swap_idx, swap_idx, swap_idx],
        swaption_volatilities=[SimpleQuote(0.20), SimpleQuote(0.20), SimpleQuote(0.20)],
        swaption_volatility_type=VolatilityType.ShiftedLognormal,
        swaption_shift=float(section["shift"]),
        settings=MarkovFunctionalSettings(
            custom_smile=custom,
            custom_smile_factory=ProbeCustomSmileFactory() if custom else None,
        ),
    )


def _assert_configuration(key: str) -> None:
    section = _REF[key]
    # Confound-free by construction: C++ calibrates the same three points we do.
    assert section["n_calibration_points"] == 3
    model = _build(section)
    assert model.get_numeraire_date().serial_number() == section["numeraire_date_serial"]
    loose(model.numeraire_time(), section["numeraire_time"])
    for row in section["numeraire"]:
        loose(
            model.numeraire(row["t"], row["y"]),
            row["value"],
            reason=f"{key} numeraire(t={row['t']}, y={row['y']})",
        )
    for row in section["zerobond_t0"]:
        loose(
            model.zerobond(row["T"], 0.0, 0.0),
            row["value"],
            reason=f"{key} zerobond(T={row['T']})",
        )


def test_baseline_1y() -> None:
    """Control: shift 0, no custom smile."""
    _assert_configuration("baseline_1y")


def test_shifted_lognormal_smile() -> None:
    """shift=0.02 against lowerRateBound=0.001, so the rate floor is NEGATIVE."""
    assert _REF["shifted_1y"]["shift"] == 0.02
    _assert_configuration("shifted_1y")


def test_custom_smile() -> None:
    """ModelSettings::CustomSmile — inverse_digital_call replaces the inversion."""
    assert _REF["custom_smile_1y"]["custom_smile"] is True
    _assert_configuration("custom_smile_1y")


def test_the_shift_is_not_discarded() -> None:
    """A port that accepted `shift` and dropped it would give baseline == shifted."""
    baseline = [row["value"] for row in _REF["baseline_1y"]["numeraire"]]
    shifted = [row["value"] for row in _REF["shifted_1y"]["numeraire"]]
    assert baseline != shifted

    got_baseline = _build(_REF["baseline_1y"])
    got_shifted = _build(_REF["shifted_1y"])
    assert [
        got_baseline.numeraire(r["t"], r["y"]) for r in _REF["baseline_1y"]["numeraire"]
    ] != [got_shifted.numeraire(r["t"], r["y"]) for r in _REF["shifted_1y"]["numeraire"]]


def test_the_custom_smile_branch_is_taken() -> None:
    """Likewise: a port that ignored the factory would give baseline == custom."""
    baseline = [row["value"] for row in _REF["baseline_1y"]["numeraire"]]
    custom = [row["value"] for row in _REF["custom_smile_1y"]["numeraire"]]
    assert baseline != custom

    got_baseline = _build(_REF["baseline_1y"])
    got_custom = _build(_REF["custom_smile_1y"])
    assert [
        got_baseline.numeraire(r["t"], r["y"]) for r in _REF["baseline_1y"]["numeraire"]
    ] != [
        got_custom.numeraire(r["t"], r["y"]) for r in _REF["custom_smile_1y"]["numeraire"]
    ]


def test_custom_smile_requires_a_factory() -> None:
    """C++ markovfunctional.cpp:413-417 dereferences customSmileFactory_."""
    yts = FlatForward(_TODAY, SimpleQuote(0.03), Actual365Fixed())
    cal = TARGET()
    expiries = [cal.advance_period(_TODAY, Period(1, TimeUnit.Years))]
    swap_idx = EuriborSwapIsdaFixA(Period(1, TimeUnit.Years), yts, yts)
    with pytest.raises(LibraryException):
        MarkovFunctional(
            term_structure=yts,
            reversion=0.01,
            volatility=[0.01, 0.01],
            smile_step_dates=expiries,
            swap_indexes=[swap_idx],
            swaption_volatilities=[SimpleQuote(0.20)],
            settings=MarkovFunctionalSettings(custom_smile=True),
        )


def test_backfill_gap_is_exactly_this_big() -> None:
    """Measure the omitted back-fill instead of hiding it inside a tolerance.

    C++ ``MarkovFunctional`` (markovfunctional.cpp:169-203) repeatedly adds
    calibration points at intermediate payment dates until the numeraire date is
    covered. This port takes the max payment date over the INPUT expiries and
    stops. The consequence is not calendar noise — which is what the
    ``test_markov_functional`` tolerance rationale used to claim before this was
    traced — it is a materially smaller calibration set whenever the swaption
    tenor outruns the expiry spacing.

    This test fails if the gap changes in either direction, so porting the
    back-fill will visibly retire it rather than silently pass.
    """
    section = _REF["baseline"]
    assert section["tenor_years"] == 10
    assert section["n_calibration_points"] == 12
    assert len(section["calibration_expiry_serials"]) == 12

    model = _build(section)
    # This port calibrates only the three input expiries...
    assert len(model.model_outputs().expiries) == 3
    assert [d.serial_number() for d in model.model_outputs().expiries] == (
        section["calibration_expiry_serials"][:3]
    )
    # ...so its numeraire date is the third point's last payment, three days
    # earlier than the date C++ reaches after back-filling.
    assert section["numeraire_date_serial"] - model.get_numeraire_date().serial_number() == 3

    # And at the 1Y tenor, where the back-fill is a no-op, there is no gap at
    # all — which is what makes the value assertions above meaningful.
    assert _REF["baseline_1y"]["n_calibration_points"] == 3
    assert (
        _build(_REF["baseline_1y"]).get_numeraire_date().serial_number()
        == _REF["baseline_1y"]["numeraire_date_serial"]
    )
