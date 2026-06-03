"""Smoke suite for the sample programs.

Mirrors the Java ``AllSamples`` driver: for every sample in the COMPLETE /
INCOMPLETE buckets, import the module and call ``run()``, asserting it
completes without raising. The PENDING bucket is exercised by a skipped test
that mirrors the Java ``@Ignore`` on un-ported samples.

Adding a sample name to ``all_samples.COMPLETE`` automatically tests it here —
no edit to this file is needed.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol

import pytest

from pquantlib_samples import all_samples, repo, swap


class _Sample(Protocol):
    def run(self) -> None: ...


def _import_sample(mod: str) -> ModuleType:
    return importlib.import_module(f"pquantlib_samples.{mod}")


def _run_sample(mod: str, capsys: pytest.CaptureFixture[str]) -> str:
    """Import a sample module, call ``run()``, and return its captured stdout."""
    sample: _Sample = _import_sample(mod)  # type: ignore[assignment]
    sample.run()
    captured = capsys.readouterr()
    # A sample is only useful if it actually printed something (parity with the
    # Java originals, which all write to stdout).
    assert captured.out.strip(), f"sample {mod!r} produced no stdout"
    return captured.out


@pytest.mark.parametrize("mod", all_samples.COMPLETE)
def test_complete_samples_run(mod: str, capsys: pytest.CaptureFixture[str]) -> None:
    _run_sample(mod, capsys)


@pytest.mark.parametrize("mod", all_samples.INCOMPLETE)
def test_incomplete_samples_run(mod: str, capsys: pytest.CaptureFixture[str]) -> None:
    _run_sample(mod, capsys)


@pytest.mark.skip(reason="pending bucket — mirrors Java @Ignore")
@pytest.mark.parametrize("mod", all_samples.PENDING)
def test_pending_samples(mod: str) -> None:  # pragma: no cover
    _import_sample(mod).run()  # type: ignore[attr-defined]


# --- light numeric cross-checks on the samples that print a pinnable value ---
# These are NOT a full numeric-parity gate (samples are validated primarily by
# run-to-completion); they pin the one headline number each financial sample
# prints, so a regression in the underlying core surfaces here too.


def test_swap_fair_rate_makes_npv_vanish() -> None:
    result = swap.compute()
    # Re-pricing at the fair fixed rate must collapse the NPV to ~0.
    assert abs(result.fair_npv_check) < 1e-6
    # Sanity: the 4% input coupon is above the ~3.09% fair rate, so a payer
    # swap has negative NPV.
    assert result.npv < 0.0
    assert 0.0 < result.fair_rate < result.input_fixed_rate


def test_repo_clean_forward_matches_fincad() -> None:
    result = repo.compute()
    # FINCAD reference clean forward price = 88.2408 (BFWD.htm example);
    # our NullCalendar / 0-settlement-day assumptions reproduce it to ~1e-3.
    assert abs(result.clean_forward_price - 88.2408) < 1e-2
    # The bond was priced from its clean price 89.97693786, so re-reading the
    # clean price round-trips to the input.
    assert abs(result.bond_clean_price - 89.97693786) < 1e-6
