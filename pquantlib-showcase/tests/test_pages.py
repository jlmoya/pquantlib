"""Headless page smoke tests via Streamlit's AppTest harness.

Each page script is executed end-to-end with its default widget values (and,
for pages with a top-level branch selector, each branch). A page that raises is
a failed test — this is the regression net for the whole UI layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_ROOT = Path(__file__).resolve().parent.parent
_PAGES = [_ROOT / "app.py", *sorted((_ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.name)
def test_page_runs_without_exception(page: Path) -> None:
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception, f"{page.name} raised: {[str(e) for e in at.exception]}"


def test_yield_curve_bootstrap_branch() -> None:
    at = AppTest.from_file(str(_ROOT / "pages" / "2_Yield_Curves.py"), default_timeout=60).run()
    at.radio[0].set_value("Bootstrapped from deposits").run()
    assert not at.exception


def test_swap_ois_branch() -> None:
    at = AppTest.from_file(str(_ROOT / "pages" / "4_Interest_Rate_Swaps.py"), default_timeout=60).run()
    at.radio[0].set_value("OIS (vs SOFR)").run()
    assert not at.exception


def test_vanilla_put_branch() -> None:
    at = AppTest.from_file(str(_ROOT / "pages" / "5_Vanilla_Options.py"), default_timeout=120).run()
    at.radio[0].set_value("Put").run()
    assert not at.exception


def test_exotics_all_families() -> None:
    page = str(_ROOT / "pages" / "7_Exotic_Options.py")
    for family in ("Single barrier", "Double barrier", "Asian (geometric)"):
        at = AppTest.from_file(page, default_timeout=60).run()
        at.radio[0].set_value(family).run()
        assert not at.exception, f"exotic family {family!r} failed"
