"""Streamlit presentation helpers — consistent header, theme, and metric cards."""

from __future__ import annotations

from collections.abc import Sequence

import pquantlib
import streamlit as st

_CSS = """
<style>
  /* --- Natural full-page scroll -------------------------------------------
     Streamlit nests the main area and the sidebar in independent, fixed-height
     scroll regions locked to the viewport, which hides the bottom of tall
     pages. Letting the app flow naturally makes the whole window scroll, so a
     normal scroll gesture reaches the bottom of both the content and sidebar. */
  [data-testid="stApp"],
  [data-testid="stAppViewContainer"] { height: auto !important; overflow: visible !important; }
  section[data-testid="stMain"] { height: auto !important; overflow: visible !important; }
  section[data-testid="stSidebar"] { height: auto !important; min-height: 100vh; }
  [data-testid="stSidebarContent"],
  [data-testid="stSidebarUserContent"] { height: auto !important; overflow: visible !important; }

  .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1100px; }
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-bottom: 2.5rem; }
  h1, h2, h3 { font-family: Inter, system-ui, sans-serif; letter-spacing: -0.01em; }
  .pq-hero { background: linear-gradient(120deg,#1e3a8a 0%,#2563eb 55%,#0ea5e9 100%);
             color: #fff; padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.4rem; }
  .pq-hero h1 { color:#fff; margin:0 0 .25rem 0; font-size:1.7rem; }
  .pq-hero p  { color:#dbeafe; margin:0; font-size:0.98rem; }
  .pq-pill { display:inline-block; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;
             padding:2px 10px; border-radius:999px; font-size:0.78rem; margin-right:6px; font-weight:600; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; }
  .pq-note { color:#475569; font-size:0.9rem; border-left:3px solid #93c5fd; padding:.2rem 0 .2rem .8rem; }
</style>
"""


def setup_page(title: str, icon: str = "📈") -> None:
    """Call once at the top of every page."""
    st.set_page_config(page_title=f"{title} · PQuantLib", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pills: Sequence[str] = ()) -> None:
    pill_html = "".join(f'<span class="pq-pill">{p}</span>' for p in pills)
    st.markdown(
        f'<div class="pq-hero"><h1>{title}</h1><p>{subtitle}</p>'
        f'{"<div style=\"margin-top:.7rem\">" + pill_html + "</div>" if pills else ""}</div>',
        unsafe_allow_html=True,
    )


def about_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### PQuantLib `v{pquantlib.__version__}`")
        st.caption(
            "A pure-Python port of QuantLib v1.42.1 — functional 1:1 with the C++ "
            "library across **4048** cross-validated tests."
        )
        st.markdown(
            "Every number on these pages is computed **live** by PQuantLib: curves, "
            "instruments, processes, pricing engines, models and calibration."
        )
        st.divider()
        st.caption("Use the page menu above to explore each domain.")


def note(text: str) -> None:
    st.markdown(f'<div class="pq-note">{text}</div>', unsafe_allow_html=True)


def metric_cards(items: Sequence[tuple[str, str, str | None]]) -> None:
    """Render a row of metric cards. Each item is (label, value, delta-or-None)."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items, strict=True):
        with col:
            st.metric(label, value, delta)
