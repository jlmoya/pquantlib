"""Interest-rate swaps: vanilla fixed-vs-Euribor and overnight-indexed (OIS)."""

from __future__ import annotations

import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import swaps

ui.setup_page("Interest-Rate Swaps", icon="🔁")
ui.about_sidebar()
ui.hero(
    "Interest-rate swaps",
    "Par rates, mark-to-market, and DV01 for vanilla and OIS swaps.",
    pills=["VanillaSwap", "OvernightIndexedSwap", "DiscountingSwapEngine"],
)

with st.sidebar:
    kind = st.radio("Swap type", ["Vanilla (vs Euribor 6M)", "OIS (vs SOFR)"])
    notional = st.number_input("Notional", 100_000.0, 100_000_000.0, 1_000_000.0, step=100_000.0)
    years = st.slider("Tenor (years)", 1, 30, 5)
    fixed_rate = st.slider("Fixed rate", 0.0, 0.08, 0.03, 0.0005, format="%.4f")
    curve_rate = st.slider("Curve level", 0.0, 0.08, 0.03, 0.0005, format="%.4f")

if kind.startswith("Vanilla"):
    res = swaps.price_vanilla_swap(notional, years, fixed_rate, curve_rate)
else:
    res = swaps.price_ois(notional, years, fixed_rate, curve_rate)

ui.metric_cards(
    [
        ("NPV (payer)", f"{res.npv:,.0f}", None),
        (
            "Fair (par) rate",
            f"{res.fair_rate:.4%}",
            f"{(fixed_rate - res.fair_rate) * 1e4:+.1f} bps vs fixed",
        ),
        ("Fixed-leg DV01", f"{abs(res.fixed_leg_bps):,.2f}", None),
    ]
)
ui.note(
    "A <b>payer</b> swap pays the fixed rate and receives floating. At the fair rate the "
    "NPV is zero by construction; away from it, the mark-to-market is the PV of the rate difference."
)

st.divider()
st.subheader("Mark-to-market vs. contractual fixed rate")
grid = swaps.default_fixed_rate_grid()
rates, npvs = swaps.npv_vs_fixed_rate(notional, years, curve_rate, grid)
fig = charts.line(rates, npvs, xtitle="Contractual fixed rate", ytitle="Swap NPV (payer)", name="NPV")
fig.add_hline(y=0.0, line=dict(color=charts.MUTED, width=1))
fig.add_vline(
    x=res.fair_rate,
    line=dict(color=charts.ACCENT3, width=1.5, dash="dot"),
    annotation_text=f"par {res.fair_rate:.3%}",
    annotation_position="top",
)
fig.update_xaxes(tickformat=".1%")
st.plotly_chart(fig, use_container_width=True)
ui.note("The NPV crosses zero exactly at the par rate — PQuantLib's solver and pricing agree.")
