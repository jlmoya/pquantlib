"""Fixed-rate bonds: clean/dirty price, yield-to-maturity, accrued, and DV01."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import bonds

ui.setup_page("Bonds", icon="🏦")
ui.about_sidebar()
ui.hero(
    "Fixed-rate bonds",
    "Discounted-cashflow pricing with full clean/dirty/yield mechanics.",
    pills=["FixedRateBond", "DiscountingBondEngine"],
)

with st.sidebar:
    coupon = st.slider("Coupon", 0.0, 0.12, 0.05, 0.0025, format="%.4f")
    years = st.slider("Maturity (years)", 1, 30, 5)
    curve_rate = st.slider("Discount rate", 0.0, 0.12, 0.04, 0.0025, format="%.4f")
    freq = st.selectbox("Coupon frequency", [("Semiannual", 6), ("Annual", 12)], format_func=lambda x: x[0])
    face = st.number_input("Face value", 1.0, 1000.0, 100.0)

res = bonds.price_bond(coupon, years, curve_rate, freq[1], face)
ui.metric_cards(
    [
        ("Clean price", f"{res.clean_price:.4f}", None),
        ("Dirty price", f"{res.dirty_price:.4f}", f"accrued {res.accrued:.4f}"),
        ("Yield to maturity", f"{res.ytm:.4%}", None),
        ("NPV", f"{res.npv:.4f}", None),
    ]
)
prem = "premium" if res.clean_price > face else "discount"
ui.note(
    f"With a {coupon:.2%} coupon discounted at {curve_rate:.2%}, the bond trades at a "
    f"<b>{prem}</b> to its {face:g} face value — coupon vs. market-rate, exactly as theory predicts."
)

st.divider()
left, right = st.columns([1.2, 1])
with left:
    st.subheader("Price vs. discount rate (DV01)")
    grid = bonds.default_rate_grid()
    rates, prices = bonds.price_vs_curve(coupon, years, freq[1], face, grid)
    fig = charts.line(rates, prices, xtitle="Discount rate", ytitle="Clean price", name="Clean price")
    fig.add_vline(x=curve_rate, line=dict(color=charts.ACCENT3, width=1.5, dash="dot"))
    fig.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)
    ui.note("The convex price-yield curve is the source of duration and convexity risk.")

with right:
    st.subheader("Cashflows")
    df = pd.DataFrame([(cf.date, cf.amount) for cf in res.cashflows], columns=["Date", "Amount"])
    st.dataframe(df.style.format({"Amount": "{:.4f}"}), use_container_width=True, hide_index=True, height=380)
