"""Time machinery: calendars, day-count conventions, schedule generation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pquantlib_showcase import ui
from pquantlib_showcase.quant import dates

ui.setup_page("Time & Market Data", icon="📅")
ui.about_sidebar()
ui.hero(
    "Time & market data",
    "The plumbing under every price: calendars, day counts, and schedules.",
    pills=["pquantlib.time", "daycounters"],
)

with st.sidebar:
    st.subheader("Date")
    y = st.number_input("Year", 2000, 2099, 2026)
    m = st.selectbox("Month", list(range(1, 13)), index=5)
    d = st.number_input("Day", 1, 31, 15)
    cal_name = st.selectbox("Calendar", list(dates.CALENDARS))

st.subheader("Calendar & business-day adjustment")
facts = dates.calendar_facts(int(y), int(m), int(d), cal_name)
ui.metric_cards(
    [
        (facts.iso, facts.weekday, None),
        ("Business day?", "Yes" if facts.is_business_day else "No", None),
        ("Holiday?", "Yes" if facts.is_holiday else "No", None),
    ]
)
ui.note(
    f"Adjusted to a good business day — Following: <b>{facts.following}</b> · "
    f"Modified Following: <b>{facts.modified_following}</b> · Preceding: <b>{facts.preceding}</b>"
)

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Day-count conventions")
    st.caption("Year fraction from the chosen date over the horizon below — conventions disagree by design.")
    horizon = st.slider("Horizon (months)", 1, 36, 12)
    rows = dates.day_count_table(int(y), int(m), int(d), int(horizon))
    st.dataframe(
        pd.DataFrame(rows, columns=["Convention", "Day count", "Year fraction"]).style.format(
            {"Year fraction": "{:.6f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Coupon schedule")
    st.caption("Backward-generated, modified-following adjusted payment dates.")
    tenor = st.selectbox(
        "Coupon frequency",
        [("Monthly", 1), ("Quarterly", 3), ("Semiannual", 6), ("Annual", 12)],
        index=2,
        format_func=lambda x: x[0],
    )
    yrs = st.slider("Maturity (years)", 1, 10, 3)
    sched = dates.schedule_table(int(y), int(m), int(d), tenor[1], int(yrs), cal_name)
    st.dataframe(
        pd.DataFrame({"#": range(1, len(sched) + 1), "Payment date": sched}),
        use_container_width=True,
        hide_index=True,
        height=320,
    )
