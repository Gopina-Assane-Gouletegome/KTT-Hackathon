"""
dashboard.py — S2.T1.2 Stunting Risk Heatmap Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit.components.v1 import html as st_html
import json, os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rwanda Stunting Risk Dashboard",
    page_icon="🇷🇼",
    layout="wide",
)

# ── Load / score data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    scored_path = "data/households_scored.csv"
    if not os.path.exists(scored_path):
        import risk_scorer
        df = risk_scorer.score_all()
        df.drop(columns=["top_drivers"], errors="ignore").to_csv(scored_path, index=False)
    df = pd.read_csv(scored_path)
    return df

@st.cache_data
def load_geojson():
    with open("data/districts.geojson") as f:
        return json.load(f)

hh = load_data()
geojson = load_geojson()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
all_districts = ["All"] + sorted(hh["district"].unique().tolist())
sel_district  = st.sidebar.selectbox("District", all_districts)
risk_threshold = st.sidebar.slider(
    "Risk score threshold (High ≥)",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
)
show_markers = st.sidebar.checkbox("Show household markers", value=False)

# ── Filter households ─────────────────────────────────────────────────────────
filtered = hh.copy()
if sel_district != "All":
    filtered = filtered[filtered["district"] == sel_district]
filtered["high_risk"] = filtered["risk_score"] >= risk_threshold

# ── Title & KPIs ──────────────────────────────────────────────────────────────
st.title("🇷🇼 Rwanda Stunting Risk Heatmap")
st.caption("Synthetic NISR-style data · S2.T1.2 · AIMS KTT Hackathon")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Households shown",  f"{len(filtered):,}")
c2.metric("High-risk (≥ threshold)", f"{filtered['high_risk'].sum():,}",
          f"{filtered['high_risk'].mean():.1%}")
c3.metric("Mean risk score",   f"{filtered['risk_score'].mean():.3f}")
c4.metric("Districts covered", filtered["district"].nunique())

st.divider()

# ── Build choropleth ──────────────────────────────────────────────────────────
sector_agg = (
    filtered.groupby(["district", "sector"])
    .agg(mean_risk=("risk_score", "mean"),
         n_high=("high_risk", "sum"),
         n_total=("household_id", "count"))
    .reset_index()
)
district_agg = (
    filtered.groupby("district")
    .agg(mean_risk=("risk_score", "mean"),
         n_high=("high_risk", "sum"),
         n_total=("household_id", "count"))
    .reset_index()
)

# Attach risk to geojson features
for feat in geojson["features"]:
    d = feat["properties"]["district"]
    row = district_agg[district_agg["district"] == d]
    feat["properties"]["mean_risk"] = float(row["mean_risk"].values[0]) if len(row) else 0.0
    feat["properties"]["n_high"]    = int(row["n_high"].values[0])    if len(row) else 0

m = folium.Map(location=[-1.95, 30.10], zoom_start=9, tiles="CartoDB positron")

folium.Choropleth(
    geo_data=geojson,
    data=district_agg,
    columns=["district", "mean_risk"],
    key_on="feature.properties.district",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name="Mean Stunting Risk Score",
    bins=[0.0, 0.15, 0.25, 0.35, 0.45, 0.60, 1.0],
).add_to(m)

# District tooltips
for feat in geojson["features"]:
    p = feat["properties"]
    folium.GeoJson(
        feat,
        style_function=lambda x: {"fillOpacity": 0, "color": "#333", "weight": 1.5},
        tooltip=folium.Tooltip(
            f"<b>{p['district']}</b><br>"
            f"Mean risk: {p['mean_risk']:.3f}<br>"
            f"High-risk HH: {p['n_high']}"
        ),
    ).add_to(m)

# Optional household markers
if show_markers:
    mc = MarkerCluster().add_to(m)
    for _, row in filtered[filtered["high_risk"]].iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="#c0392b",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"HH: {row['household_id']} | Score: {row['risk_score']:.3f} | Sector: {row['sector']}",
        ).add_to(mc)

map_html = m._repr_html_()

# ── Layout: map + sector table ────────────────────────────────────────────────
col_map, col_tbl = st.columns([3, 2])

with col_map:
    st.subheader("District Risk Choropleth")
    st_html(map_html, height=500)

with col_tbl:
    st.subheader("Sector-level Summary")
    display_df = sector_agg.copy()
    display_df["mean_risk"] = display_df["mean_risk"].round(3)
    display_df["high_risk_%"] = (display_df["n_high"] / display_df["n_total"] * 100).round(1)
    display_df = display_df.sort_values("mean_risk", ascending=False)
    st.dataframe(
        display_df.rename(columns={
            "district": "District", "sector": "Sector",
            "mean_risk": "Mean Risk", "n_high": "# High Risk",
            "n_total": "Total HH", "high_risk_%": "% High Risk"
        }),
        use_container_width=True,
        height=460,
    )

st.divider()

# ── Risk distribution chart ───────────────────────────────────────────────────
st.subheader("Risk Score Distribution by District")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, min(len(filtered["district"].unique()), 5),
                          figsize=(14, 3), sharey=True)
if not hasattr(axes, "__iter__"):
    axes = [axes]

districts_shown = sorted(filtered["district"].unique())
for ax, dist in zip(axes, districts_shown):
    subset = filtered[filtered["district"] == dist]["risk_score"]
    ax.hist(subset, bins=20, color="#e74c3c", alpha=0.75, edgecolor="white")
    ax.axvline(risk_threshold, color="#2c3e50", linewidth=1.5, linestyle="--")
    ax.set_title(dist, fontsize=9)
    ax.set_xlabel("Risk Score", fontsize=8)
    ax.tick_params(labelsize=7)

axes[0].set_ylabel("# Households", fontsize=8)
fig.suptitle("Risk score distributions (dashed = threshold)", fontsize=10)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ── Top high-risk households table ────────────────────────────────────────────
st.subheader(f"Top High-Risk Households (score ≥ {risk_threshold})")
top_hh = (
    filtered[filtered["high_risk"]]
    .sort_values("risk_score", ascending=False)
    .head(20)[["household_id", "district", "sector", "risk_score",
                "water_source", "sanitation_tier", "income_band",
                "avg_meal_count", "children_under5"]]
    .reset_index(drop=True)
)
st.dataframe(top_hh, use_container_width=True)

st.caption("Dashboard v1.0 · All household IDs are anonymised · Data is synthetic.")