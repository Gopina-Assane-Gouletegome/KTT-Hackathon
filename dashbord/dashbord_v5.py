"""
dashboard.py — S2.T1.2 Stunting Risk Heatmap Dashboard  (v3)
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rwanda Stunting Risk Dashboard",
    page_icon="🇷🇼",
    layout="wide",
)

# ── Helpers from risk_scorer ──────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
import risk_scorer as rs

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/households_scored.csv"
    if not os.path.exists(path):
        df = rs.score_all()
        df.drop(columns=["top_drivers"], errors="ignore").to_csv(path, index=False)
    return pd.read_csv(path)

@st.cache_data
def load_geojson():
    with open("data/districts.geojson") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return rs.load_model()

hh      = load_data()
geojson = load_geojson()
model   = load_model()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
all_districts  = ["All"] + sorted(hh["district"].unique().tolist())
sel_district   = st.sidebar.selectbox("District", all_districts)
all_sectors    = ["All"] + sorted(hh["sector"].unique().tolist())
sel_sector     = st.sidebar.selectbox("Sector", all_sectors)
risk_threshold = st.sidebar.slider("Risk threshold (High ≥)", 0.10, 0.90, 0.50, 0.05)
show_markers   = st.sidebar.checkbox("Show household markers on map", value=False)

# Apply filters
filtered = hh.copy()
if sel_district != "All":
    filtered = filtered[filtered["district"] == sel_district]
if sel_sector != "All":
    filtered = filtered[filtered["sector"] == sel_sector]
filtered["risk_label"] = pd.cut(
    filtered["risk_score"],
    bins=[-0.001, 0.30, 0.50, 1.01],
    labels=["Low", "Medium", "High"],
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🇷🇼 Rwanda Stunting Risk Heatmap")
st.caption("Synthetic NISR-style data · S2.T1.2 · AIMS KTT Hackathon")

# ══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════
TAB_OVERVIEW, TAB_MAP, TAB_GRAPHS, TAB_SECTORS, TAB_TABLES, TAB_PREDICT = st.tabs([
    "📊 Overview",
    "🗺️ Map",
    "📈 Charts",
    "🏘️ Sector Aggregates",
    "📋 Tables",
    "🔮 Live Prediction",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW  (descriptive statistics)
# ══════════════════════════════════════════════════════════════════════════════
with TAB_OVERVIEW:
    st.subheader("Descriptive Statistics")
    if sel_district != "All" or sel_sector != "All":
        scope = []
        if sel_district != "All": scope.append(f"District: **{sel_district}**")
        if sel_sector   != "All": scope.append(f"Sector: **{sel_sector}**")
        st.info("Filtered to " + "  ·  ".join(scope))

    total   = len(filtered)
    n_high  = (filtered["risk_label"] == "High").sum()
    n_med   = (filtered["risk_label"] == "Medium").sum()
    n_low   = (filtered["risk_label"] == "Low").sum()
    mean_sc = filtered["risk_score"].mean()
    med_sc  = filtered["risk_score"].median()
    std_sc  = filtered["risk_score"].std()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Households",         f"{total:,}")
    c2.metric("🔴 High Risk",             f"{n_high:,}", f"{n_high/total*100:.1f}%")
    c3.metric("🟡 Medium Risk",           f"{n_med:,}",  f"{n_med/total*100:.1f}%")
    c4.metric("🟢 Low Risk",              f"{n_low:,}",  f"{n_low/total*100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mean Risk Score",  f"{mean_sc:.4f}")
    c6.metric("Median Risk Score", f"{med_sc:.4f}")
    c7.metric("Std Dev",           f"{std_sc:.4f}")
    c8.metric("Districts in view", filtered["district"].nunique())

    st.divider()

    # Breakdown table by district
    st.subheader("Breakdown by District")
    dist_summary = []
    for d in sorted(filtered["district"].unique()):
        sub   = filtered[filtered["district"] == d]
        hi    = (sub["risk_label"] == "High").sum()
        me    = (sub["risk_label"] == "Medium").sum()
        lo    = (sub["risk_label"] == "Low").sum()
        dist_summary.append({
            "District":       d,
            "Total HH":       len(sub),
            "🔴 High":        hi,
            "% High":         f"{hi/len(sub)*100:.1f}%",
            "🟡 Medium":      me,
            "% Medium":       f"{me/len(sub)*100:.1f}%",
            "🟢 Low":         lo,
            "Mean Score":     round(sub["risk_score"].mean(), 4),
            "Median Score":   round(sub["risk_score"].median(), 4),
        })
    st.dataframe(pd.DataFrame(dist_summary), width="stretch", hide_index=True)

    st.divider()

    # Breakdown table by sector (top 20)
    st.subheader("Breakdown by Sector (top 20 by mean risk)")
    sec_summary = []
    for (d, s), sub in filtered.groupby(["district", "sector"]):
        hi = (sub["risk_label"] == "High").sum()
        me = (sub["risk_label"] == "Medium").sum()
        lo = (sub["risk_label"] == "Low").sum()
        sec_summary.append({
            "District":    d,
            "Sector":      s,
            "Total HH":    len(sub),
            "🔴 High":     hi,
            "% High":      f"{hi/len(sub)*100:.1f}%",
            "🟡 Medium":   me,
            "🟢 Low":      lo,
            "Mean Score":  round(sub["risk_score"].mean(), 4),
        })
    sec_df = (pd.DataFrame(sec_summary)
              .sort_values("Mean Score", ascending=False)
              .head(20)
              .reset_index(drop=True))
    st.dataframe(sec_df, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with TAB_MAP:
    st.subheader("District Risk Choropleth")
    st.caption("Hover over a district for a quick summary. Click a household marker for details.")

    district_agg = (
        filtered.groupby("district")
        .agg(mean_risk=("risk_score", "mean"),
             n_high=("risk_label", lambda x: (x == "High").sum()),
             n_total=("household_id", "count"))
        .reset_index()
    )

    # Update geojson properties
    gj = json.loads(json.dumps(geojson))
    for feat in gj["features"]:
        d   = feat["properties"]["district"]
        row = district_agg[district_agg["district"] == d]
        feat["properties"]["mean_risk"] = float(row["mean_risk"].values[0]) if len(row) else 0.0
        feat["properties"]["n_high"]    = int(row["n_high"].values[0])      if len(row) else 0
        feat["properties"]["n_total"]   = int(row["n_total"].values[0])     if len(row) else 0

    m = folium.Map(location=[-1.95, 30.10], zoom_start=9, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=gj,
        data=district_agg,
        columns=["district", "mean_risk"],
        key_on="feature.properties.district",
        fill_color="YlOrRd",
        fill_opacity=0.65,
        line_opacity=0.5,
        legend_name="Mean Stunting Risk Score",
        bins=[0.0, 0.15, 0.25, 0.35, 0.45, 0.60, 1.0],
    ).add_to(m)

    # Tooltip only — no popup, small box
    for feat in gj["features"]:
        p   = feat["properties"]
        pct = p["n_high"] / p["n_total"] * 100 if p["n_total"] else 0
        folium.GeoJson(
            feat,
            style_function=lambda x: {
                "fillOpacity": 0,
                "color": "#444",
                "weight": 1.5,
            },
            highlight_function=lambda x: {
                "fillOpacity": 0.15,
                "color": "#222",
                "weight": 2.5,
            },
            tooltip=folium.Tooltip(
                f"<b style='font-size:13px'>{p['district']}</b><br>"
                f"Mean risk: <b>{p['mean_risk']:.3f}</b><br>"
                f"High-risk HH: <b>{p['n_high']}</b> ({pct:.0f}%)",
                sticky=True,
            ),
            popup=None,
        ).add_to(m)

    # Household markers (optional)
    if show_markers:
        mc = MarkerCluster().add_to(m)
        for _, row in filtered[filtered["risk_label"] == "High"].iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color="#c0392b",
                fill=True,
                fill_opacity=0.75,
                tooltip=folium.Tooltip(
                    f"<b>{row['household_id']}</b><br>"
                    f"Score: {row['risk_score']:.3f}<br>"
                    f"Sector: {row['sector']}",
                    sticky=False,
                ),
                popup=folium.Popup(
                    f"<b>{row['household_id']}</b><br>"
                    f"District: {row['district']}<br>"
                    f"Sector: {row['sector']}<br>"
                    f"Risk score: <b>{row['risk_score']:.3f}</b><br>"
                    f"Water: {row['water_source']}<br>"
                    f"Income band: {row['income_band']}<br>"
                    f"Meals/day: {row['avg_meal_count']}",
                    max_width=200,
                ),
            ).add_to(mc)

    st.iframe(m._repr_html_(), height=520)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with TAB_GRAPHS:
    st.subheader("Visual Analytics")

    CHART_OPTIONS = {
        "Risk Score Distribution by District": "dist",
        "Risk Level Breakdown (Donut)":        "donut",
        "Feature Importance":                  "feat_imp",
        "Risk by Water Source":                "water",
        "Risk by Income Band":                 "income",
        "Risk by Sanitation Tier":             "sanit",
    }
    chosen = st.multiselect(
        "Select charts to display",
        list(CHART_OPTIONS.keys()),
        default=["Risk Score Distribution by District",
                 "Feature Importance",
                 "Risk Level Breakdown (Donut)"],
    )

    if not chosen:
        st.info("Select at least one chart above.")

    # ── Risk distribution histograms ──────────────────
    if "Risk Score Distribution by District" in chosen:
        st.markdown("#### Risk Score Distribution by District")
        districts_shown = sorted(filtered["district"].unique())
        n = len(districts_shown)
        if n == 0:
            st.warning("No data for selected filters.")
        else:
            PALETTE = ["#e74c3c","#e67e22","#f39c12","#27ae60","#2980b9"]
            fig, axes = plt.subplots(1, n, figsize=(min(3.8*n, 16), 3.5), sharey=True)
            if n == 1: axes = [axes]
            fig.patch.set_facecolor("#ffffff")
            for i, (ax, dist) in enumerate(zip(axes, districts_shown)):
                subset = filtered[filtered["district"] == dist]["risk_score"]
                col    = PALETTE[i % len(PALETTE)]
                counts, edges = np.histogram(subset, bins=20, range=(0,1))
                w = edges[1] - edges[0]
                for j in range(len(counts)):
                    ax.bar(edges[j], counts[j], width=w*0.88,
                           color=col, alpha=0.5 + 0.5*(edges[j]),
                           linewidth=0, edgecolor="white")
                ax.axvline(risk_threshold, color="#2c3e50", linewidth=1.5,
                           linestyle="--", label="Threshold")
                ax.set_title(dist, fontsize=9, fontweight="bold", pad=6)
                ax.set_xlabel("Risk Score", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.spines[["top","right"]].set_visible(False)
                ax.set_xlim(0, 1)
                ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            axes[0].set_ylabel("# Households", fontsize=8)
            fig.suptitle(f"Risk score distributions  (dashed = threshold {risk_threshold})",
                         fontsize=9.5, y=1.03)
            plt.tight_layout(w_pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

    # ── Feature Importance ────────────────────────────
    if "Feature Importance" in chosen:
        st.markdown("#### Feature Importance (Logistic Regression Coefficients)")
        if model is not None:
            coefs = model.named_steps["lr"].coef_[0]
            feat_labels = {
                "water_risk":      "Unsafe Water Source",
                "sanitation_risk": "Poor Sanitation",
                "income_risk":     "Low Income Band",
                "meal_risk":       "Insufficient Meals/Day",
                "children_under5": "Children Under 5",
            }
            names  = [feat_labels[f] for f in rs.FEATURE_NAMES]
            values = list(coefs)
            order  = np.argsort(values)
            names_s  = [names[i]  for i in order]
            values_s = [values[i] for i in order]
            colors_s = ["#e74c3c" if v > 0 else "#3498db" for v in values_s]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            bars = ax.barh(names_s, values_s, color=colors_s, height=0.55, edgecolor="white")
            ax.axvline(0, color="#2c3e50", linewidth=0.8)
            for bar, val in zip(bars, values_s):
                ax.text(val + (0.04 if val >= 0 else -0.04),
                        bar.get_y() + bar.get_height()/2,
                        f"{val:+.3f}",
                        va="center", ha="left" if val >= 0 else "right",
                        fontsize=8.5, color="#2c3e50", fontweight="bold")
            ax.set_xlabel("LR Coefficient (positive = higher risk)", fontsize=9)
            ax.set_title("Feature Importance for Stunting Risk Prediction", fontsize=10, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.3, linewidth=0.5)
            red_patch   = mpatches.Patch(color="#e74c3c", label="Increases risk")
            blue_patch  = mpatches.Patch(color="#3498db", label="Decreases risk")
            ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Coefficients from Logistic Regression trained on 300 gold-labelled households. "
                       "Larger positive coefficient = stronger predictor of stunting risk.")
        else:
            st.warning("Model not found. Run `python risk_scorer.py` first.")

    # ── Donut ─────────────────────────────────────────
    if "Risk Level Breakdown (Donut)" in chosen:
        st.markdown("#### Risk Level Breakdown")
        col_d, col_sp = st.columns([1, 1])
        with col_d:
            n_low_ = (filtered["risk_label"] == "Low").sum()
            n_med_ = (filtered["risk_label"] == "Medium").sum()
            n_hi_  = (filtered["risk_label"] == "High").sum()
            sizes  = [n_low_, n_med_, n_hi_]
            labels = [f"Low ({n_low_:,})", f"Medium ({n_med_:,})", f"High ({n_hi_:,})"]
            colors = ["#27ae60", "#e67e22", "#e74c3c"]
            fig, ax = plt.subplots(figsize=(5, 4))
            wedges, _, autotexts = ax.pie(
                sizes, colors=colors, autopct="%1.1f%%",
                startangle=90, pctdistance=0.78,
                wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontsize(9); at.set_fontweight("bold")
            ax.text(0, 0, f"{sum(sizes):,}\nhouseholds", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#2c3e50", linespacing=1.5)
            ax.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
                      ncol=3, fontsize=8.5, frameon=False)
            ax.set_title("Risk Level Distribution", fontsize=10, fontweight="bold", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Water source ──────────────────────────────────
    if "Risk by Water Source" in chosen:
        st.markdown("#### Mean Risk Score by Water Source")
        col_w, col_sp = st.columns([1, 1])
        with col_w:
            ws = (filtered.groupby("water_source")["risk_score"]
                  .agg(["mean","count"]).reset_index()
                  .sort_values("mean", ascending=True))
            bar_cols = ["#e74c3c" if v >= 0.5 else "#e67e22" if v >= 0.3 else "#27ae60"
                        for v in ws["mean"]]
            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.barh(ws["water_source"], ws["mean"], color=bar_cols, height=0.5, edgecolor="white")
            ax.axvline(risk_threshold, color="#2c3e50", linewidth=1.3, linestyle="--")
            for _, row in ws.iterrows():
                ax.text(row["mean"] + 0.006, list(ws["water_source"]).index(row["water_source"]),
                        f'{row["mean"]:.3f}  (n={int(row["count"]):,})',
                        va="center", fontsize=8, color="#2c3e50")
            ax.set_xlim(0, ws["mean"].max() * 1.4)
            ax.set_xlabel("Mean Risk Score", fontsize=9)
            ax.set_title("Risk Score by Water Source", fontsize=10, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.3, linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Income band ───────────────────────────────────
    if "Risk by Income Band" in chosen:
        st.markdown("#### Mean Risk Score by Income Band")
        col_i, col_sp = st.columns([1, 1])
        with col_i:
            ib = (filtered.groupby("income_band")["risk_score"]
                  .agg(["mean","count"]).reset_index()
                  .sort_values("income_band"))
            band_labs = {1:"Band 1\n(Lowest)", 2:"Band 2", 3:"Band 3", 4:"Band 4\n(Highest)"}
            ib["label"] = ib["income_band"].map(band_labs)
            grad = ["#e74c3c","#e67e22","#f39c12","#27ae60"]
            x    = np.arange(len(ib))
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            bars = ax.bar(x, ib["mean"], color=grad[:len(ib)], width=0.5, edgecolor="white")
            ax.axhline(risk_threshold, color="#2c3e50", linewidth=1.3, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(ib["label"], fontsize=8.5)
            ax.set_ylabel("Mean Risk Score", fontsize=9)
            ax.set_ylim(0, ib["mean"].max() * 1.3)
            for bar, cnt in zip(bars, ib["count"]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"n={int(cnt):,}", ha="center", fontsize=7.5, color="#555")
            ax.set_title("Risk Score by Income Band", fontsize=10, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Sanitation tier ───────────────────────────────
    if "Risk by Sanitation Tier" in chosen:
        st.markdown("#### Mean Risk Score by Sanitation Tier")
        col_s, col_sp = st.columns([1, 1])
        with col_s:
            st_agg = (filtered.groupby("sanitation_tier")["risk_score"]
                      .agg(["mean","count"]).reset_index()
                      .sort_values("sanitation_tier"))
            tier_labs = {1:"Tier 1\n(Worst)", 2:"Tier 2", 3:"Tier 3\n(Best)"}
            st_agg["label"] = st_agg["sanitation_tier"].map(tier_labs)
            grad = ["#e74c3c","#e67e22","#27ae60"]
            x    = np.arange(len(st_agg))
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(x, st_agg["mean"], color=grad[:len(st_agg)], width=0.45, edgecolor="white")
            ax.axhline(risk_threshold, color="#2c3e50", linewidth=1.3, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(st_agg["label"], fontsize=8.5)
            ax.set_ylabel("Mean Risk Score", fontsize=9)
            ax.set_ylim(0, st_agg["mean"].max() * 1.3)
            for bar, cnt in zip(bars, st_agg["count"]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"n={int(cnt):,}", ha="center", fontsize=7.5, color="#555")
            ax.set_title("Risk Score by Sanitation Tier", fontsize=10, fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SECTOR AGGREGATES
# ══════════════════════════════════════════════════════════════════════════════
with TAB_SECTORS:
    st.subheader("Sector-Level Aggregate Risk")
    st.caption("Each sector's risk = mean of all household risk scores in that sector.")

    sec_agg = (
        filtered.groupby(["district", "sector"])
        .agg(
            mean_risk  = ("risk_score", "mean"),
            median_risk= ("risk_score", "median"),
            std_risk   = ("risk_score", "std"),
            min_risk   = ("risk_score", "min"),
            max_risk   = ("risk_score", "max"),
            n_total    = ("household_id", "count"),
            n_high     = ("risk_label", lambda x: (x == "High").sum()),
            n_medium   = ("risk_label", lambda x: (x == "Medium").sum()),
            n_low      = ("risk_label", lambda x: (x == "Low").sum()),
        )
        .reset_index()
    )
    sec_agg["pct_high"]    = (sec_agg["n_high"]   / sec_agg["n_total"] * 100).round(1)
    sec_agg["mean_risk"]   = sec_agg["mean_risk"].round(4)
    sec_agg["median_risk"] = sec_agg["median_risk"].round(4)
    sec_agg["std_risk"]    = sec_agg["std_risk"].round(4)
    sec_agg["min_risk"]    = sec_agg["min_risk"].round(4)
    sec_agg["max_risk"]    = sec_agg["max_risk"].round(4)
    sec_agg_sorted = sec_agg.sort_values("mean_risk", ascending=False).reset_index(drop=True)

    # Summary stats across sectors
    st.markdown("#### Summary Statistics Across All Sectors")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Number of sectors",          sec_agg["sector"].nunique())
    sc2.metric("Highest-risk sector",        sec_agg_sorted.iloc[0]["sector"],
               f"Score {sec_agg_sorted.iloc[0]['mean_risk']:.4f}")
    sc3.metric("Lowest-risk sector",         sec_agg_sorted.iloc[-1]["sector"],
               f"Score {sec_agg_sorted.iloc[-1]['mean_risk']:.4f}")
    sc4.metric("Overall mean sector risk",   f"{sec_agg['mean_risk'].mean():.4f}")

    st.divider()

    # Sector stats table
    st.markdown("#### Sector Aggregate Statistics Table")
    display_sec = sec_agg_sorted.rename(columns={
        "district":   "District",
        "sector":     "Sector",
        "mean_risk":  "Mean Risk",
        "median_risk":"Median Risk",
        "std_risk":   "Std Dev",
        "min_risk":   "Min Risk",
        "max_risk":   "Max Risk",
        "n_total":    "Total HH",
        "n_high":     "# High",
        "n_medium":   "# Medium",
        "n_low":      "# Low",
        "pct_high":   "% High",
    })
    st.dataframe(display_sec, width="stretch", hide_index=True)

    st.divider()

    # Barplot of aggregate risk per sector
    st.markdown("#### Aggregate Risk Score per Sector")
    n_sec   = len(sec_agg_sorted)
    fig_h   = max(3.5, n_sec * 0.42)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sector_labels = sec_agg_sorted["sector"] + " (" + sec_agg_sorted["district"].str[:3] + ")"
    bar_cols      = ["#e74c3c" if v >= 0.5 else "#e67e22" if v >= 0.3 else "#27ae60"
                     for v in sec_agg_sorted["mean_risk"]]
    bars = ax.barh(sector_labels[::-1], sec_agg_sorted["mean_risk"][::-1],
                   color=bar_cols[::-1], height=0.65, edgecolor="white")
    ax.axvline(risk_threshold, color="#2c3e50", linewidth=1.5,
               linestyle="--", label=f"Threshold ({risk_threshold})")
    for bar, val in zip(bars, sec_agg_sorted["mean_risk"][::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7.5, color="#2c3e50")
    ax.set_xlabel("Mean Risk Score (aggregated per sector)", fontsize=9)
    ax.set_title("Aggregate Stunting Risk Score by Sector", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0, sec_agg_sorted["mean_risk"].max() * 1.18)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    red_p   = mpatches.Patch(color="#e74c3c", label="High risk (≥0.50)")
    amber_p = mpatches.Patch(color="#e67e22", label="Medium risk (0.30–0.49)")
    green_p = mpatches.Patch(color="#27ae60", label="Low risk (<0.30)")
    ax.legend(handles=[red_p, amber_p, green_p], fontsize=8,
              loc="lower right", framealpha=0.8)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — TABLES
# ══════════════════════════════════════════════════════════════════════════════
with TAB_TABLES:
    st.subheader("Data Tables")

    table_choice = st.radio(
        "What would you like to view?",
        ["Sector-Level Summary", "Top High-Risk Households"],
        horizontal=True,
    )

    if table_choice == "Sector-Level Summary":
        st.markdown("#### Sector-Level Summary")
        st.caption(f"Showing {len(filtered):,} households across "
                   f"{filtered['district'].nunique()} district(s) · "
                   f"filtered to: District={sel_district}, Sector={sel_sector}")
        sec_tbl = (
            filtered.groupby(["district", "sector"])
            .agg(mean_risk  = ("risk_score", "mean"),
                 n_high     = ("risk_label", lambda x: (x == "High").sum()),
                 n_medium   = ("risk_label", lambda x: (x == "Medium").sum()),
                 n_low      = ("risk_label", lambda x: (x == "Low").sum()),
                 n_total    = ("household_id", "count"))
            .reset_index()
        )
        sec_tbl["mean_risk"]   = sec_tbl["mean_risk"].round(4)
        sec_tbl["pct_high"]    = (sec_tbl["n_high"] / sec_tbl["n_total"] * 100).round(1)
        sec_tbl = sec_tbl.sort_values("mean_risk", ascending=False).reset_index(drop=True)
        st.dataframe(
            sec_tbl.rename(columns={
                "district":"District","sector":"Sector",
                "mean_risk":"Mean Risk","n_high":"# High","n_medium":"# Medium",
                "n_low":"# Low","n_total":"Total HH","pct_high":"% High",
            }),
            width="stretch", hide_index=True,
        )

    else:  # Top High-Risk Households
        st.markdown("#### Top High-Risk Households")
        n_show = st.slider("Number of households to show", 10, 100, 20, 10)
        st.caption(f"Showing top {n_show} households with highest risk score")
        top_hh = (
            filtered.sort_values("risk_score", ascending=False)
            .head(n_show)[["household_id","district","sector","risk_score","risk_label",
                            "water_source","sanitation_tier","income_band",
                            "avg_meal_count","children_under5"]]
            .reset_index(drop=True)
        )
        st.dataframe(top_hh, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with TAB_PREDICT:
    st.subheader("🔮 Real-Time Household Risk Prediction")
    st.caption("Enter household characteristics below to instantly predict stunting risk.")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### Household Information")
        water_source    = st.selectbox("Water Source",
                                       ["piped","protected_well","unprotected_well","river"],
                                       help="piped = lowest risk, river = highest risk")
        sanitation_tier = st.selectbox("Sanitation Tier",
                                       [3, 2, 1],
                                       format_func=lambda x: {3:"Tier 3 (Best — flush toilet)",
                                                               2:"Tier 2 (Improved latrine)",
                                                               1:"Tier 1 (Open defecation)"}[x])
        income_band     = st.selectbox("Income Band",
                                       [4, 3, 2, 1],
                                       format_func=lambda x: {4:"Band 4 (Highest)",
                                                               3:"Band 3",
                                                               2:"Band 2",
                                                               1:"Band 1 (Lowest)"}[x])
        avg_meal_count  = st.selectbox("Average Meals per Day",
                                       [3, 2, 1],
                                       format_func=lambda x: {3:"3 meals/day",
                                                               2:"2 meals/day",
                                                               1:"1 meal/day"}[x])
        children_under5 = st.slider("Number of Children Under 5", 0, 5, 1)
        predict_btn     = st.button("🔍 Predict Risk", type="primary", width="stretch")

    with col_result:
        st.markdown("#### Prediction Result")
        if predict_btn:
            household_input = {
                "water_source":    water_source,
                "sanitation_tier": sanitation_tier,
                "income_band":     income_band,
                "avg_meal_count":  avg_meal_count,
                "children_under5": children_under5,
            }
            result = rs.score(household_input, model)
            score_val = result["risk_score"]
            label_val = result["risk_label"]

            # Score gauge
            color_map = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"}
            bg_map    = {"High": "#fdecea",  "Medium": "#fef9e7",  "Low": "#eafaf1"}
            col  = color_map[label_val]
            bg   = bg_map[label_val]

            st.markdown(f"""
            <div style="background:{bg};border:2px solid {col};border-radius:12px;
                        padding:20px 24px;margin-bottom:16px;text-align:center;">
              <div style="font-size:13px;color:#555;margin-bottom:6px;font-weight:600;
                          text-transform:uppercase;letter-spacing:0.08em;">
                Stunting Risk Score
              </div>
              <div style="font-size:52px;font-weight:800;color:{col};line-height:1;">
                {score_val:.3f}
              </div>
              <div style="font-size:18px;font-weight:700;color:{col};margin-top:6px;">
                {label_val.upper()} RISK
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.markdown(f"**Risk Score: {score_val:.3f}**")
            st.progress(float(score_val))

            # Top drivers
            st.markdown("**Top Risk Drivers:**")
            for i, (driver, val, hint) in enumerate(result["top_drivers"], 1):
                with st.expander(f"#{i} — {driver}"):
                    st.markdown(f"**Feature value:** `{val}`")
                    st.markdown(f"💡 **Suggested intervention:** {hint}")

        else:
            st.info("Fill in the household details on the left and click **Predict Risk**.")
            st.markdown("""
            **How it works:**
            - The model uses Logistic Regression trained on 300 labelled households
            - It evaluates 5 risk factors: water source, sanitation, income, meals, and number of children
            - **AUC-ROC: 0.984** — highly accurate risk prediction
            - Scores range 0.0 (no risk) → 1.0 (maximum risk)
            - Threshold ≥ 0.50 → **High Risk**
            - Threshold 0.30–0.49 → **Medium Risk**
            - Threshold < 0.30 → **Low Risk**
            """)

st.divider()
st.caption("Dashboard v3.0 · All household IDs are anonymised · Data is synthetic (NISR-style) · Model: LR AUC=0.984")