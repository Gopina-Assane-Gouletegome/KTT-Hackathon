"""
dashboard.py — S2.T1.2 Stunting Risk Heatmap Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os

# ══════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Rwanda Stunting Risk Dashboard",
    page_icon="🇷🇼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
.stApp { background: #0f1117; }

.section-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7280; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content:''; flex:1; height:1px; background:#2a2d3e; }

.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px; }
.kpi-card {
    background:#1a1d27; border:1px solid #2a2d3e; border-radius:12px;
    padding:16px 20px; position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:12px 12px 0 0;
}
.kpi-card.red::before   { background: linear-gradient(90deg,#ef4444,#f97316); }
.kpi-card.amber::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-card.blue::before  { background: linear-gradient(90deg,#3b82f6,#6366f1); }
.kpi-card.green::before { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-label { font-size:11px; color:#6b7280; font-weight:500; letter-spacing:0.04em; margin-bottom:6px; }
.kpi-value { font-size:28px; font-weight:700; color:#f9fafb; line-height:1; }
.kpi-delta { font-size:12px; color:#ef4444; margin-top:4px; font-family:'DM Mono',monospace; }

.stTabs [data-baseweb="tab-list"] {
    background:#1a1d27; border-radius:10px; padding:4px; gap:2px; border:1px solid #2a2d3e;
}
.stTabs [data-baseweb="tab"] {
    background:transparent; border-radius:8px; color:#6b7280;
    font-weight:500; font-size:13px; padding:8px 18px;
}
.stTabs [aria-selected="true"] { background:#2a2d3e !important; color:#f9fafb !important; }

label { color:#9ca3af !important; font-size:12px !important; }
.stSelectbox > div > div { background:#1a1d27 !important; border-color:#2a2d3e !important; color:#f9fafb !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    path = "data/households_scored.csv"
    if not os.path.exists(path):
        import risk_scorer
        df = risk_scorer.score_all()
        df.drop(columns=["top_drivers"], errors="ignore").to_csv(path, index=False)
    return pd.read_csv(path)

@st.cache_data
def load_geojson():
    with open("data/districts.geojson") as f:
        return json.load(f)

hh      = load_data()
geojson = load_geojson()

# ══════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
  <div style="font-size:36px;line-height:1;">🇷🇼</div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#f9fafb;letter-spacing:-0.02em;">
      Rwanda Stunting Risk Dashboard
    </div>
    <div style="font-size:12px;color:#6b7280;margin-top:2px;">
      Synthetic NISR-style data &nbsp;·&nbsp; S2.T1.2 &nbsp;·&nbsp; AIMS KTT Hackathon
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  TOP FILTER BAR
# ══════════════════════════════════════════════════════
st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
with fc1:
    all_districts = ["All"] + sorted(hh["district"].unique().tolist())
    sel_district  = st.selectbox("District", all_districts, key="dist")
with fc2:
    risk_threshold = st.slider("Risk threshold (High ≥)", 0.10, 0.90, 0.50, 0.05)
with fc3:
    all_sectors = ["All"] + sorted(hh["sector"].unique().tolist())
    sel_sector  = st.selectbox("Sector", all_sectors, key="sect")
with fc4:
    show_markers = st.checkbox("Show HH markers", value=False)

# Apply filters
filtered = hh.copy()
if sel_district != "All":
    filtered = filtered[filtered["district"] == sel_district]
if sel_sector != "All":
    filtered = filtered[filtered["sector"] == sel_sector]
filtered["high_risk"] = filtered["risk_score"] >= risk_threshold

# ══════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════
st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)

total_hh   = len(filtered)
n_high     = int(filtered["high_risk"].sum())
pct_high   = filtered["high_risk"].mean() * 100
mean_score = filtered["risk_score"].mean()
n_dist     = filtered["district"].nunique()

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card red">
    <div class="kpi-label">HOUSEHOLDS SHOWN</div>
    <div class="kpi-value">{total_hh:,}</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">HIGH-RISK HOUSEHOLDS</div>
    <div class="kpi-value">{n_high:,}</div>
    <div class="kpi-delta">▲ {pct_high:.1f}% of total</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">MEAN RISK SCORE</div>
    <div class="kpi-value">{mean_score:.3f}</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">DISTRICTS ACTIVE</div>
    <div class="kpi-value">{n_dist}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  AGGREGATIONS
# ══════════════════════════════════════════════════════
sector_agg = (
    filtered.groupby(["district", "sector"])
    .agg(mean_risk=("risk_score","mean"),
         n_high=("high_risk","sum"),
         n_total=("household_id","count"))
    .reset_index()
)
district_agg = (
    filtered.groupby("district")
    .agg(mean_risk=("risk_score","mean"),
         n_high=("high_risk","sum"),
         n_total=("household_id","count"))
    .reset_index()
)

# ══════════════════════════════════════════════════════
#  TABS: GRAPHICS  |  TABLES
# ══════════════════════════════════════════════════════
tab_graphics, tab_tables = st.tabs(["📊  Graphics", "📋  Tables"])

# ─────────────────────────────────────────────────────
#  TAB 1 — GRAPHICS
# ─────────────────────────────────────────────────────
with tab_graphics:

    st.markdown('<div class="section-label" style="margin-top:14px;">Choose charts to display</div>',
                unsafe_allow_html=True)

    CHART_OPTIONS = [
        "🗺️  Choropleth Map",
        "📊  Risk Distribution by District",
        "🍩  Risk Level Breakdown",
        "📈  Risk by Water Source",
        "🏠  Risk by Income Band",
    ]
    selected_charts = st.multiselect(
        "Select charts",
        CHART_OPTIONS,
        default=CHART_OPTIONS[:3],
        label_visibility="collapsed",
    )

    if not selected_charts:
        st.info("Select at least one chart above to display.")

    # ── MAP ───────────────────────────────────────────
    if "🗺️  Choropleth Map" in selected_charts:
        st.markdown('<div class="section-label" style="margin-top:20px;">District Choropleth</div>',
                    unsafe_allow_html=True)

        gj_copy = json.loads(json.dumps(geojson))
        for feat in gj_copy["features"]:
            d   = feat["properties"]["district"]
            row = district_agg[district_agg["district"] == d]
            feat["properties"]["mean_risk"] = float(row["mean_risk"].values[0]) if len(row) else 0.0
            feat["properties"]["n_high"]    = int(row["n_high"].values[0])      if len(row) else 0
            feat["properties"]["n_total"]   = int(row["n_total"].values[0])     if len(row) else 0

        m = folium.Map(location=[-1.95, 30.10], zoom_start=9, tiles="CartoDB dark_matter")

        folium.Choropleth(
            geo_data=gj_copy,
            data=district_agg,
            columns=["district","mean_risk"],
            key_on="feature.properties.district",
            fill_color="YlOrRd",
            fill_opacity=0.65,
            line_opacity=0.4,
            legend_name="Mean Stunting Risk Score",
            bins=[0.0, 0.15, 0.25, 0.35, 0.45, 0.60, 1.0],
        ).add_to(m)

        # Compact sticky tooltip, no popup
        for feat in gj_copy["features"]:
            p   = feat["properties"]
            pct = p["n_high"] / p["n_total"] * 100 if p["n_total"] else 0
            folium.GeoJson(
                feat,
                style_function=lambda x: {"fillOpacity": 0, "color": "rgba(255,255,255,0.15)", "weight": 1.2},
                highlight_function=lambda x: {"fillOpacity": 0.12, "color": "#ffffff", "weight": 2},
                tooltip=folium.Tooltip(
                    f"""<div style="font-family:sans-serif;background:#1a1d27;color:#f9fafb;
                        border:1px solid #3a3d4e;border-radius:8px;padding:9px 13px;
                        font-size:12px;line-height:1.7;box-shadow:0 4px 12px rgba(0,0,0,.5);">
                        <b style="color:#f97316;font-size:13px;">{p['district']}</b><br>
                        Risk &nbsp;<b style="color:#fbbf24;">{p['mean_risk']:.3f}</b> &nbsp;|&nbsp;
                        High-risk &nbsp;<b style="color:#ef4444;">{p['n_high']}</b>
                        &nbsp;<span style="color:#6b7280;">({pct:.0f}%)</span>
                    </div>""",
                    sticky=True,
                ),
                popup=None,
            ).add_to(m)

        if show_markers:
            mc = MarkerCluster().add_to(m)
            for _, row in filtered[filtered["high_risk"]].iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=3, color="#ef4444", fill=True, fill_opacity=0.7,
                    tooltip=folium.Tooltip(
                        f"<span style='font-size:11px;'>{row['household_id']} | {row['risk_score']:.3f}</span>",
                        sticky=True,
                    ),
                ).add_to(mc)

        st.iframe(m._repr_html_(), height=430)

    # ── DISTRIBUTION HISTOGRAMS ───────────────────────
    if "📊  Risk Distribution by District" in selected_charts:
        st.markdown('<div class="section-label" style="margin-top:20px;">Risk Score Distribution by District</div>',
                    unsafe_allow_html=True)

        districts_shown = sorted(filtered["district"].unique())
        n = len(districts_shown)
        if n == 0:
            st.warning("No data for selected filters.")
        else:
            PALETTE = ["#ef4444","#f97316","#f59e0b","#10b981","#3b82f6"]
            fig, axes = plt.subplots(1, n, figsize=(min(3.6 * n, 14), 3.2), sharey=True)
            if n == 1:
                axes = [axes]
            fig.patch.set_facecolor("#1a1d27")

            for i, (ax, dist) in enumerate(zip(axes, districts_shown)):
                subset = filtered[filtered["district"] == dist]["risk_score"]
                color  = PALETTE[i % len(PALETTE)]
                ax.set_facecolor("#13151f")
                counts, bin_edges = np.histogram(subset, bins=20, range=(0,1))
                w = bin_edges[1] - bin_edges[0]
                for j in range(len(counts)):
                    ax.bar(bin_edges[j], counts[j], width=w*0.88,
                           color=color, alpha=0.35 + 0.65*(bin_edges[j]), linewidth=0)
                ax.axvline(risk_threshold, color="#fff", linewidth=1.2, linestyle="--", alpha=0.55)
                ax.set_title(dist, fontsize=9, fontweight="600", color="#f9fafb", pad=6)
                ax.set_xlabel("Score", fontsize=7.5, color="#6b7280")
                ax.tick_params(labelsize=7, colors="#6b7280")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#2a2d3e")
                ax.set_xlim(0, 1)
                ax.grid(axis="y", color="#2a2d3e", linewidth=0.5, linestyle=":")

            axes[0].set_ylabel("Households", fontsize=7.5, color="#6b7280")
            fig.suptitle("Risk score distributions  (dashed = threshold)",
                         fontsize=9, color="#9ca3af", y=1.02)
            plt.tight_layout(w_pad=1.2)
            st.pyplot(fig)
            plt.close(fig)

    # ── DONUT ─────────────────────────────────────────
    if "🍩  Risk Level Breakdown" in selected_charts:
        st.markdown('<div class="section-label" style="margin-top:20px;">Risk Level Breakdown</div>',
                    unsafe_allow_html=True)

        n_low = (filtered["risk_score"] < 0.30).sum()
        n_med = ((filtered["risk_score"] >= 0.30) & (filtered["risk_score"] < 0.50)).sum()
        n_hi_ = (filtered["risk_score"] >= 0.50).sum()
        sizes  = [n_low, n_med, n_hi_]
        labels = ["Low", "Medium", "High"]
        colors = ["#10b981","#f59e0b","#ef4444"]

        col_donut, col_space = st.columns([1, 1])
        with col_donut:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#1a1d27")
            ax.set_facecolor("#1a1d27")
            wedges, _, autotexts = ax.pie(
                sizes, colors=colors, autopct="%1.1f%%", startangle=90,
                pctdistance=0.78,
                wedgeprops={"width": 0.52, "edgecolor": "#1a1d27", "linewidth": 2.5},
            )
            for at in autotexts:
                at.set_fontsize(9); at.set_color("#f9fafb"); at.set_fontweight("600")
            total = sum(sizes)
            ax.text(0, 0.08, f"{total:,}", ha="center", va="center",
                    fontsize=20, fontweight="700", color="#f9fafb")
            ax.text(0, -0.2, "households", ha="center", va="center",
                    fontsize=8, color="#6b7280")
            patches = [mpatches.Patch(color=c, label=f"{l}  ({v:,})")
                       for c, l, v in zip(colors, labels, sizes)]
            ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False, labelcolor="#9ca3af", fontsize=9)
            ax.set_title("Risk Level Distribution", fontsize=10, color="#f9fafb",
                         fontweight="600", pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── WATER SOURCE ──────────────────────────────────
    if "📈  Risk by Water Source" in selected_charts:
        st.markdown('<div class="section-label" style="margin-top:20px;">Mean Risk by Water Source</div>',
                    unsafe_allow_html=True)

        ws_agg = (filtered.groupby("water_source")["risk_score"]
                  .agg(["mean","count"]).reset_index()
                  .sort_values("mean", ascending=True))

        col_ws, col_sp = st.columns([1, 1])
        with col_ws:
            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            fig.patch.set_facecolor("#1a1d27")
            ax.set_facecolor("#13151f")
            bar_colors = ["#ef4444" if v >= 0.5 else "#f59e0b" if v >= 0.3 else "#10b981"
                          for v in ws_agg["mean"]]
            ax.barh(ws_agg["water_source"], ws_agg["mean"],
                    color=bar_colors, height=0.52, linewidth=0)
            ax.axvline(risk_threshold, color="#fff", linewidth=1.0, linestyle="--", alpha=0.5)
            for _, row in ws_agg.iterrows():
                ax.text(row["mean"] + 0.006, ws_agg.index.get_loc(ws_agg[ws_agg["water_source"]==row["water_source"]].index[0]),
                        f'{row["mean"]:.3f}  n={int(row["count"]):,}',
                        va="center", fontsize=7.5, color="#9ca3af")
            ax.set_xlim(0, max(ws_agg["mean"]) * 1.35)
            ax.set_xlabel("Mean Risk Score", fontsize=8, color="#6b7280")
            ax.tick_params(labelsize=8, colors="#9ca3af")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a2d3e")
            ax.grid(axis="x", color="#2a2d3e", linewidth=0.5, linestyle=":")
            ax.set_title("Risk by Water Source", fontsize=10, color="#f9fafb", fontweight="600", pad=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── INCOME BAND ───────────────────────────────────
    if "🏠  Risk by Income Band" in selected_charts:
        st.markdown('<div class="section-label" style="margin-top:20px;">Mean Risk by Income Band</div>',
                    unsafe_allow_html=True)

        ib_agg = (filtered.groupby("income_band")["risk_score"]
                  .agg(["mean","count"]).reset_index()
                  .sort_values("income_band"))
        band_labels = {1:"Band 1\n(Lowest)", 2:"Band 2", 3:"Band 3", 4:"Band 4\n(Highest)"}
        ib_agg["label"] = ib_agg["income_band"].map(band_labels)

        col_ib, col_sp2 = st.columns([1, 1])
        with col_ib:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            fig.patch.set_facecolor("#1a1d27")
            ax.set_facecolor("#13151f")
            grad  = ["#ef4444","#f97316","#f59e0b","#10b981"]
            x     = np.arange(len(ib_agg))
            bars  = ax.bar(x, ib_agg["mean"], color=grad[:len(ib_agg)], width=0.55, linewidth=0)
            ax.axhline(risk_threshold, color="#fff", linewidth=1.0, linestyle="--", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(ib_agg["label"], fontsize=8, color="#9ca3af")
            ax.set_ylabel("Mean Risk Score", fontsize=8, color="#6b7280")
            ax.set_ylim(0, max(ib_agg["mean"]) * 1.25)
            for bar, cnt in zip(bars, ib_agg["count"]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"n={int(cnt):,}", ha="center", fontsize=7.5, color="#6b7280")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a2d3e")
            ax.tick_params(colors="#6b7280")
            ax.grid(axis="y", color="#2a2d3e", linewidth=0.5, linestyle=":")
            ax.set_title("Risk by Income Band", fontsize=10, color="#f9fafb", fontweight="600", pad=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ─────────────────────────────────────────────────────
#  TAB 2 — TABLES
# ─────────────────────────────────────────────────────
with tab_tables:
    st.markdown('<div class="section-label" style="margin-top:14px;">Sector-Level Summary</div>',
                unsafe_allow_html=True)

    disp = sector_agg.copy()
    disp["mean_risk"]   = disp["mean_risk"].round(3)
    disp["high_risk_%"] = (disp["n_high"] / disp["n_total"] * 100).round(1)
    disp = disp.sort_values("mean_risk", ascending=False).reset_index(drop=True)
    st.dataframe(
        disp.rename(columns={
            "district":"District","sector":"Sector",
            "mean_risk":"Mean Risk","n_high":"# High Risk",
            "n_total":"Total HH","high_risk_%":"% High Risk",
        }),
        width="stretch", height=300,
    )

    st.markdown('<div class="section-label" style="margin-top:20px;">Top High-Risk Households</div>',
                unsafe_allow_html=True)

    top_hh = (
        filtered[filtered["high_risk"]]
        .sort_values("risk_score", ascending=False)
        .head(30)[["household_id","district","sector","risk_score",
                   "water_source","sanitation_tier","income_band",
                   "avg_meal_count","children_under5"]]
        .reset_index(drop=True)
    )
    st.dataframe(top_hh, width="stretch", height=380)

    st.markdown(
        '<p style="font-size:11px;color:#4b5563;margin-top:12px;">'
        'Dashboard v2.0 &nbsp;·&nbsp; All IDs anonymised &nbsp;·&nbsp; Synthetic NISR-style data'
        '</p>',
        unsafe_allow_html=True,
    )