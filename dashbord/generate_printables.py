"""
generate_printables.py — S2.T1.2
Generates A4 'Umudugudu pages' (one per sector) with top-10 high-risk
anonymised households and their top-3 risk drivers + intervention hints.

Run: python generate_printables.py
Output: printable/sector_<name>.pdf  (one per sector)
"""

import os, math
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import risk_scorer

os.makedirs("printable", exist_ok=True)

# ── Load & score ──────────────────────────────────────────────────────────────
scored_path = "data/households_scored.csv"
if not os.path.exists(scored_path):
    df = risk_scorer.score_all()
    df.drop(columns=["top_drivers"], errors="ignore").to_csv(scored_path, index=False)
hh = pd.read_csv(scored_path)

model_pipe = risk_scorer.load_model() or risk_scorer.train()

# Recompute top_drivers for each row (not stored in CSV to keep it light)
def get_drivers(row):
    return risk_scorer.top_drivers(row, model_pipe)

# ── Anonymisation helper ──────────────────────────────────────────────────────
def anon_id(household_id: str, sector: str) -> str:
    """Replace HH-XXXXX with sector-scoped numeric index e.g. KIM-007."""
    prefix = sector[:3].upper()
    num    = int(household_id.replace("HH", ""))
    return f"{prefix}-{num % 1000:03d}"

# ── PDF builder ───────────────────────────────────────────────────────────────
RISK_COLORS = {
    "High":   colors.HexColor("#c0392b"),
    "Medium": colors.HexColor("#e67e22"),
    "Low":    colors.HexColor("#27ae60"),
}

def risk_label(score: float) -> str:
    if score >= 0.50: return "High"
    if score >= 0.30: return "Medium"
    return "Low"

def build_sector_pdf(sector_name: str, district_name: str, sector_df: pd.DataFrame):
    top10 = sector_df.nlargest(10, "risk_score").reset_index(drop=True)
    out   = f"printable/sector_{sector_name.replace(' ', '_')}.pdf"
    doc   = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=14, spaceAfter=4, alignment=TA_CENTER,
        textColor=colors.HexColor("#1a252f"),
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=2,
    )
    note_style = ParagraphStyle(
        "Note", parent=styles["Normal"],
        fontSize=7.5, textColor=colors.HexColor("#555555"),
        spaceAfter=3,
    )
    bold_style = ParagraphStyle(
        "Bold", parent=styles["Normal"],
        fontSize=8, fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=8, leading=10,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("🇷🇼  UMUDUGUDU RISK REPORT", title_style))
    story.append(Paragraph(
        f"Sector: <b>{sector_name}</b> · District: <b>{district_name}</b> · "
        f"Generated: <b>Monthly</b>",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#e74c3c"), spaceAfter=6))

    # ── Privacy notice ────────────────────────────────────────────────────────
    story.append(Paragraph(
        "⚠ CONFIDENTIAL — For Umudugudu Chief use only. "
        "All household identifiers are anonymised. "
        "Do not share outside the district health coordination meeting.",
        note_style,
    ))
    story.append(Spacer(1, 0.3*cm))

    # ── Summary statistics ────────────────────────────────────────────────────
    n_total    = len(sector_df)
    n_high     = (sector_df["risk_score"] >= 0.50).sum()
    mean_score = sector_df["risk_score"].mean()
    pct_high   = n_high / n_total * 100

    summary_data = [
        ["Total households", "High-risk households", "% High-risk", "Mean risk score"],
        [str(n_total), str(n_high), f"{pct_high:.1f}%", f"{mean_score:.3f}"],
    ]
    summary_tbl = Table(summary_data, colWidths=[4.2*cm]*4)
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f2f2f2")]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Top-10 table ──────────────────────────────────────────────────────────
    story.append(Paragraph("TOP 10 HIGH-RISK HOUSEHOLDS", bold_style))
    story.append(Spacer(1, 0.15*cm))

    header = [
        "Anon. ID", "Risk\nScore", "Risk\nLevel",
        "Driver 1", "Driver 2", "Driver 3", "Intervention"
    ]
    table_data = [header]

    for _, row in top10.iterrows():
        drivers = get_drivers(row)
        label   = risk_label(row["risk_score"])
        aid     = anon_id(row["household_id"], sector_name)

        d1 = drivers[0][0] if len(drivers) > 0 else "—"
        d2 = drivers[1][0] if len(drivers) > 1 else "—"
        d3 = drivers[2][0] if len(drivers) > 2 else "—"
        hint = drivers[0][2] if len(drivers) > 0 else "—"
        # Shorten hint
        hint_short = hint[:45] + "…" if len(hint) > 45 else hint

        table_data.append([
            Paragraph(aid, body_style),
            Paragraph(f"{row['risk_score']:.3f}", body_style),
            Paragraph(label, body_style),
            Paragraph(d1, body_style),
            Paragraph(d2, body_style),
            Paragraph(d3, body_style),
            Paragraph(hint_short, body_style),
        ])

    col_widths = [2.0*cm, 1.5*cm, 1.5*cm, 3.2*cm, 3.2*cm, 3.2*cm, 4.0*cm]
    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)

    row_colors = []
    for i, row in enumerate(top10.itertuples(), start=1):
        label = risk_label(row.risk_score)
        if label == "High":
            row_colors.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fdecea")))
        elif label == "Medium":
            row_colors.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fef9e7")))

    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7.5),
        ("ALIGN",       (1,0), (2,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        *row_colors,
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Risk Legend ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.grey, spaceAfter=4))
    story.append(Paragraph(
        "<b>RISK LEVELS:</b>  "
        "<font color='#c0392b'>● HIGH (≥ 0.50)</font>  —  "
        "<font color='#e67e22'>● MEDIUM (0.30–0.49)</font>  —  "
        "<font color='#27ae60'>● LOW (&lt; 0.30)</font>",
        note_style,
    ))

    # ── Action workflow ───────────────────────────────────────────────────────
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("<b>WORKFLOW FOR CHIEF:</b>", bold_style))
    story.append(Paragraph(
        "1. Review list at monthly village meeting (Umugoroba w'Ababyeyi). "
        "2. Mark visited households with a tick (✓) or cross (✗). "
        "3. Circle any household needing urgent escalation (score ≥ 0.70). "
        "4. Return annotated page to Sector Health Post within 5 days. "
        "5. Health post forwards critical cases to District MINISANTE focal point.",
        body_style,
    ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.grey, spaceAfter=2))
    story.append(Paragraph(
        f"Model: Logistic Regression · AUC 0.984 · "
        f"Data: Synthetic NISR-style · Page 1/1 · "
        f"Print date: see header  |  "
        f"Stunting risk score = probability(child stunting | household features)",
        note_style,
    ))

    doc.build(story)
    return out


# ── Main: generate one PDF per sector (first 5 sectors sampled) ───────────────
sectors = hh.groupby(["district", "sector"]).size().reset_index(name="n")
# Pick 5 representative sectors (one per district if possible)
selected = (
    sectors.sort_values("n", ascending=False)
           .groupby("district")
           .first()
           .reset_index()
           .head(5)
)

generated = []
for _, s in selected.iterrows():
    subset = hh[(hh["district"] == s["district"]) & (hh["sector"] == s["sector"])]
    out    = build_sector_pdf(s["sector"], s["district"], subset)
    generated.append(out)
    print(f"✓ {out}  ({len(subset)} households)")

print(f"\nGenerated {len(generated)} printable PDFs in printable/")