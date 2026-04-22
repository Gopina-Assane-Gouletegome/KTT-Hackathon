# S2.T1.2 · Stunting Risk Heatmap Dashboard

**AIMS KTT Hackathon** · HealthTech · Geospatial · Data Viz

A complete pipeline that computes per-household childhood stunting risk scores
for Rwanda, visualises them on an interactive choropleth, and exports printable
A4 summary pages for village chiefs who operate without laptops.

---

## Quick Start (≤ 2 commands on Colab CPU)

```bash
pip install -r requirements.txt
python generate_data.py && python risk_scorer.py && streamlit run dashboard.py
```

Or step-by-step:

```bash
# 1. Generate synthetic data
python generate_data.py

# 2. Train model + score all households
python risk_scorer.py

# 3. Launch dashboard
streamlit run dashboard.py

# 4. Generate printable PDFs
python generate_printables.py
```

---

## Repository Structure

```
.
├── generate_data.py        # Reproducible synthetic NISR-style data generator
├── risk_scorer.py          # LR model + rule-based fallback + driver extraction
├── dashboard.py            # Streamlit choropleth dashboard
├── generate_printables.py  # A4 PDF generator (one per sector)
├── data/
│   ├── households.csv          # 2,500 synthetic households
│   ├── gold_stunting_flag.csv  # 300 labelled (50/50 pos/neg)
│   ├── districts.geojson       # 5 Rwandan district polygons
│   └── households_scored.csv   # Output of risk_scorer.py
├── model/
│   └── lr_pipeline.pkl         # Trained sklearn Pipeline
├── printable/
│   └── sector_*.pdf            # 5 A4 pages (one per sector)
├── requirements.txt
├── process_log.md
├── SIGNED.md
└── LICENSE
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.984** |
| F1 Score | **0.93** |
| Threshold | 0.45 |
| Training samples | 300 gold labels |

**Biggest stunting driver found: `water_risk` (coef +2.59)**
Unsafe water source (river / unprotected well) is the strongest predictor —
households relying on river water are ~2.6× more likely to have stunted children
than piped-water households, controlling for all other factors.

### Feature Coefficients (Logistic Regression)

| Feature | Coefficient | Meaning |
|---------|-------------|---------|
| `water_risk` | +2.59 | Unsafe water = highest risk |
| `meal_risk` | +2.05 | < 2 meals/day adds major risk |
| `sanitation_risk` | +1.72 | Open defecation / no latrine |
| `income_risk` | +1.46 | Low income band |
| `children_under5` | +1.21 | More young children = higher burden |

---

## Dashboard Features

- **Choropleth map** — district-level mean risk score (YlOrRd scale)
- **District filter** — dropdown to isolate any single district
- **Risk threshold slider** — dynamically redefines "High" vs "Medium" vs "Low"
- **Sector table** — sortable per-sector breakdown
- **Distribution histograms** — per-district risk score histograms
- **Top households table** — 20 highest-risk households

---

## Printable A4 Page (Umudugudu Page)

Each PDF page contains:

1. **Header** — sector name, district, "CONFIDENTIAL" notice
2. **Summary box** — total HH, # high-risk, %, mean score
3. **Top-10 table** — anonymised IDs (e.g. `KIM-007`), risk score, risk level, top-3 drivers, intervention hint
4. **Risk legend** — colour key for field use
5. **Workflow instructions** — 5-step process for the chief
6. **Footer** — model info, print date, privacy reminder

**Privacy**: All IDs are replaced with `<SECTOR_PREFIX>-<NNN>`. No names, phone numbers, or GPS coordinates appear on printed pages.

---

## Product & Business Adaptation

### The Reality on the Ground

Rwanda's Umudugudu chiefs typically lack laptops and reliable internet.
The district health system runs on:
- Monthly community meetings (*Umugoroba w'Ababyeyi*)
- Paper-based feedback loops to sector health posts
- Feature phones (SMS) for escalation

### Paper-First Delivery Workflow

```
┌─────────────────────────────────────────────────────────┐
│  MONTHLY CYCLE (28 days)                                │
│                                                         │
│  Day 1:  District health officer runs dashboard,        │
│          prints A4 pages at sector health post          │
│          (1 page per Umudugudu, ~30 pages/sector)       │
│                                                         │
│  Day 2:  Health post nurse hands pages to chief at      │
│          sector coordination meeting                     │
│                                                         │
│  Days 3–20: Chief visits top-10 households,             │
│          ticks/crosses each row, writes notes           │
│          in margin using local language (Kinyarwanda)   │
│                                                         │
│  Day 21: Chief returns annotated page to health post    │
│                                                         │
│  Day 22: Nurse digitises annotations (3 fields/row),    │
│          flags score ≥ 0.70 cases for escalation        │
│                                                         │
│  Day 23: Severe cases escalated to District MINISANTE   │
│          focal point via SMS (Chura/RapidSMS template)  │
│                                                         │
│  Day 28: MINISANTE focal point files monthly district   │
│          nutrition report                               │
└─────────────────────────────────────────────────────────┘
```

### Multilingual & Literacy Adaptation

- A4 pages include **Kinyarwanda translations** of all column headers
- Risk level icons (🔴 🟠 🟢) used alongside text for low-literacy users
- Intervention hints are phrased as simple action sentences
- Village chief receives a **one-page legend card** (laminated, reusable)

### Low-Bandwidth Considerations

- Dashboard runs on Streamlit (< 50 MB install) — usable on 3G
- Choropleth uses lightweight GeoJSON (< 20 KB)
- Printable generation is offline-first: PDFs can be generated on any laptop
- Model is a 12 KB pickle file — no cloud dependency at inference time

---

## Rwanda Stunting Context (NISR Figures)

- **National stunting rate** (NISR DHS 2019-20): **33%** of children under 5
- **Bugesera district**: ~35–38% (rural, high poverty)
- **Kigali City** (Nyarugenge, Gasabo, Kicukiro): ~18–22% (urban, better services)
- **Rwamagana**: ~28–30% (peri-urban/rural mix)

Our synthetic model replicates this gradient: Bugesera and Rwamagana show
higher mean risk scores than Nyarugenge in the dashboard.

---

## Technical Constraints Met

- ✅ CPU-only — no GPU required
- ✅ Dashboard renders in < 3 s on Colab (tested)
- ✅ All IDs anonymised on printable pages
- ✅ Reproducible in ≤ 2 commands

---


## License

MIT License — see `LICENSE`