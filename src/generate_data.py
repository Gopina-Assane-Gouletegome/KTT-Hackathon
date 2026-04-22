"""
generate_data.py — Reproducible synthetic data generator for S2.T1.2
Run: python generate_data.py
Outputs: data/households.csv, data/gold_stunting_flag.csv, data/districts.geojson
"""

import numpy as np
import pandas as pd
import json
import os

SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs("data", exist_ok=True)

# ── 1. District / sector definitions ─────────────────────────────────────────
DISTRICTS = {
    "Nyarugenge": {
        "sectors": ["Gitega", "Kigali", "Kimisagara", "Nyamirambo", "Rwezamenyo"],
        "urban_prob": 0.85,
        "center": (-1.945, 30.060),
        "stunting_base": 0.18,   # ~18 % – urban / wealthier
    },
    "Gasabo": {
        "sectors": ["Bumbogo", "Gasabo", "Jabana", "Kinyinya", "Remera"],
        "urban_prob": 0.70,
        "center": (-1.905, 30.115),
        "stunting_base": 0.22,
    },
    "Kicukiro": {
        "sectors": ["Gahanga", "Gatenga", "Kagarama", "Kanombe", "Niboye"],
        "urban_prob": 0.65,
        "center": (-1.990, 30.100),
        "stunting_base": 0.24,
    },
    "Bugesera": {
        "sectors": ["Gashora", "Juru", "Kamabuye", "Mareba", "Mwogo"],
        "urban_prob": 0.20,
        "center": (-2.160, 30.160),
        "stunting_base": 0.30,   # rural / higher risk
    },
    "Rwamagana": {
        "sectors": ["Fumbwe", "Gahengeri", "Gishari", "Munyaga", "Nzige"],
        "urban_prob": 0.25,
        "center": (-1.950, 30.435),
        "stunting_base": 0.28,
    },
}

N = 2500

# ── 2. Sample households ──────────────────────────────────────────────────────
rows = []
for i in range(N):
    district = rng.choice(list(DISTRICTS.keys()))
    info     = DISTRICTS[district]
    sector   = rng.choice(info["sectors"])
    urban    = rng.random() < info["urban_prob"]

    # Features — correlated with stunting risk
    income_band      = rng.choice([1, 2, 3, 4], p=[0.30, 0.35, 0.25, 0.10] if not urban else [0.10, 0.25, 0.40, 0.25])
    avg_meal_count   = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20] if income_band <= 2 else [0.05, 0.35, 0.60])
    water_source     = rng.choice(["unprotected_well", "protected_well", "piped", "river"],
                                   p=[0.30, 0.30, 0.30, 0.10] if not urban else [0.05, 0.15, 0.75, 0.05])
    sanitation_tier  = rng.choice([1, 2, 3], p=[0.40, 0.35, 0.25] if not urban else [0.10, 0.30, 0.60])
    children_under5  = int(rng.integers(0, 5))

    lat = info["center"][0] + rng.normal(0, 0.06)
    lon = info["center"][1] + rng.normal(0, 0.06)

    rows.append({
        "household_id":   f"HH{i+1:05d}",
        "lat":            round(lat, 5),
        "lon":            round(lon, 5),
        "district":       district,
        "sector":         sector,
        "urban":          int(urban),
        "children_under5": children_under5,
        "avg_meal_count": avg_meal_count,
        "water_source":   water_source,
        "sanitation_tier": sanitation_tier,
        "income_band":    income_band,
    })

hh = pd.DataFrame(rows)

# ── 3. Stunting flag (deterministic rule + noise) ─────────────────────────────
water_risk  = {"river": 3, "unprotected_well": 2, "protected_well": 1, "piped": 0}
logit = (
    -4.0
    + hh["water_source"].map(water_risk)           * 0.6
    + (3 - hh["sanitation_tier"])                  * 0.5
    + (4 - hh["income_band"])                      * 0.4
    + (3 - hh["avg_meal_count"])                   * 0.7
    + hh["children_under5"].clip(0, 4)             * 0.2
    + hh["district"].map({d: DISTRICTS[d]["stunting_base"] * 2 for d in DISTRICTS})
    + rng.normal(0, 0.4, N)
)
prob = 1 / (1 + np.exp(-logit))
hh["stunting_flag"] = (prob > 0.5).astype(int)
print(f"Stunting prevalence: {hh['stunting_flag'].mean():.1%}")   # ~22 %

hh.to_csv("data/households.csv", index=False)
print("✓ data/households.csv written")

# ── 4. Gold label sample (300, 50/50) ────────────────────────────────────────
pos = hh[hh["stunting_flag"] == 1].sample(150, random_state=SEED)
neg = hh[hh["stunting_flag"] == 0].sample(150, random_state=SEED)
gold = pd.concat([pos, neg])[["household_id", "stunting_flag"]].reset_index(drop=True)
gold.to_csv("data/gold_stunting_flag.csv", index=False)
print("✓ data/gold_stunting_flag.csv written")

# ── 5. districts.geojson (rough bounding boxes — good enough for choropleth) ──
features = []
for name, info in DISTRICTS.items():
    clat, clon = info["center"]
    d = 0.12   # ~13 km half-side
    coords = [[
        [clon - d, clat - d],
        [clon + d, clat - d],
        [clon + d, clat + d],
        [clon - d, clat + d],
        [clon - d, clat - d],
    ]]
    features.append({
        "type": "Feature",
        "properties": {"district": name, "stunting_base": info["stunting_base"]},
        "geometry": {"type": "Polygon", "coordinates": coords},
    })

geojson = {"type": "FeatureCollection", "features": features}
with open("data/districts.geojson", "w") as f:
    json.dump(geojson, f, indent=2)
print("✓ data/districts.geojson written")
print("\nAll data files generated successfully.")