"""
risk_scorer.py — S2.T1.2 Stunting Risk Heatmap Dashboard
Computes a per-household risk score using Logistic Regression trained on
gold_stunting_flag.csv, with a rule-based fallback.

Usage:
    python risk_scorer.py                  # trains, evaluates, saves model
    from risk_scorer import score          # score a single household dict
"""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)

# ── Feature engineering ───────────────────────────────────────────────────────
WATER_RISK = {"river": 3, "unprotected_well": 2, "protected_well": 1, "piped": 0}

FEATURE_NAMES = [
    "water_risk",       # 0–3  (higher = riskier)
    "sanitation_risk",  # 0–2  (3 - tier)
    "income_risk",      # 0–3  (4 - band)
    "meal_risk",        # 0–2  (3 - avg_meal_count)
    "children_under5",  # 0–4  clipped
]

def featurise(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of numeric features ready for the model."""
    f = pd.DataFrame()
    f["water_risk"]      = df["water_source"].map(WATER_RISK).fillna(1)
    f["sanitation_risk"] = (3 - df["sanitation_tier"]).clip(0, 2)
    f["income_risk"]     = (4 - df["income_band"]).clip(0, 3)
    f["meal_risk"]       = (3 - df["avg_meal_count"]).clip(0, 2)
    f["children_under5"] = df["children_under5"].clip(0, 4)
    return f[FEATURE_NAMES]


# ── Rule-based scorer (interpretable fallback) ────────────────────────────────
def rule_score(row: pd.Series) -> float:
    """
    Weighted sum of risk factors → probability-like score in [0, 1].
    Weights derived from domain knowledge (NISR / WHO WASH literature).
    """
    s = 0.0
    s += WATER_RISK.get(row["water_source"], 1) * 0.25    # max 0.75
    s += max(0, 3 - row["sanitation_tier"])     * 0.20    # max 0.40
    s += max(0, 4 - row["income_band"])         * 0.20    # max 0.60
    s += max(0, 3 - row["avg_meal_count"])      * 0.25    # max 0.50
    s += min(row["children_under5"], 4)         * 0.05    # max 0.20
    return min(s / 2.45, 1.0)   # normalise to [0,1]


# ── Top-3 driver extraction ───────────────────────────────────────────────────
DRIVER_LABELS = {
    "water_risk":       "Unsafe water source",
    "sanitation_risk":  "Poor sanitation",
    "income_risk":      "Low household income",
    "meal_risk":        "Insufficient meals per day",
    "children_under5":  "Multiple children under 5",
}

INTERVENTION_MAP = {
    "water_risk":       "Provide water purification tablets / connect to protected well",
    "sanitation_risk":  "WASH upgrade — latrine construction programme",
    "income_risk":      "Enroll in Ubudehe social-protection cash-transfer",
    "meal_risk":        "Distribute community nutrition kit (Plumpy'Nut + porridge)",
    "children_under5":  "Refer to community health worker for child growth monitoring",
}

def top_drivers(row: pd.Series, model_pipe=None, n: int = 3):
    """
    Return top-n (driver_label, raw_value, intervention_hint) tuples.
    Uses model coefficients if available, otherwise raw risk values.
    """
    f = featurise(row.to_frame().T).iloc[0]
    if model_pipe is not None:
        coefs = model_pipe.named_steps["lr"].coef_[0]
        scores = {k: abs(f[k] * coefs[i]) for i, k in enumerate(FEATURE_NAMES)}
    else:
        scores = dict(f)
    ranked = sorted(scores, key=scores.get, reverse=True)[:n]
    return [(DRIVER_LABELS[k], round(float(f[k]), 2), INTERVENTION_MAP[k]) for k in ranked]


# ── Public API ────────────────────────────────────────────────────────────────
MODEL_PATH = "model/lr_pipeline.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
    return None


def score(household: dict, model_pipe=None) -> dict:
    """
    Score a single household dict.
    Returns {"risk_score": float, "risk_label": str, "top_drivers": list}
    """
    row = pd.Series(household)
    if model_pipe is None:
        model_pipe = load_model()

    if model_pipe is not None:
        X = featurise(row.to_frame().T)
        prob = float(model_pipe.predict_proba(X)[0, 1])
    else:
        prob = rule_score(row)

    label = "High" if prob >= 0.5 else ("Medium" if prob >= 0.3 else "Low")
    drivers = top_drivers(row, model_pipe)
    return {"risk_score": round(prob, 4), "risk_label": label, "top_drivers": drivers}


# ── Training ──────────────────────────────────────────────────────────────────
def train(households_path="data/households.csv",
          gold_path="data/gold_stunting_flag.csv"):
    hh   = pd.read_csv(households_path)
    gold = pd.read_csv(gold_path)
    merged = hh.drop(columns=["stunting_flag"], errors="ignore").merge(gold, on="household_id")

    X = featurise(merged)
    y = merged["stunting_flag"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipe.fit(X, y)

    # Threshold calibration — find threshold maximising F1 on training set
    probs = pipe.predict_proba(X)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.05):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n── Logistic Regression (trained on {len(merged)} gold labels) ──")
    print(f"Best threshold: {best_t:.2f}  |  Training F1: {best_f1:.3f}")
    print(f"AUC-ROC: {roc_auc_score(y, probs):.3f}")
    print("\nClassification report (threshold = {:.2f}):".format(best_t))
    print(classification_report(y, (probs >= best_t).astype(int),
                                 target_names=["No stunting", "Stunting"]))

    print("\nFeature coefficients:")
    for name, coef in zip(FEATURE_NAMES, pipe.named_steps["lr"].coef_[0]):
        print(f"  {name:22s}: {coef:+.3f}")

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(pipe, fh)
    print(f"\n✓ Model saved → {MODEL_PATH}")
    return pipe


def score_all(households_path="data/households.csv") -> pd.DataFrame:
    """Score every household and return a DataFrame with risk columns added."""
    hh   = pd.read_csv(households_path)
    pipe = load_model() or train()
    X    = featurise(hh)
    hh["risk_score"] = pipe.predict_proba(X)[:, 1].round(4)
    hh["risk_label"] = pd.cut(
        hh["risk_score"],
        bins=[0, 0.30, 0.50, 1.01],
        labels=["Low", "Medium", "High"],
        right=False,
    )
    # Attach top drivers as a list column (used by printable generator)
    model_pipe = pipe
    def _drivers(row):
        return top_drivers(row, model_pipe)
    hh["top_drivers"] = [_drivers(r) for _, r in hh.iterrows()]
    return hh


if __name__ == "__main__":
    pipe = train()
    scored = score_all()
    print(f"\nRisk distribution across {len(scored)} households:")
    print(scored["risk_label"].value_counts())
    print("\nSample score call:")
    example = scored.iloc[0].to_dict()
    result  = score(example, pipe)
    print(f"  household_id: {example['household_id']}")
    print(f"  risk_score  : {result['risk_score']}")
    print(f"  risk_label  : {result['risk_label']}")
    print("  top drivers :")
    for label, val, hint in result["top_drivers"]:
        print(f"    • {label} (value={val}) → {hint}")
    scored.drop(columns=["top_drivers"]).to_csv("data/households_scored.csv", index=False)
    print("\n✓ data/households_scored.csv written")
