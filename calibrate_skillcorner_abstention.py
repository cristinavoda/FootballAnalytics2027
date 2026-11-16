import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score

ROOT = Path(r"E:\Hackaton 2027\FootballAnalytics2027")
ART = ROOT / "artifacts"

CSV = ART / "skillcorner" / "skillcorner_features.csv.gz"
MODEL = ART / "skillcorner" / "models" / "best_model_h_a_invariant.joblib"
METRICS = ART / "skillcorner" / "models" / "training_metrics_invariant.json"

validation_ids = {"1899585", "1996436", "2007448"}

metrics = json.loads(METRICS.read_text(encoding="utf-8"))
columns = metrics["features"]

print("Leyendo partidos de validación...")
df = pd.read_csv(CSV)

df["match_id"] = df["match_id"].astype(str)
df = df[df["match_id"].isin(validation_ids)].copy()

print("Filas:", len(df))
print("Partidos:", sorted(df["match_id"].unique()))

model = joblib.load(MODEL)

X = df[columns]
y = df["label"].astype(str)

probs = model.predict_proba(X)
classes = list(model.classes_)
pred = np.array(classes)[np.argmax(probs, axis=1)]
confidence = probs.max(axis=1)

print("\n=== CALIBRACIÓN DE CONFIANZA ===")
print("Accuracy sin abstención:", round(accuracy_score(y, pred), 4))

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    accepted = confidence >= threshold
    coverage = accepted.mean()

    if accepted.sum():
        accuracy = accuracy_score(y[accepted], pred[accepted])
    else:
        accuracy = np.nan

    print(
        f"Umbral {threshold:.2f} | "
        f"Cobertura {coverage*100:.2f}% | "
        f"Abstención {(1-coverage)*100:.2f}% | "
        f"Accuracy aceptadas {accuracy:.4f} | "
        f"Frames {accepted.sum()}"
    )

print("\n=== RESULTADOS POR PARTIDO ===")
for match_id, group in df.assign(
    prediction=pred,
    confidence=confidence
).groupby("match_id"):

    print(
        match_id,
        "| Accuracy:", round(
            accuracy_score(group["label"], group["prediction"]), 4
        ),
        "| Confianza media:", round(group["confidence"].mean(), 4)
    )
