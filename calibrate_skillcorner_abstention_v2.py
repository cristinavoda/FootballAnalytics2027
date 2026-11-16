import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

ROOT = Path(r"E:\Hackaton 2027\FootballAnalytics2027")
ART = ROOT / "artifacts"

validation_ids = {"1899585", "1996436", "2007448"}

df = pd.read_csv(ART / "skillcorner" / "skillcorner_features.csv.gz")
df["match_id"] = df["match_id"].astype(str)

# Solo los tres partidos de validación
df = df[df["match_id"].isin(validation_ids)].copy()

# EXCLUIR UNKNOWN
df = df[df["label"].isin(["H", "A"])].copy()

print("Frames etiquetados:", len(df))
print("Distribución:")
print(df["label"].value_counts().to_string())

model = joblib.load(
    ART / "skillcorner" / "models" / "best_model_h_a_invariant.joblib"
)

metrics = __import__("json").loads(
    (ART / "skillcorner" / "models" /
     "training_metrics_invariant.json").read_text(encoding="utf-8")
)

columns = metrics["features"]

X = df[columns]
y = df["label"].to_numpy()

probs = model.predict_proba(X)
classes = list(model.classes_)

pred = np.array(classes)[np.argmax(probs, axis=1)]
confidence = probs.max(axis=1)

print("\n=== VALIDACIÓN CORREGIDA ===")
print("Accuracy:", round(accuracy_score(y, pred), 4))

print("\n=== CALIBRACIÓN DE CONFIANZA ===")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    accepted = confidence >= threshold
    coverage = accepted.mean()
    accuracy = accuracy_score(y[accepted], pred[accepted])

    print(
        f"Umbral {threshold:.2f} | "
        f"Cobertura {coverage*100:.2f}% | "
        f"Abstención {(1-coverage)*100:.2f}% | "
        f"Accuracy aceptadas {accuracy:.4f} | "
        f"Frames {accepted.sum()}"
    )

print("\n=== POR PARTIDO ===")

df["prediction"] = pred
df["confidence"] = confidence

for match_id, group in df.groupby("match_id"):
    print(
        match_id,
        "| Frames:", len(group),
        "| Accuracy:", round(
            accuracy_score(group["label"], group["prediction"]), 4
        ),
        "| Confianza:", round(group["confidence"].mean(), 4)
    )
