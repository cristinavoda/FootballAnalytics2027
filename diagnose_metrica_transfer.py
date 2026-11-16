import pandas as pd
import numpy as np
from pathlib import Path

root = Path(r"E:\Hackaton 2027\FootballAnalytics2027\artifacts\metrica_transfer")

features = pd.read_csv(root / "metrica_features_invariant.csv.gz")
pred = pd.read_csv(root / "metrica_predictions_HA.csv")

df = features[["frame_id", "ball_detected"]].merge(
    pred, on="frame_id", validate="one_to_one"
)

df["confidence"] = df[["probability_H", "probability_A"]].max(axis=1)

print("\n=== DISTRIBUCIÓN SEGÚN VISIBILIDAD DEL BALÓN ===")

for visible, group in df.groupby("ball_detected"):
    name = "BALÓN VISIBLE" if visible == 1 else "BALÓN NO VISIBLE"

    print(f"\n{name}: {len(group)} frames ({len(group)/len(df)*100:.2f}%)")
    print("Predicciones:")
    print(group["prediction_HA"].value_counts().to_string())
    print("Probabilidad media H:", round(group["probability_H"].mean(), 4))
    print("Confianza media:", round(group["confidence"].mean(), 4))

    for threshold in [0.55, 0.60, 0.70, 0.80, 0.90]:
        pct = (group["confidence"] >= threshold).mean() * 100
        print(f"Confianza >= {threshold:.2f}: {pct:.2f}%")

print("\n=== DISTRIBUCIÓN GLOBAL DE CONFIANZA ===")
print(df["confidence"].describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
).round(4).to_string())

print("\n=== PREDICCIONES POR BLOQUES DE 5 MINUTOS ===")
df["minute_block"] = (df["frame_id"] - 1) // (25 * 60 * 5)

summary = df.groupby("minute_block").agg(
    frames=("frame_id", "size"),
    pct_H=("prediction_HA", lambda x: (x == "H").mean() * 100),
    pct_ball_visible=("ball_detected", "mean")
)

summary["pct_ball_visible"] *= 100
print(summary.round(2).to_string())

out = root / "metrica_transfer_diagnostics.csv"
summary.to_csv(out)
print("\nInforme guardado:", out)
