from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "artifacts" / "temporal_features_v1.csv"
OUTPUT = ROOT / "artifacts" / "predictions_tech_cia_v1.csv"

print(f"Leyendo: {INPUT}")
df = pd.read_csv(INPUT)

print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns)}")

signals = {
    "dist_margin_min_m_mean_1s": -1.0,
    "dist_margin_top3_m_mean_1s": -0.8,
    "players_5m_margin_mean_1s": 0.7,
    "players_10m_margin_mean_1s": 0.5,
    "speed_margin_kph_mean_1s": 0.25,

    "dist_margin_min_m_mean_3s": -1.0,
    "dist_margin_top3_m_mean_3s": -0.8,
    "players_5m_margin_mean_3s": 0.7,
    "players_10m_margin_mean_3s": 0.5,
    "speed_margin_kph_mean_3s": 0.25,

    "dist_margin_min_m_trend_1s": -0.4,
    "dist_margin_top3_m_trend_1s": -0.3,
    "players_5m_margin_trend_1s": 0.25,
    "players_10m_margin_trend_1s": 0.20,
}

score = pd.Series(0.0, index=df.index)
used = []

for col, weight in signals.items():
    if col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        scale = x.abs().median()

        if pd.notna(scale) and scale > 1e-9:
            z = x / scale
            score += z.fillna(0.0) * weight
            used.append(col)

print("\nSeñales utilizadas:")
for col in used:
    print(f"  {col}")

collective = pd.Series(0.0, index=df.index)

collective_cols = {
    "visible_count_margin_mean_1s": 0.20,
    "compactness_margin_m_mean_1s": -0.15,
}

for col, weight in collective_cols.items():
    if col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        scale = x.abs().median()

        if pd.notna(scale) and scale > 1e-9:
            collective += (x / scale).fillna(0.0) * weight

score += collective

threshold = 0.15

label = np.where(
    score < -threshold,
    "H",
    np.where(score > threshold, "A", "D")
)

df_out = pd.DataFrame({
    "frame_id": df["frame_id"].astype(np.int64),
    "label": label,
    "score": score.astype(np.float32),
})

assert len(df_out) == 199319, f"Filas incorrectas: {len(df_out)}"
assert df_out["frame_id"].is_unique, "frame_id duplicados"
assert df_out["frame_id"].min() == 1
assert df_out["frame_id"].max() == 199319
assert df_out["label"].isin(["H", "A", "D"]).all()

df_out.to_csv(OUTPUT, index=False)

print("\n=== TECH-CIA V1 ===")
print(f"Salida: {OUTPUT}")
print(f"Filas: {len(df_out):,}")

print("\nDistribución:")
print(df_out["label"].value_counts().sort_index())

print("\nPorcentaje:")
print(
    (df_out["label"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)

print("\nScore:")
print(df_out["score"].describe().round(3))
