from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

FEATURES = BASE / "artifacts/derived/tech_cia_features_v4_rg2.csv"
SKILLCORNER = BASE / "artifacts/metrica_transfer/metrica_predictions_HA.csv"
OUTPUT = BASE / "artifacts/derived/predictions_tech_cia_v2.csv"

THRESHOLD = 0.12
SMOOTHING_FRAMES = 5

print("Cargando características...")
df = pd.read_csv(FEATURES)
ha = pd.read_csv(SKILLCORNER)

for name, data in [("características", df), ("SkillCorner", ha)]:
    if "frame_id" not in data.columns:
        raise ValueError(f"Falta frame_id en {name}")
    if data["frame_id"].duplicated().any():
        raise ValueError(f"Hay frame_id duplicados en {name}")

print("Características:", df.shape)
print("Predicciones SkillCorner:", ha.shape)

# Alineación exacta por frame_id.
df = df.merge(
    ha[["frame_id", "probability_H", "probability_A"]],
    on="frame_id",
    how="left",
    validate="one_to_one"
)

if df[["probability_H", "probability_A"]].isna().any().any():
    raise ValueError("Hay frames sin probabilidades SkillCorner.")

# Normalización robusta sin utilizar etiquetas de posesión.
# Señal positiva = ventaja Barcelona; negativa = ventaja Athletic.
def robust_signal(column):
    x = pd.to_numeric(df[column], errors="coerce")
    median = x.median()
    mad = (x - median).abs().median()
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-9:
        scale = x.std()

    if not np.isfinite(scale) or scale < 1e-9:
        return pd.Series(np.nan, index=df.index)

    z = ((x - median) / scale).clip(-3, 3)
    return np.tanh(z / 2)

def group_signal(columns):
    signals = [
        robust_signal(c)
        for c in columns
        if c in df.columns
    ]
    if not signals:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(signals, axis=1).mean(axis=1)

# 1. Modelo externo SkillCorner.
skill_signal = (
    df["probability_H"] - df["probability_A"]
).clip(-1, 1)

# 2. Geometría relacional RG2.
# Los márgenes positivos representan ventaja Barcelona.
rg_columns = [
    "rg_nearest_proximity_advantage",
    "rg_centroid_proximity_advantage",
    "rg_mean_distance_advantage",
    "rg_n5_margin",
    "rg_n10_margin",
    "rg_density_margin_5",
    "rg_density_margin_10",
]
rg_signal = group_signal(rg_columns)

# 3. Dinámica colectiva instantánea.
collective_columns = [
    "players_5m_margin",
    "players_10m_margin",
    "speed_margin_kph",
    "toward_margin_5m",
    "toward_margin_10m",
]
collective_signal = group_signal(collective_columns)

# 4. Persistencia temporal: medias de un segundo.
temporal_columns = [
    "players_5m_margin_mean_1s",
    "players_10m_margin_mean_1s",
    "speed_margin_kph_mean_1s",
]
temporal_signal = group_signal(temporal_columns)

# Combinación de dinámica instantánea y temporal.
dynamic_signal = pd.concat(
    [collective_signal, temporal_signal], axis=1
).mean(axis=1)

# Fusión ponderada. Si falta un bloque, se redistribuye
# su peso entre los bloques disponibles.
blocks = pd.DataFrame({
    "skill": skill_signal,
    "geometry": rg_signal,
    "dynamics": dynamic_signal,
})

weights = pd.Series({
    "skill": 0.40,
    "geometry": 0.35,
    "dynamics": 0.25,
})

available_weights = blocks.notna().mul(weights, axis=1)
weighted_values = blocks.fillna(0).mul(weights, axis=1)

df["score_raw"] = (
    weighted_values.sum(axis=1)
    / available_weights.sum(axis=1).replace(0, np.nan)
)

if df["score_raw"].isna().any():
    raise ValueError("Hay frames sin señales suficientes para predecir.")

# Suavizado temporal corto: mediana de cinco frames.
df["score_smoothed"] = (
    df["score_raw"]
    .rolling(
        window=SMOOTHING_FRAMES,
        center=True,
        min_periods=1
    )
    .median()
)

# H = Barcelona; A = Athletic; D = disputa/ambigüedad.
df["label"] = np.select(
    [
        df["score_smoothed"] > THRESHOLD,
        df["score_smoothed"] < -THRESHOLD,
    ],
    ["H", "A"],
    default="D",
)

# Archivo nuevo: no se modifica ningún original.
result = df[[
    "frame_id",
    "label",
    "score_raw",
    "score_smoothed",
    "probability_H",
    "probability_A",
]].copy()

result.to_csv(OUTPUT, index=False)

print("\nPREDICCIÓN TECH-CIA V2 COMPLETADA")
print("Frames:", len(result))
print("\nDistribución:")
print(result["label"].value_counts())
print("\nPorcentajes:")
print((result["label"].value_counts(normalize=True) * 100).round(2))
print("\nUmbral:", THRESHOLD)
print("Suavizado:", SMOOTHING_FRAMES, "frames")
print("\nArchivo:", OUTPUT)
print("\nOriginales intactos.")
