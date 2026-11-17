
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TECH-CIA V3 — FUSIÓN DE SEÑALES
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURES_FILE = (
    ROOT / "artifacts" / "derived" /
    "tech_cia_features_v4_rg2.csv"
)

SKILLCORNER_FILE = (
    ROOT / "artifacts" / "metrica_transfer" /
    "metrica_predictions_HA.csv"
)

OUTPUT_FILE = (
    ROOT / "artifacts" / "derived" /
    "predictions_tech_cia_v3.csv"
)

FPS = 25

# Suavizado causal: utiliza el frame actual y los anteriores.
SMOOTHING_FRAMES = 5

# Umbral inicial de abstención. No está optimizado con etiquetas.
ABSTENTION_THRESHOLD = 0.12


# ============================================================
# 1. CARGA Y VALIDACIÓN
# ============================================================

print("\nTECH-CIA V3")
print("=" * 55)

for path in [FEATURES_FILE, SKILLCORNER_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra: {path}")

features = pd.read_csv(FEATURES_FILE)
skill = pd.read_csv(SKILLCORNER_FILE)

for name, df in [
    ("features", features),
    ("SkillCorner", skill),
]:
    if "frame_id" not in df.columns:
        raise ValueError(f"{name}: falta frame_id")

    if df["frame_id"].duplicated().any():
        raise ValueError(f"{name}: frame_id duplicados")

features["frame_id"] = pd.to_numeric(
    features["frame_id"], errors="raise"
).astype("int64")

skill["frame_id"] = pd.to_numeric(
    skill["frame_id"], errors="raise"
).astype("int64")

required_skill = [
    "frame_id",
    "probability_H",
    "probability_A",
]

missing = [
    c for c in required_skill
    if c not in skill.columns
]

if missing:
    raise ValueError(
        f"Faltan columnas SkillCorner: {missing}"
    )

df = features.merge(
    skill[required_skill],
    on="frame_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not (df["_merge"] == "both").all():
    raise ValueError(
        "Hay frames sin correspondencia en SkillCorner"
    )

df = df.drop(columns="_merge")
df = df.sort_values("frame_id").reset_index(drop=True)


# ============================================================
# 2. FUNCIONES DE NORMALIZACIÓN ROBUSTA
# ============================================================

def robust_signal(series):
    """
    Convierte una variable en una señal acotada [-1, 1].
    No utiliza etiquetas de posesión.
    """

    x = pd.to_numeric(series, errors="coerce")

    median = x.median()
    mad = (x - median).abs().median()

    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-9:
        scale = x.std()

    if not np.isfinite(scale) or scale < 1e-9:
        return pd.Series(
            np.nan, index=x.index, dtype=float
        )

    z = ((x - median) / scale).clip(-3, 3)

    return np.tanh(z / 2)


def group_signal(columns):
    """
    Promedia las señales disponibles de un bloque.
    """

    signals = []

    for column, sign in columns:
        if column not in df.columns:
            continue

        signal = robust_signal(df[column] * sign)
        signals.append(signal)

    if not signals:
        return pd.Series(
            np.nan, index=df.index, dtype=float
        )

    return pd.concat(signals, axis=1).mean(
        axis=1, skipna=True
    )


# ============================================================
# 3. SEÑAL SKILLCORNER
# ============================================================

prob_h = pd.to_numeric(
    df["probability_H"], errors="coerce"
)

prob_a = pd.to_numeric(
    df["probability_A"], errors="coerce"
)

if prob_h.isna().any() or prob_a.isna().any():
    raise ValueError("Probabilidades SkillCorner incompletas")

if (
    ((prob_h < 0) | (prob_h > 1)).any()
    or ((prob_a < 0) | (prob_a > 1)).any()
):
    raise ValueError("Probabilidades fuera de [0, 1]")

skill_signal = (prob_h - prob_a).clip(-1, 1)


# ============================================================
# 4. GEOMETRÍA RELACIONAL
# ============================================================

geometry_signal = group_signal([
    ("rg_nearest_proximity_advantage", +1),
    ("rg_centroid_proximity_advantage", +1),
    ("rg_mean_distance_advantage", +1),
    ("rg_n5_margin", +1),
    ("rg_n10_margin", +1),
    ("rg_density_margin_5", +1),
    ("rg_density_margin_10", +1),
])


# ============================================================
# 5. PROXIMIDAD Y DENSIDAD
# ============================================================

proximity_signal = group_signal([
    ("min_distance_margin_m", -1),
    ("top3_distance_margin_m", -1),
    ("players_5m_margin", +1),
    ("players_10m_margin", +1),
])


# ============================================================
# 6. FUSIÓN DE BLOQUES
# ============================================================

blocks = pd.DataFrame({
    "skill": skill_signal,
    "geometry": geometry_signal,
    "proximity": proximity_signal,
})

# Pesos iniciales explícitos, no entrenados.
weights = {
    "skill": 0.40,
    "geometry": 0.35,
    "proximity": 0.25,
}

weighted = blocks.mul(
    pd.Series(weights), axis=1
)

available_weight = (
    blocks.notna()
    .mul(pd.Series(weights), axis=1)
    .sum(axis=1)
)

score_raw = (
    weighted.sum(axis=1, min_count=1)
    / available_weight.replace(0, np.nan)
)


# ============================================================
# 7. SUAVIZADO TEMPORAL CAUSAL
# ============================================================

frame_gap = df["frame_id"].diff()

# Evita suavizar a través de huecos en la secuencia.
segments = frame_gap.ne(1).cumsum()

score_smoothed = (
    score_raw
    .groupby(segments)
    .transform(
        lambda s: s.rolling(
            window=SMOOTHING_FRAMES,
            min_periods=1,
        ).median()
    )
)


# ============================================================
# 8. CLASIFICACIÓN
# ============================================================

labels = np.select(
    [
        score_smoothed > ABSTENTION_THRESHOLD,
        score_smoothed < -ABSTENTION_THRESHOLD,
    ],
    [
        "H",
        "A",
    ],
    default="X",
)

# D requiere una fuente independiente que confirme
# que el balón está fuera de juego.
# No se deduce de la incertidumbre ni de la ausencia del balón.


# ============================================================
# 9. EXPORTACIÓN
# ============================================================

output = pd.DataFrame({
    "frame": df["frame_id"] - 1,
    "label": labels,
    "score": score_smoothed,
    "score_raw": score_raw,
    "skill_signal": skill_signal,
    "geometry_signal": geometry_signal,
    "proximity_signal": proximity_signal,
})

if output["frame"].duplicated().any():
    raise ValueError("Frames duplicados en la salida")

if not output["label"].isin(["H", "A", "D", "X"]).all():
    raise ValueError("Etiqueta no válida")

OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)

output.to_csv(OUTPUT_FILE, index=False)

print("\nEjecución completada.")
print("Archivo:", OUTPUT_FILE)
print("Frames:", len(output))
print("\nDistribución de etiquetas:")
print(output["label"].value_counts(dropna=False))

print("\nIntegridad:")
print("Frames únicos:", output["frame"].is_unique)
print("Scores finitos:", np.isfinite(
    output["score"].dropna()
).all())
print("TECH-CIA V3: OK")
