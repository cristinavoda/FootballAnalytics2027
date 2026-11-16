
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "artifacts" / "derived" / "radial_velocity_near_ball.csv"
OUTPUT = ROOT / "artifacts" / "derived" / "team_toward_dynamics.csv"

WINDOWS = [5, 10, 25]

df = pd.read_csv(INPUT)

# Variables originales que vamos a estudiar
variables = [
    "barca_toward_10m",
    "athletic_toward_10m",
    "barca_n_10m",
    "athletic_n_10m",
]

required = {"frame_id", *variables}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Faltan columnas: {missing}")

if df["frame_id"].duplicated().any():
    raise ValueError("Hay frame_id duplicados.")

if not df["frame_id"].is_monotonic_increasing:
    raise ValueError("Los frames no están ordenados.")

# Verificar continuidad de los frames
frame_id = df["frame_id"]

print("=" * 72)
print("DINÁMICA INDIVIDUAL DE CADA EQUIPO")
print("=" * 72)

# Calcular variaciones retrospectivas
for variable in variables:

    for k in WINDOWS:

        previous = df[variable].shift(k)

        valid_gap = (frame_id - frame_id.shift(k)) == k

        delta = (df[variable] - previous).where(valid_gap)

        column = f"{variable}_delta_{k}"
        df[column] = delta

# --------------------------------------------------
# RESUMEN DE LAS VARIABLES DE MOVIMIENTO
# --------------------------------------------------

print("\n1. VARIACIONES DE MOVIMIENTO")
print("-" * 72)

for team in ["barca", "athletic"]:

    variable = f"{team}_toward_10m"

    print(f"\n{team.upper()}")

    for k in WINDOWS:

        column = f"{variable}_delta_{k}"
        values = df[column].dropna()

        print(
            f"{k:2d} frames | "
            f"N={len(values):,} | "
            f"media={values.mean():+.4f} | "
            f"mediana={values.median():+.4f} | "
            f"media_abs={values.abs().mean():.4f} | "
            f"P90_abs={values.abs().quantile(.90):.4f}"
        )

# --------------------------------------------------
# CAMBIOS EN EL NÚMERO DE JUGADORES
# --------------------------------------------------

print("\n2. VARIACIÓN DEL NÚMERO DE JUGADORES")
print("-" * 72)

for team in ["barca", "athletic"]:

    variable = f"{team}_n_10m"

    print(f"\n{team.upper()}")

    for k in WINDOWS:

        column = f"{variable}_delta_{k}"
        values = df[column].dropna()

        print(
            f"{k:2d} frames | "
            f"N={len(values):,} | "
            f"media={values.mean():+.4f} | "
            f"media_abs={values.abs().mean():.4f} | "
            f"sin_cambio={(values == 0).mean() * 100:.2f}%"
        )

# --------------------------------------------------
# COMPROBAR DESCOMPOSICIÓN DEL MARGEN
# --------------------------------------------------

print("\n3. COMPROBACIÓN DEL MARGEN")
print("-" * 72)

for k in WINDOWS:

    barca = df[f"barca_toward_10m_delta_{k}"]
    athletic = df[f"athletic_toward_10m_delta_{k}"]
    margin = df["toward_margin_10m"].diff(k)

    valid_gap = (frame_id - frame_id.shift(k)) == k
    margin = margin.where(valid_gap)

    reconstructed = barca - athletic

    comparable = (
        barca.notna()
        & athletic.notna()
        & margin.notna()
    )

    if comparable.any():

        error = (
            margin[comparable] - reconstructed[comparable]
        ).abs()

        print(
            f"{k:2d} frames | "
            f"N={comparable.sum():,} | "
            f"error máximo={error.max():.10f}"
        )

# --------------------------------------------------
# CORRELACIONES
# --------------------------------------------------

print("\n4. CORRELACIÓN ENTRE EQUIPOS")
print("-" * 72)

for k in WINDOWS:

    columns = [
        f"barca_toward_10m_delta_{k}",
        f"athletic_toward_10m_delta_{k}",
    ]

    print(f"\n{k} frames")
    print(
        df[columns].corr(method="spearman").round(3).to_string()
    )

# --------------------------------------------------
# GUARDADO
# --------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 72)
print("ARCHIVO GENERADO")
print(OUTPUT)
print("Datos originales sin modificar.")
