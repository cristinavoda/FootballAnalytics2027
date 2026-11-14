
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "temporal_features_v1.csv"

print(f"Leyendo: {INPUT}")

df = pd.read_csv(
    INPUT,
    usecols=[
        "dist_margin_min_m",
        "compactness_margin_m"
    ]
)

# Eliminar únicamente filas sin datos en las dos variables
df = df.dropna(
    subset=["dist_margin_min_m", "compactness_margin_m"]
)

# Grupo A: Barcelona está más cerca del balón
grupo_a = df[df["dist_margin_min_m"] < 0]["compactness_margin_m"]

# Grupo B: Athletic está más cerca del balón
grupo_b = df[df["dist_margin_min_m"] > 0]["compactness_margin_m"]

print("\n=== COMPARACIÓN DE DISPERSIÓN ===")

for nombre, grupo in [
    ("GRUPO A - Barcelona más cerca", grupo_a),
    ("GRUPO B - Athletic más cerca", grupo_b)
]:
    print(f"\n{nombre}")
    print(f"Frames: {len(grupo):,}")
    print(f"Media: {grupo.mean():.4f} m")
    print(f"Mediana: {grupo.median():.4f} m")
    print(f"Q1 (25%): {grupo.quantile(0.25):.4f} m")
    print(f"Q3 (75%): {grupo.quantile(0.75):.4f} m")
    print(f"Mínimo: {grupo.min():.4f} m")
    print(f"Máximo: {grupo.max():.4f} m")

print("\nFrames excluidos por empate exacto de distancias:")
print(f"{(df['dist_margin_min_m'] == 0).sum():,}")
