
from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path(
    "artifacts/TechCIA_GeometricPossession_Experiment01_minute94.csv"
)

df = pd.read_csv(CSV)

if len(df) != 1500:
    raise ValueError(f"Se esperaban 1500 filas y hay {len(df)}")

# Seis ventanas consecutivas de 250 frames = 10 segundos.
first_frame = int(df["frame_id"].min())

df["window_10s"] = (
    (df["frame_id"] - first_frame) // 250 + 1
)

print("\n============================================")
print("TECH-CIA | ANALISIS TEMPORAL GEOMETRICO")
print("============================================")

# --------------------------------------------------
# 1. Comparación de proximidad por frame.
# Una distancia menor significa mayor proximidad.
# --------------------------------------------------

print("\n1. PROXIMIDAD: porcentaje de frames")

for feature in (
    "nearest_m",
    "nearest3_mean_m",
    "mean_distance_m",
    "median_distance_m",
):

    b = df[f"Barcelona_{feature}"].to_numpy()
    a = df[f"Athletic_{feature}"].to_numpy()

    valid = np.isfinite(b) & np.isfinite(a)

    b_closer = np.sum(b[valid] < a[valid])
    a_closer = np.sum(a[valid] < b[valid])
    ties = np.sum(b[valid] == a[valid])

    total = np.sum(valid)

    print(f"\n{feature}")
    print(f"Barcelona más cerca: {100*b_closer/total:.1f}%")
    print(f"Athletic más cerca:  {100*a_closer/total:.1f}%")
    print(f"Empates:             {100*ties/total:.1f}%")

# --------------------------------------------------
# 2. Densidad: qué equipo tiene más jugadores
# dentro de cada radio en cada frame.
# --------------------------------------------------

print("\n2. DENSIDAD: porcentaje de frames")

for radius in (2, 5, 10, 15):

    feature = f"within_{radius}m"

    b = df[f"Barcelona_{feature}"].to_numpy()
    a = df[f"Athletic_{feature}"].to_numpy()

    valid = np.isfinite(b) & np.isfinite(a)

    b_more = np.sum(b[valid] > a[valid])
    a_more = np.sum(a[valid] > b[valid])
    ties = np.sum(b[valid] == a[valid])

    total = np.sum(valid)

    print(f"\nRadio {radius} m")
    print(f"Barcelona más jugadores: {100*b_more/total:.1f}%")
    print(f"Athletic más jugadores:  {100*a_more/total:.1f}%")
    print(f"Empates:                 {100*ties/total:.1f}%")

# --------------------------------------------------
# 3. Evolución en seis ventanas de 10 segundos.
# Medianas de distancia y cobertura.
# Medias de densidad.
# --------------------------------------------------

print("\n3. EVOLUCION POR VENTANAS DE 10 SEGUNDOS")

for window, part in df.groupby("window_10s"):

    print(f"\nVentana {window}: "
          f"{part['frame_id'].min()} - "
          f"{part['frame_id'].max()}")

    features = (
        ("nearest_m", "mediana"),
        ("nearest3_mean_m", "mediana"),
        ("within_5m", "media"),
        ("within_10m", "media"),
        ("angular_coverage_deg", "mediana"),
    )

    for feature, statistic in features:

        b = part[f"Barcelona_{feature}"]
        a = part[f"Athletic_{feature}"]

        if statistic == "mediana":
            b_value = b.median()
            a_value = a.median()
        else:
            b_value = b.mean()
            a_value = a.mean()

        print(
            f"{feature:24s} | "
            f"Barça: {b_value:8.3f} | "
            f"Athletic: {a_value:8.3f}"
        )

print("\nAnálisis temporal terminado.")
