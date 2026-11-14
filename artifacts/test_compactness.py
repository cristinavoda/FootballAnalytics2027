
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

DATA_FILE = BASE / "tracking_atd.npz"
GEOMETRY_FILE = BASE / "geometry_basic_features.csv"

print("Cargando tracking original...")
data = np.load(DATA_FILE)
players = data["players"]

print("Cargando compactness de referencia...")
geometry = pd.read_csv(GEOMETRY_FILE)

assert len(players) == 199319
assert len(geometry) == 199319

def calculate_rms(points):
    valid = points[np.isfinite(points).all(axis=1)]

    if len(valid) < 2:
        return np.nan

    xy = valid * np.array([105.0, 68.0])
    centroid = xy.mean(axis=0)
    distances = np.linalg.norm(xy - centroid, axis=1)

    return np.sqrt(np.mean(distances ** 2))


teams = {
    "Barcelona": (0, 24, "barcelona_compactness_m"),
    "Athletic": (24, 46, "athletic_compactness_m"),
}

for team, (start, end, csv_col) in teams.items():

    calculated = np.array([
        calculate_rms(players[i, start:end, :2])
        for i in range(len(players))
    ])

    reference = geometry[csv_col].to_numpy(dtype=float)

    valid = np.isfinite(calculated) & np.isfinite(reference)

    calc = calculated[valid]
    ref = reference[valid]

    errors = np.abs(calc - ref)

    correlation = (
        np.corrcoef(calc, ref)[0, 1]
        if len(calc) > 1 else np.nan
    )

    mae = errors.mean()
    max_error = errors.max()
    rmse = np.sqrt(np.mean((calc - ref) ** 2))

    print(f"\n{'=' * 55}")
    print(f"{team.upper()} — AUDITORÍA RMS")
    print(f"{'=' * 55}")

    print(f"Frames totales:       {len(players):,}")
    print(f"Frames comparables:   {valid.sum():,}")
    print(f"Cobertura:            {valid.mean() * 100:.2f}%")
    print(f"MAE:                  {mae:.6f} m")
    print(f"Error máximo:         {max_error:.6f} m")
    print(f"RMSE:                 {rmse:.6f} m")
    print(f"Correlación Pearson:  {correlation:.8f}")

print("\nAuditoría terminada. No se han modificado archivos.")
