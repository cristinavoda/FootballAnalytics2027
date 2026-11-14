
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

TEMPORAL = ROOT / "temporal_features_v1.csv"
GEOMETRY = ROOT / "geometry_basic_features.csv"

# Intervalos de las cinco rachas positivas y negativas más largas
intervals = [
    ("Positiva", 43276, 49695),
    ("Positiva", 125303, 131207),
    ("Positiva", 104122, 108936),
    ("Positiva", 65058, 68966),
    ("Positiva", 148782, 152656),
    ("Negativa", 165589, 168820),
    ("Negativa", 161251, 164249),
    ("Negativa", 56406, 59080),
    ("Negativa", 178166, 180767),
    ("Negativa", 62506, 65057),
]

geometry = pd.read_csv(
    GEOMETRY,
    usecols=[
        "frame",
        "barcelona_count",
        "athletic_count"
    ]
)

print("=== COBERTURA GENERAL ===")

for team, col in [
    ("Barcelona", "barcelona_count"),
    ("Athletic", "athletic_count")
]:
    print(f"\n{team}")
    print(geometry[col].describe().round(2))
    print(f"Frames con menos de 5 jugadores: {(geometry[col] < 5).sum():,}")
    print(f"Frames con menos de 8 jugadores: {(geometry[col] < 8).sum():,}")

print("\n=== CALIDAD DEL TRACKING EN RACHAS LARGAS ===")

for sign, start, end in intervals:

    segment = geometry[
        geometry["frame"].between(start, end)
    ]

    print(f"\n{sign}: frames {start} - {end}")
    print(f"Frames encontrados: {len(segment):,}")

    for team, col in [
        ("Barcelona", "barcelona_count"),
        ("Athletic", "athletic_count")
    ]:
        print(
            f"{team}: "
            f"media={segment[col].mean():.2f}, "
            f"mínimo={segment[col].min()}, "
            f"mediana={segment[col].median():.2f}, "
            f"frames con menos de 8={ (segment[col] < 8).sum():,}"
        )
