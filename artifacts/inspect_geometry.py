import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent

df = pd.read_csv(base / "geometry_basic_features.csv")

print("COLUMNAS:")
print(df.columns.tolist())

print("\nPRIMERAS 8 FILAS:")
print(df.head(8).to_string(index=False))

print("\nESTADÍSTICAS:")
cols = [
    "barcelona_compactness_m",
    "athletic_compactness_m",
    "centroid_distance_m"
]

existing = [c for c in cols if c in df.columns]

print(df[existing].describe().round(4).to_string())

print("\nFILAS:", len(df))
