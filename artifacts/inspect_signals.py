import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent
df = pd.read_csv(base / "temporal_features_v1.csv")

cols = [
    "compactness_margin_m",
    "barcelona_compactness_m",
    "athletic_compactness_m",
    "dist_margin_min_m_trend_1s",
    "dist_margin_top3_m_trend_1s",
    "players_5m_margin_trend_1s",
    "players_10m_margin_trend_1s",
    "speed_margin_kph_trend_1s",
]

print("=== ESTADÍSTICAS DE VARIABLES ===")

for col in cols:
    if col not in df.columns:
        print(f"\n{col}: NO EXISTE")
        continue

    s = pd.to_numeric(df[col], errors="coerce").dropna()

    print(f"\n{col}")
    print(s.describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    ).round(4).to_string())

print("\n=== FIN ===")
