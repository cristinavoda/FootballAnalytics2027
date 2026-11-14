
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "temporal_features_v1.csv"

columns = [
    "barcelona_visible_count",
    "athletic_visible_count",
    "compactness_margin_m",
    "dist_margin_min_m",
    "compactness_margin_m_mean_1s",
    "dist_margin_min_m_mean_1s",
    "compactness_margin_m_mean_3s",
    "dist_margin_min_m_mean_3s",
]

df = pd.read_csv(INPUT, usecols=columns)

print("Frames totales:", len(df))

# Cobertura mínima: al menos 8 jugadores visibles por equipo
good_coverage = (
    (df["barcelona_visible_count"] >= 8) &
    (df["athletic_visible_count"] >= 8)
)

filtered = df.loc[good_coverage].copy()

print("Frames con cobertura suficiente:", len(filtered))
print(
    "Porcentaje:",
    round(100 * len(filtered) / len(df), 2),
    "%"
)

pairs = [
    (
        "Instantánea",
        "compactness_margin_m",
        "dist_margin_min_m"
    ),
    (
        "Media 1 segundo",
        "compactness_margin_m_mean_1s",
        "dist_margin_min_m_mean_1s"
    ),
    (
        "Media 3 segundos",
        "compactness_margin_m_mean_3s",
        "dist_margin_min_m_mean_3s"
    ),
]

def analyze(data, label):
    print(f"\n=== {label} ===")

    for name, x, y in pairs:
        valid = data[[x, y]].dropna()

        pearson = valid[x].corr(valid[y], method="pearson")
        rank_x = valid[x].rank()
        rank_y = valid[y].rank()
        spearman = rank_x.corr(rank_y)

        print(f"\n{name}")
        print("N:", len(valid))
        print("Pearson:", round(pearson, 5))
        print("Spearman:", round(spearman, 5))


analyze(df, "MUESTRA COMPLETA")
analyze(filtered, "SOLO COBERTURA >= 8")
