
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "temporal_features_v1.csv"

df = pd.read_csv(
    INPUT,
    usecols=[
        "dist_margin_min_m",
        "compactness_margin_m"
    ]
)

df = df.dropna(
    subset=["dist_margin_min_m", "compactness_margin_m"]
)

# Separar escenarios según proximidad al balón
groups = {
    "Barcelona más cerca": df[df["dist_margin_min_m"] < 0],
    "Athletic más cerca": df[df["dist_margin_min_m"] > 0]
}

print("\n=== DISTRIBUCIÓN DE SIGNOS DE COMPACTNESS ===")

for name, group in groups.items():

    values = group["compactness_margin_m"]

    positive = (values > 0).sum()
    negative = (values < 0).sum()
    zero = (values == 0).sum()
    total = len(values)

    print(f"\n{name}")
    print(f"Total de frames: {total:,}")

    print(
        f"Positiva: {positive:,} "
        f"({positive / total * 100:.2f} %)"
    )

    print(
        f"Negativa: {negative:,} "
        f"({negative / total * 100:.2f} %)"
    )

    print(
        f"Cero: {zero:,} "
        f"({zero / total * 100:.2f} %)"
    )

    print(
        f"Comprobación: "
        f"{(positive + negative + zero) / total * 100:.2f} %"
    )
