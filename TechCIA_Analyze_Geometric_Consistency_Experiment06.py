
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path(
    "artifacts/TechCIA_GeometricPossession_Experiment05_MultiWindow.csv"
)

OUTPUT = Path(
    "artifacts/TechCIA_Geometric_Consistency_Experiment06.csv"
)

df = pd.read_csv(SOURCE)

if len(df) != 4500:
    raise ValueError(f"Se esperaban 4500 frames, hay {len(df)}")

# Diferencias orientadas:
# Positivo = la característica favorece espacialmente a Barcelona.
# Negativo = la característica favorece espacialmente a Athletic.

metrics = {}

# Distancias: cuanto menor, mayor proximidad.
for feature in (
    "nearest_m",
    "nearest3_mean_m",
    "mean_distance_m",
    "median_distance_m",
):
    metrics[feature] = (
        df[f"Athletic_{feature}"] -
        df[f"Barcelona_{feature}"]
    )

# Densidad relativa: mayor proporción dentro del radio.
for radius in (2, 5, 10, 15):
    feature = f"share_within_{radius}m"

    metrics[feature] = (
        df[f"Barcelona_{feature}"] -
        df[f"Athletic_{feature}"]
    )

# Cobertura angular: mayor cobertura.
metrics["angular_coverage_deg"] = (
    df["Barcelona_angular_coverage_deg"] -
    df["Athletic_angular_coverage_deg"]
)

# Dispersión angular: menor R = distribución más dispersa.
metrics["angular_dispersion"] = (
    df["Athletic_angular_resultant_R"] -
    df["Barcelona_angular_resultant_R"]
)


def summarize(values, segment, feature):

    values = pd.Series(values).dropna()

    positive = int((values > 1e-9).sum())
    negative = int((values < -1e-9).sum())
    ties = int((values.abs() <= 1e-9).sum())

    n = len(values)

    if n == 0:
        return None

    return {
        "segment": segment,
        "feature": feature,
        "frames": n,
        "median_difference": values.median(),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "barcelona_positive_pct": 100 * positive / n,
        "athletic_positive_pct": 100 * negative / n,
        "tie_pct": 100 * ties / n,
    }


rows = []

groups = [
    ("TOTAL", df),
    *list(df.groupby("segment", sort=False))
]

for segment, part in groups:

    for feature, difference in metrics.items():

        values = difference.loc[part.index]

        result = summarize(
            values,
            segment,
            feature
        )

        if result is not None:
            rows.append(result)

result_df = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

result_df.to_csv(
    OUTPUT,
    index=False,
    float_format="%.5f"
)

print("\n==========================================")
print("TECH-CIA | CONSISTENCIA GEOMETRICA 06")
print("==========================================")

for segment, part in result_df.groupby(
    "segment",
    sort=False
):

    print(f"\n========== {segment} ==========")

    for _, row in part.iterrows():

        print(
            f"{row['feature']:25s} | "
            f"n={int(row['frames']):4d} | "
            f"mediana={row['median_difference']:8.4f} | "
            f"Barça={row['barcelona_positive_pct']:5.1f}% | "
            f"Athletic={row['athletic_positive_pct']:5.1f}% | "
            f"empate={row['tie_pct']:5.1f}%"
        )

print("\nCSV generado:")
print(OUTPUT.resolve())
