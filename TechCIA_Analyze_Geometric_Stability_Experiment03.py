
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path(
    "artifacts/TechCIA_GeometricPossession_Experiment01_minute94.csv"
)

OUTPUT = Path(
    "artifacts/TechCIA_Geometric_Stability_Experiment03.csv"
)

df = pd.read_csv(SOURCE)

if len(df) != 1500:
    raise ValueError(f"Se esperaban 1500 frames, encontrados: {len(df)}")

first_frame = int(df["frame_id"].min())

df["window_10s"] = (
    (df["frame_id"] - first_frame) // 250 + 1
)

# Definición de diferencias:
# Positivo en distancia: Barcelona está más cerca.
# Positivo en densidad: Barcelona tiene mayor proporción próxima.
# Positivo en cobertura: Barcelona ocupa más ángulos.

metrics = {}

for feature in (
    "nearest_m",
    "nearest3_mean_m",
    "mean_distance_m",
    "median_distance_m",
):
    metrics[feature] = (
        df[f"Athletic_{feature}"] -
        df[f"Barcelona_{feature}"],
        "Distancia: positivo = Barcelona más cerca"
    )

for radius in (2, 5, 10, 15):
    feature = f"share_within_{radius}m"

    metrics[feature] = (
        df[f"Barcelona_{feature}"] -
        df[f"Athletic_{feature}"],
        f"Densidad relativa: positivo = Barcelona"
    )

metrics["angular_coverage_deg"] = (
    df["Barcelona_angular_coverage_deg"] -
    df["Athletic_angular_coverage_deg"],
    "Cobertura angular: positivo = Barcelona"
)

metrics["largest_angular_gap_deg"] = (
    df["Athletic_largest_angular_gap_deg"] -
    df["Barcelona_largest_angular_gap_deg"],
    "Hueco angular: positivo = menor hueco Barcelona"
)

metrics["angular_resultant_R"] = (
    df["Barcelona_angular_resultant_R"] -
    df["Athletic_angular_resultant_R"],
    "Concentración angular: positivo = Barcelona"
)


def summarize(values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return {
            "n": 0,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "positive_pct": np.nan,
            "negative_pct": np.nan,
            "tie_pct": np.nan,
        }

    return {
        "n": len(values),
        "median": values.median(),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "positive_pct": 100 * (values > 1e-9).mean(),
        "negative_pct": 100 * (values < -1e-9).mean(),
        "tie_pct": 100 * (values.abs() <= 1e-9).mean(),
    }


rows = []

for metric, (difference, interpretation) in metrics.items():

    for window in [0, 1, 2, 3, 4, 5, 6]:

        if window == 0:
            values = difference
            window_label = "TOTAL"
        else:
            mask = df["window_10s"] == window
            values = difference[mask]
            window_label = str(window)

        result = summarize(values)

        rows.append({
            "metric": metric,
            "interpretation": interpretation,
            "window": window_label,
            **result
        })

result_df = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT, index=False, float_format="%.6f")

print("\n==========================================")
print("TECH-CIA | ESTABILIDAD GEOMÉTRICA 03")
print("==========================================")

for metric in metrics:

    print(f"\n========== {metric} ==========")

    subset = result_df[
        result_df["metric"] == metric
    ]

    for _, row in subset.iterrows():

        print(
            f"Ventana {row['window']:>5} | "
            f"n={int(row['n']):3d} | "
            f"mediana={row['median']:8.4f} | "
            f"Q25={row['q25']:8.4f} | "
            f"Q75={row['q75']:8.4f} | "
            f"positivo={row['positive_pct']:5.1f}% | "
            f"negativo={row['negative_pct']:5.1f}%"
        )

print("\nCSV generado:")
print(OUTPUT.resolve())
