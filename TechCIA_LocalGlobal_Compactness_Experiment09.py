
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(
    "artifacts/TechCIA_Geometric_Compactness_Experiment08.csv"
)

OUTPUT = Path(
    "artifacts/TechCIA_LocalGlobal_Compactness_Experiment09.csv"
)

df = pd.read_csv(SOURCE)

if len(df) != 4500:
    raise ValueError(
        f"Se esperaban 4500 filas, encontradas {len(df)}"
    )

TEAMS = ("Barcelona", "Athletic")

METRICS = {
    "hull_area_m2": "hull_area_m2",
    "pairwise_mean_m": "pairwise_mean_m",
    "stretch_index_m": "stretch_index_m",
    "width_m": "width_m",
    "depth_m": "depth_m",
}

rows = []

for team in TEAMS:

    for metric, output_name in METRICS.items():

        local = df[f"{team}_nearest5_{metric}"]
        global_ = df[f"{team}_outfield_{metric}"]

        # Evitar divisiones por cero y datos ausentes.
        valid = (
            local.notna()
            & global_.notna()
            & (global_ > 0)
        )

        ratio = pd.Series(
            np.nan,
            index=df.index
        )

        ratio.loc[valid] = (
            local.loc[valid] / global_.loc[valid]
        )

        df[f"{team}_ratio_{output_name}"] = ratio

for segment, part in df.groupby("segment", sort=False):

    for team in TEAMS:

        for metric in METRICS:

            column = f"{team}_ratio_{metric}"

            values = part[column].dropna()

            rows.append({
                "segment": segment,
                "team": team,
                "metric": metric,
                "valid_frames": len(values),
                "median_ratio": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
            })

summary = pd.DataFrame(rows)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False,
    float_format="%.6f"
)

print("\n==========================================")
print("TECH-CIA | LOCAL / GLOBAL COMPACTNESS 09")
print("==========================================")

for segment in df["segment"].drop_duplicates():

    print(f"\n========== {segment} ==========")

    selected = summary[
        summary["segment"] == segment
    ]

    for metric in METRICS:

        print(f"\n{metric}")

        for team in TEAMS:

            row = selected[
                (selected["team"] == team)
                & (selected["metric"] == metric)
            ].iloc[0]

            print(
                f"{team:12s} | "
                f"n={row['valid_frames']:4d} | "
                f"mediana={row['median_ratio']:.4f} | "
                f"Q25={row['q25']:.4f} | "
                f"Q75={row['q75']:.4f}"
            )

print("\nCSV generado:")
print(OUTPUT.resolve())
