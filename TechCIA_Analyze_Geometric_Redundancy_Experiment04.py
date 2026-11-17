
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "artifacts/TechCIA_GeometricPossession_Experiment01_minute94.csv"
)

OUTPUT_FEATURES = Path(
    "artifacts/TechCIA_Geometric_Experiment04_1s_features.csv"
)

OUTPUT_CORR = Path(
    "artifacts/TechCIA_Geometric_Experiment04_Spearman.csv"
)


def main():

    df = pd.read_csv(SOURCE)

    if len(df) != 1500:
        raise ValueError("El CSV no contiene 1500 frames.")

    features = pd.DataFrame(index=df.index)

    # Signo positivo = Barcelona presenta el valor espacial
    # más favorable en la característica correspondiente.

    # Proximidad individual y colectiva.
    for metric in (
        "nearest_m",
        "nearest3_mean_m",
        "mean_distance_m",
        "median_distance_m",
    ):
        features[metric] = (
            df[f"Athletic_{metric}"]
            - df[f"Barcelona_{metric}"]
        )

    # Proporción de jugadores dentro de cada radio.
    for radius in (2, 5, 10, 15):

        column = f"share_within_{radius}m"

        features[column] = (
            df[f"Barcelona_{column}"]
            - df[f"Athletic_{column}"]
        )

    # Geometría angular.
    features["angular_coverage"] = (
        df["Barcelona_angular_coverage_deg"]
        - df["Athletic_angular_coverage_deg"]
    )

    # Positivo = Barcelona más disperso angularmente.
    features["angular_dispersion"] = (
        df["Athletic_angular_resultant_R"]
        - df["Barcelona_angular_resultant_R"]
    )

    # Agregación temporal: 25 frames = 1 segundo.
    features["second"] = (
        df["frame_id"] - df["frame_id"].min()
    ) // 25

    features_1s = (
        features
        .groupby("second")
        .median(numeric_only=True)
    )

    # Correlaciones de Spearman sobre 60 observaciones.
    correlation = features_1s.corr(method="spearman")

    OUTPUT_FEATURES.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features_1s.to_csv(
        OUTPUT_FEATURES,
        float_format="%.6f"
    )

    correlation.to_csv(
        OUTPUT_CORR,
        float_format="%.4f"
    )

    print("\n==========================================")
    print("TECH-CIA | REDUNDANCIA GEOMETRICA 04")
    print("==========================================")

    print("\nObservaciones de un segundo:", len(features_1s))

    print("\nCORRELACIONES DE SPEARMAN")
    print(correlation.round(2).to_string())

    # Pares más correlacionados, sin contar la diagonal.
    upper = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1
        ).astype(bool)
    )

    pairs = (
        upper.stack()
        .rename("spearman_rho")
        .reset_index()
    )

    pairs.columns = ["feature_1", "feature_2", "spearman_rho"]

    pairs["absolute_rho"] = pairs["spearman_rho"].abs()

    pairs = pairs.sort_values(
        "absolute_rho",
        ascending=False
    )

    print("\nPARES CON MAYOR CORRELACION")
    print(pairs.head(15).to_string(index=False))

    print("\nArchivos generados:")
    print(OUTPUT_FEATURES.resolve())
    print(OUTPUT_CORR.resolve())


if __name__ == "__main__":
    main()
