
import pandas as pd

path = "artifacts/temporal_features_v1.csv"

groups = {
    "INSTANTANEO": (
        "compactness_margin_m",
        [
            "dist_margin_min_m",
            "dist_margin_top3_m",
            "players_5m_margin",
            "players_10m_margin",
        ],
    ),
    "MEDIA_1S": (
        "compactness_margin_m_mean_1s",
        [
            "dist_margin_min_m_mean_1s",
            "dist_margin_top3_m_mean_1s",
            "players_5m_margin_mean_1s",
            "players_10m_margin_mean_1s",
        ],
    ),
    "MEDIA_3S": (
        "compactness_margin_m_mean_3s",
        [
            "dist_margin_min_m_mean_3s",
            "dist_margin_top3_m_mean_3s",
            "players_5m_margin_mean_3s",
            "players_10m_margin_mean_3s",
        ],
    ),
    "TENDENCIAS_1S": (
        "compactness_margin_m_trend_1s",
        [
            "dist_margin_min_m_trend_1s",
            "dist_margin_top3_m_trend_1s",
            "players_5m_margin_trend_1s",
            "players_10m_margin_trend_1s",
        ],
    ),
}

columns = set()

for compactness, signals in groups.values():
    columns.add(compactness)
    columns.update(signals)

print("Leyendo variables seleccionadas...", flush=True)
df = pd.read_csv(path, usecols=list(columns))

print(f"Filas cargadas: {len(df):,}", flush=True)

for group_name, (compactness, signals) in groups.items():
    print(f"\n{'=' * 65}")
    print(group_name)
    print("=" * 65)

    for signal in signals:
        pair = df[[compactness, signal]].dropna()

        x = pair[compactness]
        y = pair[signal]

        pearson = x.corr(y, method="pearson")

        print(f"Calculando Spearman para {signal}...", flush=True)

        rank_x = x.rank(method="average")
        rank_y = y.rank(method="average")
        spearman = rank_x.corr(rank_y, method="pearson")

        print(
            f"{signal:43s} "
            f"N={len(pair):7,d}  "
            f"Pearson={pearson:+.5f}  "
            f"Spearman={spearman:+.5f}",
            flush=True,
        )

print("\nAuditoría terminada. No se han modificado archivos.")
