import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent

v2 = pd.read_csv(base / "possession_features_v2.csv")
pos = pd.read_csv(base / "possession_features.csv")
tmp = pd.read_csv(base / "temporal_features_v1.csv")


def audit(df, margin, barcelona, athletic):
    print(f"\n--- {margin} ---")

    required = [margin, barcelona, athletic]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print("Columnas ausentes:", missing)
        return

    m = pd.to_numeric(df[margin], errors="coerce")
    b = pd.to_numeric(df[barcelona], errors="coerce")
    a = pd.to_numeric(df[athletic], errors="coerce")

    valid = m.notna() & b.notna() & a.notna()

    if not valid.any():
        print("Sin datos válidos.")
        return

    errors = {
        "Barcelona - Athletic": (m[valid] - (b[valid] - a[valid])).abs(),
        "Athletic - Barcelona": (m[valid] - (a[valid] - b[valid])).abs()
    }

    print("Filas válidas:", int(valid.sum()))

    for name, error in errors.items():
        print(
            f"{name}: "
            f"coincidencia={(error < 1e-5).mean()*100:.2f}% | "
            f"error máximo={error.max():.8f}"
        )


print("=" * 60)
print("AUDITORÍA DE SIGNOS - TECH-CIA")
print("=" * 60)

print("\n1. POSSESSION FEATURES V2")

for margin, b, a in [
    ("min_distance_margin_m",
     "barcelona_min_distance_m", "athletic_min_distance_m"),
    ("top3_distance_margin_m",
     "barcelona_top3_mean_m", "athletic_top3_mean_m"),
    ("visible_count_margin",
     "barcelona_visible_count", "athletic_visible_count"),
    ("compactness_margin_m",
     "barcelona_compactness_m", "athletic_compactness_m")
]:
    audit(v2, margin, b, a)


print("\n2. TEMPORAL FEATURES V1")

for margin, b, a in [
    ("dist_margin_min_m",
     "barcelona_min_distance_m", "athletic_min_distance_m"),
    ("dist_margin_top3_m",
     "barcelona_top3_mean_m", "athletic_top3_mean_m"),
    ("players_5m_margin",
     "barcelona_players_5m", "athletic_players_5m"),
    ("players_10m_margin",
     "barcelona_players_10m", "athletic_players_10m"),
    ("speed_margin_kph",
     "barcelona_nearest_speed_kph", "athletic_nearest_speed_kph"),
    ("visible_count_margin",
     "barcelona_visible_count", "athletic_visible_count"),
    ("compactness_margin_m",
     "barcelona_compactness_m", "athletic_compactness_m")
]:
    audit(tmp, margin, b, a)


print("\n3. COHERENCIA ENTRE ARCHIVOS")

merged = pos[
    ["frame_id", "min_distance_margin_m", "top3_distance_margin_m"]
].merge(
    tmp[["frame_id", "dist_margin_min_m", "dist_margin_top3_m"]],
    on="frame_id",
    how="inner"
)

for original, temporal in [
    ("min_distance_margin_m", "dist_margin_min_m"),
    ("top3_distance_margin_m", "dist_margin_top3_m")
]:
    error = (
        pd.to_numeric(merged[original], errors="coerce")
        - pd.to_numeric(merged[temporal], errors="coerce")
    ).abs()

    error = error.dropna()

    print(
        f"{original} vs {temporal}: "
        f"coincidencia={(error < 1e-5).mean()*100:.2f}% | "
        f"error máximo={error.max():.8f}"
    )

print("\n4. RESUMEN DE ARCHIVOS")
print("possession_features:", len(pos))
print("possession_features_v2:", len(v2))
print("temporal_features_v1:", len(tmp))
print("Filas comparadas:", len(merged))

print("\nAUDITORÍA TERMINADA")
