import os
import numpy as np
import pandas as pd


# ============================================================
# TECH-CIA — RELATIONAL GEOMETRY V1
# ============================================================
#
# Entrada:
#   artifacts/tracking_atd.npz
#
# Salida:
#   artifacts/derived/relational_geometry_v1.csv
#
# No modifica ningún dato original.
#
# Team convention already used by the project:
#   tracks 0:23  -> Barcelona / Home
#   tracks 24:45 -> Athletic / Away
#
# La geometría se calcula en el sistema espacial del tracking.
# No se aplican conversiones arbitrarias de coordenadas.
# ============================================================


ROOT = os.path.dirname(os.path.abspath(__file__))

NPZ_PATH = os.path.join(
    ROOT, "artifacts", "tracking_atd.npz"
)

OUT_DIR = os.path.join(
    ROOT, "artifacts", "derived"
)

OUT_PATH = os.path.join(
    OUT_DIR, "relational_geometry_v1.csv"
)


# -----------------------------
# Parámetros
# -----------------------------

# Radios expresados en unidades del sistema espacial del tracking.
# 5 y 10 corresponden a la escala física si el tracking ya está
# en metros; si no, conservan igualmente su utilidad como relaciones
# espaciales sólo si esa escala está validada.
RADIUS_5 = 5.0
RADIUS_10 = 10.0

EPS = 1e-8


# ============================================================
# CARGA
# ============================================================

print("\n========== TECH-CIA RELATIONAL GEOMETRY V1 ==========")

data = np.load(NPZ_PATH)

frame_ids = data["frame_ids"]
players = data["players"]
ball = data["ball"]

n_frames = len(frame_ids)

print(f"Frames:  {n_frames}")
print(f"Players: {players.shape}")
print(f"Ball:    {ball.shape}")


# ============================================================
# EQUIPOS
# ============================================================

barca = players[:, 0:24, :2]
athletic = players[:, 24:46, :2]

ball_xy = ball[:, :2]


# ============================================================
# UTILIDADES
# ============================================================

def finite_xy(x):
    return np.isfinite(x).all(axis=-1)


def centroid(points):
    valid = finite_xy(points)

    count = valid.sum(axis=1)

    summed = np.where(
        valid[..., None],
        points,
        0.0
    ).sum(axis=1)

    result = np.full(
        (points.shape[0], 2),
        np.nan,
        dtype=np.float32
    )

    ok = count > 0
    result[ok] = summed[ok] / count[ok, None]

    return result, count


def distance_to_point(points, target):
    valid_points = finite_xy(points)
    valid_target = finite_xy(target)

    diff = points - target[:, None, :]

    dist = np.sqrt(
        np.nansum(diff * diff, axis=-1)
    )

    dist[
        ~valid_points | ~valid_target[:, None]
    ] = np.nan

    return dist


def nearest_and_mean(dist):
    nearest = np.nanmin(dist, axis=1)

    mean = np.nanmean(dist, axis=1)

    nearest_count = np.sum(
        np.isfinite(dist),
        axis=1
    )

    nearest[nearest_count == 0] = np.nan
    mean[nearest_count == 0] = np.nan

    return nearest, mean


def count_within(dist, radius):
    return np.sum(
        np.isfinite(dist) & (dist <= radius),
        axis=1
    ).astype(np.float32)


# ============================================================
# CENTROIDES
# ============================================================

print("\nCalculating team centroids...")

barca_centroid, barca_valid = centroid(barca)
athletic_centroid, athletic_valid = centroid(athletic)

centroid_diff = (
    barca_centroid - athletic_centroid
)

centroid_distance = np.sqrt(
    np.nansum(
        centroid_diff * centroid_diff,
        axis=1
    )
)

centroid_distance[
    (barca_valid == 0) |
    (athletic_valid == 0)
] = np.nan


# ============================================================
# DISTANCIAS AL BALÓN
# ============================================================

print("Calculating player-ball relations...")

barca_ball_dist = distance_to_point(
    barca,
    ball_xy
)

athletic_ball_dist = distance_to_point(
    athletic,
    ball_xy
)


# ============================================================
# PROXIMIDAD
# ============================================================

print("Calculating proximity features...")

barca_nearest, barca_mean_dist = nearest_and_mean(
    barca_ball_dist
)

athletic_nearest, athletic_mean_dist = nearest_and_mean(
    athletic_ball_dist
)

nearest_margin = (
    athletic_nearest - barca_nearest
)

# Positivo -> Barcelona está más cerca del balón.
# Negativo -> Athletic está más cerca.
barca_proximity_advantage = (
    athletic_nearest - barca_nearest
)


# ============================================================
# OCUPACIÓN ALREDEDOR DEL BALÓN
# ============================================================

print("Calculating spatial occupancy...")

barca_n5 = count_within(
    barca_ball_dist,
    RADIUS_5
)

athletic_n5 = count_within(
    athletic_ball_dist,
    RADIUS_5
)

barca_n10 = count_within(
    barca_ball_dist,
    RADIUS_10
)

athletic_n10 = count_within(
    athletic_ball_dist,
    RADIUS_10
)

n5_margin = (
    barca_n5 - athletic_n5
)

n10_margin = (
    barca_n10 - athletic_n10
)


# ============================================================
# DENSIDAD RELATIVA
# ============================================================

print("Calculating local density...")

barca_density_5 = (
    barca_n5 /
    np.maximum(barca_valid, 1)
)

athletic_density_5 = (
    athletic_n5 /
    np.maximum(athletic_valid, 1)
)

barca_density_10 = (
    barca_n10 /
    np.maximum(barca_valid, 1)
)

athletic_density_10 = (
    athletic_n10 /
    np.maximum(athletic_valid, 1)
)

density_margin_5 = (
    barca_density_5 -
    athletic_density_5
)

density_margin_10 = (
    barca_density_10 -
    athletic_density_10
)


# ============================================================
# DISTANCIA BALÓN -> CENTROIDE
# ============================================================

print("Calculating ball-centroid relations...")

ball_to_barca = np.sqrt(
    np.nansum(
        (ball_xy - barca_centroid) ** 2,
        axis=1
    )
)

ball_to_athletic = np.sqrt(
    np.nansum(
        (ball_xy - athletic_centroid) ** 2,
        axis=1
    )
)

ball_to_barca[
    ~finite_xy(ball_xy) |
    ~finite_xy(barca_centroid)
] = np.nan

ball_to_athletic[
    ~finite_xy(ball_xy) |
    ~finite_xy(athletic_centroid)
] = np.nan

centroid_proximity_advantage = (
    ball_to_athletic -
    ball_to_barca
)


# ============================================================
# EQUIPO CON MAYOR PROXIMIDAD COLECTIVA
# ============================================================

# Positivo -> ventaja Barcelona
# Negativo -> ventaja Athletic

mean_distance_advantage = (
    athletic_mean_dist -
    barca_mean_dist
)


# ============================================================
# CONCENTRACIÓN RELATIVA
# ============================================================

total_n5 = barca_n5 + athletic_n5
total_n10 = barca_n10 + athletic_n10

barca_share_5 = (
    barca_n5 /
    np.maximum(total_n5, 1)
)

barca_share_10 = (
    barca_n10 /
    np.maximum(total_n10, 1)
)

athletic_share_5 = (
    athletic_n5 /
    np.maximum(total_n5, 1)
)

athletic_share_10 = (
    athletic_n10 /
    np.maximum(total_n10, 1)
)


# ============================================================
# DATAFRAME
# ============================================================

features = pd.DataFrame({
    "frame_id": frame_ids,

    # Centroids
    "rg_barca_centroid_x": barca_centroid[:, 0],
    "rg_barca_centroid_y": barca_centroid[:, 1],
    "rg_athletic_centroid_x": athletic_centroid[:, 0],
    "rg_athletic_centroid_y": athletic_centroid[:, 1],

    # Team separation
    "rg_team_centroid_distance": centroid_distance,

    # Ball -> centroid
    "rg_ball_to_barca_centroid": ball_to_barca,
    "rg_ball_to_athletic_centroid": ball_to_athletic,
    "rg_centroid_proximity_advantage": centroid_proximity_advantage,

    # Nearest player
    "rg_barca_nearest_ball": barca_nearest,
    "rg_athletic_nearest_ball": athletic_nearest,
    "rg_nearest_proximity_advantage": nearest_margin,

    # Mean player-ball distance
    "rg_barca_mean_ball_distance": barca_mean_dist,
    "rg_athletic_mean_ball_distance": athletic_mean_dist,
    "rg_mean_distance_advantage": mean_distance_advantage,

    # Occupancy
    "rg_barca_n5": barca_n5,
    "rg_athletic_n5": athletic_n5,
    "rg_n5_margin": n5_margin,

    "rg_barca_n10": barca_n10,
    "rg_athletic_n10": athletic_n10,
    "rg_n10_margin": n10_margin,

    # Density
    "rg_barca_density_5": barca_density_5,
    "rg_athletic_density_5": athletic_density_5,
    "rg_density_margin_5": density_margin_5,

    "rg_barca_density_10": barca_density_10,
    "rg_athletic_density_10": athletic_density_10,
    "rg_density_margin_10": density_margin_10,

    # Local team shares
    "rg_barca_share_5": barca_share_5,
    "rg_athletic_share_5": athletic_share_5,
    "rg_barca_share_10": barca_share_10,
    "rg_athletic_share_10": athletic_share_10,

    # Valid tracking
    "rg_barca_valid": barca_valid,
    "rg_athletic_valid": athletic_valid,
})

os.makedirs(OUT_DIR, exist_ok=True)

features.to_csv(
    OUT_PATH,
    index=False
)

print("\n========== DONE ==========")
print(f"Output: {OUT_PATH}")
print(f"Rows:   {len(features)}")
print(f"Cols:   {len(features.columns)}")

print("\nNew relational features:")
for c in features.columns:
    if c != "frame_id":
        print(f"  {c}")

print("\nNo original file was modified.")
