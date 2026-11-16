import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
OUT_DIR = os.path.join(ART, "derived")

data = np.load(os.path.join(ART, "tracking_atd.npz"))

frame_ids = data["frame_ids"]
players = data["players"][:, :, :2].astype(np.float32).copy()
ball = data["ball"][:, :2].astype(np.float32).copy()

# Coordenadas normalizadas -> metros
players[:, :, 0] = (players[:, :, 0] - 0.5) * 105
players[:, :, 1] = (players[:, :, 1] - 0.5) * 68
ball[:, 0] = (ball[:, 0] - 0.5) * 105
ball[:, 1] = (ball[:, 1] - 0.5) * 68

barca = players[:, :24]
athletic = players[:, 24:46]

def valid_xy(x):
    return np.isfinite(x).all(axis=-1)

def centroid(team):
    valid = valid_xy(team)
    count = valid.sum(axis=1)
    total = np.where(valid[..., None], team, 0).sum(axis=1)
    result = np.full((len(team), 2), np.nan, dtype=np.float32)
    ok = count > 0
    result[ok] = total[ok] / count[ok, None]
    return result, count

def distances(team, target):
    valid = valid_xy(team)
    target_valid = valid_xy(target)
    diff = team - target[:, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist[~valid | ~target_valid[:, None]] = np.nan
    return dist

def nearest_mean(dist):
    with np.errstate(all="ignore"):
        nearest = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        mean = np.nanmean(dist, axis=1)
    count = np.isfinite(dist).sum(axis=1)
    nearest[count == 0] = np.nan
    mean[count == 0] = np.nan
    return nearest, mean

def count_radius(dist, radius):
    return (
        np.isfinite(dist) & (dist <= radius)
    ).sum(axis=1).astype(np.float32)

barca_centroid, barca_valid = centroid(barca)
athletic_centroid, athletic_valid = centroid(athletic)

barca_dist = distances(barca, ball)
athletic_dist = distances(athletic, ball)

barca_nearest, barca_mean = nearest_mean(barca_dist)
athletic_nearest, athletic_mean = nearest_mean(athletic_dist)

barca_n5 = count_radius(barca_dist, 5.0)
athletic_n5 = count_radius(athletic_dist, 5.0)
barca_n10 = count_radius(barca_dist, 10.0)
athletic_n10 = count_radius(athletic_dist, 10.0)

barca_to_centroid = np.linalg.norm(
    ball - barca_centroid, axis=1
)
athletic_to_centroid = np.linalg.norm(
    ball - athletic_centroid, axis=1
)

barca_to_centroid[
    ~valid_xy(ball) | ~valid_xy(barca_centroid)
] = np.nan
athletic_to_centroid[
    ~valid_xy(ball) | ~valid_xy(athletic_centroid)
] = np.nan

total5 = barca_n5 + athletic_n5
total10 = barca_n10 + athletic_n10

features = pd.DataFrame({
    "frame_id": frame_ids,
    "rg_barca_centroid_x": barca_centroid[:, 0],
    "rg_barca_centroid_y": barca_centroid[:, 1],
    "rg_athletic_centroid_x": athletic_centroid[:, 0],
    "rg_athletic_centroid_y": athletic_centroid[:, 1],
    "rg_team_centroid_distance": np.linalg.norm(
        barca_centroid - athletic_centroid, axis=1
    ),
    "rg_ball_to_barca_centroid": barca_to_centroid,
    "rg_ball_to_athletic_centroid": athletic_to_centroid,
    "rg_centroid_proximity_advantage":
        athletic_to_centroid - barca_to_centroid,
    "rg_barca_nearest_ball": barca_nearest,
    "rg_athletic_nearest_ball": athletic_nearest,
    "rg_nearest_proximity_advantage":
        athletic_nearest - barca_nearest,
    "rg_barca_mean_ball_distance": barca_mean,
    "rg_athletic_mean_ball_distance": athletic_mean,
    "rg_mean_distance_advantage": athletic_mean - barca_mean,
    "rg_barca_n5": barca_n5,
    "rg_athletic_n5": athletic_n5,
    "rg_n5_margin": barca_n5 - athletic_n5,
    "rg_barca_n10": barca_n10,
    "rg_athletic_n10": athletic_n10,
    "rg_n10_margin": barca_n10 - athletic_n10,
    "rg_barca_density_5": barca_n5 / np.maximum(barca_valid, 1),
    "rg_athletic_density_5": athletic_n5 / np.maximum(athletic_valid, 1),
    "rg_density_margin_5":
        barca_n5 / np.maximum(barca_valid, 1)
        - athletic_n5 / np.maximum(athletic_valid, 1),
    "rg_barca_density_10": barca_n10 / np.maximum(barca_valid, 1),
    "rg_athletic_density_10": athletic_n10 / np.maximum(athletic_valid, 1),
    "rg_density_margin_10":
        barca_n10 / np.maximum(barca_valid, 1)
        - athletic_n10 / np.maximum(athletic_valid, 1),
    "rg_barca_share_5": barca_n5 / np.maximum(total5, 1),
    "rg_athletic_share_5": athletic_n5 / np.maximum(total5, 1),
    "rg_barca_share_10": barca_n10 / np.maximum(total10, 1),
    "rg_athletic_share_10": athletic_n10 / np.maximum(total10, 1),
    "rg_barca_valid": barca_valid,
    "rg_athletic_valid": athletic_valid,
})

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "relational_geometry_v2.csv")
features.to_csv(out, index=False)

print("RG V2 COMPLETADA")
print("Frames:", len(features))
print("Columnas:", len(features.columns))
print("Salida:", out)
print("Distancias y radios expresados en metros.")
print("RG V1 y datos originales intactos.")
