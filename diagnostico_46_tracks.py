import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(".")
TRACKING_FILE = BASE / "artifacts" / "tracking_atd.npz"
VELOCITY_FILE = BASE / "artifacts" / "derived" / "velocities_raw.npz"
OUT_DIR = BASE / "artifacts" / "derived"

START_FRAME = 149490
END_FRAME = 149530

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar los datos originales y las velocidades ya calculadas
tracking = np.load(TRACKING_FILE)
velocity = np.load(VELOCITY_FILE)

print("Claves del archivo de tracking:", tracking.files)
print("Claves del archivo de velocidades:", velocity.files)

required_tracking = ["frame_ids", "players", "ball"]
required_velocity = [
    "frame_ids", "speed", "provider_speed",
    "position_valid", "velocity_valid"
]

for key in required_tracking:
    if key not in tracking.files:
        raise KeyError(f"Falta la clave '{key}' en tracking_atd.npz")

for key in required_velocity:
    if key not in velocity.files:
        raise KeyError(f"Falta la clave '{key}' en velocities_raw.npz")

frames_t = tracking["frame_ids"]
players = tracking["players"]
ball = tracking["ball"]

frames_v = velocity["frame_ids"]
speed = velocity["speed"]
provider_speed = velocity["provider_speed"]
position_valid = velocity["position_valid"]
velocity_valid = velocity["velocity_valid"]

# Comprobar que los dos archivos tienen la misma secuencia de frames
if not np.array_equal(frames_t, frames_v):
    raise ValueError("Los frame_ids de tracking y velocidades no coinciden.")

if players.ndim != 3 or players.shape[1] != 46:
    raise ValueError(f"Forma inesperada de players: {players.shape}")

if speed.shape != (len(frames_t), 46):
    raise ValueError(f"Forma inesperada de speed: {speed.shape}")

# Seleccionar el intervalo solicitado
selected = np.where(
    (frames_t >= START_FRAME) & (frames_t <= END_FRAME)
)[0]

if len(selected) == 0:
    raise ValueError("No se encontraron frames en el intervalo solicitado.")

print(f"\nFrames seleccionados: {len(selected)}")
print(f"Tracks por frame: {players.shape[1]}")
print(f"Forma de ball: {ball.shape}")

def finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.median(values)),
        float(np.percentile(values, 90)),
        float(np.max(values))
    )

# ---------------------------------------------------------
# 1. RESUMEN POR FRAME: los 46 tracks
# ---------------------------------------------------------
summary_rows = []

for i in selected:
    derived = speed[i].astype(float)
    provider = provider_speed[i].astype(float)

    valid_pos = position_valid[i].astype(bool)
    valid_vel = velocity_valid[i].astype(bool)

    valid_derived = derived[valid_vel & np.isfinite(derived)]
    valid_provider = provider[np.isfinite(provider)]

    median_speed, p90_speed, max_speed = finite_stats(valid_derived)
    median_provider, p90_provider, max_provider = finite_stats(valid_provider)

    # Número de velocidades derivadas superiores a 10 m/s
    count_over_10 = int(np.sum(valid_derived > 10))
    count_over_15 = int(np.sum(valid_derived > 15))

    # Posición del balón y desplazamiento entre frames consecutivos.
    ball_x = float(ball[i, 0]) if np.isfinite(ball[i, 0]) else np.nan
    ball_y = float(ball[i, 1]) if np.isfinite(ball[i, 1]) else np.nan

    if i > 0:
        previous_ball = ball[i - 1, :2].astype(float)
        current_ball = ball[i, :2].astype(float)
        if np.all(np.isfinite(previous_ball)) and np.all(np.isfinite(current_ball)):
            ball_displacement = float(np.linalg.norm(current_ball - previous_ball))
        else:
            ball_displacement = np.nan
    else:
        ball_displacement = np.nan

    summary_rows.append({
        "frame": int(frames_t[i]),
        "tracks_with_position": int(np.sum(valid_pos)),
        "tracks_with_velocity": int(np.sum(valid_vel)),
        "derived_speed_median": median_speed,
        "derived_speed_p90": p90_speed,
        "derived_speed_max": max_speed,
        "derived_speed_over_10": count_over_10,
        "derived_speed_over_15": count_over_15,
        "provider_speed_median": median_provider,
        "provider_speed_p90": p90_provider,
        "provider_speed_max": max_provider,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_displacement_coordinate_units": ball_displacement
    })

summary = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "resumen_46_tracks_149490_149530.csv"
summary.to_csv(summary_path, index=False, float_format="%.4f")

# ---------------------------------------------------------
# 2. DETALLE: una fila por track y frame
# ---------------------------------------------------------
detail_rows = []

for i in selected:
    for track_idx in range(46):
        x = float(players[i, track_idx, 0])
        y = float(players[i, track_idx, 1])

        detail_rows.append({
            "frame": int(frames_t[i]),
            "track": track_idx + 1,
            "team": "BAR" if track_idx < 24 else "ATH",
            "x": x,
            "y": y,
            "position_valid": bool(position_valid[i, track_idx]),
            "velocity_valid": bool(velocity_valid[i, track_idx]),
            "derived_speed": float(speed[i, track_idx]),
            "provider_speed": float(provider_speed[i, track_idx])
        })

detail = pd.DataFrame(detail_rows)
detail_path = OUT_DIR / "detalle_46_tracks_149490_149530.csv"
detail.to_csv(detail_path, index=False, float_format="%.4f")

# ---------------------------------------------------------
# 3. INFORME EN CONSOLA
# ---------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("\n=== RESUMEN POR FRAME ===")
print(summary.to_string(index=False))

print("\n=== FRAMES CON MAYOR VELOCIDAD MÁXIMA DERIVADA ===")
print(
    summary.sort_values("derived_speed_max", ascending=False)
    .head(10)[[
        "frame",
        "tracks_with_velocity",
        "derived_speed_median",
        "derived_speed_p90",
        "derived_speed_max",
        "derived_speed_over_10",
        "provider_speed_median"
    ]].to_string(index=False)
)

print("\n=== ARCHIVOS GENERADOS ===")
print(summary_path)
print(detail_path)
print("\nDiagnóstico terminado.")
