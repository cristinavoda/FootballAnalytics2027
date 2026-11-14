from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path("artifacts/tracking_atd.npz")
OUTPUT_DIR = Path("artifacts")

print("Abriendo tracking...")
data = np.load(SOURCE)

frame_ids = data["frame_ids"]
players = data["players"]
ball = data["ball"]

n_frames, n_players, n_channels = players.shape

# -------------------------------------------------
# 1. VISTA LEGIBLE DE LOS PRIMEROS 200 FRAMES
# -------------------------------------------------

preview = {
    "frame_id": frame_ids[:200],
    "ball_x": ball[:200, 0],
    "ball_y": ball[:200, 1],
    "ball_speed": ball[:200, 2],
}

for i in range(n_players):
    preview[f"player{i+1:02d}_x"] = players[:200, i, 0]
    preview[f"player{i+1:02d}_y"] = players[:200, i, 1]
    preview[f"player{i+1:02d}_speed"] = players[:200, i, 2]

pd.DataFrame(preview).to_csv(
    OUTPUT_DIR / "tracking_preview.csv",
    index=False,
    float_format="%.5f"
)

# -------------------------------------------------
# 2. CALIDAD DE CADA TRACK
# -------------------------------------------------

track_rows = []

for i in range(n_players):
    positions = players[:, i, :2]
    valid = np.isfinite(positions).all(axis=1)

    speeds = players[:, i, 2]
    valid_speeds = speeds[np.isfinite(speeds)]

    track_rows.append({
        "track": i + 1,
        "valid_positions": int(valid.sum()),
        "missing_positions": int((~valid).sum()),
        "missing_percent": round((~valid).mean() * 100, 2),
        "valid_speeds": int(len(valid_speeds)),
        "mean_speed": (
            round(float(valid_speeds.mean()), 3)
            if len(valid_speeds) else np.nan
        ),
    })

pd.DataFrame(track_rows).to_csv(
    OUTPUT_DIR / "tracking_quality_by_track.csv",
    index=False
)

# -------------------------------------------------
# 3. CALIDAD DE CADA FRAME
# -------------------------------------------------

valid_player_positions = np.isfinite(
    players[:, :, :2]
).all(axis=2)

valid_ball_positions = np.isfinite(
    ball[:, :2]
).all(axis=1)

frame_quality = pd.DataFrame({
    "frame_id": frame_ids,
    "players_with_position": valid_player_positions.sum(axis=1),
    "ball_available": valid_ball_positions,
})

frame_quality.to_csv(
    OUTPUT_DIR / "tracking_quality_by_frame.csv",
    index=False
)

# -------------------------------------------------
# 4. RESUMEN
# -------------------------------------------------

print("\n========== DIAGNÓSTICO ==========")
print("Frames:", n_frames)
print("Tracks:", n_players)

print(
    "Jugadores con posición por frame:",
    round(float(valid_player_positions.sum(axis=1).mean()), 2),
    "de",
    n_players
)

print(
    "Frames con balón localizado:",
    f"{valid_ball_positions.sum():,}",
    f"({valid_ball_positions.mean() * 100:.2f} %)"
)

print(
    "Frames sin posición de balón:",
    f"{(~valid_ball_positions).sum():,}"
)

print("\nArchivos generados:")
print(" - tracking_preview.csv")
print(" - tracking_quality_by_track.csv")
print(" - tracking_quality_by_frame.csv")

print("\nDiagnóstico terminado.")
