
import numpy as np

path = r"artifacts\tracking_atd.npz"
data = np.load(path)

frames = data["frame_ids"]
players = data["players"]
ball = data["ball"]

fps = 25
window = fps * 60  # ventanas de 60 segundos

valid_players = np.isfinite(players[:, :, :2]).all(axis=2)
valid_ball = np.isfinite(ball[:, :2]).all(axis=1)

players_per_frame = valid_players.sum(axis=1)

print("\n========== IDENTIFICADORES DE FRAME ==========")
print("Frames totales:", len(frames))
print("Primer frame_id:", frames[0])
print("Último frame_id:", frames[-1])
print("Diferencia mínima entre IDs:", np.diff(frames).min())
print("Diferencia mediana entre IDs:", np.median(np.diff(frames)))
print("Diferencia máxima entre IDs:", np.diff(frames).max())
print("Saltos entre IDs:", np.sum(np.diff(frames) != 1))
print("Duración aproximada (min):", round(len(frames) / fps / 60, 2))

print("\n========== DISPONIBILIDAD POR MINUTO ==========")
print("MINUTO | FRAME_ID_INICIO-FIN | JUGADORES_MEDIOS | MIN-MAX | PELOTA_%")

for start in range(0, len(frames), window):
    end = min(start + window, len(frames))

    avg_players = players_per_frame[start:end].mean()
    min_players = players_per_frame[start:end].min()
    max_players = players_per_frame[start:end].max()
    ball_pct = valid_ball[start:end].mean() * 100

    minute = start / fps / 60

    print(
        f"{minute:6.1f} | "
        f"{frames[start]}-{frames[end-1]} | "
        f"{avg_players:7.2f} | "
        f"{min_players:2d}-{max_players:2d} | "
        f"{ball_pct:6.2f}%"
    )
