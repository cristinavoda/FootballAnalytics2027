
import numpy as np

# =====================================================
# TECH-CIA - AUDITORIA DEL INTERVALO DE REFERENCIA 94
# =====================================================

PATH = r"artifacts\tracking_atd.npz"

START_FRAME = 141001
END_FRAME = 142500
FPS = 25
WINDOW_SECONDS = 10

data = np.load(PATH)

frame_ids = data["frame_ids"]
players = data["players"]
ball = data["ball"]

# Localizar los frames por su identificador real
start_idx = np.searchsorted(frame_ids, START_FRAME)
end_idx = np.searchsorted(frame_ids, END_FRAME, side="right")

ids = frame_ids[start_idx:end_idx]
p = players[start_idx:end_idx]
b = ball[start_idx:end_idx]

if len(ids) != END_FRAME - START_FRAME + 1:
    raise ValueError("El intervalo no contiene los 1500 frames esperados.")

# Posición válida = X e Y disponibles
valid_players = np.isfinite(p[:, :, :2]).all(axis=2)
valid_ball = np.isfinite(b[:, :2]).all(axis=1)

players_per_frame = valid_players.sum(axis=1)

def longest_false_run(values):
    """Mayor secuencia consecutiva de valores False."""
    missing = ~values
    padded = np.concatenate(([False], missing, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) == 0:
        return 0

    return int(np.max(ends - starts))


print("\n========== IDENTIFICACIÓN DEL INTERVALO ==========")
print("Frame inicial:", ids[0])
print("Frame final:", ids[-1])
print("Frames analizados:", len(ids))
print("Duración (segundos):", len(ids) / FPS)

print("\n========== ESTABILIDAD POR VENTANAS DE 10 SEGUNDOS ==========")
print("VENTANA | JUGADORES MEDIA | MIN-MAX | PELOTA % | MAYOR HUECO PELOTA")

window_frames = FPS * WINDOW_SECONDS

for start in range(0, len(ids), window_frames):
    end = min(start + window_frames, len(ids))

    count = players_per_frame[start:end]
    ball_part = valid_ball[start:end]

    print(
        f"{ids[start]}-{ids[end-1]} | "
        f"{count.mean():6.2f} | "
        f"{count.min():2d}-{count.max():2d} | "
        f"{ball_part.mean()*100:6.2f}% | "
        f"{longest_false_run(ball_part)} frames"
    )

print("\n========== DISPONIBILIDAD DE CADA SLOT ==========")
print("SLOT | FRAMES VALIDOS | DISPONIBILIDAD % | HUECO MAXIMO")

for slot in range(p.shape[1]):
    presence = valid_players[:, slot]
    valid_count = int(presence.sum())
    percentage = presence.mean() * 100
    max_gap = longest_false_run(presence)

    print(
        f"{slot:02d} | "
        f"{valid_count:4d}/{len(ids)} | "
        f"{percentage:7.2f}% | "
        f"{max_gap:4d}"
    )

print("\n========== RESUMEN GLOBAL ==========")
print("Jugadores válidos medios:", round(players_per_frame.mean(), 2))
print("Mínimo de jugadores:", int(players_per_frame.min()))
print("Máximo de jugadores:", int(players_per_frame.max()))
print("Pelota disponible:", round(valid_ball.mean() * 100, 2), "%")
print("Slots con alguna posición:", int(valid_players.any(axis=0).sum()))
print("Slots presentes en TODOS los frames:",
      int(valid_players.all(axis=0).sum()))
print("Mayor hueco consecutivo de pelota:",
      longest_false_run(valid_ball), "frames")
