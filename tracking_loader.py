from pathlib import Path
import numpy as np

# Archivos originales: solo lectura
SOURCE = Path(
    r"E:\Hackaton 2027\data\tactical\BAR-ATH_tactical_atd.txt"
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "tracking_atd.npz"

EXPECTED_FRAMES = 199319
N_PLAYERS = 46
N_CHANNELS = 3

# Estructuras numéricas
frame_ids = np.zeros(EXPECTED_FRAMES, dtype=np.int32)

players = np.full(
    (EXPECTED_FRAMES, N_PLAYERS, N_CHANNELS),
    np.nan,
    dtype=np.float32
)

ball = np.full(
    (EXPECTED_FRAMES, N_CHANNELS),
    np.nan,
    dtype=np.float32
)

print("Cargando tracking original...")
print("Origen:", SOURCE)

total = 0
errors = []

with SOURCE.open("r", encoding="utf-8-sig") as file:

    for line_number, line in enumerate(file, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split(":", 2)

        if len(parts) != 3:
            errors.append((line_number, "Separación de campos"))
            continue

        frame_text, players_text, ball_text = parts

        player_groups = players_text.split(";")
        ball_values = ball_text.split(",")

        if len(player_groups) != N_PLAYERS:
            errors.append((line_number, "Número de jugadores"))
            continue

        if len(ball_values) != N_CHANNELS:
            errors.append((line_number, "Canales del balón"))
            continue

        if total >= EXPECTED_FRAMES:
            errors.append((line_number, "Frames adicionales"))
            break

        try:
            frame_id = int(frame_text)

            if frame_id != total + 1:
                errors.append((line_number, "Frame fuera de secuencia"))
                continue

            frame_ids[total] = frame_id

            for player_index, group in enumerate(player_groups):

                values = np.fromstring(group, sep=",", dtype=np.float32)

                if values.size != N_CHANNELS:
                    raise ValueError(
                        f"Track {player_index + 1}: "
                        f"{values.size} canales"
                    )

                players[total, player_index, :] = values

            ball_values_array = np.array(
                [float(value) for value in ball_values],
                dtype=np.float32
            )

            ball[total, :] = ball_values_array

            total += 1

        except Exception as exc:
            errors.append((line_number, str(exc)))

if errors:
    print("\nERRORES ENCONTRADOS:")
    for error in errors[:10]:
        print(error)

    raise RuntimeError(
        f"Conversión detenida: {len(errors)} errores."
    )

if total != EXPECTED_FRAMES:
    raise RuntimeError(
        f"Se esperaban {EXPECTED_FRAMES} frames y se cargaron {total}."
    )

# Guardamos una representación derivada, sin modificar el original
np.savez_compressed(
    OUTPUT,
    frame_ids=frame_ids,
    players=players,
    ball=ball
)

# Diagnóstico automático
missing_player_positions = np.isnan(players[:, :, :2]).any(axis=2)
missing_ball_positions = np.isnan(ball[:, :2]).any(axis=1)

print("\n========== RESULTADO ==========")
print("Frames cargados:", total)
print("Jugadores por frame:", N_PLAYERS)
print("Canales por jugador:", N_CHANNELS)
print("Dimensiones players:", players.shape)
print("Dimensiones ball:", ball.shape)

print(
    "Posiciones de jugador ausentes:",
    f"{missing_player_positions.sum():,}",
    "/",
    f"{missing_player_positions.size:,}"
)

print(
    "Posiciones del balón ausentes:",
    f"{missing_ball_positions.sum():,}",
    "/",
    f"{missing_ball_positions.size:,}"
)

print("Primer frame:", frame_ids[0])
print("Último frame:", frame_ids[-1])
print("Archivo generado:", OUTPUT.resolve())
print("Tamaño generado:", round(OUTPUT.stat().st_size / 1024 / 1024, 2), "MB")

print("\nConversión completada correctamente.")
