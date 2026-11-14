
from pathlib import Path
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parent

TRACKING_PATH = ROOT / "artifacts" / "tracking_atd.npz"
VELOCITIES_PATH = ROOT / "artifacts" / "derived" / "velocities_raw.npz"

OUTPUT_PATH = ROOT / "artifacts" / "derived" / "pitch_control_features.npz"

FPS = 25
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


# ============================================================
# CARGA Y VALIDACIÓN
# ============================================================

def load_data():

    tracking = np.load(TRACKING_PATH, allow_pickle=False)
    velocities = np.load(VELOCITIES_PATH, allow_pickle=False)

    frame_ids = tracking["frame_ids"]

    if not np.array_equal(frame_ids, velocities["frame_ids"]):
        raise ValueError("Los frame_ids no coinciden.")

    players = tracking["players"]
    ball = tracking["ball"]

    vx = velocities["vx"]
    vy = velocities["vy"]

    position_valid = velocities["position_valid"]
    velocity_valid = velocities["velocity_valid"]

    n_frames, n_players, _ = players.shape

    if velocity_valid.shape != (n_frames - 1, n_players):
        raise ValueError("Dimensiones inesperadas en velocity_valid.")

    # velocity_valid[i] representa la transición
    # entre los frames i e i+1.
    # Las velocidades se almacenaron en la fila i+1.

    velocity_valid_full = np.zeros(
        (n_frames, n_players),
        dtype=bool
    )

    velocity_valid_full[1:] = velocity_valid

    return (
        frame_ids,
        players,
        ball,
        vx,
        vy,
        position_valid,
        velocity_valid_full
    )


# ============================================================
# DISTANCIA Y COMPONENTES DE VELOCIDAD HACIA EL BALÓN
# ============================================================

def calculate_geometric_features(
    players,
    ball,
    vx,
    vy,
    position_valid,
    velocity_valid
):

    # Coordenadas normalizadas -> metros
    player_x = players[:, :, 0] * FIELD_LENGTH
    player_y = players[:, :, 1] * FIELD_WIDTH

    ball_x = ball[:, 0] * FIELD_LENGTH
    ball_y = ball[:, 1] * FIELD_WIDTH

    # Vector desde cada jugador hacia el balón
    dx = ball_x[:, None] - player_x
    dy = ball_y[:, None] - player_y

    # Distancia euclídea
    distance_to_ball = np.hypot(dx, dy)

    player_position_ok = (
        position_valid
        & np.isfinite(player_x)
        & np.isfinite(player_y)
    )

    ball_valid = (
        np.isfinite(ball_x)
        & np.isfinite(ball_y)
    )

    distance_valid = (
        player_position_ok
        & ball_valid[:, None]
    )

    distance_to_ball[~distance_valid] = np.nan

    # Dirección unitaria hacia el balón
    direction_valid = (
        distance_valid
        & (distance_to_ball > 1e-6)
    )

    ux = np.full(distance_to_ball.shape, np.nan)
    uy = np.full(distance_to_ball.shape, np.nan)

    ux[direction_valid] = (
        dx[direction_valid] / distance_to_ball[direction_valid]
    )

    uy[direction_valid] = (
        dy[direction_valid] / distance_to_ball[direction_valid]
    )

    # Validación de velocidad
    speed_valid = (
        velocity_valid
        & np.isfinite(vx)
        & np.isfinite(vy)
    )

    kinematic_valid = (
        direction_valid
        & speed_valid
    )

    # Proyección de velocidad hacia el balón
    velocity_toward_ball = np.full(
        distance_to_ball.shape,
        np.nan
    )

    velocity_toward_ball[kinematic_valid] = (
        vx[kinematic_valid] * ux[kinematic_valid]
        + vy[kinematic_valid] * uy[kinematic_valid]
    )

    # Componente perpendicular firmada
    velocity_lateral = np.full(
        distance_to_ball.shape,
        np.nan
    )

    velocity_lateral[kinematic_valid] = (
        -vx[kinematic_valid] * uy[kinematic_valid]
        + vy[kinematic_valid] * ux[kinematic_valid]
    )

    return {
        "distance_to_ball": distance_to_ball,
        "distance_valid": distance_valid,
        "velocity_toward_ball": velocity_toward_ball,
        "velocity_lateral": velocity_lateral,
        "kinematic_valid": kinematic_valid,
        "ball_valid": ball_valid
    }


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("Cargando tracking y velocidades...")

    (
        frame_ids,
        players,
        ball,
        vx,
        vy,
        position_valid,
        velocity_valid
    ) = load_data()

    print("Calculando variables geométricas...")

    features = calculate_geometric_features(
        players,
        ball,
        vx,
        vy,
        position_valid,
        velocity_valid
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        frame_ids=frame_ids,
        distance_to_ball=features["distance_to_ball"],
        distance_valid=features["distance_valid"],
        velocity_toward_ball=features["velocity_toward_ball"],
        velocity_lateral=features["velocity_lateral"],
        kinematic_valid=features["kinematic_valid"],
        ball_valid=features["ball_valid"]
    )

    print("\n--- RESULTADOS ---")

    print("Frames:", len(frame_ids))
    print("Jugadores:", players.shape[1])

    print(
        "Distancias válidas:",
        np.sum(features["distance_valid"])
    )

    print(
        "Velocidades hacia el balón válidas:",
        np.sum(features["kinematic_valid"])
    )

    print(
        "Frames con balón disponible:",
        np.sum(features["ball_valid"])
    )

    print(
        "Mediana distancia (m):",
        round(
            float(np.nanmedian(features["distance_to_ball"])),
            3
        )
    )

    print(
        "Mediana velocidad hacia balón (m/s):",
        round(
            float(np.nanmedian(features["velocity_toward_ball"])),
            3
        )
    )

    print("\nArchivo guardado en:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

