
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TECH-CIA | GEOMETRIC POSSESSION | EXPERIMENT 01
# Exploración espacial sin ML, etiquetas ni vídeo.
# ============================================================

NPZ_PATH = Path("artifacts/tracking_atd.npz")

OUTPUT_PATH = Path(
    "artifacts/TechCIA_GeometricPossession_Experiment01_minute94.csv"
)

# Intervalo de referencia: frame_id inclusivo.
FRAME_START = 141001
FRAME_END = 142500

# Dimensiones corregidas del campo, en metros.
PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

# Escalas descriptivas para explorar densidad.
DENSITY_RADII_M = (2.0, 5.0, 10.0, 15.0)

# Correspondencia confirmada mediante XML + Kloppy.
TEAMS = {
    "Barcelona": slice(0, 24),
    "Athletic": slice(24, 46),
}


def angular_features(dx, dy):
    """
    Distribución angular de los jugadores vistos desde el balón.

    X: horizontal.
    Y: vertical invertida para usar convención matemática.
    """

    angles = np.mod(np.arctan2(-dy, dx), 2 * np.pi)

    if len(angles) < 2:
        return np.nan, np.nan, np.nan

    angles = np.sort(angles)

    circular_gaps = np.diff(
        np.concatenate(
            [angles, [angles[0] + 2 * np.pi]]
        )
    )

    largest_gap = np.max(circular_gaps)

    coverage = 2 * np.pi - largest_gap

    resultant = np.hypot(
        np.mean(np.cos(angles)),
        np.mean(np.sin(angles))
    )

    return (
        np.degrees(coverage),
        np.degrees(largest_gap),
        resultant
    )


def extract_team_features(frame_players, ball_xy, team_slice, team_name):
    """
    Calcula las características geométricas de un equipo
    respecto al balón para un único frame.
    """

    positions = frame_players[team_slice, :2]

    valid = np.isfinite(positions).all(axis=1)
    positions = positions[valid]

    result = {
        f"{team_name}_n_players": len(positions)
    }

    # Si el balón no está disponible, no inventamos su posición.
    ball_valid = np.isfinite(ball_xy).all()

    if not ball_valid or len(positions) == 0:
        result.update({
            f"{team_name}_nearest_m": np.nan,
            f"{team_name}_nearest3_mean_m": np.nan,
            f"{team_name}_mean_distance_m": np.nan,
            f"{team_name}_median_distance_m": np.nan,
            f"{team_name}_angular_coverage_deg": np.nan,
            f"{team_name}_largest_angular_gap_deg": np.nan,
            f"{team_name}_angular_resultant_R": np.nan,
        })

        for radius in DENSITY_RADII_M:
            key = str(int(radius))
            result[f"{team_name}_within_{key}m"] = np.nan
            result[f"{team_name}_share_within_{key}m"] = np.nan

        return result

    # Diferencias espaciales en metros.
    dx = (positions[:, 0] - ball_xy[0]) * PITCH_LENGTH_M
    dy = (positions[:, 1] - ball_xy[1]) * PITCH_WIDTH_M

    distances = np.hypot(dx, dy)
    sorted_distances = np.sort(distances)

    # Distancias.
    result[f"{team_name}_nearest_m"] = sorted_distances[0]

    result[f"{team_name}_nearest3_mean_m"] = (
        np.mean(sorted_distances[:3])
        if len(sorted_distances) >= 3
        else np.nan
    )

    result[f"{team_name}_mean_distance_m"] = np.mean(distances)
    result[f"{team_name}_median_distance_m"] = np.median(distances)

    # Densidad a distintas escalas.
    for radius in DENSITY_RADII_M:
        count = int(np.sum(distances <= radius))
        key = str(int(radius))

        result[f"{team_name}_within_{key}m"] = count
        result[f"{team_name}_share_within_{key}m"] = (
            count / len(distances)
        )

    # Geometría angular.
    coverage, largest_gap, resultant = angular_features(dx, dy)

    result[f"{team_name}_angular_coverage_deg"] = coverage
    result[f"{team_name}_largest_angular_gap_deg"] = largest_gap
    result[f"{team_name}_angular_resultant_R"] = resultant

    return result


def main():

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el tracking: {NPZ_PATH}"
        )

    data = np.load(NPZ_PATH)

    frame_ids = data["frame_ids"]
    players = data["players"]
    ball = data["ball"]

    start_idx = np.searchsorted(frame_ids, FRAME_START)
    end_idx = np.searchsorted(
        frame_ids,
        FRAME_END,
        side="right"
    )

    if start_idx >= len(frame_ids):
        raise ValueError("Frame inicial fuera del dataset.")

    if frame_ids[start_idx] != FRAME_START:
        raise ValueError("No se encuentra el frame inicial exacto.")

    if end_idx == 0 or frame_ids[end_idx - 1] != FRAME_END:
        raise ValueError("No se encuentra el frame final exacto.")

    rows = []

    for idx in range(start_idx, end_idx):

        frame_id = int(frame_ids[idx])
        ball_xy = ball[idx, :2]

        ball_valid = bool(np.isfinite(ball_xy).all())

        row = {
            "frame_id": frame_id,
            "video_frame": frame_id - 1,
            "ball_valid": ball_valid,
            "ball_x": ball_xy[0],
            "ball_y": ball_xy[1],
        }

        for team_name, team_slice in TEAMS.items():

            features = extract_team_features(
                players[idx],
                ball_xy,
                team_slice,
                team_name
            )

            row.update(features)

        rows.append(row)

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.6f"
    )

    # ========================================================
    # RESUMEN EXPLORATORIO
    # ========================================================

    print("\n==========================================")
    print("TECH-CIA | EXPERIMENTO GEOMÉTRICO 01")
    print("==========================================")

    print("Frames analizados:", len(df))
    print("Primer frame:", int(df["frame_id"].iloc[0]))
    print("Último frame:", int(df["frame_id"].iloc[-1]))

    ball_percentage = 100 * df["ball_valid"].mean()

    print(f"\nDisponibilidad del balón: {ball_percentage:.2f}%")

    for team in TEAMS:

        print(f"\n========== {team.upper()} ==========")

        n = df[f"{team}_n_players"]

        print(
            "Jugadores válidos por frame:",
            f"media={n.mean():.2f}",
            f"mínimo={n.min()}",
            f"máximo={n.max()}"
        )

        for feature in (
            "nearest_m",
            "nearest3_mean_m",
            "mean_distance_m",
            "median_distance_m",
            "angular_coverage_deg",
            "largest_angular_gap_deg",
            "angular_resultant_R",
        ):

            column = f"{team}_{feature}"

            print(
                f"{feature}:",
                f"{df[column].mean():.4f}"
            )

        for radius in DENSITY_RADII_M:

            column = f"{team}_within_{int(radius)}m"

            print(
                f"Jugadores dentro de {radius:.0f} m:",
                f"{df[column].mean():.3f}"
            )

    print("\nCSV generado:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()

