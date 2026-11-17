
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TECH-CIA | GEOMETRIC COMPACTNESS | EXPERIMENT 08
# Exploración espacial sin ML, etiquetas ni vídeo.
# ============================================================

NPZ_PATH = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_Geometric_Compactness_Experiment08.csv"
)

INTERVALS = {
    "A_127501_129000": (127501, 129000),
    "B_132001_133500": (132001, 133500),
    "C_141001_142500": (141001, 142500),
}

PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

MIN_SHAPE_PLAYERS = 8
NEAREST_PLAYERS = 5

TEAMS = {
    "Barcelona": {
        "full": slice(0, 24),
        "outfield": slice(0, 23),
    },
    "Athletic": {
        "full": slice(24, 46),
        "outfield": slice(24, 45),
    },
}


def convex_hull_area(points):
    """Calcula el área del convex hull mediante el algoritmo monotónico."""

    points = np.unique(
        np.asarray(points, dtype=float),
        axis=0
    )

    if len(points) < 3:
        return np.nan

    points = sorted(map(tuple, points.tolist()))

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []

    for point in points:
        while (
            len(lower) >= 2
            and cross(lower[-2], lower[-1], point) <= 0
        ):
            lower.pop()

        lower.append(point)

    upper = []

    for point in reversed(points):
        while (
            len(upper) >= 2
            and cross(upper[-2], upper[-1], point) <= 0
        ):
            upper.pop()

        upper.append(point)

    hull = lower[:-1] + upper[:-1]

    if len(hull) < 3:
        return 0.0

    x = np.array([p[0] for p in hull])
    y = np.array([p[1] for p in hull])

    return float(0.5 * abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    ))


def to_metres(positions):
    """Convierte coordenadas normalizadas a metros."""

    result = np.asarray(positions, dtype=float).copy()

    result[:, 0] *= PITCH_LENGTH_M
    result[:, 1] *= PITCH_WIDTH_M

    return result


def get_positions(frame_players, team_slice):
    """Devuelve solamente las posiciones X/Y válidas."""

    positions = frame_players[team_slice, :2]

    valid = np.isfinite(positions).all(axis=1)

    return positions[valid]


def shape_metrics(points, min_players):
    """Mide expansión y compacidad de un conjunto de posiciones."""

    points = np.asarray(points, dtype=float)
    n = len(points)

    result = {
        "n": n,
        "shape_valid": n >= min_players,
        "hull_area_m2": np.nan,
        "depth_m": np.nan,
        "width_m": np.nan,
        "pairwise_mean_m": np.nan,
        "pairwise_median_m": np.nan,
        "stretch_index_m": np.nan,
        "centroid_rms_m": np.nan,
    }

    if n < min_players:
        return result

    # Extensión longitudinal y lateral.
    result["depth_m"] = np.ptp(points[:, 0])
    result["width_m"] = np.ptp(points[:, 1])

    # Área ocupada por el contorno exterior.
    result["hull_area_m2"] = convex_hull_area(points)

    # Centroide del equipo.
    centroid = np.mean(points, axis=0)

    radial_distances = np.linalg.norm(
        points - centroid,
        axis=1
    )

    # Distancia media al centroide.
    result["stretch_index_m"] = np.mean(radial_distances)

    # Dispersión RMS respecto al centroide.
    result["centroid_rms_m"] = np.sqrt(
        np.mean(radial_distances ** 2)
    )

    # Distancias entre todos los pares de jugadores.
    differences = (
        points[:, None, :]
        - points[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2
    )

    pairwise = distances[
        np.triu_indices(n, k=1)
    ]

    result["pairwise_mean_m"] = np.mean(pairwise)
    result["pairwise_median_m"] = np.median(pairwise)

    return result


def analyze_frame(frame_players, ball_xy):

    result = {}

    ball_valid = np.isfinite(ball_xy).all()

    if ball_valid:
        ball_m = np.array([
            ball_xy[0] * PITCH_LENGTH_M,
            ball_xy[1] * PITCH_WIDTH_M,
        ])
    else:
        ball_m = None

    for team, slices in TEAMS.items():

        # 1. Equipo completo, incluido el portero.
        full_positions = get_positions(
            frame_players,
            slices["full"]
        )

        # 2. Jugadores de campo, sin el portero.
        outfield_positions = get_positions(
            frame_players,
            slices["outfield"]
        )

        full_m = to_metres(full_positions)
        outfield_m = to_metres(outfield_positions)

        groups = {
            "full": full_m,
            "outfield": outfield_m,
        }

        for group_name, positions in groups.items():

            features = shape_metrics(
                positions,
                MIN_SHAPE_PLAYERS
            )

            for metric, value in features.items():
                result[f"{team}_{group_name}_{metric}"] = value

        # 3. Cinco jugadores de campo más próximos al balón.
        if (
            ball_valid
            and len(outfield_m) >= NEAREST_PLAYERS
        ):

            distances = np.linalg.norm(
                outfield_m - ball_m,
                axis=1
            )

            nearest_indices = np.argsort(
                distances
            )[:NEAREST_PLAYERS]

            nearest_five = outfield_m[nearest_indices]

            local_features = shape_metrics(
                nearest_five,
                NEAREST_PLAYERS
            )

        else:
            local_features = shape_metrics(
                np.empty((0, 2)),
                NEAREST_PLAYERS
            )

        for metric, value in local_features.items():
            result[f"{team}_nearest5_{metric}"] = value

    return result


def main():

    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)

    data = np.load(NPZ_PATH)

    frame_ids = data["frame_ids"]
    players = data["players"]
    ball = data["ball"]

    rows = []

    for segment, (first, last) in INTERVALS.items():

        start = np.searchsorted(frame_ids, first)
        end = np.searchsorted(
            frame_ids,
            last,
            side="right"
        )

        if (
            start >= len(frame_ids)
            or frame_ids[start] != first
        ):
            raise ValueError(
                f"No se encuentra el frame inicial {first}"
            )

        if (
            end <= start
            or frame_ids[end - 1] != last
        ):
            raise ValueError(
                f"No se encuentra el frame final {last}"
            )

        print(f"Procesando {segment}...")

        for idx in range(start, end):

            row = {
                "segment": segment,
                "frame_id": int(frame_ids[idx]),
                "ball_valid": bool(
                    np.isfinite(ball[idx, :2]).all()
                ),
            }

            row.update(
                analyze_frame(
                    players[idx],
                    ball[idx, :2]
                )
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT,
        index=False,
        float_format="%.6f"
    )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print("\n==========================================")
    print("TECH-CIA | COMPACTNESS EXPERIMENT 08")
    print("==========================================")

    for segment, part in df.groupby(
        "segment",
        sort=False
    ):

        print(f"\n========== {segment} ==========")

        print(
            "Disponibilidad del balón:",
            f"{100 * part['ball_valid'].mean():.2f}%"
        )

        for team in TEAMS:

            print(f"\n--- {team} ---")

            for group in ("full", "outfield", "nearest5"):

                prefix = f"{team}_{group}"

                print(f"\n  {group}:")

                n_mean = part[f"{prefix}_n"].mean()

                valid_pct = (
                    100 * part[f"{prefix}_shape_valid"].mean()
                )

                print(f"    Jugadores medios: {n_mean:.2f}")
                print(f"    Frames válidos: {valid_pct:.2f}%")

                for metric in (
                    "hull_area_m2",
                    "depth_m",
                    "width_m",
                    "pairwise_mean_m",
                    "stretch_index_m",
                    "centroid_rms_m",
                ):

                    column = f"{prefix}_{metric}"

                    value = part[column].median()

                    print(f"    {metric}: {value:.3f}")

        # Comparación de jugadores de campo.
        print("\n--- DIFERENCIAS: JUGADORES DE CAMPO ---")

        for metric in (
            "hull_area_m2",
            "depth_m",
            "width_m",
            "pairwise_mean_m",
            "stretch_index_m",
        ):

            b = part[f"Barcelona_outfield_{metric}"]
            a = part[f"Athletic_outfield_{metric}"]

            valid = b.notna() & a.notna()

            if not valid.any():
                continue

            difference = b[valid] - a[valid]

            positive = 100 * (
                difference > 1e-9
            ).mean()

            negative = 100 * (
                difference < -1e-9
            ).mean()

            ties = 100 * (
                difference.abs() <= 1e-9
            ).mean()

            print(
                f"{metric:22s} | "
                f"Mediana B-A={difference.median():8.3f} | "
                f"Barça mayor={positive:5.1f}% | "
                f"Athletic mayor={negative:5.1f}% | "
                f"Igual={ties:5.1f}%"
            )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()

