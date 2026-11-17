
from pathlib import Path
import numpy as np
import pandas as pd

NPZ = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_Compactness_FixedK_Experiment11.csv"
)

INTERVALS = {
    "A_127501_129000": (127501, 129000),
    "B_132001_133500": (132001, 133500),
    "C_141001_142500": (141001, 142500),
}

TEAMS = {
    "Barcelona": slice(0, 23),
    "Athletic": slice(24, 45),
}

LENGTH_M = 105.3
WIDTH_M = 68.0
K_VALUES = (3, 5)


def hull_area(points):

    points = sorted(
        set(map(tuple, np.asarray(points).tolist()))
    )

    if len(points) < 3:
        return np.nan

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []

    for p in points:
        while (
            len(lower) >= 2
            and cross(lower[-2], lower[-1], p) <= 0
        ):
            lower.pop()
        lower.append(p)

    upper = []

    for p in reversed(points):
        while (
            len(upper) >= 2
            and cross(upper[-2], upper[-1], p) <= 0
        ):
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]

    x = np.array([p[0] for p in hull])
    y = np.array([p[1] for p in hull])

    return float(0.5 * abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    ))


def shape(points):

    n = len(points)

    centroid = points.mean(axis=0)

    radial = np.linalg.norm(
        points - centroid,
        axis=1
    )

    delta = (
        points[:, None, :]
        - points[None, :, :]
    )

    distances = np.linalg.norm(delta, axis=2)

    pairwise = distances[np.triu_indices(n, k=1)]

    return {
        "hull_area_m2": hull_area(points),
        "width_m": np.ptp(points[:, 1]),
        "depth_m": np.ptp(points[:, 0]),
        "pairwise_mean_m": pairwise.mean(),
        "stretch_index_m": radial.mean(),
    }


def main():

    data = np.load(NPZ)

    frame_ids = data["frame_ids"]
    players = data["players"]
    ball = data["ball"]

    rows = []

    for segment, (first, last) in INTERVALS.items():

        start = np.searchsorted(frame_ids, first)
        end = np.searchsorted(frame_ids, last, side="right")

        if (
            start >= len(frame_ids)
            or end <= start
            or frame_ids[start] != first
            or frame_ids[end - 1] != last
        ):
            raise ValueError(f"Intervalo incorrecto: {segment}")

        print(f"Procesando {segment}...")

        for idx in range(start, end):

            row = {
                "segment": segment,
                "frame_id": int(frame_ids[idx]),
                "ball_valid": bool(
                    np.isfinite(ball[idx, :2]).all()
                ),
            }

            if not row["ball_valid"]:
                continue

            ball_m = np.array([
                ball[idx, 0] * LENGTH_M,
                ball[idx, 1] * WIDTH_M,
            ])

            for team, team_slice in TEAMS.items():

                positions = players[idx, team_slice, :2]

                valid = np.isfinite(positions).all(axis=1)
                positions = positions[valid].copy()

                positions[:, 0] *= LENGTH_M
                positions[:, 1] *= WIDTH_M

                distances = np.linalg.norm(
                    positions - ball_m,
                    axis=1
                )

                row[f"{team}_n_15m"] = int(
                    np.sum(distances <= 15)
                )

                order = np.argsort(distances)

                for k in K_VALUES:

                    if len(order) < k:
                        raise ValueError(
                            f"Menos de {k} posiciones válidas "
                            f"en {team}, frame {frame_ids[idx]}"
                        )

                    selected = positions[order[:k]]

                    prefix = f"{team}_nearest{k}"

                    row[f"{prefix}_kth_distance_m"] = (
                        distances[order[k - 1]]
                    )

                    for metric, value in shape(selected).items():
                        row[f"{prefix}_{metric}"] = value

            rows.append(row)

    df = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, float_format="%.6f")

    print("\n==========================================")
    print("TECH-CIA | FIXED-K COMPACTNESS 11")
    print("==========================================")

    metrics = (
        "hull_area_m2",
        "width_m",
        "depth_m",
        "pairwise_mean_m",
        "stretch_index_m",
    )

    for segment, part in df.groupby("segment", sort=False):

        print(f"\n========== {segment} ==========")

        print(
            "Media jugadores dentro de 15 m | "
            f"Barça={part['Barcelona_n_15m'].mean():.2f} | "
            f"Athletic={part['Athletic_n_15m'].mean():.2f}"
        )

        for k in K_VALUES:

            print(f"\n--- Comparación de {k} jugadores ---")

            b_distance = part[
                f"Barcelona_nearest{k}_kth_distance_m"
            ].median()

            a_distance = part[
                f"Athletic_nearest{k}_kth_distance_m"
            ].median()

            print(
                f"Distancia del jugador {k}: "
                f"Barça={b_distance:.2f} m | "
                f"Athletic={a_distance:.2f} m"
            )

            for metric in metrics:

                b = part[f"Barcelona_nearest{k}_{metric}"]
                a = part[f"Athletic_nearest{k}_{metric}"]

                difference = (b - a).dropna()

                print(
                    f"{metric:22s} | "
                    f"Mediana B-A={difference.median():8.3f} | "
                    f"Barça mayor={100*(difference > 1e-9).mean():5.1f}% | "
                    f"Athletic mayor={100*(difference < -1e-9).mean():5.1f}%"
                )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
