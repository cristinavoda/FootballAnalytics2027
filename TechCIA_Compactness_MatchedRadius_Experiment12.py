
from pathlib import Path
import numpy as np
import pandas as pd

NPZ = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_Compactness_MatchedRadius_Experiment12.csv"
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
RADIUS_M = 15.0
K_VALUES = (3, 5)


def hull_area(points):

    points = sorted(
        set(map(tuple, np.asarray(points).tolist()))
    )

    if len(points) < 3:
        return 0.0

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

    if len(hull) < 3:
        return 0.0

    x = np.array([p[0] for p in hull])
    y = np.array([p[1] for p in hull])

    return float(0.5 * abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    ))


def calculate_shape(points):

    n = len(points)

    centroid = points.mean(axis=0)

    radial = np.linalg.norm(
        points - centroid,
        axis=1
    )

    differences = (
        points[:, None, :] - points[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2
    )

    pairs = distances[np.triu_indices(n, k=1)]

    return {
        "hull_area_m2": hull_area(points),
        "width_m": np.ptp(points[:, 1]),
        "depth_m": np.ptp(points[:, 0]),
        "pairwise_mean_m": pairs.mean(),
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
        end = np.searchsorted(
            frame_ids, last, side="right"
        )

        if (
            start >= len(frame_ids)
            or end <= start
            or frame_ids[start] != first
            or frame_ids[end - 1] != last
        ):
            raise ValueError(f"Intervalo incorrecto: {segment}")

        print(f"Procesando {segment}...")

        for idx in range(start, end):

            ball_xy = ball[idx, :2]

            if not np.isfinite(ball_xy).all():
                continue

            ball_m = np.array([
                ball_xy[0] * LENGTH_M,
                ball_xy[1] * WIDTH_M,
            ])

            candidates = {}

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

                local_mask = distances <= RADIUS_M

                local_positions = positions[local_mask]
                local_distances = distances[local_mask]

                candidates[team] = {
                    "positions": local_positions,
                    "distances": local_distances,
                }

            row = {
                "segment": segment,
                "frame_id": int(frame_ids[idx]),
                "radius_m": RADIUS_M,
                "Barcelona_n_15m": len(
                    candidates["Barcelona"]["positions"]
                ),
                "Athletic_n_15m": len(
                    candidates["Athletic"]["positions"]
                ),
            }

            for k in K_VALUES:

                for team in TEAMS:

                    group = candidates[team]
                    order = np.argsort(group["distances"])

                    prefix = f"{team}_k{k}"

                    if len(order) >= k:

                        selected = order[:k]

                        points = group["positions"][selected]

                        row[f"{prefix}_kth_distance_m"] = (
                            group["distances"][selected[-1]]
                        )

                        for metric, value in calculate_shape(points).items():
                            row[f"{prefix}_{metric}"] = value

                    else:

                        row[f"{prefix}_kth_distance_m"] = np.nan

                        for metric in (
                            "hull_area_m2",
                            "width_m",
                            "depth_m",
                            "pairwise_mean_m",
                            "stretch_index_m",
                        ):
                            row[f"{prefix}_{metric}"] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT,
        index=False,
        float_format="%.6f"
    )

    print("\n==========================================")
    print("TECH-CIA | MATCHED RADIUS COMPACTNESS 12")
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

        for k in K_VALUES:

            common = (
                (part["Barcelona_n_15m"] >= k)
                & (part["Athletic_n_15m"] >= k)
            )

            paired = part.loc[common]

            print(f"\n--- K={k}, radio máximo 15 m ---")

            print(
                f"Frames comparables: {len(paired)}/{len(part)} "
                f"({100*len(paired)/len(part):.1f}%)"
            )

            for metric in metrics:

                b = paired[f"Barcelona_k{k}_{metric}"]
                a = paired[f"Athletic_k{k}_{metric}"]

                difference = (b - a).dropna()

                if len(difference) == 0:
                    continue

                print(
                    f"{metric:22s} | "
                    f"Mediana B-A={difference.median():8.3f} | "
                    f"Barça mayor={100*(difference > 1e-9).mean():5.1f}% | "
                    f"Athletic mayor={100*(difference < -1e-9).mean():5.1f}%"
                )

            for team in TEAMS:

                column = f"{team}_k{k}_kth_distance_m"

                print(
                    f"Distancia del jugador {k}, {team}:",
                    f"{paired[column].median():.2f} m"
                )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
