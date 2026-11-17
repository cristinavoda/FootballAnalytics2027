
from pathlib import Path
import numpy as np
import pandas as pd

NPZ = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_RadialAngular_Decomposition_Experiment13.csv"
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

    x = np.array([p[0] for p in hull])
    y = np.array([p[1] for p in hull])

    return float(0.5 * abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    ))


def analyze_group(points, ball_m):

    relative = points - ball_m

    radii = np.linalg.norm(relative, axis=1)

    # Y se invierte para utilizar la convención angular matemática.
    angles = np.mod(
        np.arctan2(-relative[:, 1], relative[:, 0]),
        2 * np.pi
    )

    angle_differences = []

    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):

            delta = abs(angles[i] - angles[j])

            delta = min(
                delta,
                2 * np.pi - delta
            )

            angle_differences.append(delta)

    angle_differences = np.asarray(angle_differences)

    pairwise_distances = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):

            pairwise_distances.append(
                np.linalg.norm(points[i] - points[j])
            )

    pairwise_distances = np.asarray(pairwise_distances)

    return {
        "mean_radius_m": radii.mean(),
        "median_radius_m": np.median(radii),
        "radial_std_m": radii.std(),
        "radial_span_m": np.ptp(radii),
        "kth_radius_m": radii.max(),

        "mean_angular_separation_deg": np.degrees(
            angle_differences.mean()
        ),

        # Distancia media entre direcciones unitarias.
        # No depende de la distancia radial al balón.
        "angular_chord_mean": np.mean(
            2 * np.sin(angle_differences / 2)
        ),

        "pairwise_mean_m": pairwise_distances.mean(),
        "hull_area_m2": hull_area(points),
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
            raise ValueError(
                f"Intervalo incorrecto: {segment}"
            )

        print(f"Procesando {segment}...")

        for idx in range(start, end):

            ball_xy = ball[idx, :2]

            if not np.isfinite(ball_xy).all():
                continue

            ball_m = np.array([
                ball_xy[0] * LENGTH_M,
                ball_xy[1] * WIDTH_M,
            ])

            row = {
                "segment": segment,
                "frame_id": int(frame_ids[idx]),
            }

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

                row[f"{team}_n_15m"] = len(local_positions)

                for k in K_VALUES:

                    prefix = f"{team}_k{k}"

                    if len(local_positions) < k:
                        continue

                    order = np.argsort(local_distances)[:k]

                    selected = local_positions[order]

                    metrics = analyze_group(
                        selected,
                        ball_m
                    )

                    for metric, value in metrics.items():
                        row[f"{prefix}_{metric}"] = value

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

    print("\n==========================================")
    print("TECH-CIA | RADIAL / ANGULAR DECOMPOSITION 13")
    print("==========================================")

    metrics = (
        "mean_radius_m",
        "radial_std_m",
        "mean_angular_separation_deg",
        "angular_chord_mean",
        "pairwise_mean_m",
        "hull_area_m2",
    )

    for segment, part in df.groupby("segment", sort=False):

        print(f"\n========== {segment} ==========")

        for k in K_VALUES:

            paired = part.dropna(
                subset=[
                    f"Barcelona_k{k}_{metrics[0]}",
                    f"Athletic_k{k}_{metrics[0]}",
                ]
            )

            print(f"\n--- K={k}, radio máximo 15 m ---")
            print(
                f"Frames comparables: {len(paired)}/{len(part)}"
            )

            for metric in metrics:

                b = paired[f"Barcelona_k{k}_{metric}"]
                a = paired[f"Athletic_k{k}_{metric}"]

                difference = (b - a).dropna()

                if len(difference) == 0:
                    continue

                print(
                    f"{metric:32s} | "
                    f"Mediana B-A={difference.median():9.4f} | "
                    f"Barça mayor={100*(difference > 1e-9).mean():5.1f}% | "
                    f"Athletic mayor={100*(difference < -1e-9).mean():5.1f}%"
                )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()

