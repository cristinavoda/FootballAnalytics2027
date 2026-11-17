
from pathlib import Path
import numpy as np
import pandas as pd

NPZ = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_Compactness_15m_Experiment10.csv"
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
MIN_PLAYERS = 3


def hull_area(points):

    points = np.unique(points, axis=0)

    if len(points) < 3:
        return np.nan

    points = sorted(map(tuple, points.tolist()))

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


def local_shape(points):

    n = len(points)

    result = {
        "n": n,
        "hull_area_m2": np.nan,
        "width_m": np.nan,
        "depth_m": np.nan,
        "pairwise_mean_m": np.nan,
        "stretch_index_m": np.nan,
    }

    if n < MIN_PLAYERS:
        return result

    result["hull_area_m2"] = hull_area(points)
    result["width_m"] = np.ptp(points[:, 1])
    result["depth_m"] = np.ptp(points[:, 0])

    centroid = points.mean(axis=0)

    radial = np.linalg.norm(
        points - centroid,
        axis=1
    )

    result["stretch_index_m"] = radial.mean()

    differences = (
        points[:, None, :] - points[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2
    )

    pairs = distances[np.triu_indices(n, k=1)]

    result["pairwise_mean_m"] = pairs.mean()

    return result


def main():

    data = np.load(NPZ)

    frame_ids = data["frame_ids"]
    players = data["players"]
    ball = data["ball"]

    rows = []

    for segment, (first, last) in INTERVALS.items():

        start = np.searchsorted(frame_ids, first)
        end = np.searchsorted(frame_ids, last, side="right")

        if frame_ids[start] != first or frame_ids[end - 1] != last:
            raise ValueError(f"Intervalo incompleto: {segment}")

        print(f"Procesando {segment}...")

        for idx in range(start, end):

            ball_xy = ball[idx, :2]
            ball_valid = np.isfinite(ball_xy).all()

            row = {
                "segment": segment,
                "frame_id": int(frame_ids[idx]),
                "ball_valid": bool(ball_valid),
            }

            for team, team_slice in TEAMS.items():

                positions = players[idx, team_slice, :2]

                valid = np.isfinite(positions).all(axis=1)
                positions = positions[valid]

                positions_m = positions.copy()
                positions_m[:, 0] *= LENGTH_M
                positions_m[:, 1] *= WIDTH_M

                if ball_valid and len(positions_m):

                    ball_m = np.array([
                        ball_xy[0] * LENGTH_M,
                        ball_xy[1] * WIDTH_M,
                    ])

                    distances = np.linalg.norm(
                        positions_m - ball_m,
                        axis=1
                    )

                    local = positions_m[
                        distances <= RADIUS_M
                    ]

                    fifth_distance = (
                        np.sort(distances)[4]
                        if len(distances) >= 5
                        else np.nan
                    )

                else:
                    local = np.empty((0, 2))
                    fifth_distance = np.nan

                row[f"{team}_n_outfield"] = len(positions_m)
                row[f"{team}_n_15m"] = len(local)
                row[f"{team}_fifth_nearest_m"] = fifth_distance

                features = local_shape(local)

                for metric, value in features.items():
                    row[f"{team}_local_{metric}"] = value

            rows.append(row)

    df = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, float_format="%.6f")

    print("\n==========================================")
    print("TECH-CIA | TRUE LOCAL COMPACTNESS 10")
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

        for team in TEAMS:

            n = part[f"{team}_n_15m"]

            print(f"\n{team}")
            print("Media de jugadores dentro de 15 m:", f"{n.mean():.2f}")
            print("Frames con al menos 3 jugadores:", f"{100*(n >= 3).mean():.1f}%")
            print("Distancia del quinto más próximo, mediana:",
                  f"{part[f'{team}_fifth_nearest_m'].median():.2f} m")

        common = (
            (part["Barcelona_n_15m"] >= MIN_PLAYERS)
            & (part["Athletic_n_15m"] >= MIN_PLAYERS)
            & part["ball_valid"]
        )

        paired = part.loc[common]

        print(
            "\nFrames comparables (ambos equipos >= 3):",
            f"{len(paired)}/{len(part)}",
            f"({100*len(paired)/len(part):.1f}%)"
        )

        for metric in metrics:

            b = paired[f"Barcelona_local_{metric}"]
            a = paired[f"Athletic_local_{metric}"]

            difference = b - a

            valid = difference.dropna()

            if len(valid) == 0:
                continue

            print(
                f"{metric:22s} | "
                f"mediana B-A={valid.median():8.3f} | "
                f"Barça mayor={100*(valid > 1e-9).mean():5.1f}% | "
                f"Athletic mayor={100*(valid < -1e-9).mean():5.1f}%"
            )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
