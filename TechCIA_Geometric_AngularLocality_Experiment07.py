
from pathlib import Path

import numpy as np
import pandas as pd


NPZ_PATH = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_Geometric_AngularLocality_Experiment07.csv"
)

INTERVALS = {
    "A_127501_129000": (127501, 129000),
    "B_132001_133500": (132001, 133500),
    "C_141001_142500": (141001, 142500),
}

TEAMS = {
    "Barcelona": slice(0, 24),
    "Athletic": slice(24, 46),
}

PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

RADII = {
    "all": None,
    "5m": 5.0,
    "10m": 10.0,
    "15m": 15.0,
}


def calculate_angles(dx, dy):

    if len(dx) < 2:
        return np.nan, np.nan

    # Convención matemática: Y positivo hacia arriba.
    angles = np.mod(
        np.arctan2(-dy, dx),
        2 * np.pi
    )

    angles = np.sort(angles)

    gaps = np.diff(
        np.concatenate([
            angles,
            [angles[0] + 2 * np.pi]
        ])
    )

    coverage = 2 * np.pi - np.max(gaps)

    resultant = np.hypot(
        np.mean(np.cos(angles)),
        np.mean(np.sin(angles))
    )

    return np.degrees(coverage), resultant


def calculate_team(frame_players, ball_xy, team_slice):

    positions = frame_players[team_slice, :2]

    valid = np.isfinite(positions).all(axis=1)
    positions = positions[valid]

    if not np.isfinite(ball_xy).all():
        return {
            radius: (0, np.nan, np.nan)
            for radius in RADII
        }

    if len(positions) == 0:
        return {
            radius: (0, np.nan, np.nan)
            for radius in RADII
        }

    dx = (
        positions[:, 0] - ball_xy[0]
    ) * PITCH_LENGTH_M

    dy = (
        positions[:, 1] - ball_xy[1]
    ) * PITCH_WIDTH_M

    distances = np.hypot(dx, dy)

    result = {}

    for label, radius in RADII.items():

        if radius is None:
            selected = np.ones(
                len(positions),
                dtype=bool
            )
        else:
            selected = distances <= radius

        n = int(selected.sum())

        coverage, resultant = calculate_angles(
            dx[selected],
            dy[selected]
        )

        result[label] = (
            n,
            coverage,
            resultant
        )

    return result


def main():

    data = np.load(NPZ_PATH)

    frame_ids = data["frame_ids"]
    players = data["players"]
    ball = data["ball"]

    rows = []

    for segment, (first, last) in INTERVALS.items():

        start = np.searchsorted(
            frame_ids, first
        )

        end = np.searchsorted(
            frame_ids, last, side="right"
        )

        if start >= len(frame_ids) or frame_ids[start] != first:
            raise ValueError(
                f"Frame inicial no encontrado: {first}"
            )

        if end <= start or frame_ids[end - 1] != last:
            raise ValueError(
                f"Frame final no encontrado: {last}"
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

            for team, team_slice in TEAMS.items():

                results = calculate_team(
                    players[idx],
                    ball[idx, :2],
                    team_slice
                )

                for radius, (n, coverage, resultant) in results.items():

                    prefix = f"{team}_{radius}"

                    row[f"{prefix}_n"] = n
                    row[f"{prefix}_coverage"] = coverage
                    row[f"{prefix}_resultant"] = resultant

            rows.append(row)

    result = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT,
        index=False,
        float_format="%.6f"
    )

    print("\n==========================================")
    print("TECH-CIA | GEOMETRIA ANGULAR LOCAL 07")
    print("==========================================")

    for segment, part in result.groupby(
        "segment",
        sort=False
    ):

        print(f"\n========== {segment} ==========")

        for radius in RADII:

            print(f"\n--- Radio: {radius} ---")

            for team in TEAMS:

                prefix = f"{team}_{radius}"

                n_mean = part[f"{prefix}_n"].mean()

                coverage = part[
                    f"{prefix}_coverage"
                ].median()

                resultant = part[
                    f"{prefix}_resultant"
                ].median()

                print(
                    f"{team:12s} | "
                    f"N medio={n_mean:5.2f} | "
                    f"Cobertura={coverage:8.2f}° | "
                    f"R={resultant:7.4f}"
                )

            b = part[
                f"Barcelona_{radius}_coverage"
            ]

            a = part[
                f"Athletic_{radius}_coverage"
            ]

            valid = b.notna() & a.notna()

            if valid.any():

                delta = b[valid] - a[valid]

                print(
                    "Barcelona mayor cobertura:",
                    f"{100 * (delta > 1e-9).mean():.1f}%"
                )

                print(
                    "Athletic mayor cobertura:",
                    f"{100 * (delta < -1e-9).mean():.1f}%"
                )

                print(
                    "Empate:",
                    f"{100 * (delta.abs() <= 1e-9).mean():.1f}%"
                )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()

