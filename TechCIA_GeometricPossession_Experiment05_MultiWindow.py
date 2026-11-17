
from pathlib import Path

import numpy as np
import pandas as pd

from TechCIA_GeometricPossession_Experiment01 import (
    extract_team_features,
    TEAMS,
)

NPZ_PATH = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_GeometricPossession_Experiment05_MultiWindow.csv"
)

INTERVALS = {
    "A_127501_129000": (127501, 129000),
    "B_132001_133500": (132001, 133500),
    "C_141001_142500": (141001, 142500),
}


def main():

    data = np.load(NPZ_PATH)

    frame_ids = data["frame_ids"]
    players = data["players"]
    ball = data["ball"]

    rows = []

    for segment, (first, last) in INTERVALS.items():

        start = np.searchsorted(frame_ids, first)
        end = np.searchsorted(frame_ids, last, side="right")

        if start >= len(frame_ids) or frame_ids[start] != first:
            raise ValueError(f"No se encuentra frame inicial: {first}")

        if end <= start or frame_ids[end - 1] != last:
            raise ValueError(f"No se encuentra frame final: {last}")

        print(f"\nProcesando {segment}: {first}-{last}")

        for idx in range(start, end):

            frame_id = int(frame_ids[idx])
            ball_xy = ball[idx, :2]

            row = {
                "segment": segment,
                "frame_id": frame_id,
                "ball_valid": bool(np.isfinite(ball_xy).all()),
            }

            for team, team_slice in TEAMS.items():

                features = extract_team_features(
                    players[idx],
                    ball_xy,
                    team_slice,
                    team
                )

                row.update(features)

            rows.append(row)

    result = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT,
        index=False,
        float_format="%.6f"
    )

    print("\n==========================================")
    print("TECH-CIA | COMPARACION MULTI-INTERVALO")
    print("==========================================")

    for segment, part in result.groupby("segment", sort=False):

        print(f"\n========== {segment} ==========")

        print("Frames:", len(part))
        print(
            "Disponibilidad del balón:",
            f"{100 * part['ball_valid'].mean():.2f}%"
        )

        for team in TEAMS:

            print(f"\n{team}")

            print(
                "Jugadores válidos por frame:",
                f"{part[f'{team}_n_players'].mean():.2f}"
            )

            for feature in (
                "nearest_m",
                "nearest3_mean_m",
                "mean_distance_m",
                "median_distance_m",
                "angular_coverage_deg",
                "angular_resultant_R",
            ):

                column = f"{team}_{feature}"

                print(
                    f"{feature}: "
                    f"{part[column].median():.4f}"
                )

            for radius in (2, 5, 10, 15):

                column = f"{team}_share_within_{radius}m"

                print(
                    f"Proporción dentro de {radius} m:",
                    f"{part[column].mean():.4f}"
                )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
