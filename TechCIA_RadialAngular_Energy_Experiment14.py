
from pathlib import Path
import numpy as np
import pandas as pd

NPZ = Path("artifacts/tracking_atd.npz")

OUTPUT = Path(
    "artifacts/TechCIA_RadialAngular_Energy_Experiment14.csv"
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


def decompose_group(points, ball_m):

    relative = points - ball_m

    radii = np.linalg.norm(relative, axis=1)

    angles = np.arctan2(
        -relative[:, 1],
        relative[:, 0]
    )

    i, j = np.triu_indices(len(points), k=1)

    delta = np.arctan2(
        np.sin(angles[i] - angles[j]),
        np.cos(angles[i] - angles[j])
    )

    # Separación radial al cuadrado.
    radial_sq = (radii[i] - radii[j]) ** 2

    # Separación angular al cuadrado.
    angular_sq = (
        4
        * radii[i]
        * radii[j]
        * np.sin(delta / 2) ** 2
    )

    # Distancia euclídea observada al cuadrado.
    actual_sq = np.sum(
        (points[i] - points[j]) ** 2,
        axis=1
    )

    radial_component = radial_sq.mean()
    angular_component = angular_sq.mean()
    total_component = actual_sq.mean()

    calculated_total = (
        radial_component + angular_component
    )

    error = abs(calculated_total - total_component)

    if total_component > 1e-12:

        radial_share = (
            100 * radial_component / total_component
        )

        angular_share = (
            100 * angular_component / total_component
        )

    else:
        radial_share = np.nan
        angular_share = np.nan

    return {
        "mean_radius_m": radii.mean(),
        "radial_std_m": radii.std(),

        "radial_component_m2": radial_component,
        "angular_component_m2": angular_component,
        "total_component_m2": total_component,

        "radial_share_pct": radial_share,
        "angular_share_pct": angular_share,

        "pairwise_rms_m": np.sqrt(total_component),
        "decomposition_error_m2": error,
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

            for k in K_VALUES:

                row = {
                    "segment": segment,
                    "frame_id": int(frame_ids[idx]),
                    "k": k,
                }

                for team, team_slice in TEAMS.items():

                    positions = players[idx, team_slice, :2]

                    valid = np.isfinite(
                        positions
                    ).all(axis=1)

                    positions = positions[valid].copy()

                    positions[:, 0] *= LENGTH_M
                    positions[:, 1] *= WIDTH_M

                    distances = np.linalg.norm(
                        positions - ball_m,
                        axis=1
                    )

                    local = positions[
                        distances <= RADIUS_M
                    ]

                    prefix = team

                    row[f"{prefix}_n_15m"] = len(local)

                    if len(local) >= k:

                        order = np.argsort(
                            np.linalg.norm(
                                local - ball_m,
                                axis=1
                            )
                        )

                        selected = local[order[:k]]

                        result = decompose_group(
                            selected,
                            ball_m
                        )

                        for metric, value in result.items():
                            row[f"{prefix}_{metric}"] = value

                    else:

                        for metric in (
                            "mean_radius_m",
                            "radial_std_m",
                            "radial_component_m2",
                            "angular_component_m2",
                            "total_component_m2",
                            "radial_share_pct",
                            "angular_share_pct",
                            "pairwise_rms_m",
                            "decomposition_error_m2",
                        ):
                            row[f"{prefix}_{metric}"] = np.nan

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
    print("TECH-CIA | RADIAL / ANGULAR ENERGY 14")
    print("==========================================")

    metrics = (
        "radial_share_pct",
        "angular_share_pct",
        "radial_component_m2",
        "angular_component_m2",
        "pairwise_rms_m",
    )

    for (segment, k), part in df.groupby(
        ["segment", "k"],
        sort=False
    ):

        common = (
            (part["Barcelona_n_15m"] >= k)
            & (part["Athletic_n_15m"] >= k)
        )

        paired = part.loc[common]

        print(
            f"\n========== {segment} | K={k} =========="
        )

        print(
            f"Frames comparables: {len(paired)}/{len(part)}"
        )

        for metric in metrics:

            b = paired[f"Barcelona_{metric}"]
            a = paired[f"Athletic_{metric}"]

            difference = (b - a).dropna()

            if len(difference) == 0:
                continue

            print(
                f"{metric:25s} | "
                f"Barça mediana={b.median():8.3f} | "
                f"Athletic mediana={a.median():8.3f} | "
                f"B-A={difference.median():8.3f} | "
                f"Barça mayor={100*(difference > 1e-9).mean():5.1f}%"
            )

        errors = pd.concat([
            paired["Barcelona_decomposition_error_m2"],
            paired["Athletic_decomposition_error_m2"],
        ])

        print(
            "Error máximo de la descomposición:",
            f"{errors.max():.10f} m²"
        )

    print("\nCSV generado:")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
