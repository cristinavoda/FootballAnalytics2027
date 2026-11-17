
from pathlib import Path
import pandas as pd

SOURCE = Path(
    "artifacts/TechCIA_Geometric_AngularLocality_Experiment07.csv"
)

df = pd.read_csv(SOURCE)

RADII = ("all", "5m", "10m", "15m")
TEAMS = ("Barcelona", "Athletic")

print("\n==========================================")
print("TECH-CIA | AUDITORIA DE SOPORTE ANGULAR")
print("==========================================")

for segment, part in df.groupby("segment", sort=False):

    print(f"\n========== {segment} ==========")

    for radius in RADII:

        valid_masks = {}

        print(f"\n--- Radio {radius} ---")

        for team in TEAMS:

            column = f"{team}_{radius}_n"

            # Para calcular cobertura hacen falta
            # al menos dos jugadores distintos.
            valid = (
                part["ball_valid"]
                & (part[column] >= 2)
            )

            valid_masks[team] = valid

            count = int(valid.sum())
            percentage = 100 * count / len(part)

            three_or_more = int(
                (
                    part["ball_valid"]
                    & (part[column] >= 3)
                ).sum()
            )

            print(
                f"{team:12s} | "
                f"n >= 2: {count:4d} "
                f"({percentage:5.1f}%) | "
                f"n >= 3: {three_or_more:4d} "
                f"({100*three_or_more/len(part):5.1f}%)"
            )

        common = (
            valid_masks["Barcelona"]
            & valid_masks["Athletic"]
        )

        common_count = int(common.sum())

        print(
            f"Comparación simultánea válida: "
            f"{common_count}/{len(part)} "
            f"({100*common_count/len(part):.1f}%)"
        )

print("\nAuditoría terminada.")
