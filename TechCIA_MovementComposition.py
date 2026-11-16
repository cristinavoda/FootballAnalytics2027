
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "artifacts" / "derived" / "team_toward_dynamics.csv"
OUTPUT = ROOT / "artifacts" / "derived" / "movement_composition_analysis.csv"

WINDOWS = [5, 10, 25]
TEAMS = ["barca", "athletic"]

df = pd.read_csv(INPUT)

print("=" * 75)
print("RELACIÓN ENTRE MOVIMIENTO RADIAL Y COMPOSICIÓN")
print("=" * 75)

results = []

for team in TEAMS:

    print(f"\n{'=' * 75}")
    print(team.upper())
    print("=" * 75)

    for k in WINDOWS:

        movement_col = f"{team}_toward_10m_delta_{k}"
        count_col = f"{team}_n_10m_delta_{k}"

        data = df[[movement_col, count_col]].dropna()

        if data.empty:
            print(f"{k} frames: sin datos")
            continue

        movement = data[movement_col]
        count_change = data[count_col]

        stable = data[count_col] == 0
        changing = data[count_col] != 0

        
        pearson = movement.corr(count_change, method="pearson")

        # Spearman: correlación de Pearson sobre los rangos
        movement_ranks = movement.rank(method="average")
        count_ranks = count_change.rank(method="average")
        spearman = movement_ranks.corr(count_ranks, method="pearson")

        stable_data = movement[stable]
        changing_data = movement[changing]

        print(f"\n--- {k} frames ---")
        print(f"Observaciones: {len(data):,}")
        print(f"Correlación Spearman: {spearman:.4f}")
        print(f"Correlación Pearson:  {pearson:.4f}")

        print(
            f"Grupo estable: {stable.sum():,} "
            f"({stable.mean() * 100:.2f}%)"
        )
        print(
            f"Grupo cambiante: {changing.sum():,} "
            f"({changing.mean() * 100:.2f}%)"
        )

        print(
            "Movimiento medio absoluto cuando el recuento permanece estable:",
            f"{stable_data.abs().mean():.4f}"
        )

        print(
            "Movimiento medio absoluto cuando el recuento cambia:",
            f"{changing_data.abs().mean():.4f}"
        )

        print(
            "Mediana absoluta, recuento estable:",
            f"{stable_data.abs().median():.4f}"
        )

        print(
            "Mediana absoluta, recuento cambiante:",
            f"{changing_data.abs().median():.4f}"
        )

        results.append({
            "team": team,
            "window_frames": k,
            "n_observations": len(data),
            "spearman": spearman,
            "pearson": pearson,
            "stable_n": int(stable.sum()),
            "changing_n": int(changing.sum()),
            "stable_abs_mean": stable_data.abs().mean(),
            "changing_abs_mean": changing_data.abs().mean(),
            "stable_abs_median": stable_data.abs().median(),
            "changing_abs_median": changing_data.abs().median(),
        })

# --------------------------------------------------
# GUARDAR RESUMEN
# --------------------------------------------------

summary = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT, index=False)

print("\n" + "=" * 75)
print("RESUMEN COMPARATIVO")
print("=" * 75)

print(
    summary[
        [
            "team",
            "window_frames",
            "spearman",
            "pearson",
            "stable_abs_mean",
            "changing_abs_mean",
        ]
    ].round(4).to_string(index=False)
)

print("\nArchivo generado:")
print(OUTPUT)
print("\nLos archivos de origen no se han modificado.")
