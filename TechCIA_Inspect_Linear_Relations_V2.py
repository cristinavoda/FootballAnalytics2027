
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    ROOT / "artifacts" / "derived" /
    "tech_cia_features_v2.csv"
)

PAIRS_FILE = (
    ROOT / "artifacts" / "derived" /
    "feature_correlations_v2.csv"
)

OUTPUT_FILE = (
    ROOT / "artifacts" / "derived" /
    "linear_relations_v2.csv"
)

df = pd.read_csv(DATA_FILE)
pairs = pd.read_csv(PAIRS_FILE)

results = []

print("\n========== LINEAR RELATIONS V2 ==========\n")

for _, pair in pairs.iterrows():

    col1 = pair["feature_1"]
    col2 = pair["feature_2"]

    valid = df[[col1, col2]].dropna()

    x = pd.to_numeric(
        valid[col1], errors="coerce"
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        valid[col2], errors="coerce"
    ).to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        continue

    # Igualdad numérica, independientemente del dtype
    equal_pct = np.isclose(
        x, y, rtol=1e-9, atol=1e-12
    ).mean() * 100

    # Diferencias directas
    differences = np.abs(x - y)

    max_difference = differences.max()
    mean_difference = differences.mean()

    # Relación lineal: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept
    residuals = y - predicted

    results.append({
        "feature_1": col1,
        "feature_2": col2,
        "rows": len(x),
        "equal_pct": round(equal_pct, 4),
        "max_abs_difference": max_difference,
        "mean_abs_difference": mean_difference,
        "slope": slope,
        "intercept": intercept,
        "residual_std": residuals.std(),
        "residual_max_abs": np.abs(residuals).max(),
    })

result = pd.DataFrame(results)

result = result.sort_values(
    "residual_std",
    ascending=True
)

result.to_csv(OUTPUT_FILE, index=False)

print(result.to_string(index=False))

print("\nInforme guardado:")
print(OUTPUT_FILE)

print("\nInspección completada.")

