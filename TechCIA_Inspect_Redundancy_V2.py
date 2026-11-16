
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "artifacts" / "derived" / "tech_cia_features_v2.csv"
CORR_FILE = ROOT / "artifacts" / "derived" / "feature_correlations_v2.csv"

df = pd.read_csv(DATA_FILE)
pairs = pd.read_csv(CORR_FILE)

print("\n========== REDUNDANCY INSPECTION V2 ==========\n")

# 1. Detectar columnas exactamente duplicadas
# Incluye la comparación de posiciones NaN.

columns = [c for c in df.columns if c != "frame_id"]

duplicate_groups = []
checked = set()

for i, col1 in enumerate(columns):

    if col1 in checked:
        continue

    group = [col1]

    for col2 in columns[i + 1:]:

        if df[col1].equals(df[col2]):
            group.append(col2)

    if len(group) > 1:
        duplicate_groups.append(group)
        checked.update(group)

print("--- DUPLICADOS EXACTOS ---")

if duplicate_groups:
    for group in duplicate_groups:
        print(" | ".join(group))
else:
    print("No se encontraron duplicados exactos.")

# 2. Examinar los pares correlacionados
print("\n--- CORRELACIONES Y MUESTRA COMPARTIDA ---")

results = []

for _, pair in pairs.iterrows():

    col1 = pair["feature_1"]
    col2 = pair["feature_2"]

    valid = df[[col1, col2]].dropna()

    if len(valid) < 2:
        continue

    corr = valid[col1].corr(valid[col2])

    results.append({
        "feature_1": col1,
        "feature_2": col2,
        "correlation": corr,
        "common_rows": len(valid),
        "common_pct": round(len(valid) / len(df) * 100, 2),
        "same_values_pct": round(
            (valid[col1] == valid[col2]).mean() * 100, 2
        )
    })

result_df = pd.DataFrame(results)

if not result_df.empty:
    result_df = result_df.sort_values(
        "correlation",
        key=lambda s: s.abs(),
        ascending=False
    )

    print(result_df.to_string(index=False))

# 3. Guardar el informe
output_file = (
    ROOT / "artifacts" / "derived" /
    "redundancy_inspection_v2.csv"
)

result_df.to_csv(output_file, index=False)

print("\nInforme guardado en:")
print(output_file)

print("\nInspección completada.")
