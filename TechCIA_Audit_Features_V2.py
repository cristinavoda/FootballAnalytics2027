
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    ROOT / "artifacts" / "derived" /
    "tech_cia_features_v2.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "derived"

AUDIT_FILE = OUTPUT_DIR / "feature_audit_v2.csv"
CORRELATION_FILE = OUTPUT_DIR / "feature_correlations_v2.csv"

CORRELATION_THRESHOLD = 0.95

print("\n========== TECH-CIA FEATURE AUDIT V2 ==========\n")

# --------------------------------------------------
# 1. Cargar y comprobar el dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

assert "frame_id" in df.columns, "Falta frame_id"
assert df["frame_id"].is_unique, "Hay frame_id duplicados"
assert df["frame_id"].is_monotonic_increasing, (
    "Los frame_id no están ordenados"
)

feature_columns = [
    col for col in df.columns
    if col != "frame_id"
]

print("Frames:", len(df))
print("Columnas totales:", len(df.columns))
print("Features analizadas:", len(feature_columns))

# --------------------------------------------------
# 2. Auditoría individual de cada feature
# --------------------------------------------------

audit_rows = []

for col in feature_columns:

    series = df[col]

    missing = int(series.isna().sum())
    non_missing = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))

    row = {
        "feature": col,
        "dtype": str(series.dtype),
        "non_missing": non_missing,
        "missing": missing,
        "missing_pct": round(
            missing / len(df) * 100, 3
        ),
        "unique_values": unique,
        "constant": unique <= 1,
    }

    if pd.api.types.is_numeric_dtype(series):

        values = series.dropna()

        if len(values) > 0:
            row.update({
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p01": values.quantile(0.01),
                "median": values.median(),
                "p99": values.quantile(0.99),
                "max": values.max(),
                "zero_count": int((values == 0).sum()),
                "zero_pct": round(
                    (values == 0).mean() * 100, 3
                ),
            })

    audit_rows.append(row)

audit = pd.DataFrame(audit_rows)

audit = audit.sort_values(
    by=["missing_pct", "unique_values"],
    ascending=[False, True]
)

audit.to_csv(AUDIT_FILE, index=False)

# --------------------------------------------------
# 3. Detectar features constantes
# --------------------------------------------------

constant_features = audit.loc[
    audit["constant"], "feature"
].tolist()

# --------------------------------------------------
# 4. Detectar features con muchos valores ausentes
# --------------------------------------------------

high_missing = audit.loc[
    audit["missing_pct"] >= 50,
    ["feature", "missing_pct", "unique_values"]
]

# --------------------------------------------------
# 5. Correlaciones entre features numéricas
# --------------------------------------------------

numeric = df[feature_columns].select_dtypes(
    include="number"
)

# Excluir columnas constantes: no aportan correlación útil
numeric = numeric.loc[
    :, numeric.nunique(dropna=True) > 1
]

correlation_matrix = numeric.corr(
    method="pearson",
    min_periods=100
)

pairs = []

columns = correlation_matrix.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        corr = correlation_matrix.iloc[i, j]

        if pd.notna(corr) and abs(corr) >= CORRELATION_THRESHOLD:
            pairs.append({
                "feature_1": columns[i],
                "feature_2": columns[j],
                "correlation": corr,
                "absolute_correlation": abs(corr),
            })

correlations = pd.DataFrame(pairs)

if not correlations.empty:
    correlations = correlations.sort_values(
        "absolute_correlation",
        ascending=False
    )

correlations.to_csv(CORRELATION_FILE, index=False)

# --------------------------------------------------
# 6. Informe en consola
# --------------------------------------------------

print("\n--- FEATURES CONSTANTES ---")

if constant_features:
    for feature in constant_features:
        print(feature)
else:
    print("Ninguna")

print("\n--- FEATURES CON >= 50% DE VALORES AUSENTES ---")

if len(high_missing):
    print(high_missing.to_string(index=False))
else:
    print("Ninguna")

print("\n--- CORRELACIONES ALTAS (|r| >= 0.95) ---")

if correlations.empty:
    print("No se han encontrado pares.")
else:
    print(correlations.to_string(index=False))

print("\n--- RESULTADO ---")
print("Auditoría:", AUDIT_FILE)
print("Correlaciones:", CORRELATION_FILE)
print("Features analizadas:", len(feature_columns))
print("Pares altamente correlacionados:", len(correlations))

print("\nAuditoría completada.")
