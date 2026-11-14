from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "artifacts" / "derived" / "tech_cia_features_v2.csv"
OUT = ROOT / "artifacts" / "derived" / "missingness_audit_v2.csv"

df = pd.read_csv(FILE)

report = pd.DataFrame({
    "feature": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (df.isna().mean().values * 100).round(2)
})

report = report.sort_values(
    "missing_percent", ascending=False
)

report.to_csv(OUT, index=False)

print("\n========== MISSINGNESS AUDIT V2 ==========")
print("Total de celdas:", df.shape[0] * df.shape[1])
print("Celdas ausentes:", int(df.isna().sum().sum()))
print("Porcentaje global:", round(df.isna().sum().sum() /
      (df.shape[0] * df.shape[1]) * 100, 2), "%")

print("\n20 variables con más ausencias:")
print(report.head(20).to_string(index=False))

print("\nAusencias en variables de dinámica:")
dynamics = report[report["feature"].str.contains(
    "speed|acc|valid", case=False, regex=True
)]
print(dynamics.to_string(index=False))

print("\nInforme guardado:", OUT)
