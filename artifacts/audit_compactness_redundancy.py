
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "temporal_features_v1.csv"

columns = [
    "barcelona_visible_count",
    "athletic_visible_count",
    "compactness_margin_m_mean_3s",
    "dist_margin_min_m_mean_3s",
    "dist_margin_top3_m_mean_3s",
    "players_5m_margin_mean_3s",
    "players_10m_margin_mean_3s",
]

df = pd.read_csv(INPUT, usecols=columns)

# Filtrar frames con al menos 8 jugadores visibles por equipo
mask = (
    (df["barcelona_visible_count"] >= 8) &
    (df["athletic_visible_count"] >= 8)
)

df = df.loc[mask].copy()

# Renombrar para facilitar la lectura
df = df.rename(columns={
    "compactness_margin_m_mean_3s": "compactness",
    "dist_margin_min_m_mean_3s": "min_distance",
    "dist_margin_top3_m_mean_3s": "top3_distance",
    "players_5m_margin_mean_3s": "players_5m",
    "players_10m_margin_mean_3s": "players_10m",
})

features = [
    "compactness",
    "min_distance",
    "top3_distance",
    "players_5m",
    "players_10m",
]

valid = df[features].dropna()

print("=== MUESTRA ===")
print("Frames con cobertura suficiente:", len(df))
print("Frames válidos para las cinco señales:", len(valid))

# Correlación de Pearson
print("\n=== CORRELACIÓN DE PEARSON ===")
print(valid.corr(method="pearson").round(3))

# Correlación de Spearman calculada mediante rangos
print("\n=== CORRELACIÓN DE SPEARMAN ===")
ranks = valid.rank()
print(ranks.corr(method="pearson").round(3))

# Asociación parcial mediante regresión lineal auxiliar:
# residualizamos compactación frente a las cuatro señales restantes.
predictors = [
    "top3_distance",
    "players_5m",
    "players_10m",
]

# Incluimos distancia mínima como señal de referencia en el
# modelo auxiliar para medir qué parte de compactación no explican
# las cuatro señales de proximidad.
predictors = [
    "min_distance",
    "top3_distance",
    "players_5m",
    "players_10m",
]

X = valid[predictors].to_numpy()
y = valid["compactness"].to_numpy()

# Intercepto
X = pd.DataFrame(X)
X.insert(0, "intercept", 1.0)

coefficients = __import__("numpy").linalg.lstsq(
    X.to_numpy(), y, rcond=None
)[0]

residuals = y - X.to_numpy() @ coefficients

# Asociación entre compactación residual y distancia mínima
residual_corr = pd.Series(residuals).corr(
    valid["min_distance"].reset_index(drop=True)
)

print("\n=== INFORMACIÓN RESIDUAL ===")
print("Correlación compactación residual vs distancia mínima:")
print(round(residual_corr, 5))
