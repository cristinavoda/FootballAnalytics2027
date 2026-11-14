from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "temporal_features_v1.csv"
FEATURES_BASE = [
    "dist_margin_top3_m_mean_3s",
    "players_5m_margin_mean_3s",
    "players_10m_margin_mean_3s",
]

FEATURE_COMPACTNESS = "compactness_margin_m_mean_3s"

TARGET = "dist_margin_min_m_mean_3s"

VISIBILITY = [
    "barcelona_visible_count",
    "athletic_visible_count",
]

USECOLS = list(dict.fromkeys(
    FEATURES_BASE + [FEATURE_COMPACTNESS] + VISIBILITY
))
USECOLS = list(dict.fromkeys(
    FEATURES_BASE + [FEATURE_COMPACTNESS, TARGET] + VISIBILITY
))

print("Leyendo datos...")
df = pd.read_csv(INPUT, usecols=USECOLS)

# ---------------------------------------------------------
# 1. FILTRO DE COBERTURA
# ---------------------------------------------------------

mask = (
    (df["barcelona_visible_count"] >= 8) &
    (df["athletic_visible_count"] >= 8)
)

df = df.loc[mask].copy()

print("\n=== MUESTRA ===")
print(f"Frames con cobertura >= 8: {len(df):,}")

# ---------------------------------------------------------
# 2. ELIMINAR FILAS INCOMPLETAS
# ---------------------------------------------------------

model_cols = FEATURES_BASE + [FEATURE_COMPACTNESS]

valid = df.dropna(subset=model_cols).copy()

print(f"Frames válidos: {len(valid):,}")

# ---------------------------------------------------------
# 3. DIVISIÓN TEMPORAL 70 / 30
# ---------------------------------------------------------

n = len(valid)
split = int(n * 0.70)

train = valid.iloc[:split].copy()
test = valid.iloc[split:].copy()

print("\n=== DIVISIÓN TEMPORAL ===")
print(f"Train: {len(train):,} frames")
print(f"Test : {len(test):,} frames")

# ---------------------------------------------------------
# 4. REGRESIÓN LINEAL MANUAL
# ---------------------------------------------------------

def fit_predict(train_df, test_df, features):

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=float)

    X_test = test_df[features].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    # Intercepto
    X_train = np.column_stack([
        np.ones(len(X_train)),
        X_train
    ])

    X_test = np.column_stack([
        np.ones(len(X_test)),
        X_test
    ])

    beta = np.linalg.lstsq(
        X_train,
        y_train,
        rcond=None
    )[0]

    prediction = X_test @ beta

    return y_test, prediction, beta


def metrics(y, pred):

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot

    rmse = np.sqrt(np.mean((y - pred) ** 2))

    correlation = np.corrcoef(y, pred)[0, 1]

    return r2, rmse, correlation


# ---------------------------------------------------------
# 5. MODELO A
# ---------------------------------------------------------

y, pred_a, beta_a = fit_predict(
    train,
    test,
    FEATURES_BASE
)

r2_a, rmse_a, corr_a = metrics(y, pred_a)

# ---------------------------------------------------------
# 6. MODELO B
# ---------------------------------------------------------

features_b = FEATURES_BASE + [FEATURE_COMPACTNESS]

y, pred_b, beta_b = fit_predict(
    train,
    test,
    features_b
)

r2_b, rmse_b, corr_b = metrics(y, pred_b)

# ---------------------------------------------------------
# 7. RESULTADOS
# ---------------------------------------------------------

print("\n=== MODELO A: PROXIMIDAD ===")
print("Variables:")
for f in FEATURES_BASE:
    print("  -", f)

print(f"R²   : {r2_a:.5f}")
print(f"RMSE : {rmse_a:.5f}")
print(f"Corr : {corr_a:.5f}")

print("\n=== MODELO B: PROXIMIDAD + COMPACTACIÓN ===")
print("Variables:")
for f in features_b:
    print("  -", f)

print(f"R²   : {r2_b:.5f}")
print(f"RMSE : {rmse_b:.5f}")
print(f"Corr : {corr_b:.5f}")

# ---------------------------------------------------------
# 8. GANANCIA
# ---------------------------------------------------------

delta_r2 = r2_b - r2_a
delta_rmse = rmse_a - rmse_b

print("\n=== GANANCIA DE COMPACTACIÓN ===")
print(f"Δ R²       : {delta_r2:+.5f}")
print(f"Δ RMSE     : {delta_rmse:+.5f}")

if delta_r2 > 0:
    print("Resultado: la compactación mejora el R².")
else:
    print("Resultado: la compactación NO mejora el R².")

if delta_rmse > 0:
    print("También reduce el RMSE.")
elif delta_rmse < 0:
    print("También aumenta el RMSE.")
else:
    print("El RMSE no cambia.")

# ---------------------------------------------------------
# 9. COEFICIENTE DE COMPACTACIÓN
# ---------------------------------------------------------

print("\n=== COEFICIENTES MODELO B ===")

for name, value in zip(
    ["intercept"] + features_b,
    beta_b
):
    print(f"{name}: {value:+.6f}")