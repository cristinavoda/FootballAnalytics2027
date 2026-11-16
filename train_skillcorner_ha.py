import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

ROOT = Path.cwd()
DATA = ROOT / "artifacts" / "skillcorner" / "skillcorner_features.csv.gz"
OUT = ROOT / "artifacts" / "skillcorner" / "models"
OUT.mkdir(parents=True, exist_ok=True)

TEST_MATCHES = ["1874553", "2017461"]

VAL_MATCHES = ["1899585", "1996436", "2007448"]

FEATURES = [
    "ball_x", "ball_y", "ball_z", "ball_detected",
    "players_total", "players_detected",
    "home_count", "home_detected",
    "away_count", "away_detected",
    "home_centroid_x", "home_centroid_y",
    "away_centroid_x", "away_centroid_y",
    "home_spread_x", "home_spread_y",
    "away_spread_x", "away_spread_y",
    "home_centroid_ball_dist", "away_centroid_ball_dist",
    "home_min_ball_dist", "away_min_ball_dist",
    "home_mean_ball_dist", "away_mean_ball_dist",
    "home_near_2m", "home_near_5m", "home_near_10m",
    "away_near_2m", "away_near_5m", "away_near_10m"
]

USECOLS = ["match_id", "frame", "label"] + FEATURES

# --------------------------------------------------
# CARGA MUESTREADA: 1 FRAME DE CADA 5
# --------------------------------------------------

print("Leyendo dataset...")

parts = []

for chunk in pd.read_csv(
    DATA,
    usecols=USECOLS,
    chunksize=100_000,
    dtype={"match_id": str}
):
    chunk["frame"] = pd.to_numeric(chunk["frame"], errors="coerce")

    chunk = chunk[
        (chunk["frame"] % 5 == 0) &
        (chunk["label"].isin(["H", "A"]))
    ]

    if not chunk.empty:
        parts.append(chunk)

df = pd.concat(parts, ignore_index=True)
del parts

print("Frames etiquetados muestreados:", f"{len(df):,}")

# --------------------------------------------------
# PARTICIÓN POR PARTIDO COMPLETO
# --------------------------------------------------

test_ids = set(TEST_MATCHES)
val_ids = set(VAL_MATCHES)

train_ids = set(df["match_id"].unique()) - test_ids - val_ids

train = df[df["match_id"].isin(train_ids)]
val = df[df["match_id"].isin(val_ids)]
test = df[df["match_id"].isin(test_ids)]

def report_split(name, data):
    print(f"\n{name}:")
    print("Partidos:", sorted(data["match_id"].unique()))
    print("Filas:", f"{len(data):,}")
    print(data["label"].value_counts().to_dict())

report_split("TRAIN", train)
report_split("VALIDATION", val)
report_split("TEST", test)

X_train = train[FEATURES]
y_train = train["label"]

X_val = val[FEATURES]
y_val = val["label"]

X_test = test[FEATURES]
y_test = test["label"]

# --------------------------------------------------
# MODELO 1: REGRESIÓN LOGÍSTICA
# --------------------------------------------------

print("\nEntrenando Logistic Regression...")

logistic = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

logistic.fit(X_train, y_train)

# --------------------------------------------------
# MODELO 2: HISTOGRAM GRADIENT BOOSTING
# --------------------------------------------------

print("Entrenando HistGradientBoosting...")

boosting = HistGradientBoostingClassifier(
    max_iter=120,
    learning_rate=0.08,
    max_leaf_nodes=15,
    min_samples_leaf=50,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42
)

boosting.fit(X_train, y_train)

# --------------------------------------------------
# VALIDACIÓN
# --------------------------------------------------

models = {
    "LogisticRegression": logistic,
    "HistGradientBoosting": boosting
}

validation_scores = {}

print("\n" + "=" * 55)
print("RESULTADOS DE VALIDACIÓN")
print("=" * 55)

for name, model in models.items():
    predictions = model.predict(X_val)

    macro = f1_score(y_val, predictions, average="macro")
    validation_scores[name] = macro

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
    print("Balanced accuracy:", round(
        balanced_accuracy_score(y_val, predictions), 4
    ))
    print("Macro F1:", round(macro, 4))
    print("Matriz H/A:")
    print(confusion_matrix(y_val, predictions, labels=["H", "A"]))

# --------------------------------------------------
# SELECCIÓN Y TEST FINAL
# --------------------------------------------------

best_name = max(validation_scores, key=validation_scores.get)
best_model = models[best_name]

print("\nModelo seleccionado:", best_name)
print("Macro F1 validación:", round(validation_scores[best_name], 4))

print("\n" + "=" * 55)
print("EVALUACIÓN FINAL EN TEST")
print("=" * 55)

test_predictions = best_model.predict(X_test)

test_metrics = {
    "selected_model": best_name,
    "validation_macro_f1": float(validation_scores[best_name]),
    "test_accuracy": float(accuracy_score(y_test, test_predictions)),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "test_macro_f1": float(
        f1_score(y_test, test_predictions, average="macro")
    ),
    "test_confusion_matrix": confusion_matrix(
        y_test, test_predictions, labels=["H", "A"]
    ).tolist(),
    "train_matches": sorted(train_ids),
    "validation_matches": sorted(val_ids),
    "test_matches": sorted(test_ids),
    "sample_interval_frames": 5,
    "features": FEATURES
}

print("Accuracy:", round(test_metrics["test_accuracy"], 4))
print("Balanced accuracy:", round(test_metrics["test_balanced_accuracy"], 4))
print("Macro F1:", round(test_metrics["test_macro_f1"], 4))

print("\nInforme por clase:")
print(classification_report(
    y_test, test_predictions,
    labels=["H", "A"],
    zero_division=0
))

print("Matriz de confusión (filas reales H/A, columnas predichas H/A):")
print(np.array(test_metrics["test_confusion_matrix"]))

# --------------------------------------------------
# GUARDAR RESULTADOS
# --------------------------------------------------

joblib.dump(best_model, OUT / "best_model_h_a.joblib")

with (OUT / "training_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2, ensure_ascii=False)

print("\nMODELO GUARDADO:")
print(OUT / "best_model_h_a.joblib")
print("MÉTRICAS:")
print(OUT / "training_metrics.json")
print("\nENTRENAMIENTO TERMINADO.")
