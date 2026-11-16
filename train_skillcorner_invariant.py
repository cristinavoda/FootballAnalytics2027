import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "artifacts" / "skillcorner" / "skillcorner_features.csv.gz"
OUT = ROOT / "artifacts" / "skillcorner" / "models"
OUT.mkdir(parents=True, exist_ok=True)

TEST_MATCHES = ["1874553", "2017461"]
VAL_MATCHES = ["1899585", "1996436", "2007448"]

FEATURES = [
    "ball_detected",
    "home_detected", "away_detected",
    "home_centroid_ball_dist", "away_centroid_ball_dist",
    "home_min_ball_dist", "away_min_ball_dist",
    "home_mean_ball_dist", "away_mean_ball_dist",
    "home_near_2m", "home_near_5m", "home_near_10m",
    "away_near_2m", "away_near_5m", "away_near_10m"
]

USECOLS = ["match_id", "frame", "label"] + FEATURES

print("Leyendo SkillCorner...")
parts = []

for chunk in pd.read_csv(
    DATA, usecols=USECOLS, chunksize=100_000,
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

print("Filas:", len(df))
print("Características:", len(FEATURES))

test_ids = set(TEST_MATCHES)
val_ids = set(VAL_MATCHES)
train_ids = set(df["match_id"].unique()) - test_ids - val_ids

train = df[df["match_id"].isin(train_ids)]
val = df[df["match_id"].isin(val_ids)]
test = df[df["match_id"].isin(test_ids)]

for name, part in [("TRAIN", train), ("VALIDATION", val), ("TEST", test)]:
    print(f"\n{name}: {len(part)} filas")
    print("Partidos:", sorted(part["match_id"].unique()))
    print(part["label"].value_counts().to_dict())

X_train, y_train = train[FEATURES], train["label"]
X_val, y_val = val[FEATURES], val["label"]
X_test, y_test = test[FEATURES], test["label"]

print("\nEntrenando Logistic Regression...")
logistic = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ))
])
logistic.fit(X_train, y_train)

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

models = {
    "LogisticRegression": logistic,
    "HistGradientBoosting": boosting
}

scores = {}

print("\n=== VALIDACIÓN ===")
for name, model in models.items():
    pred = model.predict(X_val)
    score = f1_score(y_val, pred, average="macro")
    scores[name] = score

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_val, pred), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_val, pred), 4))
    print("Macro F1:", round(score, 4))
    print(confusion_matrix(y_val, pred, labels=["H", "A"]))

best_name = max(scores, key=scores.get)
best_model = models[best_name]

print("\nModelo seleccionado:", best_name)

pred = best_model.predict(X_test)

metrics = {
    "selected_model": best_name,
    "validation_macro_f1": float(scores[best_name]),
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
    "test_confusion_matrix": confusion_matrix(
        y_test, pred, labels=["H", "A"]
    ).tolist(),
    "train_matches": sorted(train_ids),
    "validation_matches": sorted(val_ids),
    "test_matches": sorted(test_ids),
    "sample_interval_frames": 5,
    "features": FEATURES
}

print("\n=== TEST FINAL ===")
print("Accuracy:", round(metrics["test_accuracy"], 4))
print("Balanced accuracy:", round(metrics["test_balanced_accuracy"], 4))
print("Macro F1:", round(metrics["test_macro_f1"], 4))
print(classification_report(y_test, pred, labels=["H", "A"]))

joblib.dump(best_model, OUT / "best_model_h_a_invariant.joblib")

with (OUT / "training_metrics_invariant.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\nModelo guardado:", OUT / "best_model_h_a_invariant.joblib")
print("Métricas guardadas:", OUT / "training_metrics_invariant.json")
print("\nSEGUNDO ENTRENAMIENTO TERMINADO.")
