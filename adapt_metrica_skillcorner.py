import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

ROOT = Path(r"E:\Hackaton 2027\FootballAnalytics2027")
ART = ROOT / "artifacts"

# Cargar tracking
data = np.load(ART / "tracking_atd.npz")
players = data["players"][:, :, :2].astype(np.float32)
ball = data["ball"][:, :2].astype(np.float32)
frame_ids = data["frame_ids"]

# Convertir coordenadas normalizadas a metros.
# Se conserva cualquier valor fuera de los límites del campo.
players[:, :, 0] = (players[:, :, 0] - 0.5) * 105
players[:, :, 1] = (players[:, :, 1] - 0.5) * 68

ball[:, 0] = (ball[:, 0] - 0.5) * 105
ball[:, 1] = (ball[:, 1] - 0.5) * 68

# Equipos: Barcelona local (24), Athletic visitante (22)
home = players[:, :24, :]
away = players[:, 24:, :]

ball_valid = np.isfinite(ball).all(axis=1)
home_valid = np.isfinite(home).all(axis=2)
away_valid = np.isfinite(away).all(axis=2)

n = len(frame_ids)

features = {
    "ball_detected": ball_valid.astype(np.float32),
    "home_detected": home_valid.sum(axis=1).astype(np.float32),
    "away_detected": away_valid.sum(axis=1).astype(np.float32),
}

def calculate_team_features(team, valid):
    centroid_dist = np.full(n, np.nan, dtype=np.float32)
    min_dist = np.full(n, np.nan, dtype=np.float32)
    mean_dist = np.full(n, np.nan, dtype=np.float32)

    near_2 = np.zeros(n, dtype=np.float32)
    near_5 = np.zeros(n, dtype=np.float32)
    near_10 = np.zeros(n, dtype=np.float32)

    for i in np.flatnonzero(ball_valid):
        positions = team[i, valid[i]]

        if len(positions) == 0:
            continue

        distances = np.linalg.norm(positions - ball[i], axis=1)

        centroid = positions.mean(axis=0)
        centroid_dist[i] = np.linalg.norm(centroid - ball[i])

        min_dist[i] = distances.min()
        mean_dist[i] = distances.mean()

        near_2[i] = np.sum(distances <= 2)
        near_5[i] = np.sum(distances <= 5)
        near_10[i] = np.sum(distances <= 10)

    return centroid_dist, min_dist, mean_dist, near_2, near_5, near_10

h = calculate_team_features(home, home_valid)
a = calculate_team_features(away, away_valid)

features.update({
    "home_centroid_ball_dist": h[0],
    "away_centroid_ball_dist": a[0],
    "home_min_ball_dist": h[1],
    "away_min_ball_dist": a[1],
    "home_mean_ball_dist": h[2],
    "away_mean_ball_dist": a[2],
    "home_near_2m": h[3],
    "home_near_5m": h[4],
    "home_near_10m": h[5],
    "away_near_2m": a[3],
    "away_near_5m": a[4],
    "away_near_10m": a[5],
})

# Orden exacto utilizado durante el entrenamiento
metrics_path = ART / "skillcorner" / "models" / "training_metrics_invariant.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
columns = metrics["features"]

X = pd.DataFrame(features, columns=columns)

# Guardar características transferidas
out_dir = ART / "metrica_transfer"
out_dir.mkdir(parents=True, exist_ok=True)

features_path = out_dir / "metrica_features_invariant.csv.gz"
X.insert(0, "frame_id", frame_ids)
X.to_csv(features_path, index=False, compression="gzip")

print("CARACTERÍSTICAS CALCULADAS")
print("Frames:", len(X))
print("Columnas:", len(X.columns) - 1)
print("Balón visible:", int(ball_valid.sum()))
print("Valores ausentes por columna:")
print(X.isna().sum().to_string())

# Cargar modelo entrenado exclusivamente con SkillCorner
model_path = ART / "skillcorner" / "models" / "best_model_h_a_invariant.joblib"
model = joblib.load(model_path)

# Predecir H/A
predictions = model.predict(X[columns])
probabilities = model.predict_proba(X[columns])

class_order = list(model.classes_)
prob_h = probabilities[:, class_order.index("H")]
prob_a = probabilities[:, class_order.index("A")]

result = pd.DataFrame({
    "frame_id": frame_ids,
    "prediction_HA": predictions,
    "probability_H": prob_h,
    "probability_A": prob_a,
})

prediction_path = out_dir / "metrica_predictions_HA.csv"
result.to_csv(prediction_path, index=False)

print("\nPREDICCIÓN TERMINADA")
print(result["prediction_HA"].value_counts().to_string())
print("\nProbabilidad media H:", round(float(prob_h.mean()), 4))
print("Probabilidad media A:", round(float(prob_a.mean()), 4))
print("\nCaracterísticas:", features_path)
print("Predicciones:", prediction_path)
