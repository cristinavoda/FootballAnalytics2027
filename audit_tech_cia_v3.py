
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

FEATURES = ROOT / "artifacts/derived/tech_cia_features_v4_rg2.csv"
PREDICTIONS = ROOT / "artifacts/derived/predictions_tech_cia_v3.csv"

EXPECTED_ROWS = 199_319

features = pd.read_csv(FEATURES)
pred = pd.read_csv(PREDICTIONS)

print("\n=== AUDITORÍA TÉCNICA TECH-CIA V3 ===")

# 1. Continuidad de la secuencia original
ids = pd.to_numeric(features["frame_id"], errors="coerce")

expected_ids = np.arange(1, EXPECTED_ROWS + 1)

assert ids.notna().all(), "Hay frame_id no numéricos"
assert ids.is_unique, "Hay frame_id duplicados"
assert np.array_equal(ids.to_numpy(), expected_ids), (
    "La secuencia original no es exactamente 1..199319"
)

print("PASS | Secuencia original completa y ordenada")

# 2. Continuidad en la salida
frames = pd.to_numeric(pred["frame"], errors="coerce")

expected_frames = np.arange(EXPECTED_ROWS)

assert frames.notna().all(), "Hay frames no numéricos"
assert frames.is_unique, "Hay frames duplicados"
assert np.array_equal(frames.to_numpy(), expected_frames), (
    "La salida no es exactamente 0..199318"
)

print("PASS | Salida completa, ordenada y sin duplicados")

# 3. Comprobar huecos y segmentos temporales
gaps = np.diff(ids.to_numpy())

assert np.all(gaps == 1), "Hay saltos en frame_id"

print("PASS | No existen huecos en la secuencia")

# 4. Verificar columnas y etiquetas
required = {"frame", "label"}
assert required.issubset(pred.columns)

assert pred["label"].notna().all()
assert pred["label"].isin(["H", "A", "D", "X"]).all()

print("PASS | Etiquetas válidas y sin vacíos")


# 5. Comprobar señales de los tres bloques
signal_cols = [
    "skill_signal",
    "geometry_signal",
    "proximity_signal",
]

assert set(signal_cols).issubset(pred.columns)

signals = pred[signal_cols].apply(
    pd.to_numeric, errors="coerce"
)

print("\n=== CONTROL INTERNO DE SEÑALES ===")

for col in signal_cols:
    values = signals[col]

    print(
        f"{col}: "
        f"NaN={values.isna().sum()} | "
        "Inf="
        f"{np.isinf(values.to_numpy()).sum()} | "
        f"válidos={values.notna().sum()}"
    )

    finite = values.dropna()

    if not finite.empty:
        assert np.isfinite(finite.to_numpy()).all()

# Las señales individuales pueden estar ausentes.
# El score final, en cambio, debe ser finito.
score = pd.to_numeric(pred["score"], errors="coerce")

assert score.notna().all(), (
    "Hay scores ausentes"
)
assert np.isfinite(score.to_numpy()).all(), (
    "Hay scores infinitos"
)

print("PASS | Scores finales completos y finitos")

# Correlaciones sobre pares de valores disponibles.
corr = signals.corr()

print("\nAviso de posible redundancia entre bloques:")
found = False

for i, col_a in enumerate(signal_cols):
    for col_b in signal_cols[i + 1:]:
        value = corr.loc[col_a, col_b]

        if pd.notna(value) and abs(value) >= 0.95:
            print(
                f"REVISAR | {col_a} / {col_b}: "
                "correlación absoluta >= 0.95"
            )
            found = True

if not found:
    print("No se detectan correlaciones >= 0.95")

# 6. Crear archivo de entrega con SOLO dos columnas
submission = pred[["frame", "label"]].copy()

submission_path = (
    ROOT / "artifacts/derived/tech_cia_submission.csv"
)

submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8",
    lineterminator="\n",
)

# Verificación posterior a la escritura
check = pd.read_csv(submission_path)

assert list(check.columns) == ["frame", "label"]
assert len(check) == EXPECTED_ROWS
assert check["frame"].iloc[0] == 0
assert check["frame"].iloc[-1] == EXPECTED_ROWS - 1
assert check["label"].notna().all()
assert check["label"].isin(["H", "A", "D", "X"]).all()

print("\nPASS | CSV oficial de dos columnas generado")
print("Archivo:", submission_path)
print("Filas:", len(check))
print("Columnas:", list(check.columns))
print("\nAUDITORÍA ESTRUCTURAL: OK")

