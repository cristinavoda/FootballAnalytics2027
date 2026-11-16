
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT = (
    ROOT
    / "artifacts"
    / "derived"
    / "radial_velocity_near_ball.csv"
)

OUTPUT = (
    ROOT
    / "artifacts"
    / "derived"
    / "toward_margin_dynamics.csv"
)

WINDOWS = {
    5: "0.2 s",
    10: "0.4 s",
    25: "1.0 s",
}

SOURCE_COLUMN = "toward_margin_10m"


# ============================================================
# CARGA Y VALIDACIÓN
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(f"No existe el archivo: {INPUT}")

df = pd.read_csv(INPUT)

required = {"frame_id", SOURCE_COLUMN}
missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Faltan columnas obligatorias: {sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError("El archivo de entrada está vacío.")

df["frame_id"] = pd.to_numeric(
    df["frame_id"], errors="coerce"
)

if df["frame_id"].isna().any():
    raise ValueError("Hay frame_id ausentes o no numéricos.")

if df["frame_id"].duplicated().any():
    raise ValueError("Hay frame_id duplicados.")

if not df["frame_id"].is_monotonic_increasing:
    raise ValueError("Los frame_id no están ordenados.")

if not np.all(df["frame_id"].diff().dropna() == 1):
    raise ValueError(
        "Se detectaron saltos en frame_id. "
        "No se calcularán deltas sobre una secuencia discontinua."
    )

margin = pd.to_numeric(
    df[SOURCE_COLUMN], errors="coerce"
)

# ============================================================
# CÁLCULO DE DELTAS RETROSPECTIVOS
# ============================================================

for frames, label in WINDOWS.items():

    previous = margin.shift(frames)

    # Diferencia retrospectiva:
    # M(t) - M(t-k)
    delta = margin - previous

    # Comprobación explícita de continuidad de los IDs.
    previous_frame = df["frame_id"].shift(frames)

    valid_frame_gap = (
        df["frame_id"] - previous_frame
    ) == frames

    # Ambos extremos deben tener valores válidos.
    valid_values = margin.notna() & previous.notna()

    valid = valid_frame_gap & valid_values

    delta = delta.where(valid)

    column = f"toward_margin_delta_{frames}f"

    df[column] = delta

    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    values = delta.dropna()

    print(f"\nVENTANA: {frames} frames ({label})")
    print("-" * 45)
    print("Valores válidos:", len(values))
    print(
        "Cobertura:",
        f"{len(values) / len(df) * 100:.2f}%"
    )

    if len(values):
        print("Media:", round(values.mean(), 4))
        print("Mediana:", round(values.median(), 4))
        print("Desviación estándar:", round(values.std(), 4))
        print("P10:", round(values.quantile(0.10), 4))
        print("P90:", round(values.quantile(0.90), 4))
        print("Media absoluta:", round(values.abs().mean(), 4))
        print("P90 absoluto:", round(values.abs().quantile(0.90), 4))

        positive = (values > 0).mean() * 100
        negative = (values < 0).mean() * 100

        print("Positivos:", f"{positive:.2f}%")
        print("Negativos:", f"{negative:.2f}%")

# ============================================================
# GUARDAR RESULTADO
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 55)
print("PROCESO COMPLETADO")
print("=" * 55)
print("Filas:", len(df))
print("Columnas:", len(df.columns))
print("Archivo:", OUTPUT)
print("Columnas nuevas:")

for frames in WINDOWS:
    print(f"  toward_margin_delta_{frames}f")
