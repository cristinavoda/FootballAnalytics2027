from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

temporal_file = ROOT / "artifacts" / "temporal_features_v1.csv"
dynamics_file = ROOT / "artifacts" / "derived" / "team_dynamics_v2.csv"
toward_file = ROOT / "artifacts" / "derived" / "toward_margin_dynamics.csv"

output_file = ROOT / "artifacts" / "derived" / "tech_cia_features_v2.csv"

print("Cargando datasets...")

temporal = pd.read_csv(temporal_file)
dynamics = pd.read_csv(dynamics_file)
toward = pd.read_csv(toward_file)

# ============================================================
# COMPROBACIONES DE INTEGRIDAD
# ============================================================

assert temporal["frame_id"].is_unique
assert dynamics["frame_id"].is_unique
assert toward["frame_id"].is_unique

assert temporal["frame_id"].equals(dynamics["frame_id"]), \
    "Los frame_id de temporal y dynamics no coinciden exactamente."

assert temporal["frame_id"].equals(toward["frame_id"]), \
    "Los frame_id de temporal y toward dynamics no coinciden exactamente."

# ============================================================
# UNIÓN DE FEATURES
# ============================================================

features = temporal.merge(
    dynamics,
    on="frame_id",
    how="inner",
    validate="one_to_one"
)

features = features.merge(
    toward,
    on="frame_id",
    how="inner",
    validate="one_to_one"
)

# ============================================================
# COMPROBACIONES FINALES
# ============================================================

assert len(features) == len(temporal)
assert len(features) == len(dynamics)
assert len(features) == len(toward)

assert features["frame_id"].is_unique

# Comprobar que no se ha perdido ningún frame
assert features["frame_id"].equals(temporal["frame_id"])

# Comprobación de valores infinitos
numeric = features.select_dtypes(include="number")

infinite_count = int(
    numeric.isin([float("inf"), float("-inf")]).sum().sum()
)

# ============================================================
# GUARDAR
# ============================================================

output_file.parent.mkdir(parents=True, exist_ok=True)

features.to_csv(output_file, index=False)

# ============================================================
# INFORME
# ============================================================

print("\n========== TECH-CIA FEATURES V2 ==========")
print("Archivo:", output_file)
print("Frames:", len(features))
print("Columnas:", len(features.columns))
print("Valores infinitos:", infinite_count)
print("Valores ausentes:", int(features.isna().sum().sum()))

print("\nNuevas columnas de movimiento hacia el balón:")

toward_columns = [
    "barca_toward_10m",
    "athletic_toward_10m",
    "toward_margin_10m",
    "toward_margin_delta_5f",
    "toward_margin_delta_10f",
    "toward_margin_delta_25f",
]

for column in toward_columns:
    assert column in features.columns
    print(f"  {column}")

print("\nPrimeras columnas:")
print(features.columns.tolist()[:12])

print("\nIntegridad: OK")