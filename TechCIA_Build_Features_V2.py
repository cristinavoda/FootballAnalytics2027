from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

temporal_file = ROOT / "artifacts" / "temporal_features_v1.csv"
dynamics_file = ROOT / "artifacts" / "derived" / "team_dynamics_v2.csv"
output_file = ROOT / "artifacts" / "derived" / "tech_cia_features_v2.csv"

print("Cargando datasets...")
temporal = pd.read_csv(temporal_file)
dynamics = pd.read_csv(dynamics_file)

# Comprobaciones de integridad
assert temporal["frame_id"].is_unique
assert dynamics["frame_id"].is_unique
assert temporal["frame_id"].equals(dynamics["frame_id"]), \
    "Los frame_id no coinciden exactamente."

# Unión por frame_id
features = temporal.merge(
    dynamics,
    on="frame_id",
    how="inner",
    validate="one_to_one"
)

assert len(features) == len(temporal)
assert len(features) == len(dynamics)
assert features["frame_id"].is_unique

# Comprobación de valores infinitos
numeric = features.select_dtypes(include="number")
infinite_count = int(numeric.isin([float("inf"), float("-inf")]).sum().sum())

output_file.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(output_file, index=False)

print("\n========== TECH-CIA FEATURES V2 ==========")
print("Archivo:", output_file)
print("Frames:", len(features))
print("Columnas:", len(features.columns))
print("Valores infinitos:", infinite_count)
print("Valores ausentes:", int(features.isna().sum().sum()))
print("\nPrimeras columnas:")
print(features.columns.tolist()[:12])
print("\nIntegridad: OK")
