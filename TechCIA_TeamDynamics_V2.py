from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# TECH-CIA V2 — DINÁMICA COLECTIVA
# =====================================================

ROOT = Path(__file__).resolve().parent

TRACKING_FILE = ROOT / "artifacts" / "tracking_atd.npz"
VELOCITY_FILE = ROOT / "artifacts" / "derived" / "velocities_raw.npz"

OUTPUT_DIR = ROOT / "artifacts" / "derived"
OUTPUT_FILE = OUTPUT_DIR / "team_dynamics_v2.csv"

# Índices Python: Track 1 = índice 0
BARCA = np.arange(0, 24)
ATHLETIC = np.arange(24, 46)

# =====================================================
# 1. CARGA
# =====================================================

print("Cargando tracking y velocidades...")

with np.load(TRACKING_FILE) as tracking:
    frame_ids = tracking["frame_ids"].copy()

with np.load(VELOCITY_FILE) as dynamics:
    dynamics_frame_ids = dynamics["frame_ids"].copy()
    speed = dynamics["speed"].copy()
    acceleration = dynamics["acceleration"].copy()
    velocity_valid = dynamics["velocity_valid"].copy()
    acceleration_valid = dynamics["acceleration_valid"].copy()

assert np.array_equal(frame_ids, dynamics_frame_ids)

# Alinear las transiciones de velocidad con los frames.
speed_valid = np.zeros_like(speed, dtype=bool)
speed_valid[1:] = velocity_valid

n_frames = len(frame_ids)

assert speed.shape == (n_frames, 46)
assert acceleration.shape == (n_frames, 46)
assert len(velocity_valid) == n_frames - 1
assert acceleration_valid.shape == (n_frames, 46)

print("Frames:", f"{n_frames:,}")

# =====================================================
# 2. FUNCIONES ESTADÍSTICAS
# =====================================================

def team_stats(values, indices, valid_mask=None):
    """
    Estadísticas por frame para un equipo.
    Los valores extremos no se eliminan.
    Se utiliza la mediana como medida robusta.
    """

    team = values[:, indices].copy()

    valid = np.isfinite(team)

    if valid_mask is not None:
        valid &= valid_mask[:, indices]

    clean = np.where(valid, team, np.nan)

    count = valid.sum(axis=1)

    mean = np.full(n_frames, np.nan, dtype=np.float32)
    median = np.full(n_frames, np.nan, dtype=np.float32)
    p90 = np.full(n_frames, np.nan, dtype=np.float32)

    for i in range(n_frames):
        row = clean[i]
        row = row[np.isfinite(row)]

        if len(row):
            mean[i] = np.mean(row)
            median[i] = np.median(row)
            p90[i] = np.percentile(row, 90)

    return {
        "mean": mean,
        "median": median,
        "p90": p90,
        "count": count.astype(np.int8),
    }


# =====================================================
# 3. VELOCIDAD COLECTIVA
# =====================================================

print("Calculando velocidades colectivas...")

barca_speed = team_stats(
    speed, BARCA, speed_valid
)

athletic_speed = team_stats(
    speed, ATHLETIC, speed_valid
)

# =====================================================
# 4. ACELERACIÓN COLECTIVA
# =====================================================

print("Calculando aceleraciones colectivas...")

barca_acc = team_stats(
    acceleration, BARCA, acceleration_valid
)

athletic_acc = team_stats(
    acceleration, ATHLETIC, acceleration_valid
)

# =====================================================
# 5. CONSTRUCCIÓN DEL DATAFRAME
# =====================================================

df = pd.DataFrame({
    "frame_id": frame_ids,

    "barca_speed_mean": barca_speed["mean"],
    "barca_speed_median": barca_speed["median"],
    "barca_speed_p90": barca_speed["p90"],
    "barca_speed_valid": barca_speed["count"],

    "athletic_speed_mean": athletic_speed["mean"],
    "athletic_speed_median": athletic_speed["median"],
    "athletic_speed_p90": athletic_speed["p90"],
    "athletic_speed_valid": athletic_speed["count"],

    "speed_margin_mean":
        barca_speed["mean"] - athletic_speed["mean"],

    "speed_margin_median":
        barca_speed["median"] - athletic_speed["median"],

    "barca_acc_mean": barca_acc["mean"],
    "barca_acc_median": barca_acc["median"],
    "barca_acc_p90": barca_acc["p90"],
    "barca_acc_valid": barca_acc["count"],

    "athletic_acc_mean": athletic_acc["mean"],
    "athletic_acc_median": athletic_acc["median"],
    "athletic_acc_p90": athletic_acc["p90"],
    "athletic_acc_valid": athletic_acc["count"],

    "acc_margin_mean":
        barca_acc["mean"] - athletic_acc["mean"],

    "acc_margin_median":
        barca_acc["median"] - athletic_acc["median"],
})

# =====================================================
# 6. VALIDACIÓN
# =====================================================

assert len(df) == 199319
assert df["frame_id"].is_unique
assert df["frame_id"].iloc[0] == 1
assert df["frame_id"].iloc[-1] == 199319

assert (
    df["barca_speed_valid"].between(0, 24).all()
)

assert (
    df["athletic_speed_valid"].between(0, 22).all()
)

# =====================================================
# 7. GUARDADO
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== TECH-CIA TEAM DYNAMICS V2 ==========")

print("Archivo:", OUTPUT_FILE)
print("Filas:", f"{len(df):,}")
print("Columnas:", len(df.columns))

print("\nTracks válidos por frame:")
print(
    df[
        [
            "barca_speed_valid",
            "athletic_speed_valid",
            "barca_acc_valid",
            "athletic_acc_valid",
        ]
    ].describe().round(2)
)

print("\nDistribución de diferencias de velocidad:")
print(df["speed_margin_mean"].describe().round(3))

print("\nDistribución de diferencias de aceleración:")
print(df["acc_margin_mean"].describe().round(3))

print("\nMódulo generado correctamente.")
