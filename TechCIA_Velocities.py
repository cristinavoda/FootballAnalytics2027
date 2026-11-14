
from pathlib import Path
import numpy as np

# =====================================================
# CONFIGURACIÓN
# =====================================================

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "artifacts" / "tracking_atd.npz"
OUTPUT_DIR = ROOT / "artifacts" / "derived"

FPS = 25
DT = 1 / FPS

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# =====================================================
# 1. CARGA Y VALIDACIÓN
# =====================================================

print("Cargando tracking...")

if not SOURCE.exists():
    raise FileNotFoundError(f"No existe el archivo: {SOURCE}")

data = np.load(SOURCE, allow_pickle=False)

frame_ids = data["frame_ids"]
players = data["players"]

n_frames, n_players, n_channels = players.shape

if n_channels < 2:
    raise ValueError("Se necesitan al menos dos canales de posición.")

if len(frame_ids) != n_frames:
    raise ValueError("La cantidad de frame_ids no coincide con players.")

if not np.isfinite(players[:, :, :2]).any():
    raise ValueError("No existen coordenadas válidas.")

print(f"Frames: {n_frames:,}")
print(f"Tracks: {n_players}")
print(f"Frecuencia: {FPS} FPS")

# =====================================================
# 2. POSICIONES VÁLIDAS
# =====================================================

positions = players[:, :, :2]

position_valid = np.isfinite(positions).all(axis=2)

# =====================================================
# 3. VELOCIDADES DERIVADAS
# =====================================================

# Diferencias entre frames consecutivos.
delta_x = np.diff(positions[:, :, 0], axis=0) * FIELD_LENGTH
delta_y = np.diff(positions[:, :, 1], axis=0) * FIELD_WIDTH

# Una transición solo es válida si ambos extremos
# tienen coordenadas válidas y los frames son consecutivos.
consecutive_frames = np.diff(frame_ids) == 1

velocity_valid = (
    position_valid[:-1]
    & position_valid[1:]
    & consecutive_frames[:, None]
)

vx = np.full((n_frames, n_players), np.nan, dtype=np.float32)
vy = np.full((n_frames, n_players), np.nan, dtype=np.float32)
speed = np.full((n_frames, n_players), np.nan, dtype=np.float32)

vx[1:] = np.where(
    velocity_valid,
    delta_x / DT,
    np.nan
)

vy[1:] = np.where(
    velocity_valid,
    delta_y / DT,
    np.nan
)

speed[1:] = np.hypot(vx[1:], vy[1:])

# =====================================================
# 4. VELOCIDAD ORIGINAL DEL PROVEEDOR
# =====================================================

provider_speed = players[:, :, 2].copy()



# =====================================================
# 5. ACELERACIÓN VECTORIAL
# =====================================================

ax = np.full((n_frames, n_players), np.nan, dtype=np.float32)
ay = np.full((n_frames, n_players), np.nan, dtype=np.float32)
acceleration = np.full(
    (n_frames, n_players),
    np.nan,
    dtype=np.float32
)

# Necesitamos dos transiciones consecutivas válidas:
# tres posiciones consecutivas del mismo track.

acceleration_valid = np.zeros(
    (n_frames, n_players),
    dtype=bool
)

acceleration_valid[2:] = (
    velocity_valid[1:]
    & velocity_valid[:-1]
)

ax[2:] = np.where(
    acceleration_valid[2:],
    (vx[2:] - vx[1:-1]) / DT,
    np.nan
)

ay[2:] = np.where(
    acceleration_valid[2:],
    (vy[2:] - vy[1:-1]) / DT,
    np.nan
)

acceleration[2:] = np.hypot(ax[2:], ay[2:])

# =====================================================
# 6. DIAGNÓSTICO DE ACELERACIÓN
# =====================================================

valid_acceleration = acceleration[
    np.isfinite(acceleration)
]

print("\nDistribución de aceleraciones:")
print("Aceleraciones calculadas:", len(valid_acceleration))

if len(valid_acceleration):
    for percentile in [50, 75, 90, 95, 99, 99.5, 99.9]:
        value = np.percentile(
            valid_acceleration,
            percentile
        )
        print(f"P{percentile}: {value:.3f} m/s²")

# =====================================================
# 7. DIAGNÓSTICO INICIAL
# =====================================================

valid_speeds = speed[np.isfinite(speed)]

print("\n========== RESULTADOS ==========")

print("Posiciones válidas:", f"{position_valid.sum():,}")
print("Transiciones válidas:", f"{velocity_valid.sum():,}")
print("Velocidades calculadas:", f"{len(valid_speeds):,}")

if len(valid_speeds):
    print("\nDistribución de velocidades derivadas:")
    for percentile in [50, 75, 90, 95, 99, 99.5, 99.9]:
        value = np.percentile(valid_speeds, percentile)
        print(f"P{percentile}: {value:.3f} m/s")

# =====================================================
# 8. GUARDADO DE RESULTADOS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "velocities_raw.npz"

np.savez_compressed(
    output_file,
    frame_ids=frame_ids,
    vx=vx,
    ax=ax,
ay=ay,
acceleration=acceleration,
acceleration_valid=acceleration_valid,
    vy=vy,
    speed=speed,
    position_valid=position_valid,
    velocity_valid=velocity_valid,
    provider_speed=provider_speed
)

print("\nArchivo generado:")
print(output_file)
print("Tamaño:", round(output_file.stat().st_size / 1024 / 1024, 2), "MB")
print("\nCálculo de velocidades terminado.")
