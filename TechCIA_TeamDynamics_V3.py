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
OUTPUT_FILE = OUTPUT_DIR / "team_dynamics_v3.csv"

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

    # Componentes de velocidad
    vx = dynamics["vx"].copy()
    vy = dynamics["vy"].copy()

    # Magnitud de velocidad y aceleración
    speed = dynamics["speed"].copy()
    acceleration = dynamics["acceleration"].copy()

    # Máscaras de validez
    velocity_valid = dynamics["velocity_valid"].copy()
    acceleration_valid = dynamics["acceleration_valid"].copy()

assert np.array_equal(frame_ids, dynamics_frame_ids)

# Alinear las transiciones de velocidad con los frames.
speed_valid = np.zeros_like(speed, dtype=bool)
speed_valid[1:] = velocity_valid

n_frames = len(frame_ids)

assert vx.shape == (n_frames, 46)
assert vy.shape == (n_frames, 46)
assert speed.shape == (n_frames, 46)
assert acceleration.shape == (n_frames, 46)
assert len(velocity_valid) == n_frames - 1
assert acceleration_valid.shape == (n_frames, 46)

print("Frames:", f"{n_frames:,}")
print("VX:", vx.shape)
print("VY:", vy.shape)
print("Carga completada correctamente.")


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
# =====================================================
# 4B. DIN?MICA DIRECCIONAL Y CONVERGENCIA ? V3
# =====================================================

print("Calculando coherencia direccional, convergencia y dispersi?n...")

with np.load(TRACKING_FILE) as tracking:
    players = tracking["players"].copy()

positions = players[:, :, :2].astype(np.float32)

# Coordenadas normalizadas -> metros.
positions[:, :, 0] = (positions[:, :, 0] - 0.5) * 105.0
positions[:, :, 1] = (positions[:, :, 1] - 0.5) * 68.0

DT = 1.0 / 25.0
MIN_PLAYERS = 5


def calculate_spread(points):
    """Dispersi?n RMS de un conjunto de posiciones, en metros."""
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return np.sqrt(np.mean(distances ** 2))


def collective_motion(indices):
    """
    M?tricas colectivas por frame:

    - direction_coherence: coherencia direccional.
    - team_convergence: velocidad radial hacia el centro.
    - team_spread: dispersi?n espacial en metros.
    - team_spread_delta: variaci?n de dispersi?n en m/s.

    La variaci?n de dispersi?n compara exactamente los mismos
    tracks en los dos frames consecutivos.
    """

    pos = positions[:, indices, :]
    vx_team = vx[:, indices]
    vy_team = vy[:, indices]
    speed_team = speed[:, indices]

    valid_pos = np.isfinite(pos).all(axis=2)

    valid_vel = (
        speed_valid[:, indices]
        & np.isfinite(vx_team)
        & np.isfinite(vy_team)
        & np.isfinite(speed_team)
    )

    moving = valid_vel & (speed_team > 0.1)

    coherence = np.full(n_frames, np.nan, dtype=np.float32)
    convergence = np.full(n_frames, np.nan, dtype=np.float32)
    spread = np.full(n_frames, np.nan, dtype=np.float32)
    spread_delta = np.full(n_frames, np.nan, dtype=np.float32)

    spread_delta_common = np.zeros(n_frames, dtype=np.int8)

    # -------------------------------------------------
    # A. M?TRICAS POR FRAME
    # -------------------------------------------------

    for t in range(n_frames):

        mask = moving[t]

        if mask.sum() >= MIN_PLAYERS:

            ux = vx_team[t, mask] / speed_team[t, mask]
            uy = vy_team[t, mask] / speed_team[t, mask]

            coherence[t] = np.hypot(
                ux.mean(),
                uy.mean()
            )

        p_mask = valid_pos[t]

        if p_mask.sum() >= MIN_PLAYERS:

            current = pos[t, p_mask]
            center = current.mean(axis=0)

            spread[t] = calculate_spread(current)

            v_mask = valid_vel[t] & p_mask

            if v_mask.sum() >= MIN_PLAYERS:

                p = pos[t, v_mask]

                v = np.column_stack((
                    vx_team[t, v_mask],
                    vy_team[t, v_mask]
                ))

                radial = center - p
                radial_norm = np.linalg.norm(radial, axis=1)

                usable = radial_norm > 0.01

                if usable.sum() >= MIN_PLAYERS:

                    unit_radial = (
                        radial[usable]
                        / radial_norm[usable, None]
                    )

                    convergence[t] = np.mean(
                        np.sum(v[usable] * unit_radial, axis=1)
                    )

    # -------------------------------------------------
    # B. VARIACI?N DE DISPERSI?N CORREGIDA
    # -------------------------------------------------

    for t in range(1, n_frames):

        if frame_ids[t] - frame_ids[t - 1] != 1:
            continue

        # Mismos tracks v?lidos en ambos frames.
        common = valid_pos[t] & valid_pos[t - 1]

        common_count = int(common.sum())
        spread_delta_common[t] = common_count

        if common_count < MIN_PLAYERS:
            continue

        previous_positions = pos[t - 1, common]
        current_positions = pos[t, common]

        previous_spread = calculate_spread(previous_positions)
        current_spread = calculate_spread(current_positions)

        spread_delta[t] = (
            current_spread - previous_spread
        ) / DT

    return {
        "direction_coherence": coherence,
        "team_convergence": convergence,
        "team_spread": spread,
        "team_spread_delta": spread_delta,
        "spread_delta_common": spread_delta_common,
    }


barca_motion = collective_motion(BARCA)
athletic_motion = collective_motion(ATHLETIC)

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

     
    # Dinámica colectiva - BARÇA
    "barca_direction_coherence":
        barca_motion["direction_coherence"],

    "barca_team_convergence":
        barca_motion["team_convergence"],

    "barca_team_spread":
        barca_motion["team_spread"],

    "barca_team_spread_delta":
        barca_motion["team_spread_delta"],

    # Dinámica colectiva - ATHLETIC
    "athletic_direction_coherence":
        athletic_motion["direction_coherence"],

    "athletic_team_convergence":
        athletic_motion["team_convergence"],

    "athletic_team_spread":
        athletic_motion["team_spread"],

    "athletic_team_spread_delta":
        athletic_motion["team_spread_delta"],

    # Comparativas entre equipos (Bar?a - Athletic)
    "direction_coherence_margin":
        barca_motion["direction_coherence"] - athletic_motion["direction_coherence"],

    "team_convergence_margin":
        barca_motion["team_convergence"] - athletic_motion["team_convergence"],

    "team_spread_margin":
        barca_motion["team_spread"] - athletic_motion["team_spread"],

   
})




# =====================================================
# DIAGNÓSTICO DE DINÁMICA COLECTIVA
# =====================================================

motion_columns = [
    "barca_direction_coherence",
    "barca_team_convergence",
    "barca_team_spread",
    "barca_team_spread_delta",
    "athletic_direction_coherence",
    "athletic_team_convergence",
    "athletic_team_spread",
    "athletic_team_spread_delta",
    "direction_coherence_margin",
    "team_convergence_margin",
    "team_spread_margin",
]

print("\n--- DINÁMICA COLECTIVA ---")

print(df[motion_columns].describe().round(3))

print("\n--- VALORES AUSENTES ---")

print(df[motion_columns].isna().sum())

print("\n--- COMPROBACIÓN DE COHERENCIA ---")

for column in [
    "barca_direction_coherence",
    "athletic_direction_coherence",
]:
    valid_values = df[column].dropna()

    print(
        column,
        "mínimo:", round(valid_values.min(), 3),
        "máximo:", round(valid_values.max(), 3),
    )

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
