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
# 4B. DINÁMICA DIRECCIONAL Y CONVERGENCIA
# =====================================================

print("Calculando coherencia direccional y convergencia...")

with np.load(TRACKING_FILE) as tracking:
    players = tracking["players"].copy()

positions = players[:, :, :2].astype(np.float32)

# Convertir coordenadas normalizadas a metros.
positions[:, :, 0] = (positions[:, :, 0] - 0.5) * 105.0
positions[:, :, 1] = (positions[:, :, 1] - 0.5) * 68.0

def collective_motion(indices):
    """
    Métricas colectivas por frame:
    - Coherencia direccional.
    - Convergencia radial hacia el centro del equipo.
    - Variación de dispersión.

    Requiere al menos 5 jugadores válidos.
    """

    pos = positions[:, indices, :]
    vx_team = vx[:, indices]
    vy_team = vy[:, indices]

    valid_pos = np.isfinite(pos).all(axis=2)
    valid_vel = (
        speed_valid[:, indices]
        & np.isfinite(vx_team)
        & np.isfinite(vy_team)
    )

    # Velocidad mínima para considerar una dirección significativa.
    moving = valid_vel & (speed[:, indices] > 0.1)

    coherence = np.full(n_frames, np.nan, dtype=np.float32)
    convergence = np.full(n_frames, np.nan, dtype=np.float32)
    spread = np.full(n_frames, np.nan, dtype=np.float32)

    for t in range(n_frames):
        # Coherencia direccional
        mask = moving[t]

        if mask.sum() >= 5:
            ux = vx_team[t, mask] / speed[t, indices][mask]
            uy = vy_team[t, mask] / speed[t, indices][mask]

            coherence[t] = np.hypot(ux.mean(), uy.mean())

        # Centro geométrico del equipo
        p_mask = valid_pos[t]

        if p_mask.sum() >= 5:
            current = pos[t, p_mask]
            center = current.mean(axis=0)

            distances = np.linalg.norm(current - center, axis=1)
            spread[t] = np.sqrt(np.mean(distances ** 2))

            # Convergencia radial:
            # velocidad proyectada hacia el centro del equipo.
            v_mask = valid_vel[t] & p_mask

            if v_mask.sum() >= 5:
                p = pos[t, v_mask]
                v = np.column_stack((
                    vx_team[t, v_mask],
                    vy_team[t, v_mask]
                ))

                radial = center - p
                radial_norm = np.linalg.norm(radial, axis=1)

                usable = radial_norm > 0.01

                if usable.sum() >= 5:
                    unit_radial = radial[usable] / radial_norm[usable, None]
                    convergence[t] = np.mean(
                        np.sum(v[usable] * unit_radial, axis=1)
                    )

    # Variación causal de dispersión: frame actual menos anterior.
    spread_delta = np.full(n_frames, np.nan, dtype=np.float32)


    consecutive = np.diff(frame_ids) == 1
    valid_pair = (
        np.isfinite(spread[1:])
        & np.isfinite(spread[:-1])
        & consecutive
    )
    DT = 1.0 / 25.0

    spread_delta[1:][valid_pair] = (
        spread[1:][valid_pair] - spread[:-1][valid_pair]
    ) / DT
   
    return {
        "direction_coherence": coherence,
        "team_convergence": convergence,
        "team_spread": spread,
        "team_spread_delta": spread_delta,
    }


barca_motion = collective_motion(BARCA)
athletic_motion = collective_motion(ATHLETIC)

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
