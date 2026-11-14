
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "temporal_features_v1.csv"

FPS = 25

print(f"Leyendo: {INPUT}")

df = pd.read_csv(
    INPUT,
    usecols=["frame_id", "compactness_margin_m"]
)

df = df.sort_values("frame_id").reset_index(drop=True)

# Detectar rachas consecutivas sin atravesar datos ausentes
runs = []
current_sign = None
current_length = 0
previous_frame = None

def save_run(sign, length):
    if sign is not None and length > 0:
        runs.append({
            "sign": sign,
            "frames": length,
            "duration_s": length / FPS
        })

for row in df.itertuples(index=False):

    frame = int(row.frame_id)
    value = row.compactness_margin_m

    # Un dato ausente interrumpe la racha
    if pd.isna(value):
        save_run(current_sign, current_length)
        current_sign = None
        current_length = 0
        previous_frame = frame
        continue

    # Determinar el signo
    if value > 0:
        sign = "Positiva"
    elif value < 0:
        sign = "Negativa"
    else:
        sign = "Cero"

    # Un salto de frame también interrumpe la racha
    consecutive = (
        previous_frame is not None
        and frame == previous_frame + 1
    )

    if not consecutive or sign != current_sign:
        save_run(current_sign, current_length)
        current_sign = sign
        current_length = 1
    else:
        current_length += 1

    previous_frame = frame

# Guardar la última racha
save_run(current_sign, current_length)

runs_df = pd.DataFrame(runs)

print("\n=== DURACIÓN DE RACHAS ===")
print(f"Total de rachas: {len(runs_df):,}")

for sign in ["Positiva", "Negativa", "Cero"]:

    group = runs_df[runs_df["sign"] == sign]

    if group.empty:
        continue

    durations = group["duration_s"]

    print(f"\n--- {sign.upper()} ---")
    print(f"Número de rachas: {len(group):,}")
    print(f"Duración media: {durations.mean():.3f} s")
    print(f"Mediana: {durations.median():.3f} s")
    print(f"Q1 (25%): {durations.quantile(0.25):.3f} s")
    print(f"Q3 (75%): {durations.quantile(0.75):.3f} s")
    print(f"Máxima: {durations.max():.3f} s")

    print("\nPercentiles de duración:")
    print(
        durations.quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(3)
    )

print("\n=== CONTROL ===")
print(f"Frames originales: {len(df):,}")
print(f"Rachas positivas: {(runs_df['sign'] == 'Positiva').sum():,}")
print(f"Rachas negativas: {(runs_df['sign'] == 'Negativa').sum():,}")
print(f"Rachas de cero: {(runs_df['sign'] == 'Cero').sum():,}")
