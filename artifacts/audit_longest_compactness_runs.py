
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "temporal_features_v1.csv"

FPS = 25

df = pd.read_csv(
    INPUT,
    usecols=["frame_id", "compactness_margin_m"]
).sort_values("frame_id").reset_index(drop=True)

runs = []
current_sign = None
start_frame = None
last_frame = None
length = 0

def save_run():
    if current_sign is not None:
        runs.append({
            "sign": current_sign,
            "start_frame": start_frame,
            "end_frame": last_frame,
            "frames": length,
            "duration_s": length / FPS
        })

for row in df.itertuples(index=False):
    frame = int(row.frame_id)
    value = row.compactness_margin_m

    if pd.isna(value):
        save_run()
        current_sign = None
        start_frame = None
        length = 0
        last_frame = frame
        continue

    sign = (
        "Positiva" if value > 0
        else "Negativa" if value < 0
        else "Cero"
    )

    consecutive = (
        last_frame is not None
        and frame == last_frame + 1
    )

    if not consecutive or sign != current_sign:
        save_run()
        current_sign = sign
        start_frame = frame
        length = 1
    else:
        length += 1

    last_frame = frame

save_run()

runs_df = pd.DataFrame(runs)

for sign in ["Positiva", "Negativa"]:
    print(f"\n=== 10 RACHAS MÁS LARGAS: {sign.upper()} ===")

    result = (
        runs_df[runs_df["sign"] == sign]
        .nlargest(10, "duration_s")
        .copy()
    )

    print(
        result[
            ["start_frame", "end_frame", "frames", "duration_s"]
        ].to_string(index=False)
    )

print("\n=== CONTROL DE EXTREMOS ===")
print(f"Primera racha: frame {runs_df.iloc[0]['start_frame']}")
print(f"Última racha: frame {runs_df.iloc[-1]['end_frame']}")
print(f"Total de rachas: {len(runs_df)}")
