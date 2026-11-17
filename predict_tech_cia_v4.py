
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PC_PATH = ROOT / "artifacts" / "derived" / "pitch_control_features.npz"
V3_PATH = ROOT / "artifacts" / "derived" / "predictions_tech_cia_v3.csv"
OUT_PATH = ROOT / "artifacts" / "derived" / "predictions_tech_cia_v4.csv"
SUB_PATH = ROOT / "artifacts" / "derived" / "tech_cia_submission_v4.csv"

RADIUS_M = 10.0
V3_WEIGHT = 0.90
RADIAL_WEIGHT = 0.10
THRESHOLD = 0.12
ROLLING_WINDOW = 5


def robust_signal(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    result = np.full(len(values), np.nan)
    if not valid.any():
        return result

    x = values[valid]
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-9:
        scale = np.std(x)

    if not np.isfinite(scale) or scale < 1e-9:
        result[valid] = 0.0
        return result

    z = np.clip((x - median) / scale, -3, 3)
    result[valid] = np.tanh(z / 2)
    return result


def rolling_median_by_segments(frames, values, window=5):
    result = np.full(len(values), np.nan)
    start = 0

    for i in range(1, len(frames) + 1):
        boundary = (
            i == len(frames)
            or frames[i] - frames[i - 1] != 1
        )

        if boundary:
            segment = pd.Series(values[start:i])
            result[start:i] = (
                segment.rolling(window, min_periods=1).median().to_numpy()
            )
            start = i

    return result


def main():
    v3 = pd.read_csv(V3_PATH)
    pc = np.load(PC_PATH, allow_pickle=False)

    required = {
        "frame", "label", "score_raw"
    }
    if not required.issubset(v3.columns):
        raise ValueError("Faltan columnas necesarias en V3.")

    frames = v3["frame"].to_numpy(dtype=np.int64)
    npz_frames = pc["frame_ids"].astype(np.int64) - 1

    if len(frames) != len(npz_frames):
        raise ValueError("Las dimensiones de V3 y NPZ no coinciden.")

    if not np.array_equal(frames, npz_frames):
        raise ValueError("Los identificadores de frame no coinciden.")

    distance = pc["distance_to_ball"]
    distance_valid = pc["distance_valid"]
    radial = pc["velocity_toward_ball"]
    kinematic_valid = pc["kinematic_valid"]
    ball_valid = pc["ball_valid"]

    def team_median(slots):
        d = distance[:, slots]
        v = radial[:, slots]

        valid = (
            distance_valid[:, slots]
            & kinematic_valid[:, slots]
            & np.isfinite(d)
            & np.isfinite(v)
            & ball_valid[:, None]
            & (d < RADIUS_M)
        )

        values = np.where(valid, v, np.nan)
        counts = valid.sum(axis=1)

        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=1)

        medians[counts == 0] = np.nan
        return medians, counts

    barca, barca_n = team_median(slice(0, 24))
    athletic, athletic_n = team_median(slice(24, 46))

    radial_margin = barca - athletic
    radial_signal = robust_signal(radial_margin)

    v3_raw = v3["score_raw"].to_numpy(dtype=float)

    # Si la señal radial no está disponible, se conserva V3.
    v4_raw = np.where(
        np.isfinite(radial_signal),
        V3_WEIGHT * v3_raw + RADIAL_WEIGHT * radial_signal,
        v3_raw
    )

    v4_score = rolling_median_by_segments(
        frames, v4_raw, ROLLING_WINDOW
    )

    labels = np.full(len(frames), "X", dtype="<U1")
    labels[v4_score > THRESHOLD] = "H"
    labels[v4_score < -THRESHOLD] = "A"

    output = v3.copy()
    output["radial_signal"] = radial_signal
    output["radial_margin"] = radial_margin
    output["barca_radial_count"] = barca_n
    output["athletic_radial_count"] = athletic_n
    output["score_raw"] = v4_raw
    output["score"] = v4_score
    output["label"] = labels

    output.to_csv(OUT_PATH, index=False)

    submission = pd.DataFrame({
        "frame": frames,
        "label": labels
    })
    submission.to_csv(SUB_PATH, index=False)

    assert len(output) == len(v3)
    assert np.isfinite(v4_score).all()
    assert set(np.unique(labels)).issubset({"H", "A", "D", "X"})

    print("V4 generada correctamente.")
    print("Predicciones:", OUT_PATH)


    
    print("Submission:", SUB_PATH)
    print("Filas:", len(output))
    print("Columnas:", output.columns.tolist())


if __name__ == "__main__":
    main()
