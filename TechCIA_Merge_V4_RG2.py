import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts", "derived")

files = {
    "V3": os.path.join(ART, "tech_cia_features_v3.csv"),
    "V4": os.path.join(ART, "ball_intelligence_v4.csv"),
    "RG2": os.path.join(ART, "relational_geometry_v2.csv"),
}

print("Cargando características...")
v3 = pd.read_csv(files["V3"])
v4 = pd.read_csv(files["V4"])
rg = pd.read_csv(files["RG2"])

df = v3.merge(v4, on="frame_id", validate="one_to_one")
df = df.merge(rg, on="frame_id", validate="one_to_one")

assert len(df) == 199319
assert df["frame_id"].is_unique
assert df.shape[1] == 228

out = os.path.join(ART, "tech_cia_features_v4_rg2.csv")
df.to_csv(out, index=False)

print("\nINTEGRACIÓN COMPLETADA")
print("Dimensiones:", df.shape)
print("Archivo:", out)
print("Originales intactos.")
