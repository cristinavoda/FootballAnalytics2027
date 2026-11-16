import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

V3 = os.path.join(
    ROOT, "artifacts", "derived",
    "tech_cia_features_v3.csv"
)

V4 = os.path.join(
    ROOT, "artifacts", "derived",
    "ball_intelligence_v4.csv"
)

RG = os.path.join(
    ROOT, "artifacts", "derived",
    "relational_geometry_v1.csv"
)

OUT = os.path.join(
    ROOT, "artifacts", "derived",
    "tech_cia_features_v4_rg.csv"
)

print("\n========== TECH-CIA V4 + RELATIONAL GEOMETRY ==========")

print("Loading V3...")
df_v3 = pd.read_csv(V3)

print("Loading V4...")
df_v4 = pd.read_csv(V4)

print("Loading relational geometry...")
df_rg = pd.read_csv(RG)

print(f"\nV3: {df_v3.shape}")
print(f"V4: {df_v4.shape}")
print(f"RG: {df_rg.shape}")

# Merge only by frame_id.
# No original dataframe is modified.

df = (
    df_v3
    .merge(
        df_v4,
        on="frame_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_v4")
    )
    .merge(
        df_rg,
        on="frame_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_rg")
    )
)

print(f"\nFINAL: {df.shape}")

df.to_csv(
    OUT,
    index=False
)

print(f"\nSaved:")
print(OUT)

print("\nComponents:")
print(f"  V3 features: {len(df_v3.columns)} columns")
print(f"  V4 features: {len(df_v4.columns) - 1} new columns")
print(f"  RG features: {len(df_rg.columns) - 1} new columns")
print(f"  Final:       {len(df.columns)} columns")

print("\nOriginal files untouched.")
