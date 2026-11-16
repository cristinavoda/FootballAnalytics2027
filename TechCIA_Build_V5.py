from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

FPS = 25
N = 199319

BASE = Path("artifacts/derived")
XML = Path(
    r"E:\Hackaton 2027\data\tactical\BAR-ATH_tactical_smart_tagging.xml"
)

INPUT = BASE / "tech_cia_features_v4_rg2.csv"
OUTPUT = BASE / "tech_cia_features_v5.csv"

df = pd.read_csv(INPUT)

if len(df) != N:
    raise ValueError(f"Filas inesperadas: {len(df)}")

root = ET.parse(XML).getroot()

columns = [
    "st_side_central",
    "st_side_left",
    "st_side_right",
    "st_box_players_1",
    "st_box_players_2",
    "st_box_players_3",
    "st_box_players_4",
    "st_box_players_5",
    "st_box_players_6",
    "st_box_players_7plus",
    "st_direction_horizontal",
    "st_direction_vertical",
    "st_direction_diagonal",
    "st_goal_event_active",
    "st_goal_barcelona_active",
    "st_goal_athletic_active"
]

features = {
    col: np.zeros(N, dtype=np.uint8)
    for col in columns
}

dimension_map = {
    "Side": {
        "Central": "st_side_central",
        "Left": "st_side_left",
        "Right": "st_side_right"
    },
    "Max Players in the box": {
        "1": "st_box_players_1",
        "2": "st_box_players_2",
        "3": "st_box_players_3",
        "4": "st_box_players_4",
        "5": "st_box_players_5",
        "6": "st_box_players_6",
        "7+": "st_box_players_7plus"
    },
    "Direction of ball entry": {
        "Horizontal": "st_direction_horizontal",
        "Vertical": "st_direction_vertical",
        "Diagonal": "st_direction_diagonal"
    }
}

for inst in root.findall(".//instance"):
    code = inst.findtext("code")

    start = max(
        0,
        round(float(inst.findtext("start")) * FPS)
    )
    end = min(
        N,
        round(float(inst.findtext("end")) * FPS)
    )

    if end <= start:
        continue

    labels = {
        label.findtext("group"): label.findtext("text")
        for label in inst.findall("label")
    }

    # Etiquetas tácticas multivalor
    for dimension, mapping in dimension_map.items():
        value = labels.get(dimension)
        col = mapping.get(value)

        if col:
            features[col][start:end] = 1

    # Eventos de gol: contexto, no posesión
    if code == "GOALS":
        features["st_goal_event_active"][start:end] = 1

        team = labels.get("Team")

        if team == "Barcelona":
            features["st_goal_barcelona_active"][start:end] = 1

        elif team == "Atletic Club":
            features["st_goal_athletic_active"][start:end] = 1

# No sobrescribir silenciosamente columnas existentes
duplicates = set(features).intersection(df.columns)

if duplicates:
    raise ValueError(
        f"Columnas ya existentes: {sorted(duplicates)}"
    )

for col, values in features.items():
    df[col] = values

if df.shape != (N, 244):
    raise ValueError(f"Dimensiones inesperadas: {df.shape}")

df.to_csv(OUTPUT, index=False)

print("V5 CREADO CORRECTAMENTE")
print("Archivo:", OUTPUT)
print("Dimensiones:", df.shape)
print("Columnas originales:", 228)
print("Columnas nuevas:", len(features))

print("\nCOBERTURA DE LAS NUEVAS VARIABLES")
for col in columns:
    print(f"{col:35} {int(df[col].sum()):8}")

print("\nARCHIVO ORIGINAL PRESERVADO:", INPUT.exists())
