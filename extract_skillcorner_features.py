import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "external_data" / "skillcorner-opendata" / "data" / "matches"
OUTPUT_DIR = ROOT / "artifacts" / "skillcorner"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "skillcorner_features.csv.gz"

MATCH_IDS = [
    "1874553", "1886347", "1899585", "1925299", "1927964",
    "1953632", "1959846", "1986691", "1996435", "1996436",
    "2006229", "2006363", "2007448", "2007721", "2010085",
    "2011166", "2013725", "2015213", "2016236", "2017461"
]

FIELDS = [
    "match_id", "frame", "timestamp", "period", "label",
    "ball_x", "ball_y", "ball_z", "ball_detected",
    "players_total", "players_detected",
    "home_count", "home_detected",
    "away_count", "away_detected",
    "home_centroid_x", "home_centroid_y",
    "away_centroid_x", "away_centroid_y",
    "home_spread_x", "home_spread_y",
    "away_spread_x", "away_spread_y",
    "home_centroid_ball_dist", "away_centroid_ball_dist",
    "home_min_ball_dist", "away_min_ball_dist",
    "home_mean_ball_dist", "away_mean_ball_dist",
    "home_near_2m", "home_near_5m", "home_near_10m",
    "away_near_2m", "away_near_5m", "away_near_10m"
]


def number(value):
    try:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    return sum(values) / len(values) if values else None


def spread(values):
    if len(values) < 2:
        return None
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def team_features(players, ball_x, ball_y):
    xs = [p[0] for p in players]
    ys = [p[1] for p in players]

    distances = []
    if ball_x is not None and ball_y is not None:
        distances = [
            math.hypot(x - ball_x, y - ball_y)
            for x, y in players
        ]

    return {
        "count": len(players),
        "centroid_x": mean(xs),
        "centroid_y": mean(ys),
        "spread_x": spread(xs),
        "spread_y": spread(ys),
        "centroid_ball_dist": (
            math.hypot(mean(xs) - ball_x, mean(ys) - ball_y)
            if players and ball_x is not None and ball_y is not None
            else None
        ),
        "min_ball_dist": min(distances) if distances else None,
        "mean_ball_dist": mean(distances),
        "near_2m": sum(d <= 2 for d in distances),
        "near_5m": sum(d <= 5 for d in distances),
        "near_10m": sum(d <= 10 for d in distances),
    }


# --------------------------------------------------
# EXTRACCIÓN
# --------------------------------------------------

global_labels = Counter()
total_frames = 0
unknown_players = 0
matches_processed = 0

with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()

    for match_id in MATCH_IDS:
        match_dir = DATA / match_id
        metadata_file = match_dir / f"{match_id}_match.json"
        tracking_file = match_dir / f"{match_id}_tracking_extrapolated.jsonl"

        if not metadata_file.exists() or not tracking_file.exists():
            print(f"\nOMITIDO: {match_id} - falta algún archivo")
            continue

        with metadata_file.open(encoding="utf-8") as f:
            metadata = json.load(f)

        home_id = metadata["home_team"]["id"]
        away_id = metadata["away_team"]["id"]

        # player_data.player_id coincide con players[].id
        player_to_team = {
            p["id"]: p["team_id"]
            for p in metadata["players"]
            if p.get("id") is not None
        }

        match_labels = Counter()
        match_frames = 0
        match_unknown_players = set()

        print(f"\nProcesando {match_id}: "
              f"{metadata['home_team']['short_name']} - "
              f"{metadata['away_team']['short_name']}")

        with tracking_file.open(encoding="utf-8") as tracking:
            for line in tracking:
                if not line.strip():
                    continue

                frame_data = json.loads(line)
                possession = frame_data.get("possession") or {}
                group = possession.get("group")

                if group == "home team":
                    label = "H"
                elif group == "away team":
                    label = "A"
                else:
                    label = "UNKNOWN"

                ball = frame_data.get("ball_data") or {}
                ball_x = number(ball.get("x"))
                ball_y = number(ball.get("y"))
                ball_z = number(ball.get("z"))

                ball_detected = (
                    ball.get("is_detected") is True
                    and ball_x is not None
                    and ball_y is not None
                )

                # Si el balón no está detectado, no calculamos distancias.
                if not ball_detected:
                    ball_x = None
                    ball_y = None

                home_all = 0
                away_all = 0
                home_positions = []
                away_positions = []
                players_total = 0
                players_detected = 0

                for player in frame_data.get("player_data", []):
                    player_id = player.get("player_id")
                    team_id = player_to_team.get(player_id)

                    if team_id is None:
                        if player_id is not None:
                            match_unknown_players.add(player_id)
                        continue

                    players_total += 1

                    if team_id == home_id:
                        home_all += 1
                    elif team_id == away_id:
                        away_all += 1
                    else:
                        continue

                    x = number(player.get("x"))
                    y = number(player.get("y"))
                    detected = player.get("is_detected") is True

                    if not detected or x is None or y is None:
                        continue

                    players_detected += 1

                    if team_id == home_id:
                        home_positions.append((x, y))
                    elif team_id == away_id:
                        away_positions.append((x, y))

                hf = team_features(home_positions, ball_x, ball_y)
                af = team_features(away_positions, ball_x, ball_y)

                row = {
                    "match_id": match_id,
                    "frame": frame_data.get("frame"),
                    "timestamp": frame_data.get("timestamp"),
                    "period": frame_data.get("period"),
                    "label": label,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "ball_z": ball_z,
                    "ball_detected": int(ball_detected),
                    "players_total": players_total,
                    "players_detected": players_detected,
                    "home_count": home_all,
                    "home_detected": hf["count"],
                    "away_count": away_all,
                    "away_detected": af["count"],
                }

                for prefix, features in [("home", hf), ("away", af)]:
                    for key, value in features.items():
                        row[f"{prefix}_{key}"] = value

                writer.writerow(row)

                match_frames += 1
                match_labels[label] += 1

        total_frames += match_frames
        global_labels.update(match_labels)
        unknown_players += len(match_unknown_players)
        matches_processed += 1

        print(f"  Frames: {match_frames:,}")
        print(f"  H: {match_labels['H']:,} | "
              f"A: {match_labels['A']:,} | "
              f"UNKNOWN: {match_labels['UNKNOWN']:,}")
        print(f"  IDs de jugadores sin correspondencia: "
              f"{len(match_unknown_players)}")

print("\n" + "=" * 55)
print("EXTRACCIÓN TERMINADA")
print("=" * 55)
print("Partidos procesados:", matches_processed)
print("Frames totales:", f"{total_frames:,}")
print("Etiquetas:", dict(global_labels))
print("IDs desconocidos acumulados por partido:", unknown_players)
print("Archivo:", OUTPUT_FILE)
print("Tamaño MB:", round(OUTPUT_FILE.stat().st_size / 1024**2, 2))
