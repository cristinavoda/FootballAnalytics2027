
"""
TECH-CIA
More Than A Hack 2027 — Metrica Sports

Main entry point for the team possession estimation pipeline.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("TECH-CIA | Football Analytics 2027")
    print("Team Possession Estimation")
    print("=" * 60)

    print(f"\nProject directory: {PROJECT_ROOT}")
    print("\nPipeline status: under final integration.")
    print("Competition tracking data is not included in this repository.")


if __name__ == "__main__":
    main()
