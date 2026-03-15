"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 1 — Create the "Opnew" Experiment                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS DOES:
  Creates an MLflow experiment named "Opnew" on the tracking server.
  An experiment is a container — it groups all runs, traces, datasets,
  and models for a project. Think of it like a folder.

  If the experiment already exists, MLflow reuses it (safe to re-run).

WHY IT MATTERS:
  Everything in Steps 2 and 3 writes into this experiment. Without it,
  runs would go to the "Default" experiment, which is messy.

WHAT TO CHECK IN MLFLOW UI AFTER RUNNING:
  1. Open the MLflow UI (Railway or localhost:5001)
  2. Left sidebar → "Opnew" should appear as an experiment
  3. No runs yet — just the empty experiment container

USAGE:
  python MLFLOW_STEPS/step1_create_experiment.py
  python MLFLOW_STEPS/step1_create_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make MLFLOW_STEPS/ importable ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_config import (
    EXPERIMENT_NAME,
    banner,
    info,
    resolve_tracking_uri,
    warn,
)


def step1_create_experiment(tracking_uri: str, dry_run: bool = False) -> str:
    """
    Create (or get) the "Opnew" experiment in MLflow.

    ┌─────────────────────────────────────────────────────────┐
    │ CONCEPT: MLflow Experiment                              │
    │                                                         │
    │ An experiment is the top-level container in MLflow.      │
    │ It groups:                                              │
    │   • Runs (individual evaluation executions)             │
    │   • Traces (request/response logs from your agent)      │
    │   • LoggedModels (GenAI app versions)                   │
    │   • Datasets (linked evaluation data)                   │
    │                                                         │
    │ mlflow.set_experiment("Opnew") either creates a new     │
    │ experiment or activates an existing one by that name.    │
    │ It returns an Experiment object with an experiment_id.   │
    └─────────────────────────────────────────────────────────┘

    Returns:
        The experiment_id (a string like "12").
    """
    import mlflow

    banner("STEP 1: Create MLflow Experiment")
    print(f"  Experiment name: {EXPERIMENT_NAME}")
    print(f"  Tracking URI:    {tracking_uri}")

    if dry_run:
        warn("DRY RUN — would create/get experiment, skipping.")
        return "<dry-run>"

    # ── Create or get the experiment ────────────────────────────────────
    # set_tracking_uri tells MLflow WHERE to store data.
    # set_experiment tells MLflow WHICH experiment to use.
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    info(f"Experiment ready: id={experiment_id}")
    print()
    print("  📋 WHAT TO CHECK IN MLFLOW UI:")
    print(f"     Open {tracking_uri}")
    print(f"     → Left sidebar should show experiment: \"{EXPERIMENT_NAME}\"")
    print(f"     → Experiment ID: {experiment_id}")
    print("     → No runs yet (we'll add those in Step 3)")

    return experiment_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Create Opnew experiment")
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 1 — Create Experiment                                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    tracking_uri = resolve_tracking_uri()
    experiment_id = step1_create_experiment(tracking_uri, dry_run=args.dry_run)

    banner("STEP 1 COMPLETE")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Tracking URI:  {tracking_uri}")


if __name__ == "__main__":
    main()
