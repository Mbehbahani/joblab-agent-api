#!/usr/bin/env python3
"""
Migrate all local MLflow runs to DagsHub remote tracking server.

Reads every non-default experiment + run from the local backend
(sqlite:///mlflow.db by default) and re-logs params, metrics, tags,
and artifact files to the DagsHub MLflow server.

Usage (from lambda_backend/):

    # Load your DagsHub credentials first
    $env:DAGSHUB_USER_TOKEN = "<your_token>"

    python evals/migrate_local_to_dagshub.py

    # Point to a different local backend:
    python evals/migrate_local_to_dagshub.py --local-uri sqlite:///mlflow.db

    # Dry-run (prints what would be migrated, does nothing):
    python evals/migrate_local_to_dagshub.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dagshub() -> None:
    try:
        import dagshub as _dh
    except ImportError:
        sys.exit("dagshub not installed.  Run: pip install -r evals/requirements.txt")
    _dh.init(repo_owner="Mbehbahani", repo_name="joblab-mlflow", mlflow=True)
    print("DagsHub remote tracking URI set.")


def migrate(local_uri: str, run_id: str | None, dry_run: bool) -> None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        sys.exit("mlflow not installed.  Run: pip install -r evals/requirements.txt")

    # ── source: local backend ──────────────────────────────────────────────────
    src = MlflowClient(tracking_uri=local_uri)
    runs_by_exp: dict[str, list] = {}
    experiments = []

    if run_id:
        # Single-run migration mode
        try:
            run_to_migrate = src.get_run(run_id)
            exp = src.get_experiment(run_to_migrate.info.experiment_id)
            experiments.append(exp)
            runs_by_exp[exp.experiment_id] = [run_to_migrate]
            print(f"Target: run '{run_to_migrate.info.run_name}' ({run_id}) in experiment '{exp.name}'")
        except Exception as exc:
            sys.exit(f"Error: Could not find run with id '{run_id}' in local store. Details: {exc}")
    else:
        # Bulk-migration mode (original behavior)
        experiments = [e for e in src.search_experiments() if e.name != "Default"]
        if not experiments:
            print("No experiments found in local store. Nothing to migrate.")
            return

        print(f"\nFound {len(experiments)} experiment(s) in local store ({local_uri}):")
        for e in experiments:
            runs = src.search_runs([e.experiment_id])
            runs_by_exp[e.experiment_id] = runs
            print(f"  [{e.experiment_id}] {e.name}  ({len(runs)} run(s))")


    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    # ── destination: DagsHub ───────────────────────────────────────────────────
    _load_dagshub()

    import mlflow as remote_mlflow  # tracking URI is now DagsHub after dagshub.init()
    dst = MlflowClient()            # uses the currently set tracking URI

    migrated_runs = 0
    skipped_runs = 0

    for exp in experiments:
        # Create or reuse the experiment on DagsHub
        remote_exp = dst.get_experiment_by_name(exp.name)
        if remote_exp is None:
            remote_exp_id = dst.create_experiment(exp.name)
            print(f"\n[NEW] Created experiment '{exp.name}' on DagsHub (id={remote_exp_id})")
        else:
            remote_exp_id = remote_exp.experiment_id
            print(f"\n[OK] Experiment '{exp.name}' already exists on DagsHub (id={remote_exp_id})")

        runs = runs_by_exp.get(exp.experiment_id, [])
        for run in runs:
            ri = run.info
            # Skip if a run with the same name already exists on DagsHub
            existing = dst.search_runs(
                experiment_ids=[remote_exp_id],
                filter_string=f"tags.`migrated_from_run_id` = '{ri.run_id}'",
            )
            if existing:
                print(f"  [SKIP] {ri.run_name} ({ri.run_id}) — already migrated")
                skipped_runs += 1
                continue

            print(f"  [MIGRATING] {ri.run_name} ({ri.run_id})")

            with remote_mlflow.start_run(
                experiment_id=remote_exp_id,
                run_name=ri.run_name,
            ) as new_run:
                # params
                if run.data.params:
                    remote_mlflow.log_params(run.data.params)

                # metrics (single-value; final snapshot from local store)
                if run.data.metrics:
                    remote_mlflow.log_metrics(run.data.metrics)

                # tags (skip internal mlflow. prefixed tags)
                user_tags = {
                    k: v for k, v in run.data.tags.items()
                    if not k.startswith("mlflow.")
                }
                user_tags["migrated_from_run_id"] = ri.run_id
                user_tags["migrated_from_uri"] = local_uri
                if ri.run_name:
                    user_tags["mlflow.runName"] = ri.run_name
                remote_mlflow.set_tags(user_tags)

                # artifacts
                local_artifact_dir = Path(ri.artifact_uri.replace("file:", ""))
                if local_artifact_dir.exists():
                    artifact_files = list(local_artifact_dir.rglob("*"))
                    artifact_files = [f for f in artifact_files if f.is_file()]
                    if artifact_files:
                        print(f"    Uploading {len(artifact_files)} artifact(s)…")
                        for af in artifact_files:
                            relative = af.relative_to(local_artifact_dir)
                            artifact_path = str(relative.parent) if str(relative.parent) != "." else None
                            remote_mlflow.log_artifact(str(af), artifact_path=artifact_path)
                    else:
                        print("    No artifact files found.")

            print(f"    → DagsHub run_id: {new_run.info.run_id}")
            migrated_runs += 1

    print(f"\nDone.  Migrated: {migrated_runs}  Skipped (already exist): {skipped_runs}")
    print(f"View at: https://dagshub.com/Mbehbahani/joblab-mlflow/experiments")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate local MLflow runs to DagsHub. Can do full bulk migration or a single run."
    )
    parser.add_argument(
        "--local-uri",
        default=os.getenv("LOCAL_MLFLOW_URI", "sqlite:///mlflow.db"),
        help="Local MLflow tracking URI to read from (default: sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="ID of a single run to migrate. If omitted, all runs from all experiments are migrated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without writing anything.",
    )
    args = parser.parse_args()
    migrate(args.local_uri, args.run_id, args.dry_run)


if __name__ == "__main__":
    main()
