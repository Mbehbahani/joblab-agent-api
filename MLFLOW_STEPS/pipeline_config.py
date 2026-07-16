"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Shared Configuration — Opnew Learning Pipeline                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module contains all constants and utility functions shared by:
  - step1_create_experiment.py
  - step2_register_assets.py
  - step3_run_evaluation.py

WHAT'S HERE:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ Constants                                                                  │
  │   EXPERIMENT_NAME     "Opnew" — the MLflow experiment all steps write to  │
  │   RAILWAY_URI         Railway cloud MLflow server (primary)               │
  │   LOCAL_URI           localhost:5001 fallback (same Postgres+S3 backend)  │
  │   BACKEND_URL         FastAPI agent at localhost:8000                     │
  │   PROMPT_NAME         "joblab-system-prompt" — prompt registry key        │
  │   DATASET_NAME        "opnow-baseline" — 10-case eval dataset            │
  │   JUDGE_MODEL         AI Gateway endpoint for the judge scorer            │
  │   AGENT_MODEL_NAME    "joblab-agent" — LoggedModel for your GenAI agent  │
  │   JUDGE_MODEL_NAME    "myendpoint-judge" — LoggedModel for the judge     │
  │                                                                            │
  │ Utilities                                                                  │
  │   resolve_tracking_uri()  Try Railway → localhost fallback                │
  │   check_backend()         Verify localhost:8000 is up                     │
  │   banner/info/warn         Pretty-print helpers                           │
  └────────────────────────────────────────────────────────────────────────────┘

WHY A SEPARATE CONFIG FILE:
  So you can change a value once (e.g. experiment name, model ID) and all
  3 step files pick it up automatically. No copy-paste duplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Project root (parent of MLFLOW_STEPS/) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# The experiment name for this learning session.
# All runs, traces, and models will live under this experiment.
EXPERIMENT_NAME = "Opnew"

# MLflow server candidates — resolve_tracking_uri() tries Railway first.
RAILWAY_URI = "https://mlflow-production-34b0.up.railway.app"
LOCAL_URI = "http://localhost:5001"

# Backend (your FastAPI app) — the agent we're evaluating.
BACKEND_URL = "http://localhost:8000"

# ── Prompt ─────────────────────────────────────────────────────────────────
PROMPT_NAME = "joblab-system-prompt"

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET_NAME = "opnow-baseline"
DATASET_SOURCE = PROJECT_ROOT / "evals" / "optimization_baseline_dataset.json"

# ── LLM (Agent) ───────────────────────────────────────────────────────────
# This is the model your agent uses for conversation.
LLM_MODEL_NAME = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# ── LLM Judge ─────────────────────────────────────────────────────────────
# Route the judge through MLflow AI Gateway so judge usage is traced and
# attributed to the gateway endpoint instead of bypassing it with a direct
# provider URI.
JUDGE_MODEL = "gateway:/myendpoint"

# GEPA uses a separate reflection model. Route that through AI Gateway too so
# optimizer LLM traffic is attributable to the gateway endpoint.
# NOTE: claude-3-5-sonnet and claude-3-7-sonnet are both marked Legacy by AWS
# Bedrock (access denied if unused for 15 days). The gateway endpoint named
# "claude-3-5-sonnet" must be updated in MLflow UI to use the model ID:
#   us.anthropic.claude-sonnet-4-20250514-v1:0
OPTIMIZER_MODEL = "gateway:/claude-3-5-sonnet"
OPTIMIZER_MODEL_NAME = "claude-sonnet-4-20250514-v1:0"
OPTIMIZER_MAX_METRIC_CALLS = 40
OPTIMIZATION_CANDIDATE_ALIAS = "candidate"

# ── LoggedModel names ─────────────────────────────────────────────────────
# MLflow 3.x "LoggedModel" links traces to a named GenAI app version.
# This is what shows up in the Model tab of the MLflow UI.
#
# AGENT_MODEL_NAME: Your Joblab agent (the thing being evaluated)
# JUDGE_MODEL_NAME: The Correctness judge (so you can track judge cost too)
AGENT_MODEL_NAME = "joblab-agent"
JUDGE_MODEL_NAME = "myendpoint-judge"


# ═════════════════════════════════════════════════════════════════════════════
#  PRINT UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def banner(title: str, char: str = "━") -> None:
    """Print a visible section banner."""
    width = 70
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def info(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def detail(msg: str) -> None:
    print(f"    → {msg}")


# ═════════════════════════════════════════════════════════════════════════════
#  SERVER UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def resolve_tracking_uri() -> str:
    """
    Try Railway first. If unreachable, fall back to localhost:5001.

    WHY: Both servers point to the same Postgres DB and S3 bucket.
    Railway is the cloud server; localhost:5001 runs the same MLflow
    against the same backend (started via scripts/mlflow-local.ps1).

    WHAT IT DOES:
      1. GET {RAILWAY_URI}/health — if 200, use Railway
      2. GET {LOCAL_URI}/health   — if 200, use localhost
      3. If both fail, exit with an error message
    """
    import requests

    banner("Resolving MLflow Tracking Server")
    for uri in [RAILWAY_URI, LOCAL_URI]:
        try:
            resp = requests.get(f"{uri}/health", timeout=5)
            if resp.status_code == 200:
                info(f"Connected to MLflow at: {uri}")
                return uri
        except Exception:
            warn(f"Could not reach: {uri}")

    print()
    print("  ERROR: No MLflow server reachable!")
    print(f"    Tried: {RAILWAY_URI}")
    print(f"    Tried: {LOCAL_URI}")
    print()
    print("  To start the local server:")
    print("    .\\scripts\\mlflow-local.ps1")
    sys.exit(1)


def check_backend() -> None:
    """
    Verify the FastAPI backend is running (needed for Step 3).

    The backend is your Joblab AI agent at localhost:8000.
    Start it with: uvicorn app.main:app --port 8000 --reload
    """
    import requests

    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        info(f"Backend is running at: {BACKEND_URL}")
    except Exception:
        print()
        print(f"  ERROR: Backend not reachable at {BACKEND_URL}")
        print("  Start it with: uvicorn app.main:app --port 8000 --reload")
        sys.exit(1)
