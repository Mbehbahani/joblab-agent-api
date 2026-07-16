"""
MLflow GenAI Evaluation — End-to-end agent quality scoring.

Uses mlflow.genai.evaluate() with:
  - A predict function that calls the full AI agent (/ai/ask endpoint locally)
  - Built-in LLM judges (Correctness, Guidelines, RelevanceToQuery) powered by
    Bedrock Claude as the judge model
  - Custom scorers for tool selection accuracy and scope enforcement

Prerequisites:
  - Local backend running: uvicorn app.main:app --port 8000 --reload
  - AWS credentials configured (for Bedrock judge calls)
  - MLflow tracking URI configured (results logged to MLflow)

Usage:
  python -m evals.eval_genai                        # run full evaluation
  python -m evals.eval_genai --dry-run               # show dataset only
  python -m evals.eval_genai --limit 5               # run first N cases only
  python -m evals.eval_genai --no-builtin            # skip LLM judge scorers
  python -m evals.eval_genai --judge-model bedrock:/anthropic.claude-3-5-sonnet-20241022-v2:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root so we can import app modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

from evals.optimization_contract import (
    annotate_genai_dataset,
    summarize_failure_mode_coverage_from_genai,
)

# ── Constants ───────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent / "golden_qa_dataset.json"

# Default judge model: use the same Bedrock Claude model as the agent
# Format: <provider>:/<model-name>  (litellm routing)
DEFAULT_JUDGE_MODEL = "bedrock:/us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Local backend URL (the predict function calls the running server)
_backend_url = "http://localhost:8000"


# ── Dataset ─────────────────────────────────────────────────────────────────

def load_dataset(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the golden QA evaluation dataset."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return annotate_genai_dataset(data)


# ── Predict Function ────────────────────────────────────────────────────────
# This is the function mlflow.genai.evaluate() will call for each dataset row.
# It sends the question to the running local backend and returns the answer.

def predict_fn(question: str) -> str:
    """
    Call the AI agent's /ai/ask endpoint and return the answer text.

    MLflow calls this by unpacking the dataset's `inputs` dict as kwargs,
    so the parameter name must match the key in the dataset: "question".

    Parameters
    ----------
    question : str
        The user question (from dataset inputs.question).

    Returns
    -------
    str
        The agent's answer text.
    """
    import requests

    logger.info("Predict: %s", question[:80])
    try:
        resp = requests.post(
            f"{_backend_url}/ai/ask",
            json={"prompt": question},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        logger.info("Predict OK: %d chars", len(answer))
        return answer
    except Exception as e:
        logger.error("Predict failed for '%s': %s", question[:50], e)
        return f"[ERROR] {e}"


# ── Custom Scorers ──────────────────────────────────────────────────────────

def build_custom_scorers() -> list:
    """Build custom scorers using @scorer decorator."""
    from mlflow.genai.scorers import scorer

    @scorer
    def tool_selection_accuracy(inputs, outputs, expectations):
        """
        Check if the agent used the expected tool.

        Heuristic: look for tool-name indicators in the output text.
        The agent's responses typically mention the data source or show
        structured results that reveal which tool was used.
        """
        expected_tool = expectations.get("expected_tool") if expectations else None

        # For out-of-scope questions, the agent should NOT use any tool
        if expected_tool is None:
            # Check for signs the agent incorrectly used a tool
            tool_indicators = [
                "found", "results", "job posting", "here are",
                "count", "total", "listing",
            ]
            used_tool = any(ind in outputs.lower() for ind in tool_indicators)
            if not used_tool:
                return {"score": 1.0, "rationale": "Correctly avoided using tools for out-of-scope query."}
            else:
                return {"score": 0.0, "rationale": "Agent appears to have used tools for an out-of-scope query."}

        # For in-scope questions, check if the answer looks like it came
        # from the right type of query
        tool_signatures = {
            "job_stats": ["count", "total", "statistics", "distribution", "trend", "month", "by country", "by platform", "by industry"],
            "search_jobs": ["job posting", "position", "company", "location", "here are", "found", "listing"],
            "semantic_search_jobs": ["relevant", "similar", "related", "semantic", "matching"],
        }

        expected_sigs = tool_signatures.get(expected_tool, [])
        output_lower = outputs.lower()
        matches = sum(1 for s in expected_sigs if s in output_lower)

        if matches >= 2:
            return {"score": 1.0, "rationale": f"Output matches '{expected_tool}' signature ({matches} indicators found)."}
        elif matches == 1:
            return {"score": 0.5, "rationale": f"Partial match for '{expected_tool}' ({matches} indicator)."}
        else:
            return {"score": 0.0, "rationale": f"Output does not match expected tool '{expected_tool}'."}

    @scorer
    def scope_enforcement(inputs, outputs, expectations):
        """
        For out-of-scope questions, verify the agent declines appropriately.
        For in-scope questions, verify it provides a substantive answer.
        """
        expected_tool = expectations.get("expected_tool") if expectations else None
        category = expectations.get("category", "") if expectations else ""
        output_lower = outputs.lower()

        if category == "out_of_scope" or expected_tool is None:
            # Should decline / redirect
            decline_indicators = [
                "can't help", "cannot help", "can't assist", "cannot assist",
                "not able to", "outside", "only help with", "job-related",
                "job market", "don't have", "not a", "i'm designed",
                "i specialize", "let me know if", "redirect",
            ]
            declined = any(ind in output_lower for ind in decline_indicators)
            if declined:
                return {"score": 1.0, "rationale": "Agent correctly declined out-of-scope query."}
            else:
                return {"score": 0.0, "rationale": "Agent should have declined but gave a substantive response."}
        else:
            # Should give a real answer (not empty, not an error, not a decline)
            if "[ERROR]" in outputs:
                return {"score": 0.0, "rationale": "Agent returned an error."}
            if len(outputs.strip()) < 20:
                return {"score": 0.0, "rationale": "Answer too short to be substantive."}
            return {"score": 1.0, "rationale": "Agent provided a substantive answer for in-scope query."}

    @scorer
    def response_completeness(inputs, outputs, expectations):
        """
        Check if the response is well-formed and complete (not truncated,
        has reasonable length, contains actual data).
        """
        if "[ERROR]" in outputs:
            return {"score": 0.0, "rationale": "Response contains an error."}

        word_count = len(outputs.split())
        if word_count < 5:
            return {"score": 0.0, "rationale": f"Response too short ({word_count} words)."}
        elif word_count < 20:
            return {"score": 0.5, "rationale": f"Response is brief ({word_count} words)."}
        else:
            return {"score": 1.0, "rationale": f"Response has adequate length ({word_count} words)."}

    return [tool_selection_accuracy, scope_enforcement]


# ── Built-in LLM Judge Scorers ─────────────────────────────────────────────

def build_builtin_scorers(judge_model: str) -> list:
    """
    Build MLflow built-in LLM judge scorers configured to use Bedrock.

    Single Correctness judge for fast trend tracking during learning.
    """
    from mlflow.genai.scorers import Correctness

    return [
        Correctness(model=judge_model),
    ]


# ── Main Evaluation Runner ─────────────────────────────────────────────────

def run_evaluation(
    dataset: list[dict],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    use_builtin_scorers: bool = True,
) -> Any:
    """
    Run mlflow.genai.evaluate() with the dataset, predict function, and scorers.

    Results are automatically logged to the active MLflow experiment.
    """
    import mlflow

    # Combine custom + built-in scorers
    scorers = build_custom_scorers()
    if use_builtin_scorers:
        scorers.extend(build_builtin_scorers(judge_model))

    print(f"\n  Scorers ({len(scorers)}):")
    for s in scorers:
        print(f"    - {s.name}")
    print()

    # Skip the pre-validation trace probe (doubles first-case time)
    import os
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")

    # Run evaluation — mlflow.genai.evaluate handles:
    #   1. Calling predict_fn for each dataset row
    #   2. Running all scorers against (inputs, outputs, expectations)
    #   3. Logging results to MLflow
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=scorers,
    )

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    global _backend_url

    parser = argparse.ArgumentParser(
        description="Run MLflow GenAI evaluation on the JobLab AI agent"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show dataset without running evaluation",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N cases",
    )
    parser.add_argument(
        "--no-builtin", action="store_true",
        help="Skip built-in LLM judge scorers (only custom scorers)",
    )
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Model URI for LLM judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--backend-url", type=str, default=_backend_url,
        help=f"Backend URL for predict function (default: {_backend_url})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    # Override the backend URL if specified
    _backend_url = args.backend_url

    print("=" * 65)
    print("  JobLab AI Agent — MLflow GenAI Evaluation")
    print("=" * 65)
    print(f"  Judge model:  {args.judge_model}")
    print(f"  Backend URL:  {_backend_url}")

    # Load dataset
    dataset = load_dataset(args.limit)
    print(f"  Dataset size: {len(dataset)} cases")
    failure_mode_coverage = summarize_failure_mode_coverage_from_genai(dataset)
    if failure_mode_coverage:
        print("  Failure-mode coverage:")
        for key, count in sorted(failure_mode_coverage.items()):
            print(f"    {key:35s}: {count}")

    if args.dry_run:
        print("\n--- Dry Run (no predictions or scoring) ---\n")
        for i, row in enumerate(dataset):
            q = row["inputs"]["question"]
            cat = row["expectations"].get("category", "?")
            tool = row["expectations"].get("expected_tool") or "none"
            failure_modes = ",".join(row["expectations"].get("failure_modes", [])) or "none"
            print(
                f"  [{i+1:2d}] [{cat:20s}] tool={tool:25s} "
                f"| failures={failure_modes:45s} | {q}"
            )
        return

    # Initialize MLflow
    import mlflow
    from app.config import get_settings

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    print(f"  MLflow URI:   {settings.mlflow_tracking_uri}")
    print(f"  Experiment:   {settings.mlflow_experiment_name}")
    print()

    # Verify backend is running
    import requests
    try:
        r = requests.get(f"{_backend_url}/health", timeout=5)
        r.raise_for_status()
        print("  Backend health check: OK")
    except Exception as e:
        print(f"\n  ERROR: Backend not reachable at {_backend_url}")
        print(f"  Start it first: uvicorn app.main:app --port 8000 --reload")
        print(f"  Details: {e}")
        sys.exit(1)

    # Run evaluation
    print("\n  Running evaluation...\n")
    start = time.time()

    results = run_evaluation(
        dataset=dataset,
        judge_model=args.judge_model,
        use_builtin_scorers=not args.no_builtin,
    )

    elapsed = time.time() - start
    print(f"\n  Evaluation completed in {elapsed:.1f}s")

    # Print results summary
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    # The EvaluationResult object contains a metrics dict and a pandas DataFrame
    if hasattr(results, "metrics") and results.metrics:
        print("\n  Aggregate Metrics:")
        for name, value in sorted(results.metrics.items()):
            if isinstance(value, float):
                print(f"    {name:40s}: {value:.3f}")
            else:
                print(f"    {name:40s}: {value}")

    if hasattr(results, "eval_results_table"):
        df = results.eval_results_table
        print(f"\n  Per-case results: {len(df)} rows")
        # Save to CSV
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = results_dir / f"genai_eval_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path}")

    print()


if __name__ == "__main__":
    main()
