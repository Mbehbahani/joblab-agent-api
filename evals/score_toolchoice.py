"""
Offline Evaluation Scorer — measures tool-selection accuracy and filter preservation.

This script runs golden prompts through the AI agent's tool-selection logic
(without actually executing tools against the DB) and scores:

  1. Tool Selection Accuracy:  Did the LLM pick the right tool?
  2. Filter Preservation:      Did the LLM include all expected filters?
  3. No Extra Filters:         Did the LLM avoid adding unneeded filters?
  4. Out-of-Scope Detection:   Did the LLM correctly avoid tools for non-DB questions?

Usage:
  python -m evals.score_toolchoice                         # run all evals
  python -m evals.score_toolchoice --dry-run               # show dataset without calling LLM
  python -m evals.score_toolchoice --ids eval_001 eval_005 # run specific evals

Metrics logged:
  - tool_selection_accuracy (fraction correct)
  - filter_recall           (fraction of expected filters present)
  - filter_precision        (fraction of model's filters that were expected)
  - overall_score           (harmonic mean of above)

If MLflow is available, results are logged as an experiment run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path so we can import app modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Dataset Loading ────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_dataset(ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Load golden evaluation dataset, optionally filtering by IDs."""
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    if ids:
        dataset = [d for d in dataset if d["id"] in ids]
    return dataset


# ── Tool-Choice Extraction ─────────────────────────────────────────────────

def extract_tool_choice_from_response(response: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """
    Extract the tool name and input from a Bedrock Converse API response.
    Returns (None, {}) if no tool was called.
    """
    message = response.get("output", {}).get("message", {})
    content = message.get("content", [])
    for block in content:
        if "toolUse" in block:
            tu = block["toolUse"]
            return tu.get("name"), tu.get("input", {})
    return None, {}


# ── Scoring Functions ──────────────────────────────────────────────────────

def score_tool_selection(expected_tool: str | None, actual_tool: str | None) -> float:
    """Binary score: 1.0 if correct tool, 0.0 if wrong."""
    if expected_tool is None and actual_tool is None:
        return 1.0
    if expected_tool == actual_tool:
        return 1.0
    return 0.0


def score_filter_recall(
    expected_filters: dict[str, Any] | None,
    actual_filters: dict[str, Any],
) -> float:
    """
    What fraction of expected filters are present in actual filters?
    Recall = |expected ∩ actual| / |expected|

    We check key presence AND value match (case-insensitive for strings).
    """
    if expected_filters is None:
        return 1.0  # no filters expected, recall is perfect

    if not expected_filters:
        return 1.0

    matches = 0
    for key, expected_val in expected_filters.items():
        actual_val = actual_filters.get(key)
        if actual_val is None:
            continue

        # Normalize for comparison
        if isinstance(expected_val, str) and isinstance(actual_val, str):
            if expected_val.lower().strip() in actual_val.lower().strip() or \
               actual_val.lower().strip() in expected_val.lower().strip():
                matches += 1
                continue
        if expected_val == actual_val:
            matches += 1

    return matches / len(expected_filters)


def score_filter_precision(
    expected_filters: dict[str, Any] | None,
    actual_filters: dict[str, Any],
) -> float:
    """
    What fraction of actual filters were expected?
    Precision = |expected ∩ actual_keys| / |actual_keys|

    Penalizes adding unnecessary filters (minimal filter policy).
    """
    if expected_filters is None:
        # No filters expected — if model added any, precision is 0
        return 0.0 if actual_filters else 1.0

    if not actual_filters:
        return 1.0 if not expected_filters else 0.0

    # Only count filter keys (not limit, metric, etc.)
    filter_keys = {
        "country", "is_remote", "is_research", "job_level_std",
        "job_function_std", "company_industry_std", "job_type_filled",
        "platform", "posted_start", "posted_end", "role_keyword",
        "query_text", "top_k",
    }
    actual_filter_keys = {k for k in actual_filters if k in filter_keys and actual_filters[k] is not None}
    expected_filter_keys = {k for k in expected_filters if k in filter_keys}

    if not actual_filter_keys:
        return 1.0

    matches = len(actual_filter_keys & expected_filter_keys)
    return matches / len(actual_filter_keys)


def compute_overall_score(
    tool_accuracy: float,
    filter_recall: float,
    filter_precision: float,
) -> float:
    """
    Harmonic mean of the three scores.
    Returns 0 if any score is 0 (harmonic mean property).
    """
    scores = [tool_accuracy, filter_recall, filter_precision]
    if any(s == 0 for s in scores):
        return 0.0
    n = len(scores)
    return n / sum(1.0 / s for s in scores)


# ── Evaluation Runner ──────────────────────────────────────────────────────

def run_eval_case(
    case: dict[str, Any],
    system_prompt: str,
    tool_definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run a single evaluation case:
    1. Send prompt to Claude with tools
    2. Extract tool choice
    3. Score against expected
    """
    from app.services.bedrock import invoke_claude, has_tool_use

    prompt = case["prompt"]
    expected_tool = case.get("expected_tool")
    expected_filters = case.get("expected_filters")

    # Call Claude
    start = time.time()
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    response = invoke_claude(
        messages=messages,
        system=system_prompt,
        tools=tool_definitions,
    )
    latency_ms = (time.time() - start) * 1000

    # Extract what the model chose
    actual_tool, actual_filters = extract_tool_choice_from_response(response)

    # If model didn't use a tool
    if not has_tool_use(response):
        actual_tool = None
        actual_filters = {}

    # Score
    tool_acc = score_tool_selection(expected_tool, actual_tool)
    f_recall = score_filter_recall(expected_filters, actual_filters)
    f_precision = score_filter_precision(expected_filters, actual_filters)
    overall = compute_overall_score(tool_acc, f_recall, f_precision)

    result = {
        "id": case["id"],
        "prompt": prompt,
        "category": case.get("category", ""),
        "difficulty": case.get("difficulty", ""),
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "expected_filters": expected_filters,
        "actual_filters": actual_filters,
        "tool_selection_correct": tool_acc == 1.0,
        "tool_accuracy": tool_acc,
        "filter_recall": round(f_recall, 3),
        "filter_precision": round(f_precision, 3),
        "overall_score": round(overall, 3),
        "latency_ms": round(latency_ms, 1),
    }

    return result


def run_full_eval(
    dataset: list[dict[str, Any]],
    system_prompt: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Run evaluation across all cases and compute aggregate metrics.

    Returns:
        {
            "results": [...],
            "metrics": {
                "tool_selection_accuracy": ...,
                "avg_filter_recall": ...,
                "avg_filter_precision": ...,
                "avg_overall_score": ...,
                "total_cases": ...,
                "by_category": { ... },
                "by_difficulty": { ... },
            }
        }
    """
    from app.services.prompt_policy import get_system_prompt
    from app.services.joblab_tools import TOOL_DEFINITIONS

    if system_prompt is None:
        system_prompt = get_system_prompt()
    if tool_definitions is None:
        tool_definitions = TOOL_DEFINITIONS

    results: list[dict[str, Any]] = []
    for i, case in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {case['id']}: {case['prompt'][:60]}...", flush=True)
        try:
            result = run_eval_case(case, system_prompt, tool_definitions)
            results.append(result)

            status = "PASS" if result["tool_selection_correct"] else "FAIL"
            print(
                f"    {status} | tool={result['actual_tool']} "
                f"| recall={result['filter_recall']:.2f} "
                f"| precision={result['filter_precision']:.2f} "
                f"| overall={result['overall_score']:.2f}"
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "id": case["id"],
                "prompt": case["prompt"],
                "error": str(e),
                "tool_accuracy": 0.0,
                "filter_recall": 0.0,
                "filter_precision": 0.0,
                "overall_score": 0.0,
            })

    # Compute aggregates
    total = len(results)
    metrics: dict[str, Any] = {
        "total_cases": total,
        "tool_selection_accuracy": round(
            sum(r.get("tool_accuracy", 0) for r in results) / max(total, 1), 3
        ),
        "avg_filter_recall": round(
            sum(r.get("filter_recall", 0) for r in results) / max(total, 1), 3
        ),
        "avg_filter_precision": round(
            sum(r.get("filter_precision", 0) for r in results) / max(total, 1), 3
        ),
        "avg_overall_score": round(
            sum(r.get("overall_score", 0) for r in results) / max(total, 1), 3
        ),
        "avg_latency_ms": round(
            sum(r.get("latency_ms", 0) for r in results) / max(total, 1), 1
        ),
        "error_count": sum(1 for r in results if "error" in r),
    }

    # By category
    categories: dict[str, list] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    metrics["by_category"] = {
        cat: {
            "count": len(items),
            "tool_accuracy": round(
                sum(r.get("tool_accuracy", 0) for r in items) / len(items), 3
            ),
            "avg_overall": round(
                sum(r.get("overall_score", 0) for r in items) / len(items), 3
            ),
        }
        for cat, items in sorted(categories.items())
    }

    # By difficulty
    difficulties: dict[str, list] = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        difficulties.setdefault(diff, []).append(r)

    metrics["by_difficulty"] = {
        diff: {
            "count": len(items),
            "tool_accuracy": round(
                sum(r.get("tool_accuracy", 0) for r in items) / len(items), 3
            ),
            "avg_overall": round(
                sum(r.get("overall_score", 0) for r in items) / len(items), 3
            ),
        }
        for diff, items in sorted(difficulties.items())
    }

    return {"results": results, "metrics": metrics}


# ── MLflow Report ──────────────────────────────────────────────────────────

def log_eval_to_mlflow(eval_result: dict[str, Any], policy_version: str = "") -> None:
    """Log evaluation results to MLflow as an experiment run."""
    try:
        import mlflow
        from app.config import get_settings

        s = get_settings()
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment(s.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"eval-{policy_version or 'baseline'}"):
            # Log aggregate metrics
            metrics = eval_result["metrics"]
            mlflow.log_metrics({
                "tool_selection_accuracy": metrics["tool_selection_accuracy"],
                "avg_filter_recall": metrics["avg_filter_recall"],
                "avg_filter_precision": metrics["avg_filter_precision"],
                "avg_overall_score": metrics["avg_overall_score"],
                "avg_latency_ms": metrics["avg_latency_ms"],
                "total_cases": metrics["total_cases"],
                "error_count": metrics["error_count"],
            })

            # Log params
            mlflow.log_params({
                "policy_version": policy_version,
                "dataset_size": metrics["total_cases"],
            })

            # Save full results as artifact
            mlflow.log_text(
                json.dumps(eval_result, indent=2, default=str),
                artifact_file="eval_results.json",
            )

            print("\n  MLflow run logged to experiment 'joblab-ai-eval'")

    except ImportError:
        print("\n  MLflow not available — skipping MLflow logging")
    except Exception as e:
        print(f"\n  MLflow logging failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run offline evaluation for AI agent tool selection")
    parser.add_argument("--dry-run", action="store_true", help="Show dataset without running evals")
    parser.add_argument("--ids", nargs="*", help="Run specific eval IDs only")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("JobLab AI Agent — Offline Evaluation")
    print("=" * 60)

    dataset = load_dataset(args.ids)
    print(f"\nLoaded {len(dataset)} evaluation cases")

    if args.dry_run:
        print("\n--- Dry Run (no LLM calls) ---\n")
        for case in dataset:
            print(f"  {case['id']}: [{case.get('difficulty', '?')}] [{case.get('category', '?')}]")
            print(f"    Prompt: {case['prompt']}")
            print(f"    Expected: tool={case.get('expected_tool')} filters={case.get('expected_filters')}")
            print()
        return

    print("\nRunning evaluations...\n")
    result = run_full_eval(dataset)

    # Print summary
    metrics = result["metrics"]
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total cases:              {metrics['total_cases']}")
    print(f"  Tool Selection Accuracy:  {metrics['tool_selection_accuracy']:.1%}")
    print(f"  Average Filter Recall:    {metrics['avg_filter_recall']:.1%}")
    print(f"  Average Filter Precision: {metrics['avg_filter_precision']:.1%}")
    print(f"  Average Overall Score:    {metrics['avg_overall_score']:.1%}")
    print(f"  Average Latency:          {metrics['avg_latency_ms']:.0f}ms")
    print(f"  Errors:                   {metrics['error_count']}")

    print("\n  By Difficulty:")
    for diff, data in metrics.get("by_difficulty", {}).items():
        print(f"    {diff:8s}: accuracy={data['tool_accuracy']:.1%}  overall={data['avg_overall']:.1%}  (n={data['count']})")

    print("\n  By Category:")
    for cat, data in metrics.get("by_category", {}).items():
        print(f"    {cat:25s}: accuracy={data['tool_accuracy']:.1%}  overall={data['avg_overall']:.1%}  (n={data['count']})")

    # Failed cases
    failures = [r for r in result["results"] if r.get("tool_accuracy", 0) < 1.0]
    if failures:
        print(f"\n  Failed Cases ({len(failures)}):")
        for f in failures:
            print(f"    {f['id']}: expected={f.get('expected_tool')} got={f.get('actual_tool')} | {f.get('prompt', '')[:50]}")

    # Log to MLflow
    if not args.no_mlflow:
        from app.services.prompt_policy import POLICY_VERSION
        log_eval_to_mlflow(result, policy_version=POLICY_VERSION)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to {output_path}")
    else:
        # Default: save to evals/results/
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"eval_{timestamp}.json"
        output_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to {output_path}")

    print()


if __name__ == "__main__":
    main()

