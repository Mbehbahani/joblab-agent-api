#!/usr/bin/env python3
"""
Offline evaluator for /ai/ask with MLflow tracking.

This script:
1. Loads JSONL eval cases
2. Calls the backend endpoint for each prompt
3. Scores tool selection and required-filter preservation
4. Logs metrics + artifacts to MLflow
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[idx]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            case = json.loads(line)
            if not isinstance(case, dict):
                raise ValueError(f"Line {line_no}: expected JSON object")
            if "prompt" not in case:
                raise ValueError(f"Line {line_no}: missing required field 'prompt'")
            case_id = case.get("id")
            if case_id:
                case_id = str(case_id)
                if case_id in seen_ids:
                    raise ValueError(f"Line {line_no}: duplicate id '{case_id}'")
                seen_ids.add(case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"No eval cases found in {path}")
    return cases


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "contains" in expected:
            needle = str(expected["contains"]).strip().lower()
            return needle in str(actual or "").strip().lower()
        if "any_of" in expected:
            options = expected["any_of"]
            if not isinstance(options, list):
                return False
            return any(_value_matches(actual, opt) for opt in options)
        # Unknown object matcher: fall back to strict string compare
        return str(actual).strip().lower() == str(expected).strip().lower()
    if isinstance(expected, (list, tuple, set)):
        return any(_value_matches(actual, item) for item in expected)
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except Exception:
            return False
    if expected is None:
        return actual is None
    return str(actual).strip().lower() == str(expected).strip().lower()


def _select_scored_tool_input(
    tool_calls: list[dict[str, Any]], expected_primary_tool: str | None
) -> tuple[str | None, dict[str, Any]]:
    if not tool_calls:
        return None, {}
    if expected_primary_tool:
        for tc in tool_calls:
            if tc.get("name") == expected_primary_tool:
                return expected_primary_tool, tc.get("input", {})
    first = tool_calls[0]
    return first.get("name"), first.get("input", {})


def _evaluate_case(
    case: dict[str, Any],
    session: requests.Session,
    ask_url: str,
    timeout_s: int,
    conversation_id: str,
) -> dict[str, Any]:
    prompt = str(case["prompt"])
    case_id = case.get("id") or f"case-{uuid.uuid4().hex[:8]}"
    expected_primary_tool = case.get("expected_primary_tool")
    required_filters = case.get("required_filters", {}) or {}
    expect_any_tool_call = case.get("expect_any_tool_call")
    expected_answer_contains = case.get("expected_answer_contains")
    if isinstance(expected_answer_contains, str):
        expected_answer_contains = [expected_answer_contains]
    if expected_answer_contains is None:
        expected_answer_contains = []
    if not isinstance(expected_answer_contains, list):
        raise ValueError(f"Case '{case_id}': expected_answer_contains must be a string or list")

    payload = {
        "prompt": prompt,
        "conversation_id": conversation_id,
    }

    t0 = time.perf_counter()
    err: str | None = None
    status_code = 0
    body: dict[str, Any] = {}

    try:
        resp = session.post(ask_url, json=payload, timeout=timeout_s)
        status_code = resp.status_code
        body = resp.json() if resp.content else {}
    except Exception as exc:
        err = str(exc)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    tool_calls = body.get("tool_calls") if isinstance(body, dict) else None
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    called_tools = [str(tc.get("name")) for tc in tool_calls if tc.get("name")]
    any_tool_called = len(called_tools) > 0

    primary_tool, scored_input = _select_scored_tool_input(tool_calls, expected_primary_tool)

    tool_match: bool | None = None
    if expected_primary_tool:
        tool_match = _value_matches(primary_tool, expected_primary_tool)

    tool_call_expectation_match: bool | None = None
    if isinstance(expect_any_tool_call, bool):
        tool_call_expectation_match = any_tool_called == expect_any_tool_call

    filter_total = 0
    filter_matched = 0
    for key, expected in required_filters.items():
        filter_total += 1
        actual = scored_input.get(key)
        if _value_matches(actual, expected):
            filter_matched += 1
    filter_exact = filter_total > 0 and filter_matched == filter_total
    answer_text = body.get("answer", "") if isinstance(body, dict) else ""
    answer_contains_match: bool | None = None
    if expected_answer_contains:
        answer_lower = str(answer_text).lower()
        answer_contains_match = all(
            str(piece).strip().lower() in answer_lower for piece in expected_answer_contains
        )

    return {
        "id": case_id,
        "prompt": prompt,
        "conversation_id": conversation_id,
        "conversation_group": case.get("conversation_group"),
        "expected_primary_tool": expected_primary_tool,
        "required_filters": required_filters,
        "expect_any_tool_call": expect_any_tool_call,
        "expected_answer_contains": expected_answer_contains,
        "status_code": status_code,
        "http_ok": 200 <= status_code < 300 and err is None,
        "error": err,
        "latency_ms": latency_ms,
        "primary_tool": primary_tool,
        "called_tools": called_tools,
        "any_tool_called": any_tool_called,
        "tool_match": tool_match,
        "tool_call_expectation_match": tool_call_expectation_match,
        "filter_total": filter_total,
        "filter_matched": filter_matched,
        "filter_exact": filter_exact,
        "answer_contains_match": answer_contains_match,
        "answer_preview": (body.get("answer", "")[:240] if isinstance(body, dict) else ""),
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    http_ok = sum(1 for r in results if r["http_ok"])
    latencies = [float(r["latency_ms"]) for r in results]

    labeled_tool = [r for r in results if r.get("expected_primary_tool")]
    tool_matches = sum(1 for r in labeled_tool if r.get("tool_match") is True)

    filter_cases = [r for r in results if int(r.get("filter_total", 0)) > 0]
    filter_exact_cases = sum(1 for r in filter_cases if r.get("filter_exact"))
    filter_total_keys = sum(int(r.get("filter_total", 0)) for r in filter_cases)
    filter_matched_keys = sum(int(r.get("filter_matched", 0)) for r in filter_cases)
    tool_call_expectation_cases = [
        r for r in results if isinstance(r.get("expect_any_tool_call"), bool)
    ]
    tool_call_expectation_matches = sum(
        1 for r in tool_call_expectation_cases if r.get("tool_call_expectation_match") is True
    )
    answer_contains_cases = [r for r in results if r.get("expected_answer_contains")]
    answer_contains_matches = sum(
        1 for r in answer_contains_cases if r.get("answer_contains_match") is True
    )

    return {
        "total_cases": total,
        "http_success_rate": round(http_ok / total, 4) if total else 0.0,
        "tool_labeled_cases": len(labeled_tool),
        "primary_tool_accuracy": round(tool_matches / len(labeled_tool), 4)
        if labeled_tool
        else 0.0,
        "filter_labeled_cases": len(filter_cases),
        "required_filter_exact_rate": round(filter_exact_cases / len(filter_cases), 4)
        if filter_cases
        else 0.0,
        "required_filter_coverage": round(filter_matched_keys / filter_total_keys, 4)
        if filter_total_keys
        else 0.0,
        "tool_call_expectation_cases": len(tool_call_expectation_cases),
        "tool_call_expectation_accuracy": round(
            tool_call_expectation_matches / len(tool_call_expectation_cases), 4
        )
        if tool_call_expectation_cases
        else 0.0,
        "answer_contains_cases": len(answer_contains_cases),
        "answer_contains_accuracy": round(
            answer_contains_matches / len(answer_contains_cases), 4
        )
        if answer_contains_cases
        else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(_p95(latencies), 2) if latencies else 0.0,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline /ai/ask eval and log to MLflow.")
    parser.add_argument(
        "--dataset",
        default="evals/datasets/ai_tool_eval.sample.jsonl",
        help="JSONL dataset path.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("EVAL_API_BASE_URL", "http://localhost:8000"),
        help="Backend base URL.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "joblab_ai_offline_eval"),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default=f"offline-eval-{_utc_stamp()}",
        help="MLflow run name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Run eval and write local artifacts without logging to MLflow.",
    )
    parser.add_argument(
        "--dagshub",
        action="store_true",
        default=os.getenv("DAGSHUB_ENABLE", "").lower() in ("1", "true", "yes"),
        help=(
            "Log to DagsHub remote MLflow tracking server "
            "(https://dagshub.com/Mbehbahani/joblab-mlflow). "
            "Requires DAGSHUB_USER_TOKEN env var or MLFLOW_TRACKING_USERNAME/PASSWORD."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = (repo_root / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)
    ask_url = f"{args.api_base_url.rstrip('/')}/ai/ask"
    git_sha = _get_git_sha(repo_root)
    stamp = _utc_stamp()

    cases = _load_jsonl(dataset_path)
    session = requests.Session()
    conversation_ids: dict[str, str] = {}

    results: list[dict[str, Any]] = []
    for case in cases:
        group = case.get("conversation_group")
        if group:
            key = str(group)
            if key not in conversation_ids:
                conversation_ids[key] = f"eval-{key}-{uuid.uuid4().hex[:10]}"
            conv_id = conversation_ids[key]
        else:
            case_id = case.get("id") or uuid.uuid4().hex[:10]
            conv_id = f"eval-{case_id}-{uuid.uuid4().hex[:8]}"
        results.append(_evaluate_case(case, session, ask_url, args.timeout, conv_id))

    summary = _summarize(results)
    summary["dataset"] = str(dataset_path)
    summary["api_base_url"] = args.api_base_url
    summary["ask_url"] = ask_url
    summary["git_sha"] = git_sha
    summary["timestamp_utc"] = stamp

    outputs_dir = repo_root / "evals" / "outputs"
    results_path = outputs_dir / f"{stamp}_ai_eval_results.jsonl"
    summary_path = outputs_dir / f"{stamp}_ai_eval_summary.json"
    _write_jsonl(results_path, results)
    _write_json(summary_path, summary)

    run_id = None
    if not args.skip_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise SystemExit(
                "mlflow is not installed. Run: pip install -r evals/requirements.txt"
            ) from exc

        if args.dagshub:
            try:
                import dagshub as _dagshub
            except ImportError as exc:
                raise SystemExit(
                    "dagshub is not installed. Run: pip install -r evals/requirements.txt"
                ) from exc
            # dagshub.init() sets MLFLOW_TRACKING_URI + injects auth automatically
            _dagshub.init(
                repo_owner="Mbehbahani",
                repo_name="joblab-mlflow",
                mlflow=True,
            )
        else:
            mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=args.run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags(
                {
                    "component": "ai_offline_eval",
                    "git_sha": git_sha,
                }
            )
            mlflow.log_params(
                {
                    "dataset_path": str(dataset_path),
                    "api_base_url": args.api_base_url,
                    "ask_url": ask_url,
                    "timeout_s": args.timeout,
                    "git_sha": git_sha,
                }
            )
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            mlflow.log_artifact(str(results_path))
            mlflow.log_artifact(str(summary_path))

    print("\nOffline eval completed.")
    print(f"Cases: {summary['total_cases']}")
    print(f"HTTP success rate: {summary['http_success_rate']}")
    print(f"Primary tool accuracy: {summary['primary_tool_accuracy']}")
    print(f"Required filter exact rate: {summary['required_filter_exact_rate']}")
    print(f"Required filter coverage: {summary['required_filter_coverage']}")
    print(f"Tool-call expectation accuracy: {summary['tool_call_expectation_accuracy']}")
    print(f"Answer-contains accuracy: {summary['answer_contains_accuracy']}")
    print(f"Latency avg/p95 ms: {summary['avg_latency_ms']} / {summary['p95_latency_ms']}")
    print(f"Summary artifact: {summary_path}")
    print(f"Results artifact: {results_path}")
    if run_id:
        print(f"MLflow run_id: {run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
