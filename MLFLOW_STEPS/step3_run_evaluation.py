"""
Step 3 baseline evaluation for the Opnew learning pipeline.

This step evaluates the production prompt against the baseline dataset and keeps
the tracing model honest:

- The backend is the source of AGENT / LLM / TOOL spans.
- This script measures end-to-end request latency for each eval case.
- This script links backend trace IDs to the MLflow evaluation run.
- This script logs the evaluation bundle:
  prompt + model config + judge config + score metrics + operational summaries.

Operational metrics logged here:
- eval latency (mean / median / p95)
- token usage (mean / median)
- estimated token-derived cost

Detailed per-turn metrics such as `llm_latency_ms`, `tool_latency_ms`,
`tool_rounds`, and `soft_enforcement_retries` still live in the linked backend
traces and the backend turn logger. They are not synthesized here.

When MLflow distributed tracing is available on both sides, this step also
propagates trace context to `/ai/ask` so backend spans can appear under the
evaluation `predict_fn` trace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

MODEL_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.00),
    "anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.00),
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": (3.00, 15.00),
    "anthropic.claude-3-7-sonnet-20250219-v1:0": (3.00, 15.00),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": (3.00, 15.00),
    "anthropic.claude-sonnet-4-20250514-v1:0": (3.00, 15.00),
    "us.anthropic.claude-opus-4-20250514-v1:0": (15.00, 75.00),
    "anthropic.claude-opus-4-20250514-v1:0": (15.00, 75.00),
}


sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_config import (
    AGENT_MODEL_NAME,
    BACKEND_URL,
    DATASET_NAME,
    DATASET_SOURCE,
    EXPERIMENT_NAME,
    JUDGE_MODEL,
    JUDGE_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    PROMPT_NAME,
    banner,
    check_backend,
    info,
    resolve_tracking_uri,
    warn,
)

LINKED_PROMPTS_TAG = "mlflow.linkedPrompts"


def _normalized_output_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _estimate_total_cost_usd(model_name: str, usage: dict[str, Any]) -> float:
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get((model_name or "").strip().lower())
    if not pricing:
        return 0.0
    input_rate, output_rate = pricing
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return ((input_tokens / 1_000_000) * input_rate) + ((output_tokens / 1_000_000) * output_rate)


def step3_run_baseline(tracking_uri: str, dry_run: bool = False) -> None:
    """
    Run the baseline evaluation and link the backend traces to the run.

    Important behavior:
    - The backend already emits spans. This script does not create duplicate
      synthetic spans in `predict_fn`.
    - `predict_fn` measures only end-to-end eval request latency.
    - Judge traffic is routed through `gateway:/myendpoint`.
    """
    import mlflow
    import pandas as pd
    import requests
    from mlflow import MlflowClient
    from mlflow.entities import SpanType
    from mlflow.genai import evaluate, load_prompt
    from mlflow.genai.datasets import get_dataset
    from mlflow.genai.scorers import Correctness

    from evals.mlflow_scorers import build_code_scorers

    banner("STEP 3: Run Baseline Evaluation")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    inference_profile = (
        f"{LLM_MODEL_NAME}|temperature={LLM_TEMPERATURE}|max_tokens={LLM_MAX_TOKENS}"
    )
    trace_context_header_builder = None
    try:
        from mlflow.tracing import get_tracing_context_headers_for_http_request

        trace_context_header_builder = get_tracing_context_headers_for_http_request
    except ImportError:
        trace_context_header_builder = None

    active_model = mlflow.set_active_model(name=AGENT_MODEL_NAME)
    active_model_id = active_model.model_id
    info(f"Active model set to: {AGENT_MODEL_NAME}")

    prompt_uri = f"prompts:/{PROMPT_NAME}@production"
    prompt_version = load_prompt(prompt_uri, model_id=active_model_id)
    version_num = int(prompt_version.version)
    versioned_uri = f"prompts:/{PROMPT_NAME}/{version_num}"
    linked_prompts_payload = json.dumps(
        [{"name": prompt_version.name, "version": str(prompt_version.version)}]
    )

    print(f"  Prompt URI:       {prompt_uri}")
    print(f"  Resolved version: {version_num}")
    print(f"  Template length:  {len(prompt_version.template)} chars")
    print(f"  Inference model:  {LLM_MODEL_NAME}")
    print(f"  Temperature:      {LLM_TEMPERATURE}")
    print(f"  Max tokens:       {LLM_MAX_TOKENS}")

    from mlflow.exceptions import MlflowException as _MlflowExc

    try:
        dataset = get_dataset(name=DATASET_NAME)
        dataset_records = dataset.to_dict().get("records", [])
    except _MlflowExc:
        if dry_run:
            warn(f"Dataset '{DATASET_NAME}' not registered yet (run Step 2 first).")
            with DATASET_SOURCE.open(encoding="utf-8") as f:
                dataset_records = [
                    {
                        "inputs": {"question": row["prompt"]},
                        "expectations": {"expected_tool": row.get("expected_tool") or "none"},
                    }
                    for row in json.load(f)
                ]
            dataset = None
        else:
            print(f"\n  ERROR: Dataset '{DATASET_NAME}' not found. Run Step 2 first.")
            sys.exit(1)

    print(f"  Dataset:          {DATASET_NAME} ({len(dataset_records)} records)")

    code_scorers = build_code_scorers()
    llm_judge = Correctness(model=JUDGE_MODEL)
    all_scorers = [*code_scorers, llm_judge]

    print(f"  Scorers:          {[scorer.name for scorer in all_scorers]}")
    print(f"  Judge model:      {JUDGE_MODEL}")
    print(f"  Judge logged as:  {JUDGE_MODEL_NAME}")

    print()
    print("  Dataset preview:")
    for index, record in enumerate(dataset_records, start=1):
        question = record.get("inputs", {}).get("question", "?")
        tool = record.get("expectations", {}).get("expected_tool", "none")
        print(f"    {index:2d}. [{tool:20s}] {question}")

    if dry_run:
        warn("DRY RUN - would run evaluation, skipping.")
        print()
        print("  What WOULD happen:")
        print(f"    - Send {len(dataset_records)} questions to {BACKEND_URL}/ai/ask")
        print(f"    - Score each case with {len(all_scorers)} scorers")
        print(f"    - Measure end-to-end eval latency per case")
        print(f"    - Log inference profile: {inference_profile}")
        print("    - Log token / cost operational summaries for the run")
        print("    - Propagate MLflow trace context to /ai/ask when available")
        print("    - Link backend traces for AGENT / LLM / TOOL span inspection")
        print(f"    - Log everything to experiment '{EXPERIMENT_NAME}'")
        return

    check_backend()

    token_totals: dict[str, int] = {"input": 0, "output": 0, "total": 0}
    trace_ids: list[str] = []
    lock = Lock()

    def predict_fn(question: str) -> dict[str, Any]:
        prompt = load_prompt(
            prompt_uri,
            cache_ttl_seconds=0,
            model_id=active_model_id,
        )
        started = time.perf_counter()
        with mlflow.start_span(
            name="backend_request",
            span_type=SpanType.CHAIN,
            attributes={"http.url": f"{BACKEND_URL}/ai/ask"},
        ) as span:
            span.set_inputs({"question": question})
            try:
                mlflow.update_current_trace(tags={LINKED_PROMPTS_TAG: linked_prompts_payload})
            except Exception:
                pass
            headers: dict[str, str] = {}
            if trace_context_header_builder is not None:
                try:
                    headers = trace_context_header_builder()
                except Exception:
                    headers = {}
            headers["x-mlflow-experiment-name"] = EXPERIMENT_NAME
            headers["x-mlflow-active-model-name"] = AGENT_MODEL_NAME
            headers["x-mlflow-prompt-name"] = prompt_version.name
            headers["x-mlflow-prompt-version"] = str(prompt_version.version)
            headers["x-mlflow-prompt-uri"] = versioned_uri
            response = requests.post(
                f"{BACKEND_URL}/ai/ask",
                json={"prompt": question, "system": prompt.template},
                headers=headers,
                timeout=120,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Expected /ai/ask to return a JSON object.")
            payload["eval_latency_ms"] = elapsed_ms
            span.set_outputs(
                {
                    "trace_id": payload.get("trace_id"),
                    "tool_calls_count": len(payload.get("tool_calls") or []),
                    "latency_ms": elapsed_ms,
                }
            )

        usage = payload.get("usage", {})
        with lock:
            token_totals["input"] += int(usage.get("input_tokens", 0) or 0)
            token_totals["output"] += int(usage.get("output_tokens", 0) or 0)
            token_totals["total"] += int(usage.get("total_tokens", 0) or 0)

        trace_id = payload.get("trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            with lock:
                trace_ids.append(trace_id.strip())
        return payload

    if dataset is None:
        raise RuntimeError("Dataset must exist for a non-dry-run evaluation.")

    eval_rows = dataset.to_df().to_dict(orient="records")
    for row in eval_rows:
        expectations = row.get("expectations") or {}
        if isinstance(expectations, dict):
            row["expectations"] = {key: value for key, value in expectations.items() if value is not None}
    eval_df = pd.DataFrame(eval_rows)

    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"

    run_name = f"opnew-baseline::v{version_num}::{int(time.time())}"

    print()
    print(f"  Starting evaluation run: {run_name}")
    print(f"  Active model: {AGENT_MODEL_NAME}")
    print(f"  This will make {len(dataset_records)} backend calls + {len(dataset_records)} judge calls.")
    print()

    result = None
    run_id = ""
    try:
        with mlflow.start_run(run_name=run_name) as run:
            try:
                client.link_prompt_version_to_run(run.info.run_id, prompt_version)
            except Exception:
                pass
            mlflow.set_tag("evaluation_stage", "baseline")
            mlflow.set_tag("experiment_label", EXPERIMENT_NAME)
            mlflow.set_tag(LINKED_PROMPTS_TAG, linked_prompts_payload)
            mlflow.set_tag("prompt_uri", versioned_uri)
            mlflow.set_tag("prompt_version", str(version_num))
            mlflow.set_tag("dataset_name", DATASET_NAME)
            mlflow.set_tag("dataset_record_count", str(len(dataset_records)))
            mlflow.set_tag("scorers", ",".join(scorer.name for scorer in all_scorers))
            mlflow.set_tag("judge_model", JUDGE_MODEL)
            mlflow.set_tag("judge_logged_model_name", JUDGE_MODEL_NAME)
            mlflow.set_tag("agent_model", AGENT_MODEL_NAME)
            mlflow.set_tag("agent_runtime_model", LLM_MODEL_NAME)
            mlflow.set_tag("agent_runtime_temperature", str(LLM_TEMPERATURE))
            mlflow.set_tag("agent_runtime_max_tokens", str(LLM_MAX_TOKENS))
            mlflow.set_tag("inference_profile", inference_profile)
            mlflow.set_tag(
                "span_breakdown_source",
                "propagated_backend_trace" if trace_context_header_builder is not None else "linked_backend_traces",
            )
            mlflow.set_tag("judge_trace_source", "ai_gateway")
            mlflow.set_tag(
                "distributed_trace_propagation",
                "true" if trace_context_header_builder is not None else "false",
            )

            mlflow.log_param("prompt_name", PROMPT_NAME)
            mlflow.log_param("prompt_version", version_num)
            mlflow.log_param("dataset_name", DATASET_NAME)
            mlflow.log_param("judge_model", JUDGE_MODEL)
            mlflow.log_param("judge_logged_model_name", JUDGE_MODEL_NAME)
            mlflow.log_param("agent_model", AGENT_MODEL_NAME)
            mlflow.log_param("agent_runtime_model", LLM_MODEL_NAME)
            mlflow.log_param("agent_runtime_temperature", LLM_TEMPERATURE)
            mlflow.log_param("agent_runtime_max_tokens", LLM_MAX_TOKENS)
            mlflow.log_param("inference_profile", inference_profile)
            mlflow.log_param("backend_url", BACKEND_URL)
            mlflow.log_param("scorer_names", ",".join(scorer.name for scorer in all_scorers))

            result = evaluate(
                data=eval_df,
                predict_fn=predict_fn,
                scorers=all_scorers,
            )

            operational_rows: list[dict[str, Any]] = []
            outputs_series = result.result_df["outputs"] if "outputs" in result.result_df.columns else []
            for output_value in outputs_series:
                output = _normalized_output_dict(output_value)
                usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
                model_name = str(output.get("model") or LLM_MODEL_NAME)
                tool_calls = output.get("tool_calls")
                operational_rows.append(
                    {
                        "trace_id": str(output.get("trace_id", "") or ""),
                        "eval_latency_ms": float(output.get("eval_latency_ms", 0.0) or 0.0),
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(usage.get("output_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                        "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
                        "model": model_name,
                        "total_cost_usd": _estimate_total_cost_usd(model_name, usage),
                    }
                )

            operational_summary: dict[str, float | int] = {}
            if operational_rows:
                ops_df = pd.DataFrame(operational_rows)
                operational_summary = {
                    "record_count": int(len(ops_df)),
                    "mean_eval_latency_ms": float(ops_df["eval_latency_ms"].mean()),
                    "median_eval_latency_ms": float(ops_df["eval_latency_ms"].median()),
                    "p95_eval_latency_ms": float(ops_df["eval_latency_ms"].quantile(0.95)),
                    "mean_total_tokens": float(ops_df["total_tokens"].mean()),
                    "median_total_tokens": float(ops_df["total_tokens"].median()),
                    "mean_total_cost_usd": float(ops_df["total_cost_usd"].mean()),
                    "median_total_cost_usd": float(ops_df["total_cost_usd"].median()),
                    "total_cost_usd": float(ops_df["total_cost_usd"].sum()),
                    "mean_tool_calls_count": float(ops_df["tool_calls_count"].mean()),
                }
                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in operational_summary.items()
                        if key != "record_count"
                    }
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                result_df_path = tmp_path / "result_df.csv"
                result.result_df.to_csv(result_df_path, index=False)
                mlflow.log_artifact(str(result_df_path), artifact_path="baseline_results")

                operational_rows_path = tmp_path / "operational_rows.json"
                operational_rows_path.write_text(
                    json.dumps(operational_rows, indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(operational_rows_path), artifact_path="baseline_results")

                operational_summary_path = tmp_path / "operational_summary.json"
                operational_summary_path.write_text(
                    json.dumps(operational_summary, indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(operational_summary_path), artifact_path="baseline_results")

                notes_path = tmp_path / "operational_notes.txt"
                notes_path.write_text(
                    (
                        "Detailed per-turn llm_latency_ms, tool_latency_ms, "
                        "tool_rounds, and soft_enforcement_retries remain in the "
                        "linked backend traces and backend turn logger. This Step 3 "
                        "run logs end-to-end eval latency, token usage, tool call "
                        "counts, and estimated cost, then links the backend traces "
                        "for span inspection."
                    ),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(notes_path), artifact_path="baseline_results")

            mlflow.log_dict(result.metrics, "baseline_results/metrics.json")
            mlflow.log_dict(operational_summary, "baseline_results/operational_summary.json")

            mlflow.log_metrics(
                {
                    "eval_input_tokens": float(token_totals["input"]),
                    "eval_output_tokens": float(token_totals["output"]),
                    "eval_total_tokens": float(token_totals["total"]),
                }
            )
            info(
                "Token usage: "
                f"{token_totals['input']} in / {token_totals['output']} out / {token_totals['total']} total"
            )

            unique_traces = list(dict.fromkeys(trace_ids))
            link_summary = {
                "trace_count": len(unique_traces),
                "trace_ids": unique_traces,
                "run_id": run.info.run_id,
                "linked": False,
                "link_error": None,
            }
            if unique_traces:
                try:
                    client.link_traces_to_run(trace_ids=unique_traces, run_id=run.info.run_id)
                    link_summary["linked"] = True
                    mlflow.set_tag("linked_backend_traces", "true")
                    mlflow.set_tag("linked_trace_count", str(len(unique_traces)))
                    mlflow.log_metric("linked_backend_trace_count", float(len(unique_traces)))
                    info(f"Linked {len(unique_traces)} backend traces to run")
                except Exception as exc:
                    link_summary["link_error"] = str(exc)
                    mlflow.set_tag("linked_backend_traces", "false")
                    mlflow.set_tag("linked_backend_trace_error", str(exc)[:500])
                    warn(f"Could not link traces: {exc}")
            else:
                mlflow.set_tag("linked_backend_traces", "false")
                warn("No backend trace_ids were returned by the eval responses.")
            mlflow.log_dict(link_summary, "baseline_results/linked_backend_traces.json")

            run_id = run.info.run_id
    finally:
        mlflow.clear_active_model()

    if result is None:
        raise RuntimeError("Baseline evaluation did not produce a result.")

    banner("BASELINE RESULTS")

    print("  Metrics:")
    for metric_name, metric_value in sorted(result.metrics.items()):
        bar = ""
        if isinstance(metric_value, (int, float)):
            bar = "#" * int(float(metric_value) * 20)
        print(f"    {metric_name:40s}: {metric_value} {bar}")

    print()
    print(f"  Run ID: {run_id}")
    print()
    print("  What to check in MLflow UI:")
    print(f"    1. Open: {tracking_uri}")
    print(f"    2. Select experiment: '{EXPERIMENT_NAME}'")
    print(f"    3. Open run: '{run_name}'")
    print()
    print("    Overview:")
    print("      - linked backend trace token and cost data")
    print("      - run-level eval latency / token / cost metrics")
    print("    Params:")
    print("      - prompt version")
    print("      - model ID / temperature / max tokens")
    print("      - inference_profile")
    print("    Artifacts:")
    print("      - baseline_results/result_df.csv")
    print("      - baseline_results/metrics.json")
    print("      - baseline_results/operational_rows.json")
    print("      - baseline_results/operational_summary.json")
    print("      - baseline_results/linked_backend_traces.json")
    print("    Traces:")
    print("      - if propagation is active, backend AGENT / LLM / TOOL spans appear under predict_fn")
    print("      - linked backend traces remain available for direct inspection")
    print("      - backend traces carry llm_latency_ms, tool_latency_ms, tool_rounds, retries")
    print("    Judge:")
    print("      - Correctness uses gateway:/myendpoint")
    print("      - inspect AI Gateway traces for judge traffic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Run baseline evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")
    args = parser.parse_args()

    tracking_uri = resolve_tracking_uri()
    step3_run_baseline(tracking_uri, dry_run=args.dry_run)

    banner("STEP 3 COMPLETE")
    print(f"  Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
