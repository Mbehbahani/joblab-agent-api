"""
Step 4 prompt optimization for the Opnew learning pipeline.

This step keeps the same experiment boundary as Steps 1-3:

- Experiment: Opnew
- Source prompt: prompts:/joblab-system-prompt@production
- Runtime path: /ai/ask on the local backend
- Evaluation judge: gateway:/myendpoint

What changes in Step 4:
- Prompt optimization is performed with MLflow GEPA
- GEPA reflection uses the AI Gateway endpoint gateway:/claude-3-5-sonnet
- The optimized candidate is validated against the same Opnew dataset
- The candidate alias only moves if validation beats the current baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

OBJECTIVE_WEIGHTS = {
    "joblab_tool_choice_accuracy": 0.30,
    "joblab_filter_recall": 0.25,
    "joblab_out_of_scope_refusal": 0.20,
    "correctness": 0.25,
}

LINKED_PROMPTS_TAG = "mlflow.linkedPrompts"
STEP3_CODE_SCORER_NAMES = {
    "joblab_tool_choice_accuracy",
    "joblab_filter_recall",
    "joblab_out_of_scope_refusal",
}

if os.name == "nt" and os.environ.get("PYTHONUTF8") != "1":
    relaunched_env = dict(os.environ)
    relaunched_env["PYTHONUTF8"] = "1"
    raise SystemExit(
        subprocess.call([sys.executable, *sys.argv], env=relaunched_env)
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_config import (
    AGENT_MODEL_NAME,
    BACKEND_URL,
    DATASET_NAME,
    EXPERIMENT_NAME,
    JUDGE_MODEL,
    JUDGE_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    OPTIMIZATION_CANDIDATE_ALIAS,
    OPTIMIZER_MAX_METRIC_CALLS,
    OPTIMIZER_MODEL,
    OPTIMIZER_MODEL_NAME,
    PROMPT_NAME,
    banner,
    check_backend,
    info,
    resolve_tracking_uri,
    warn,
)


def _parse_model_uri(model_uri: str) -> tuple[str, str]:
    provider, model_name = model_uri.split(":/", 1)
    return provider, model_name


def _configure_gateway_reflection_model(
    reflection_model: str,
    *,
    tracking_uri: str,
) -> tuple[str, dict[str, str | None]]:
    import mlflow
    from mlflow.genai.utils.gateway_utils import get_gateway_litellm_config

    provider, model_name = _parse_model_uri(reflection_model)
    if provider != "gateway":
        return reflection_model, {}

    mlflow.set_tracking_uri(tracking_uri)
    config = get_gateway_litellm_config(model_name)
    previous_env = {
        "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "MLFLOW_GATEWAY_URI": os.environ.get("MLFLOW_GATEWAY_URI"),
    }
    os.environ["OPENAI_API_BASE"] = config.api_base
    os.environ["OPENAI_API_KEY"] = config.api_key
    os.environ["MLFLOW_GATEWAY_URI"] = tracking_uri
    return f"openai:/{model_name}", previous_env


def _restore_env(previous_env: dict[str, str | None]) -> None:
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _probe_reflection_model(reflection_model: str, *, tracking_uri: str) -> None:
    import litellm

    effective_model, previous_env = _configure_gateway_reflection_model(
        reflection_model,
        tracking_uri=tracking_uri,
    )
    provider, model_name = _parse_model_uri(effective_model)
    litellm_model = f"{provider}/{model_name}"
    try:
        response = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": "Reply with OK only."}],
            max_tokens=5,
        )
        content = response.choices[0].message.content
        info(f"Optimizer reflection probe succeeded: {litellm_model} -> {content}")
    except Exception as exc:
        raise RuntimeError(
            "Optimizer reflection endpoint is not healthy. "
            f"Configured model '{reflection_model}' could not answer a probe request. "
            f"Underlying error: {exc}"
        ) from exc
    finally:
        _restore_env(previous_env)


def _feedback_to_float(value: Any) -> float:
    from mlflow.entities import Feedback

    if isinstance(value, Feedback):
        value = value.value
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _score_from_metrics(metrics: dict[str, Any], scorer_name: str) -> float | None:
    for key in (scorer_name, f"{scorer_name}/mean"):
        value = metrics.get(key)
        if isinstance(value, (int, float, bool)):
            return float(value)
    return None


def _objective(scores: dict[str, Any]) -> float:
    weighted_total = 0.0
    applied_weight = 0.0
    for scorer_name, weight in OBJECTIVE_WEIGHTS.items():
        if scorer_name not in scores:
            continue
        weighted_total += weight * _feedback_to_float(scores[scorer_name])
        applied_weight += weight
    if applied_weight == 0.0:
        return 0.0
    return weighted_total / applied_weight


def _aggregate_metrics(metrics: dict[str, Any]) -> float:
    weighted_total = 0.0
    applied_weight = 0.0
    for scorer_name, weight in OBJECTIVE_WEIGHTS.items():
        metric_value = _score_from_metrics(metrics, scorer_name)
        if metric_value is None:
            continue
        weighted_total += weight * metric_value
        applied_weight += weight
    if applied_weight == 0.0:
        return 0.0
    return weighted_total / applied_weight


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


def _selected_step3_code_scorers() -> list[Any]:
    from evals.mlflow_scorers import build_code_scorers

    return [
        scorer for scorer in build_code_scorers() if scorer.name in STEP3_CODE_SCORER_NAMES
    ]


def _normalize_feedback_value(value: Any) -> tuple[Any | None, str | None]:
    import math

    if value is None:
        return None, None
    if hasattr(value, "value"):
        return getattr(value, "value", None), getattr(value, "rationale", None)
    if isinstance(value, dict):
        if "value" in value:
            return value.get("value"), value.get("rationale")
        if "result" in value:
            return value.get("result"), value.get("rationale")
    if isinstance(value, float) and math.isnan(value):
        return None, None
    return value, None


def _log_trace_feedback_from_result_df(result_df: Any) -> int:
    import mlflow

    if "outputs" not in result_df.columns:
        return 0

    logged = 0
    scorer_columns = list(OBJECTIVE_WEIGHTS.keys())
    for _, row in result_df.iterrows():
        output = _normalized_output_dict(row.get("outputs"))
        trace_id = str(output.get("trace_id", "") or "").strip()
        if not trace_id:
            continue
        for scorer_name in scorer_columns:
            if scorer_name not in result_df.columns:
                continue
            value, rationale = _normalize_feedback_value(row.get(scorer_name))
            if value is None:
                continue
            try:
                mlflow.log_feedback(
                    trace_id=trace_id,
                    name=scorer_name,
                    value=value,
                    rationale=rationale,
                )
                logged += 1
            except Exception:
                continue
    return logged


def _summarize_operational_metrics(result_df: Any) -> dict[str, Any]:
    import pandas as pd

    outputs_series = result_df["outputs"] if "outputs" in result_df.columns else []
    rows: list[dict[str, Any]] = []
    for output_value in outputs_series:
        output = _normalized_output_dict(output_value)
        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        model_name = str(output.get("model") or LLM_MODEL_NAME)
        tool_calls = output.get("tool_calls")
        rows.append(
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
    if not rows:
        return {"rows": [], "summary": {}}

    ops_df = pd.DataFrame(rows)
    summary = {
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
    return {"rows": rows, "summary": summary}


def _build_train_data(dataset: Any) -> list[dict[str, Any]]:
    train_data: list[dict[str, Any]] = []
    for record in dataset.to_dict().get("records", []):
        inputs = dict(record.get("inputs", {}))
        expectations = dict(record.get("expectations", {}))
        expected_response = expectations.get("expected_response")
        if not expected_response:
            continue
        train_data.append(
            {
                "inputs": inputs,
                "outputs": {"answer": expected_response},
                "expectations": {k: v for k, v in expectations.items() if v is not None},
            }
        )
    return train_data


def _build_eval_df(dataset: Any) -> Any:
    import pandas as pd

    rows = dataset.to_df().to_dict(orient="records")
    for row in rows:
        expectations = row.get("expectations") or {}
        if isinstance(expectations, dict):
            row["expectations"] = {k: v for k, v in expectations.items() if v is not None}
    return pd.DataFrame(rows)


def _link_traces_to_run(*, client: Any, trace_ids: list[str], run_id: str) -> dict[str, Any]:
    unique_trace_ids = list(dict.fromkeys(trace_ids))
    summary = {
        "trace_count": len(unique_trace_ids),
        "trace_ids": unique_trace_ids,
        "run_id": run_id,
        "linked": False,
        "link_error": None,
    }
    if not unique_trace_ids:
        return summary
    try:
        client.link_traces_to_run(trace_ids=unique_trace_ids, run_id=run_id)
        summary["linked"] = True
    except Exception as exc:
        summary["link_error"] = str(exc)
    return summary


def _build_predict_fn(
    *,
    prompt_uri: str,
    active_model_id: str,
    trace_context_header_builder: Any,
    trace_collector: list[str],
    lock: Lock,
) -> Any:
    import mlflow
    import requests
    from mlflow.entities import SpanType
    from mlflow.genai import load_prompt

    def predict_fn(question: str) -> dict[str, Any]:
        prompt = load_prompt(
            prompt_uri,
            cache_ttl_seconds=0,
            model_id=active_model_id,
        )
        versioned_uri = f"prompts:/{prompt.name}/{prompt.version}"
        linked_prompts_payload = json.dumps([{"name": prompt.name, "version": str(prompt.version)}])
        started = time.perf_counter()
        with mlflow.start_span(
            name="backend_request",
            span_type=SpanType.CHAIN,
            attributes={"http.url": f"{BACKEND_URL}/ai/ask"},
        ) as span:
            span.set_inputs({"question": question, "prompt_uri": versioned_uri})
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
            headers["x-mlflow-prompt-name"] = prompt.name
            headers["x-mlflow-prompt-version"] = str(prompt.version)
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
        trace_id = payload.get("trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            with lock:
                trace_collector.append(trace_id.strip())
        return payload

    return predict_fn


def _run_validation_eval(
    *,
    label: str,
    prompt_uri: str,
    dataset: Any,
    active_model_id: str,
    trace_context_header_builder: Any,
    scorers: list[Any],
    client: Any,
) -> dict[str, Any]:
    import mlflow
    from mlflow.genai import evaluate, load_prompt

    eval_df = _build_eval_df(dataset)
    trace_ids: list[str] = []
    lock = Lock()
    predict_fn = _build_predict_fn(
        prompt_uri=prompt_uri,
        active_model_id=active_model_id,
        trace_context_header_builder=trace_context_header_builder,
        trace_collector=trace_ids,
        lock=lock,
    )
    prompt_version = load_prompt(prompt_uri, cache_ttl_seconds=0, model_id=active_model_id)
    linked_prompts_payload = json.dumps(
        [{"name": prompt_version.name, "version": str(prompt_version.version)}]
    )
    with mlflow.start_run(run_name=f"validation::{label}", nested=True) as run:
        try:
            client.link_prompt_version_to_run(run.info.run_id, prompt_version)
        except Exception:
            pass
        mlflow.set_tag("evaluation_stage", "optimization_validation")
        mlflow.set_tag("validation_label", label)
        mlflow.set_tag(LINKED_PROMPTS_TAG, linked_prompts_payload)
        mlflow.set_tag("prompt_uri", f"prompts:/{prompt_version.name}/{prompt_version.version}")
        mlflow.set_tag("judge_model", JUDGE_MODEL)
        mlflow.log_param("prompt_uri", f"prompts:/{prompt_version.name}/{prompt_version.version}")
        mlflow.log_param("dataset_name", DATASET_NAME)
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("scorer_names", ",".join(scorer.name for scorer in scorers))

        result = evaluate(data=eval_df, predict_fn=predict_fn, scorers=scorers)
        weighted_score = _aggregate_metrics(result.metrics)
        ops_payload = _summarize_operational_metrics(result.result_df)
        ops_summary = ops_payload.get("summary", {}) or {}

        mlflow.log_metric("weighted_validation_score", weighted_score)
        for metric_name, metric_value in ops_summary.items():
            if metric_name == "record_count":
                continue
            mlflow.log_metric(metric_name, float(metric_value))

        link_summary = _link_traces_to_run(
            client=client,
            trace_ids=trace_ids,
            run_id=run.info.run_id,
        )
        feedback_count = _log_trace_feedback_from_result_df(result.result_df)
        mlflow.log_metric("trace_feedback_assessment_count", float(feedback_count))
        mlflow.log_dict(result.metrics, f"prompt_optimization/{label}_metrics.json")
        mlflow.log_dict(ops_payload, f"prompt_optimization/{label}_operational_metrics.json")
        mlflow.log_dict(link_summary, f"prompt_optimization/{label}_linked_backend_traces.json")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result_df_path = tmp_path / f"{label}_result_df.csv"
            result.result_df.to_csv(result_df_path, index=False)
            mlflow.log_artifact(str(result_df_path), artifact_path="prompt_optimization")

        return {
            "prompt_uri": f"prompts:/{prompt_version.name}/{prompt_version.version}",
            "run_id": run.info.run_id,
            "metrics": result.metrics,
            "weighted_score": weighted_score,
            "operational_summary": ops_summary,
            "trace_link_summary": link_summary,
            "trace_feedback_assessment_count": feedback_count,
        }


def step4_optimize_prompt(tracking_uri: str, dry_run: bool = False) -> None:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.genai import load_prompt, optimize_prompts, set_prompt_alias, set_prompt_version_tag
    from mlflow.genai.datasets import get_dataset
    from mlflow.genai.optimize import GepaPromptOptimizer
    from mlflow.genai.scorers import Correctness

    banner("STEP 4: Optimize Prompt with GEPA")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    trace_context_header_builder = None
    try:
        from mlflow.tracing import get_tracing_context_headers_for_http_request

        trace_context_header_builder = get_tracing_context_headers_for_http_request
    except ImportError:
        trace_context_header_builder = None

    active_model = mlflow.set_active_model(name=AGENT_MODEL_NAME)
    active_model_id = active_model.model_id
    info(f"Active model set to: {AGENT_MODEL_NAME}")

    prompt_alias_uri = f"prompts:/{PROMPT_NAME}@production"
    source_prompt = load_prompt(prompt_alias_uri, model_id=active_model_id)
    source_prompt_uri = f"prompts:/{source_prompt.name}/{source_prompt.version}"
    dataset = get_dataset(name=DATASET_NAME)
    dataset_records = dataset.to_dict().get("records", [])
    train_data = _build_train_data(dataset)

    code_scorers = _selected_step3_code_scorers()
    llm_judge = Correctness(model=JUDGE_MODEL)
    optimization_scorers = [*code_scorers, llm_judge]

    inference_profile = (
        f"{LLM_MODEL_NAME}|temperature={LLM_TEMPERATURE}|max_tokens={LLM_MAX_TOKENS}"
    )

    print(f"  Experiment:          {EXPERIMENT_NAME}")
    print(f"  Source prompt alias: {prompt_alias_uri}")
    print(f"  Source prompt URI:   {source_prompt_uri}")
    print(f"  Dataset:             {DATASET_NAME} ({len(dataset_records)} records)")
    print(f"  Train subset:        {len(train_data)} records with expected_response")
    print(f"  Optimizer:           GEPA")
    print(f"  Optimizer model:     {OPTIMIZER_MODEL}")
    print(f"  Judge model:         {JUDGE_MODEL}")
    print(f"  Candidate alias:     {OPTIMIZATION_CANDIDATE_ALIAS}")
    print(f"  Inference profile:   {inference_profile}")
    print(f"  Scorers:             {[scorer.name for scorer in optimization_scorers]}")

    if dry_run:
        warn("DRY RUN - would run prompt optimization, skipping.")
        print()
        print("  What WOULD happen:")
        print("    - optimize the registered production prompt with GEPA")
        print("    - use gateway:/claude-3-5-sonnet for optimizer reflection")
        print("    - validate baseline vs optimized candidate on the Opnew dataset")
        print("    - keep the candidate alias unchanged unless validation improves")
        print("    - preserve judge scoring through gateway:/myendpoint")
        mlflow.clear_active_model()
        return

    check_backend()
    _probe_reflection_model(OPTIMIZER_MODEL, tracking_uri=tracking_uri)
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"
    effective_reflection_model, previous_env = _configure_gateway_reflection_model(
        OPTIMIZER_MODEL,
        tracking_uri=tracking_uri,
    )
    optimizer = GepaPromptOptimizer(
        reflection_model=effective_reflection_model,
        max_metric_calls=OPTIMIZER_MAX_METRIC_CALLS,
    )

    optimization_trace_ids: list[str] = []
    optimization_lock = Lock()
    optimization_predict_fn = _build_predict_fn(
        prompt_uri=source_prompt_uri,
        active_model_id=active_model_id,
        trace_context_header_builder=trace_context_header_builder,
        trace_collector=optimization_trace_ids,
        lock=optimization_lock,
    )

    run_name = f"opnew-optimize::v{source_prompt.version}::{int(time.time())}"
    result = None
    run_id = ""
    try:
        with mlflow.start_run(run_name=run_name) as run:
            try:
                client.link_prompt_version_to_run(run.info.run_id, source_prompt)
            except Exception:
                pass

            mlflow.set_tag("optimization_stage", "step4")
            mlflow.set_tag("experiment_label", EXPERIMENT_NAME)
            mlflow.set_tag("source_prompt_uri", source_prompt_uri)
            mlflow.set_tag("prompt_uri", source_prompt_uri)
            mlflow.set_tag("dataset_name", DATASET_NAME)
            mlflow.set_tag("optimizer_name", "GepaPromptOptimizer")
            mlflow.set_tag("optimizer_model", OPTIMIZER_MODEL)
            mlflow.set_tag("optimizer_effective_model", effective_reflection_model)
            mlflow.set_tag("optimizer_model_name", OPTIMIZER_MODEL_NAME)
            mlflow.set_tag("optimizer_internal_tracking", "disabled")
            mlflow.set_tag("judge_model", JUDGE_MODEL)
            mlflow.set_tag("judge_logged_model_name", JUDGE_MODEL_NAME)
            mlflow.set_tag("agent_model", AGENT_MODEL_NAME)
            mlflow.set_tag("agent_runtime_model", LLM_MODEL_NAME)
            mlflow.set_tag("agent_runtime_temperature", str(LLM_TEMPERATURE))
            mlflow.set_tag("agent_runtime_max_tokens", str(LLM_MAX_TOKENS))
            mlflow.set_tag("inference_profile", inference_profile)
            mlflow.set_tag("candidate_alias", OPTIMIZATION_CANDIDATE_ALIAS)
            mlflow.set_tag(
                "optimization_scorers",
                ",".join(scorer.name for scorer in optimization_scorers),
            )

            mlflow.log_param("source_prompt_uri", source_prompt_uri)
            mlflow.log_param("dataset_name", DATASET_NAME)
            mlflow.log_param("optimizer_name", "GepaPromptOptimizer")
            mlflow.log_param("optimizer_model", OPTIMIZER_MODEL)
            mlflow.log_param("optimizer_effective_model", effective_reflection_model)
            mlflow.log_param("optimizer_model_name", OPTIMIZER_MODEL_NAME)
            mlflow.log_param("optimizer_internal_tracking", "disabled")
            mlflow.log_param("optimizer_max_metric_calls", OPTIMIZER_MAX_METRIC_CALLS)
            mlflow.log_param("judge_model", JUDGE_MODEL)
            mlflow.log_param("judge_logged_model_name", JUDGE_MODEL_NAME)
            mlflow.log_param("agent_model", AGENT_MODEL_NAME)
            mlflow.log_param("agent_runtime_model", LLM_MODEL_NAME)
            mlflow.log_param("agent_runtime_temperature", LLM_TEMPERATURE)
            mlflow.log_param("agent_runtime_max_tokens", LLM_MAX_TOKENS)
            mlflow.log_param("inference_profile", inference_profile)
            mlflow.log_param("candidate_alias", OPTIMIZATION_CANDIDATE_ALIAS)
            mlflow.log_param(
                "optimization_scorers",
                ",".join(scorer.name for scorer in optimization_scorers),
            )

            result = optimize_prompts(
                predict_fn=optimization_predict_fn,
                train_data=train_data,
                prompt_uris=[source_prompt_uri],
                optimizer=optimizer,
                scorers=optimization_scorers,
                aggregation=_objective,
                enable_tracking=False,
            )

            optimized_prompt = result.optimized_prompts[0]
            optimized_prompt_uri = f"prompts:/{optimized_prompt.name}/{optimized_prompt.version}"
            mlflow.log_param("optimized_prompt_uri", optimized_prompt_uri)
            if result.initial_eval_score is not None:
                mlflow.log_metric("optimization_initial_score", float(result.initial_eval_score))
            if result.final_eval_score is not None:
                mlflow.log_metric("optimization_final_score", float(result.final_eval_score))
            if result.initial_eval_score is not None and result.final_eval_score is not None:
                mlflow.log_metric(
                    "optimization_score_improvement",
                    float(result.final_eval_score - result.initial_eval_score),
                )

            set_prompt_version_tag(
                optimized_prompt.name,
                optimized_prompt.version,
                "optimized_from_prompt_uri",
                source_prompt_uri,
            )
            set_prompt_version_tag(
                optimized_prompt.name,
                optimized_prompt.version,
                "optimizer",
                "GepaPromptOptimizer",
            )
            set_prompt_version_tag(
                optimized_prompt.name,
                optimized_prompt.version,
                "optimizer_model",
                OPTIMIZER_MODEL,
            )
            set_prompt_version_tag(
                optimized_prompt.name,
                optimized_prompt.version,
                "optimization_dataset",
                DATASET_NAME,
            )
            set_prompt_version_tag(
                optimized_prompt.name,
                optimized_prompt.version,
                "optimization_run_id",
                run.info.run_id,
            )

            baseline_validation = _run_validation_eval(
                label="baseline",
                prompt_uri=source_prompt_uri,
                dataset=dataset,
                active_model_id=active_model_id,
                trace_context_header_builder=trace_context_header_builder,
                scorers=optimization_scorers,
                client=client,
            )
            candidate_validation = _run_validation_eval(
                label="candidate",
                prompt_uri=optimized_prompt_uri,
                dataset=dataset,
                active_model_id=active_model_id,
                trace_context_header_builder=trace_context_header_builder,
                scorers=optimization_scorers,
                client=client,
            )

            baseline_score = float(baseline_validation["weighted_score"])
            candidate_score = float(candidate_validation["weighted_score"])
            score_delta = candidate_score - baseline_score
            mlflow.log_metric("baseline_validation_score", baseline_score)
            mlflow.log_metric("candidate_validation_score", candidate_score)
            mlflow.log_metric("candidate_validation_delta", score_delta)

            promoted = candidate_score > baseline_score
            mlflow.set_tag("candidate_promoted", str(promoted).lower())
            if promoted:
                set_prompt_alias(
                    optimized_prompt.name,
                    OPTIMIZATION_CANDIDATE_ALIAS,
                    optimized_prompt.version,
                )
                info(
                    f"Promoted {optimized_prompt_uri} to alias '{OPTIMIZATION_CANDIDATE_ALIAS}'"
                )
            else:
                warn("Candidate did not beat the validation baseline; alias unchanged.")

            optimization_trace_summary = {
                "trace_count": len(list(dict.fromkeys(optimization_trace_ids))),
                "trace_ids": list(dict.fromkeys(optimization_trace_ids)),
                "run_id": run.info.run_id,
                "linked": False,
                "link_error": None,
                "note": (
                    "Optimization search traces are kept separate so the run focuses on the "
                    "scored validation traces, matching the Step 3 inspection model."
                ),
            }

            summary = {
                "source_prompt_uri": source_prompt_uri,
                "optimized_prompt_uri": optimized_prompt_uri,
                "dataset_name": DATASET_NAME,
                "optimizer_name": "GepaPromptOptimizer",
                "optimizer_model": OPTIMIZER_MODEL,
                "judge_model": JUDGE_MODEL,
                "initial_eval_score": result.initial_eval_score,
                "final_eval_score": result.final_eval_score,
                "baseline_validation_score": baseline_score,
                "candidate_validation_score": candidate_score,
                "candidate_validation_delta": score_delta,
                "candidate_promoted": promoted,
                "candidate_alias": OPTIMIZATION_CANDIDATE_ALIAS,
                "optimization_trace_summary": optimization_trace_summary,
                "baseline_validation": baseline_validation,
                "candidate_validation": candidate_validation,
            }
            mlflow.log_dict(summary, "prompt_optimization/optimization_summary.json")

            comparison_rows = [
                {"label": "baseline", **baseline_validation["operational_summary"]},
                {"label": "candidate", **candidate_validation["operational_summary"]},
            ]
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                comparison_path = tmp_path / "validation_operational_comparison.json"
                comparison_path.write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
                mlflow.log_artifact(str(comparison_path), artifact_path="prompt_optimization")

            run_id = run.info.run_id
    finally:
        _restore_env(previous_env)
        mlflow.clear_active_model()

    if result is None:
        raise RuntimeError("Prompt optimization did not produce a result.")

    banner("STEP 4 RESULTS")
    print(f"  Run ID:            {run_id}")
    print(f"  Optimizer:         GepaPromptOptimizer")
    print(f"  Optimizer model:   {OPTIMIZER_MODEL}")
    print(f"  Judge model:       {JUDGE_MODEL}")
    print(f"  Source prompt:     {source_prompt_uri}")
    print(f"  Optimized prompt:  prompts:/{result.optimized_prompts[0].name}/{result.optimized_prompts[0].version}")
    print()
    print("  What to check in MLflow UI:")
    print(f"    1. Open experiment '{EXPERIMENT_NAME}'")
    print(f"    2. Open run '{run_name}'")
    print("    3. Inspect nested validation runs for baseline and candidate")
    print()
    print("    Overview:")
    print("      - optimization scores and validation deltas")
    print("      - nested validation latency / token / cost metrics")
    print("    Params / Tags:")
    print("      - optimizer_model = gateway:/claude-3-5-sonnet")
    print("      - judge_model = gateway:/myendpoint")
    print("      - candidate_alias and promotion decision")
    print("    Artifacts:")
    print("      - prompt_optimization/optimization_summary.json")
    print("      - prompt_optimization/*_metrics.json")
    print("      - prompt_optimization/*_operational_metrics.json")
    print("    Traces:")
    print("      - backend traces for optimization and validation remain linked to Opnew")
    print("      - judge traces go through AI Gateway myendpoint")
    print("      - optimizer reflection traffic goes through AI Gateway claude-3-5-sonnet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Optimize prompt with GEPA")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")
    args = parser.parse_args()

    tracking_uri = resolve_tracking_uri()
    step4_optimize_prompt(tracking_uri, dry_run=args.dry_run)

    banner("STEP 4 COMPLETE")
    print(f"  Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
