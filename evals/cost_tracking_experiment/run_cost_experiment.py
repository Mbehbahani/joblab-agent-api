"""
Standalone MLflow Bedrock cost-tracking experiment runner.

Purpose:
- Keep this separate from app/main.py and production flow.
- Let you test your own prompts/tasks and compare Bedrock models.
- Log each call as an MLflow run while Bedrock autolog captures trace details
  (tokens, latency, inputs/outputs, model metadata).

Usage examples:
  python -m evals.cost_tracking_experiment.run_cost_experiment
  python -m evals.cost_tracking_experiment.run_cost_experiment --experiment "03-model-cost-comparison"
  python -m evals.cost_tracking_experiment.run_cost_experiment --models "us.anthropic.claude-3-5-haiku-20241022-v1:0,us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  python -m evals.cost_tracking_experiment.run_cost_experiment --step6-experiment "04-production-candidate-testing"
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import boto3
import mlflow
from mlflow.entities import SpanType

try:
    # Optional: load .env if available
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass


logger = logging.getLogger(__name__)


# Optional manual pricing map (USD per 1K tokens). Fill only if you want
# explicit dollar metric on top of autolog traces.
MODEL_PRICING_USD_PER_1K: dict[str, dict[str, float]] = {
    # "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.0008, "output": 0.004},
}


def parse_args() -> argparse.Namespace:
    default_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    default_model = os.getenv(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    parser = argparse.ArgumentParser(description="Run Bedrock model cost comparison with MLflow.")
    parser.add_argument(
        "--experiment",
        default="03-model-cost-comparison",
        help="MLflow experiment name for this test batch.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=default_tracking_uri,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--region",
        default=default_region,
        help="AWS region for Bedrock runtime client.",
    )
    parser.add_argument(
        "--models",
        default=default_model,
        help="Comma-separated Bedrock model IDs to compare.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Max completion tokens.",
    )
    parser.add_argument(
        "--step6-experiment",
        default="04-production-candidate-testing",
        help="MLflow experiment name for Step 6 (tags + metadata).",
    )
    parser.add_argument(
        "--step6-prompt",
        default="Explain the concept of LLM temperature.",
        help="User prompt for Step 6 tagged runs.",
    )
    parser.add_argument(
        "--skip-step6",
        action="store_true",
        help="Skip Step 6 (tags + metadata experiment).",
    )
    return parser.parse_args()


def get_tasks() -> list[dict[str, str]]:
    # Customize these for your real AI task/prompt style.
    return [
        {
            "name": "job_scope_guardrail",
            "prompt": (
                "You are JobLab AI. Explain in 4 concise bullets what job-market "
                "questions you can answer and what you must refuse."
            ),
        },
        {
            "name": "analytics_summary_style",
            "prompt": (
                "Write a compact answer style guide for job analytics responses: "
                "1) summary sentence, 2) evidence bullets, 3) uncertainty note."
            ),
        },
    ]


def get_step6_configs(models: list[str]) -> list[dict[str, Any]]:
    baseline_model = models[0]
    creative_model = models[1] if len(models) > 1 else models[0]
    return [
        {
            "name": "baseline",
            "model": baseline_model,
            "temperature": 0.7,
            "system_prompt": "You are a helpful JobLab assistant.",
        },
        {
            "name": "creative",
            "model": creative_model,
            "temperature": 1.0,
            "system_prompt": "You are a creative job-market writing assistant.",
        },
    ]


def extract_text(response: dict[str, Any]) -> str:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(b.get("text", "") for b in blocks if "text" in b).strip()


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = MODEL_PRICING_USD_PER_1K.get(model)
    if not pricing:
        return None
    in_cost = (input_tokens / 1000.0) * pricing["input"]
    out_cost = (output_tokens / 1000.0) * pricing["output"]
    return round(in_cost + out_cost, 8)


def get_cost_metrics(model: str, input_tokens: int, output_tokens: int) -> dict[str, float] | None:
    pricing = MODEL_PRICING_USD_PER_1K.get(model)
    if not pricing:
        return None

    input_cost = round((input_tokens / 1000.0) * pricing["input"], 8)
    output_cost = round((output_tokens / 1000.0) * pricing["output"], 8)
    total_cost = round(input_cost + output_cost, 8)
    total_tokens = input_tokens + output_tokens

    effective_cost_per_1k_tokens = (
        round((total_cost / total_tokens) * 1000.0, 8) if total_tokens > 0 else 0.0
    )

    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "estimated_cost_usd": total_cost,
        "effective_cost_per_1k_tokens": effective_cost_per_1k_tokens,
    }


def llm_call_with_cost(
    client: Any,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    run_name: str,
    task_name: str,
) -> dict[str, Any]:
    with mlflow.start_run(run_name=run_name):
        with mlflow.start_span(
            name=f"cost_test::{task_name}",
            span_type=SpanType.CHAT_MODEL,
        ) as span:
            span.set_inputs(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            response = client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                },
            )

            usage = response.get("usage", {})
            input_tokens = int(usage.get("inputTokens", 0))
            output_tokens = int(usage.get("outputTokens", 0))
            total_tokens = int(usage.get("totalTokens", input_tokens + output_tokens))

            # Explicit trace-level token usage + cost attributes.
            # This guarantees trace cost visibility even when automatic pricing
            # resolution cannot derive it for the model/provider.
            span.set_attribute(
                "mlflow.chat.tokenUsage",
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            )
            span.set_attribute("mlflow.modelName", model)
            cost_metrics = get_cost_metrics(model, input_tokens, output_tokens)
            if cost_metrics is not None:
                span.set_attribute(
                    "mlflow.llm.cost",
                    {
                        "input_cost": cost_metrics["input_cost_usd"],
                        "output_cost": cost_metrics["output_cost_usd"],
                        "total_cost": cost_metrics["estimated_cost_usd"],
                    },
                )
                span.set_attribute(
                    "joblab.cost.effective_per_1k_tokens",
                    cost_metrics["effective_cost_per_1k_tokens"],
                )
            answer = extract_text(response)
            span.set_outputs({"answer": answer})

        mlflow.log_params(
            {
                "task_name": task_name,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        mlflow.log_metrics(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )
        mlflow.log_text(prompt, "prompt.txt")

        if cost_metrics is not None:
            mlflow.log_metrics(cost_metrics)

        mlflow.log_text(answer, "response.txt")
        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_metrics": cost_metrics,
        }


def run_step6_tags_and_metadata(
    client: Any,
    *,
    experiment_name: str,
    configs: list[dict[str, Any]],
    test_prompt: str,
    max_tokens: int,
) -> None:
    logger.info("Step 6: Organizing Experiments with Tags and Metadata")
    mlflow.set_experiment(experiment_name)
    logger.info("Step 6 experiment: %s", experiment_name)

    for config in configs:
        with mlflow.start_run(run_name=config["name"]):
            response = client.converse(
                modelId=config["model"],
                system=[{"text": config["system_prompt"]}],
                messages=[{"role": "user", "content": [{"text": test_prompt}]}],
                inferenceConfig={
                    "temperature": float(config["temperature"]),
                    "maxTokens": max_tokens,
                },
            )

            answer = extract_text(response)

            mlflow.set_tags(
                {
                    "config_name": str(config["name"]),
                    "task": "explanation",
                    "stage": "testing",
                    "team": "ai-research",
                    "version": "v1.0",
                    "production_candidate": str(config["name"] == "baseline").lower(),
                }
            )
            mlflow.log_dict(config, "config.json")
            mlflow.log_text(test_prompt, "prompt.txt")
            mlflow.log_text(answer, "response.txt")

            logger.info("  %s done", config["name"])

    logger.info(
        "Step 6 complete. In MLflow UI, filter tags with production_candidate=true."
    )


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        logger.error("No models provided.")
        return 1

    # Important: enable autolog BEFORE client creation.
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name=args.region)
    tasks = get_tasks()

    logger.info("Experiment: %s", args.experiment)
    logger.info("Tracking URI: %s", args.tracking_uri)
    logger.info("Models: %s", models)
    logger.info("Tasks: %s", [t["name"] for t in tasks])

    summary_rows: list[dict[str, Any]] = []

    for model in models:
        for task in tasks:
            run_name = f"{task['name']}__{model.split('/')[-1]}"
            logger.info("Running model=%s task=%s", model, task["name"])
            result = llm_call_with_cost(
                client,
                model=model,
                prompt=task["prompt"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                run_name=run_name,
                task_name=task["name"],
            )
            answer = str(result["answer"])
            preview = (answer[:180] + "...") if len(answer) > 180 else answer
            logger.info("Response preview: %s", preview)

            row = {
                "model": model,
                "task": task["name"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"],
            }
            if result["cost_metrics"] is not None:
                row.update(result["cost_metrics"])
            summary_rows.append(row)

    logger.info("Step 5: Cost Efficiency Summary")
    for row in summary_rows:
        if "estimated_cost_usd" in row:
            logger.info(
                "model=%s | task=%s | tokens=%s | cost=$%s | effective_per_1k=$%s",
                row["model"],
                row["task"],
                row["total_tokens"],
                row["estimated_cost_usd"],
                row["effective_cost_per_1k_tokens"],
            )
        else:
            logger.info(
                "model=%s | task=%s | tokens=%s | cost=NA (add pricing map entry)",
                row["model"],
                row["task"],
                row["total_tokens"],
            )

    if not args.skip_step6:
        step6_configs = get_step6_configs(models)
        run_step6_tags_and_metadata(
            client,
            experiment_name=args.step6_experiment,
            configs=step6_configs,
            test_prompt=args.step6_prompt,
            max_tokens=args.max_tokens,
        )

    logger.info("Done. Open MLflow UI and inspect traces + run metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
