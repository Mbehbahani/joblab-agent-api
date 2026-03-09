"""
AWS Bedrock service – thin wrapper around boto3 Converse API.
Keeps all Bedrock interaction in one place so it can later be swapped for
LangChain or another orchestration layer without touching routers.

Uses the Bedrock **Converse API** (not invoke_model) for:
  • Full MLflow autolog tracing with function-calling metadata
  • Unified tool calling format (toolSpec / toolUse / toolResult)
  • Structured system prompts and inference config

Supports:
  • Plain chat (messages only)
  • Tool calling (Bedrock Converse ``toolConfig`` parameter)
"""

import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ── MLflow availability ─────────────────────────────────────────────────────

_mlflow = None
try:
    import mlflow as _mlflow
except ImportError:
    pass

# ── Singleton client ────────────────────────────────────────────────────────

_bedrock_client = None

_MODEL_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
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


def _get_client(settings: Settings | None = None):
    """Return a reusable bedrock-runtime client (created once)."""
    global _bedrock_client
    if _bedrock_client is None:
        settings = settings or get_settings()
        _bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_region,
        )
    return _bedrock_client


def reset_client() -> None:
    """Force recreation of the boto3 client (e.g. after autolog is enabled)."""
    global _bedrock_client
    _bedrock_client = None


def _resolve_model_pricing(settings: Settings) -> tuple[float, float] | None:
    """Resolve USD-per-1M-token pricing for the configured model."""
    if (
        settings.bedrock_input_cost_per_million_tokens is not None
        and settings.bedrock_output_cost_per_million_tokens is not None
    ):
        return (
            settings.bedrock_input_cost_per_million_tokens,
            settings.bedrock_output_cost_per_million_tokens,
        )

    return _MODEL_PRICING_PER_MILLION_TOKENS.get(settings.bedrock_model_id.lower())


def _calculate_usage_cost(
    usage: dict[str, Any],
    settings: Settings,
) -> dict[str, float] | None:
    """Convert token usage to USD cost for MLflow span attributes."""
    pricing = _resolve_model_pricing(settings)
    if not pricing:
        return None

    input_tokens = int(usage.get("inputTokens", 0) or 0)
    output_tokens = int(usage.get("outputTokens", 0) or 0)
    input_rate, output_rate = pricing

    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


# ── Message helpers ─────────────────────────────────────────────────────────


def _ensure_content_blocks(content: Any) -> list[dict[str, Any]]:
    """
    Normalize message content to Converse API format.

    Converse requires content as a list of content blocks:
      [{"text": "..."}, {"toolUse": {...}}, {"toolResult": {...}}]

    This converts:
      - Plain string → [{"text": "..."}]
      - Already a list → returned as-is
    """
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        return content
    return [{"text": str(content)}]


def make_user_message(content: str | list[dict]) -> dict[str, Any]:
    """Build a user message in Converse format."""
    return {"role": "user", "content": _ensure_content_blocks(content)}


def make_tool_result_block(tool_use_id: str, result_content: str) -> dict[str, Any]:
    """Build a toolResult content block for Converse API."""
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "content": [{"text": result_content}],
        }
    }


# ── Public helpers ──────────────────────────────────────────────────────────


def invoke_claude(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Send a Converse API call to Claude via Bedrock.

    Parameters
    ----------
    messages : list of {"role": …, "content": [content_blocks]}
        Content blocks use Converse format:
        - Text: {"text": "..."}
        - Tool use: {"toolUse": {"toolUseId": ..., "name": ..., "input": ...}}
        - Tool result: {"toolResult": {"toolUseId": ..., "content": [{"text": ...}]}}
    system   : optional system prompt (plain string, wrapped automatically)
    tools    : optional list of tool definitions in Bedrock toolSpec format
    max_tokens / temperature : per-call overrides (fall back to settings)

    Returns
    -------
    dict : Full Converse API response with keys:
        - output.message.role, output.message.content
        - stopReason ("end_turn" | "tool_use")
        - usage (inputTokens, outputTokens, totalTokens)
    """
    settings = settings or get_settings()
    client = _get_client(settings)

    # Ensure all message content is in list-of-blocks format
    normalized_messages = []
    for msg in messages:
        normalized_messages.append({
            "role": msg["role"],
            "content": _ensure_content_blocks(msg.get("content", "")),
        })

    call_kwargs: dict[str, Any] = {
        "modelId": settings.bedrock_model_id,
        "messages": normalized_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens or settings.bedrock_max_tokens,
            "temperature": temperature if temperature is not None else settings.bedrock_temperature,
        },
    }

    if system:
        call_kwargs["system"] = [{"text": system}]

    if tools:
        call_kwargs["toolConfig"] = {"tools": tools}

    tool_count = len(tools) if tools else 0
    logger.info(
        "Invoking Bedrock (Converse) model=%s tokens=%s tools=%s",
        settings.bedrock_model_id,
        call_kwargs["inferenceConfig"]["maxTokens"],
        tool_count,
    )

    start_time = time.time()
    try:
        response = client.converse(**call_kwargs)
    except ClientError as exc:
        logger.error("Bedrock Converse invocation failed: %s", exc)
        if _mlflow:
            try:
                span = _mlflow.get_current_active_span()
                if span:
                    span.set_attributes({
                        "bedrock.error": str(exc),
                        "bedrock.model_id": settings.bedrock_model_id,
                    })
            except Exception:
                pass
        raise

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # ── MLflow span enrichment ──────────────────────────────────────────
    if _mlflow:
        try:
            span = _mlflow.get_current_active_span()
            if span:
                usage = response.get("usage", {})
                attributes: dict[str, Any] = {
                    "bedrock.model_id": settings.bedrock_model_id,
                    "bedrock.max_tokens": call_kwargs["inferenceConfig"]["maxTokens"],
                    "bedrock.temperature": call_kwargs["inferenceConfig"]["temperature"],
                    "bedrock.tool_count": tool_count,
                    "bedrock.stop_reason": response.get("stopReason", ""),
                    "bedrock.input_tokens": usage.get("inputTokens", 0),
                    "bedrock.output_tokens": usage.get("outputTokens", 0),
                    "bedrock.total_tokens": usage.get("totalTokens", 0),
                    "bedrock.latency_ms": elapsed_ms,
                    "bedrock.stopped_for_tool_use": response.get("stopReason") == "tool_use",
                    "mlflow.chat.tokenUsage": {
                        "input_tokens": usage.get("inputTokens", 0),
                        "output_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0),
                    },
                }
                usage_cost = _calculate_usage_cost(usage, settings)
                if usage_cost:
                    attributes["mlflow.llm.cost"] = usage_cost
                    attributes["bedrock.input_cost_usd"] = usage_cost["input_cost"]
                    attributes["bedrock.output_cost_usd"] = usage_cost["output_cost"]
                    attributes["bedrock.total_cost_usd"] = usage_cost["total_cost"]
                span.set_attributes(attributes)
        except Exception as e:
            logger.debug("MLflow span enrichment failed (non-fatal): %s", e)

    return response


# ── Response parsing (Converse API format) ──────────────────────────────────


def get_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the assistant message from a Converse response."""
    return response.get("output", {}).get("message", {"role": "assistant", "content": []})


def extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant's text out of a Converse API response."""
    message = get_assistant_message(response)
    content_blocks = message.get("content", [])
    return "".join(
        block["text"] for block in content_blocks if "text" in block
    )


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return a list of tool_use blocks from a Converse response.
    Each returned dict has keys: id, name, input (normalized for downstream use).
    """
    message = get_assistant_message(response)
    content_blocks = message.get("content", [])
    result = []
    for block in content_blocks:
        if "toolUse" in block:
            tu = block["toolUse"]
            result.append({
                "id": tu.get("toolUseId", ""),
                "name": tu.get("name", ""),
                "input": tu.get("input", {}),
            })
    return result


def has_tool_use(response: dict[str, Any]) -> bool:
    """True when the Converse stopReason indicates a tool call."""
    return response.get("stopReason") == "tool_use"


def get_usage(response: dict[str, Any]) -> dict[str, Any]:
    """
    Extract usage from Converse response and normalize key names
    to snake_case for downstream compatibility.
    """
    usage = response.get("usage", {})
    return {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "total_tokens": usage.get("totalTokens", 0),
    }


def quick_ask(prompt: str, *, system: str | None = None) -> str:
    """Convenience: single user prompt → assistant text."""
    resp = invoke_claude(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=system,
    )
    return extract_text(resp)
