"""
Turn Logger — per-turn tracing and observability with MLflow integration.

Logs every agent turn with structured metrics:
  - Prompt text and tool selected
  - Tool arguments and result summary
  - Confidence gate decision
  - Latency (total, tool execution, LLM inference)
  - Token usage
  - Error information

Storage backends:
  1. MLflow (primary) — when available, logs as MLflow runs for experiment tracking
  2. Structured logging (fallback) — always logs via Python logger for CloudWatch/stdout

Architecture Pattern:
  This is the "observability layer" that enables:
  - Production analytics (which tools are used, success rate, latency distribution)
  - Evaluation tracking (compare prompt versions, model versions)
  - Debugging (trace a specific turn to see what went wrong)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ── MLflow availability check ──────────────────────────────────────────────

_mlflow_available = False
try:
    import mlflow
    _mlflow_available = True
except ImportError:
    mlflow = None  # type: ignore
    logger.info("MLflow not installed — using structured logging only")


def is_mlflow_available() -> bool:
    """Check if MLflow is installed and importable."""
    return _mlflow_available


# ── Turn Data ──────────────────────────────────────────────────────────────


@dataclass
class TurnRecord:
    """
    Complete record of a single agent turn.

    A "turn" = one user prompt → agent response cycle, which may
    include multiple tool calls in a loop.
    """
    # Identity
    conversation_id: str = ""
    turn_id: str = ""

    # Input
    user_prompt: str = ""
    prompt_char_count: int = 0

    # Policy
    policy_version: str = ""
    system_prompt_chars: int = 0

    # Tool tracking
    tools_called: list[dict[str, Any]] = field(default_factory=list)
    tool_rounds: int = 0
    soft_enforcement_retries: int = 0

    # Confidence gate
    gate_outcome: str = ""
    gate_confidence: float = 0.0
    gate_reason: str = ""

    # Timing (milliseconds)
    total_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Result
    result_type: str = ""   # "answer", "error", "clarification", "handoff"
    result_length: int = 0
    error: str | None = None

    # Custom tags
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging."""
        return {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "user_prompt": self.user_prompt[:200],  # truncate for logging
            "prompt_char_count": self.prompt_char_count,
            "policy_version": self.policy_version,
            "tools_called": [t.get("name", "unknown") for t in self.tools_called],
            "tool_rounds": self.tool_rounds,
            "soft_enforcement_retries": self.soft_enforcement_retries,
            "gate_outcome": self.gate_outcome,
            "gate_confidence": self.gate_confidence,
            "gate_reason": self.gate_reason,
            "total_latency_ms": self.total_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "tool_latency_ms": self.tool_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "result_type": self.result_type,
            "result_length": self.result_length,
            "error": self.error,
        }


# ── Structured Logger (always active) ──────────────────────────────────────

def log_turn_structured(record: TurnRecord) -> None:
    """
    Log turn record as structured JSON to Python logger.
    This always works — in Lambda, it goes to CloudWatch.
    """
    log_data = record.to_dict()
    logger.info(
        "TURN_LOG | conv=%s | turn=%s | tool=%s | gate=%s(%.2f) | latency=%.0fms | tokens=%d",
        record.conversation_id[:8],
        record.turn_id[:8],
        ",".join(t.get("name", "?") for t in record.tools_called) or "none",
        record.gate_outcome,
        record.gate_confidence,
        record.total_latency_ms,
        record.total_tokens,
    )
    # Full structured log at DEBUG level
    logger.debug("TURN_LOG_FULL | %s", json.dumps(log_data, default=str))


# ── MLflow Logger ──────────────────────────────────────────────────────────


def _get_experiment_name() -> str:
    """Get experiment name from settings (lazy import to avoid circular deps)."""
    try:
        from app.config import get_settings
        return get_settings().mlflow_experiment_name
    except Exception:
        return "joblab-ai-agent-production"  # local dev fallback; Lambda sets this via MLFLOW_EXPERIMENT_NAME env var


def _ensure_experiment() -> None:
    """Create MLflow experiment if it doesn't exist."""
    if not _mlflow_available:
        return
    try:
        name = _get_experiment_name()
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            mlflow.create_experiment(name)
        mlflow.set_experiment(name)
    except Exception as e:
        logger.warning("MLflow experiment setup failed: %s", e)


def log_turn_mlflow(record: TurnRecord) -> None:
    """
    Log turn as an MLflow run with metrics, params, and tags.

    Metrics (numeric, trackable over time):
      - total_latency_ms, llm_latency_ms, tool_latency_ms
      - gate_confidence
      - input_tokens, output_tokens, total_tokens
      - tool_rounds, soft_enforcement_retries

    Params (string, per-run):
      - policy_version, gate_outcome, result_type
      - tools_called, conversation_id

    Tags:
      - Custom tags passed in the record
    """
    if not _mlflow_available:
        return

    try:
        _ensure_experiment()

        with mlflow.start_run(run_name=f"turn-{record.turn_id[:8]}"):
            # Metrics
            mlflow.log_metrics({
                "total_latency_ms": record.total_latency_ms,
                "llm_latency_ms": record.llm_latency_ms,
                "tool_latency_ms": record.tool_latency_ms,
                "gate_confidence": record.gate_confidence,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "total_tokens": record.total_tokens,
                "tool_rounds": record.tool_rounds,
                "soft_enforcement_retries": record.soft_enforcement_retries,
                "prompt_char_count": record.prompt_char_count,
                "result_length": record.result_length,
            })

            # Params
            mlflow.log_params({
                "policy_version": record.policy_version,
                "gate_outcome": record.gate_outcome,
                "gate_reason": record.gate_reason[:250],
                "result_type": record.result_type,
                "tools_called": ",".join(
                    t.get("name", "?") for t in record.tools_called
                ) or "none",
                "conversation_id": record.conversation_id[:36],
                "turn_id": record.turn_id,
                "error": (record.error or "")[:250],
            })

            # Tags
            mlflow.set_tags({
                "agent": "joblab-genai",
                "component": "ai_router",
                **record.tags,
            })

            # Log the full prompt as an artifact (for debugging)
            if record.user_prompt:
                mlflow.log_text(
                    record.user_prompt,
                    artifact_file="user_prompt.txt",
                )

    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)


# ── Combined Logger ────────────────────────────────────────────────────────


def log_turn(record: TurnRecord) -> None:
    """
    Log a turn record to all available backends.
    Always logs structured. Also logs to MLflow SDK or Lite REST client.
    """
    log_turn_structured(record)
    if _mlflow_available:
        log_turn_mlflow(record)
    else:
        # Fallback: use lightweight REST client (initialised in main.py)
        try:
            from app.services.mlflow_lite import get_lite_client
            lite = get_lite_client()
            if lite:
                lite.log_turn_async(record.to_dict())
        except Exception:
            pass  # already logged elsewhere


# ── Timer Context Manager ──────────────────────────────────────────────────


class TurnTimer:
    """
    Context-based timer for measuring latencies within a turn.

    Usage:
        timer = TurnTimer()
        timer.start()

        with timer.track("llm"):
            response = invoke_claude(...)

        with timer.track("tool"):
            result = executor(args)

        record.total_latency_ms = timer.total_ms()
        record.llm_latency_ms = timer.get("llm")
        record.tool_latency_ms = timer.get("tool")
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._segments: dict[str, float] = {}
        self._segment_start: float = 0.0

    def start(self) -> None:
        self._start = time.time()

    @contextmanager
    def track(self, name: str):
        """Track a named time segment."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start) * 1000
            # Accumulate (a segment like "llm" may be called multiple times)
            self._segments[name] = self._segments.get(name, 0) + elapsed

    def total_ms(self) -> float:
        """Total elapsed time since start() in milliseconds."""
        return (time.time() - self._start) * 1000 if self._start else 0.0

    def get(self, name: str) -> float:
        """Get accumulated time for a segment in milliseconds."""
        return round(self._segments.get(name, 0.0), 1)


# ── Aggregate Metrics (in-memory, for /metrics endpoint) ──────────────────

_METRICS: dict[str, Any] = {
    "total_turns": 0,
    "outcome_counts": {"ANSWER": 0, "ASK_CLARIFICATION": 0, "DECLINE": 0, "HANDOFF": 0},
    "tool_usage_counts": {},
    "avg_latency_ms": 0.0,
    "avg_confidence": 0.0,
    "error_count": 0,
    "_latency_sum": 0.0,
    "_confidence_sum": 0.0,
}


def update_aggregate_metrics(record: TurnRecord) -> None:
    """Update in-memory aggregate metrics from a turn record."""
    _METRICS["total_turns"] += 1
    _METRICS["_latency_sum"] += record.total_latency_ms
    _METRICS["_confidence_sum"] += record.gate_confidence

    if record.total_latency_ms > 0:
        _METRICS["avg_latency_ms"] = round(
            _METRICS["_latency_sum"] / _METRICS["total_turns"], 1
        )
    _METRICS["avg_confidence"] = round(
        _METRICS["_confidence_sum"] / _METRICS["total_turns"], 3
    )

    outcome = record.gate_outcome
    if outcome in _METRICS["outcome_counts"]:
        _METRICS["outcome_counts"][outcome] += 1

    for tc in record.tools_called:
        tool_name = tc.get("name", "unknown")
        _METRICS["tool_usage_counts"][tool_name] = (
            _METRICS["tool_usage_counts"].get(tool_name, 0) + 1
        )

    if record.error:
        _METRICS["error_count"] += 1


def get_aggregate_metrics() -> dict[str, Any]:
    """Return current aggregate metrics snapshot."""
    return {
        "total_turns": _METRICS["total_turns"],
        "outcome_counts": dict(_METRICS["outcome_counts"]),
        "tool_usage_counts": dict(_METRICS["tool_usage_counts"]),
        "avg_latency_ms": _METRICS["avg_latency_ms"],
        "avg_confidence": _METRICS["avg_confidence"],
        "error_count": _METRICS["error_count"],
        "success_rate": round(
            _METRICS["outcome_counts"]["ANSWER"] / max(_METRICS["total_turns"], 1), 3
        ),
        "clarification_rate": round(
            _METRICS["outcome_counts"]["ASK_CLARIFICATION"] / max(_METRICS["total_turns"], 1), 3
        ),
        "handoff_rate": round(
            _METRICS["outcome_counts"]["HANDOFF"] / max(_METRICS["total_turns"], 1), 3
        ),
    }
