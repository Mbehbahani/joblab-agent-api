"""
Confidence Gate — post-execution quality evaluation state machine.

After a tool executes and returns results, this module evaluates the
quality of those results and decides the appropriate outcome:

  ANSWER            → High confidence. Present results to user.
  ASK_CLARIFICATION → Ambiguous intent or weak results. Ask user to refine.
  DECLINE           → Out-of-scope or policy violation. Politely refuse.
  HANDOFF           → Repeated low confidence or critical failure. Escalate.

Architecture Pattern:
  This is a "confidence-gated automation path" — a production safety
  pattern that prevents the agent from blindly returning poor results.
  Instead, the agent adapts its behavior based on result quality signals.

Gating Features (signals used for decision):
  - result_count: number of results returned
  - top_similarity: highest similarity score (semantic search)
  - score_margin: difference between top-1 and top-2 scores
  - filter_completeness: did all requested filters apply?
  - tool_execution_success: did the tool execute without errors?
  - is_empty: zero results?
  - latency_ms: how long the tool took
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GateOutcome(str, Enum):
    """Possible outcomes of the confidence gate evaluation."""
    ANSWER = "ANSWER"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    DECLINE = "DECLINE"
    HANDOFF = "HANDOFF"


@dataclass
class GateDecision:
    """
    The output of confidence gate evaluation.

    Attributes:
        outcome: The gate decision (ANSWER, ASK_CLARIFICATION, etc.)
        confidence: Float 0.0-1.0 representing overall confidence
        reason: Human-readable explanation of why this outcome was chosen
        signals: The raw signal values used for the decision
        suggestion: Optional message to show the user (e.g. clarification question)
    """
    outcome: GateOutcome
    confidence: float
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging / API response."""
        return {
            "outcome": self.outcome.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "signals": self.signals,
            "suggestion": self.suggestion,
        }


# ── Signal Extraction ───────────────────────────────────────────────────────


def _extract_signals(
    tool_name: str,
    tool_input: dict[str, Any],
    result_data: Any,
    *,
    latency_ms: float = 0.0,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Extract quality signals from tool execution results.

    These signals are the features used by the gate to make decisions.
    """
    signals: dict[str, Any] = {
        "tool_name": tool_name,
        "latency_ms": round(latency_ms, 1),
        "error": error,
        "tool_execution_success": error is None,
    }

    if error is not None:
        signals["result_count"] = 0
        signals["is_empty"] = True
        return signals

    if isinstance(result_data, list):
        signals["result_count"] = len(result_data)
        signals["is_empty"] = len(result_data) == 0
    else:
        signals["result_count"] = 1 if result_data else 0
        signals["is_empty"] = not result_data

    # Semantic search specific signals
    if tool_name == "semantic_search_jobs" and isinstance(result_data, list) and result_data:
        similarities = [r.get("similarity", 0) for r in result_data if "similarity" in r]
        if similarities:
            signals["top_similarity"] = max(similarities)
            signals["avg_similarity"] = round(sum(similarities) / len(similarities), 4)
            if len(similarities) >= 2:
                sorted_sims = sorted(similarities, reverse=True)
                signals["score_margin"] = round(sorted_sims[0] - sorted_sims[1], 4)
            else:
                signals["score_margin"] = 0.0

    # Filter completeness: count how many filters were provided
    filter_keys = [
        "country", "is_remote", "is_research", "job_level_std",
        "job_function_std", "company_industry_std", "job_type_filled",
        "tools",
        "platform", "posted_start", "posted_end", "role_keyword",
    ]
    provided_filters = sum(1 for k in filter_keys if tool_input.get(k) is not None)
    signals["filters_applied"] = provided_filters

    # Job stats specific signals
    if tool_name == "job_stats" and isinstance(result_data, list):
        total_count = sum(r.get("count", 0) for r in result_data)
        signals["total_count"] = total_count
        signals["group_count"] = len(result_data)

    return signals


# ── Gate Logic ──────────────────────────────────────────────────────────────
# Thresholds — tune these based on your eval dataset over time.

# Semantic search confidence thresholds
SEMANTIC_HIGH_CONFIDENCE = 0.45     # similarity above this → confident
SEMANTIC_LOW_CONFIDENCE = 0.25      # similarity below this → ask clarification
# TODO: Use SEMANTIC_MIN_MARGIN in evaluate_confidence to flag ambiguous cases
#       when the margin between top-1 and top-2 semantic scores is too small.
SEMANTIC_MIN_MARGIN = 0.02          # margin below this → results too similar

# Result count thresholds
EMPTY_RESULT_THRESHOLD = 0          # exactly zero
HIGH_RESULT_THRESHOLD = 80          # suspiciously many (likely too broad)

# Latency thresholds (ms)
HIGH_LATENCY_MS = 10_000            # over 10s is worrying


def evaluate_confidence(
    tool_name: str,
    tool_input: dict[str, Any],
    result_data: Any,
    *,
    latency_ms: float = 0.0,
    error: str | None = None,
    consecutive_low_confidence: int = 0,
) -> GateDecision:
    """
    Evaluate confidence in tool results and return a gate decision.

    Parameters:
        tool_name: Which tool was called
        tool_input: The parameters passed to the tool
        result_data: The raw data returned by the tool
        latency_ms: How long execution took
        error: Error message if tool failed
        consecutive_low_confidence: How many low-confidence turns in a row

    Returns:
        GateDecision with outcome, confidence score, and explanation
    """
    signals = _extract_signals(
        tool_name, tool_input, result_data,
        latency_ms=latency_ms, error=error,
    )

    # ── Rule 1: Tool execution failed → HANDOFF or ASK ─────────────────
    if not signals["tool_execution_success"]:
        if consecutive_low_confidence >= 2:
            return GateDecision(
                outcome=GateOutcome.HANDOFF,
                confidence=0.0,
                reason=f"Tool '{tool_name}' failed after {consecutive_low_confidence} consecutive low-confidence turns",
                signals=signals,
                suggestion="I'm having trouble processing this request. Could you try rephrasing, or would you like me to try a different approach?",
            )
        return GateDecision(
            outcome=GateOutcome.ASK_CLARIFICATION,
            confidence=0.1,
            reason=f"Tool '{tool_name}' execution error: {error}",
            signals=signals,
            suggestion="I encountered an error processing your request. Could you rephrase your question?",
        )

    # ── Rule 2: Empty results → ASK_CLARIFICATION ──────────────────────
    if signals["is_empty"]:
        return GateDecision(
            outcome=GateOutcome.ASK_CLARIFICATION,
            confidence=0.3,
            reason="No results returned. Filters may be too restrictive.",
            signals=signals,
            suggestion="No results found with the current filters. Would you like to broaden your search?",
        )

    # ── Rule 3: Semantic search specific confidence ────────────────────
    if tool_name == "semantic_search_jobs":
        top_sim = signals.get("top_similarity", 0)
        margin = signals.get("score_margin", 0)

        if top_sim >= SEMANTIC_HIGH_CONFIDENCE:
            confidence = min(0.7 + (top_sim - SEMANTIC_HIGH_CONFIDENCE) * 2, 1.0)
            return GateDecision(
                outcome=GateOutcome.ANSWER,
                confidence=round(confidence, 3),
                reason=f"Strong semantic match (top similarity={top_sim:.3f})",
                signals=signals,
            )

        if top_sim < SEMANTIC_LOW_CONFIDENCE:
            return GateDecision(
                outcome=GateOutcome.ASK_CLARIFICATION,
                confidence=round(top_sim, 3),
                reason=f"Weak semantic match (top similarity={top_sim:.3f})",
                signals=signals,
                suggestion="The matches I found are not very relevant. Could you describe what you're looking for in different terms?",
            )

        # Medium confidence zone
        confidence = 0.4 + (top_sim - SEMANTIC_LOW_CONFIDENCE) * 1.5
        return GateDecision(
            outcome=GateOutcome.ANSWER,
            confidence=round(min(confidence, 0.7), 3),
            reason=f"Moderate semantic match (top similarity={top_sim:.3f}, margin={margin:.3f})",
            signals=signals,
        )

    # ── Rule 4: Suspiciously many results → might be too broad ─────────
    if signals.get("result_count", 0) >= HIGH_RESULT_THRESHOLD:
        return GateDecision(
            outcome=GateOutcome.ANSWER,
            confidence=0.6,
            reason=f"Large result set ({signals['result_count']} results). Query may be too broad.",
            signals=signals,
        )

    # ── Rule 5: High latency warning ───────────────────────────────────
    if latency_ms > HIGH_LATENCY_MS:
        logger.warning("High latency: %.1fms for tool %s", latency_ms, tool_name)

    # ── Rule 6: Normal successful result → ANSWER ──────────────────────
    result_count = signals.get("result_count", 0)
    confidence = min(0.75 + min(result_count / 50, 0.25), 1.0)

    return GateDecision(
        outcome=GateOutcome.ANSWER,
        confidence=round(confidence, 3),
        reason=f"Successful tool execution ({result_count} results)",
        signals=signals,
    )


# ── Consecutive Tracker (per conversation) ──────────────────────────────────

_LOW_CONFIDENCE_COUNTER: dict[str, int] = {}


def track_confidence(conversation_id: str, decision: GateDecision) -> int:
    """
    Track consecutive low-confidence decisions per conversation.
    Returns the current count after updating.
    """
    if decision.confidence < 0.4:
        _LOW_CONFIDENCE_COUNTER[conversation_id] = (
            _LOW_CONFIDENCE_COUNTER.get(conversation_id, 0) + 1
        )
    else:
        _LOW_CONFIDENCE_COUNTER[conversation_id] = 0

    return _LOW_CONFIDENCE_COUNTER.get(conversation_id, 0)


def get_consecutive_low_confidence(conversation_id: str) -> int:
    """Get the current consecutive low-confidence count."""
    return _LOW_CONFIDENCE_COUNTER.get(conversation_id, 0)


def reset_confidence_tracker(conversation_id: str) -> None:
    """Reset the tracker for a conversation."""
    _LOW_CONFIDENCE_COUNTER.pop(conversation_id, None)
