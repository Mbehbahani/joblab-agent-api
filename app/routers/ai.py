"""
AI router - exposes Bedrock-backed endpoints.
Supports Claude tool calling for structured Supabase queries.

Architecture: Policy / Execution / Evaluation separation
  - prompt_policy.py  -> composable system prompt (testable, swappable)
  - confidence_gate.py -> ANSWER/ASK_CLARIFICATION/DECLINE/HANDOFF state machine
  - turn_logger.py     -> per-turn tracing with MLflow + structured logging

MLflow Tracing:
  - Each /ai/ask call creates a parent trace span
  - Child spans: LLM inference, tool execution, confidence gate
  - Autolog (from main.py) also traces raw boto3 calls
"""
import json
import logging
import uuid
import time
from contextlib import nullcontext
from typing import Any

from fastapi import APIRouter, HTTPException, Header, Request

from app.config import get_settings
from app.schemas.ai import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    MlflowFlushRequest,
    MlflowFlushResponse,
)
from app.services.bedrock import (
    invoke_claude,
    extract_text,
    extract_tool_calls,
    has_tool_use,
    get_assistant_message,
    get_usage,
    make_tool_result_block,
)
from app.services.joblab_tools import TOOL_DEFINITIONS, TOOL_EXECUTORS
from app.services.conversation_memory import (
    get_last_tool,
    set_last_tool,
    get_pending_followup,
    set_pending_followup,
    clear_pending_followup,
    set_mentioned_jobs,
    get_mentioned_jobs,
)
from app.services.prompt_policy import (
    get_system_prompt,
    DEFAULT_POLICY,
    POLICY_VERSION,
)
from app.services.confidence_gate import (
    evaluate_confidence,
    track_confidence,
    get_consecutive_low_confidence,
    GateOutcome,
)
from app.services.turn_logger import (
    TurnRecord,
    TurnTimer,
    log_turn,
    update_aggregate_metrics,
    get_aggregate_metrics,
)

logger = logging.getLogger(__name__)
# ── MLflow availability ─────────────────────────────────────────────────────
_mlflow = None
_SpanType = None
try:
    import mlflow as _mlflow
    from mlflow.entities import SpanType as _SpanType
except ImportError:
    pass

_set_tracing_context_from_http_request_headers = None
if _mlflow is not None:
    try:
        from mlflow.tracing import (
            set_tracing_context_from_http_request_headers as _set_tracing_context_from_http_request_headers,
        )
    except ImportError:
        _set_tracing_context_from_http_request_headers = None

router = APIRouter(prefix="/ai", tags=["ai"])

# ── System prompt from policy layer (replaces monolithic string) ────────────
# The prompt is now composed from discrete, testable sections.
# See app/services/prompt_policy.py for the full policy.
JOBLAB_SYSTEM = get_system_prompt(DEFAULT_POLICY)

MAX_TOOL_ROUNDS = 5  # safety: prevent infinite tool loops
MAX_SOFT_ENFORCEMENT_RETRIES = 2
DB_RELATED_KEYWORDS = [
    "job",
    "jobs",
    "how many",
    "count",
    "list",
    "show",
    "find",
    "trend",
    "increase",
    "decrease",
    "growth",
    "decline",
    "month-over-month",
    "comparison",
    "compare",
    "change",
    "hiring",
    "posted",
    "research",
    "remote",
    "industry",
    "full-time",
    "part-time",
    "contract",
    "internship",
    # Semantic search triggers
    "related to",
    "about",
    "similar to",
    "positions mentioning",
    "jobs involving",
    "skills like",
    "roles that deal with",
    # Job detail follow-up triggers
    "job id",
    "link",
    "url",
    "detail",
    "more about",
]


def _safe_logged_model_name(base_name: str, version: str) -> str:
    safe_version = version.replace(".", "_")
    return f"{base_name}-v{safe_version}"


LINKED_PROMPTS_TAG = "mlflow.linkedPrompts"

def _is_database_related(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in DB_RELATED_KEYWORDS)

def _extract_jobs_from_results(tool_name: str, result_data: Any) -> list[dict[str, Any]]:
    """
    Extract slim job metadata from tool execution results.
    Works for both search_jobs (returns job rows) and semantic_search_jobs
    (returns enriched chunks with job metadata).
    Returns a list of dicts suitable for set_mentioned_jobs().
    """
    jobs: list[dict[str, Any]] = []
    if not isinstance(result_data, list):
        return jobs
    for row in result_data:
        job_id = row.get("job_id")
        if not job_id:
            continue
        jobs.append({
            "job_id": job_id,
            "actual_role": row.get("actual_role", ""),
            "company_name": row.get("company_name", ""),
            "url": row.get("url", ""),
            "posted_date": row.get("posted_date", ""),
            "country": row.get("country", ""),
        })
    return jobs

def _build_ai_job_results(tool_name: str | None, result_data: Any) -> list[dict[str, Any]] | None:
    """
    Build a frontend-friendly job list payload for AI answers.
    Only listing-style tools should expose structured job cards.
    """
    if tool_name not in {"search_jobs", "semantic_search_jobs"}:
        return None
    if not isinstance(result_data, list):
        return None

    jobs: list[dict[str, Any]] = []
    for row in result_data[:10]:
        job_id = row.get("job_id")
        if not job_id:
            continue
        jobs.append({
            "job_id": job_id,
            "actual_role": row.get("actual_role", ""),
            "company_name": row.get("company_name", ""),
            "country": row.get("country", ""),
            "location": row.get("location", ""),
            "url": row.get("url", ""),
            "posted_date": row.get("posted_date", ""),
            "job_level_std": row.get("job_level_std", ""),
            "job_function_std": row.get("job_function_std", ""),
            "job_type_filled": row.get("job_type_filled", ""),
            "platform": row.get("platform", ""),
            "is_remote": row.get("is_remote"),
            "is_research": row.get("is_research"),
            "skills": row.get("skills", ""),
            "tools": row.get("tools", ""),
        })
    return jobs or None

def _is_job_detail_followup(prompt: str) -> bool:
    """Detect if user is asking about a previously mentioned job."""
    prompt_lower = prompt.lower()
    triggers = [
        "its job id", "its id", "its link", "its url", "the link",
        "job id", "give me the link", "more about", "details on",
        "tell me about the", "which one", "the first one", "the second",
        "the third", "the last one", "more info", "more detail",
        "show me the", "open the",
    ]
    return any(t in prompt_lower for t in triggers)

# Short affirmative/vague follow-ups that imply "continue with previous context"
_AFFIRMATIVE_PATTERNS = [
    "yes", "yeah", "yep", "yup", "sure", "please", "ok", "okay",
    "go ahead", "do it", "show me", "tell me", "absolutely",
    "of course", "why not", "right", "correct", "exactly",
    "more", "details", "elaborate", "explain", "break it down",
    "breakdown", "continue", "go on", "please do",
]

# Negative patterns — user declining a follow-up offer
_NEGATIVE_PATTERNS = [
    "no", "nah", "nope", "no thanks", "no thank you",
    "not now", "not really", "never mind", "nevermind",
    "skip", "pass", "i'm good", "im good", "that's all",
    "thats all", "all good", "nothing else",
]

_NEGATED_RESEARCH_PATTERNS = [
    "non research",
    "non-research",
    "not research",
    "exclude research",
    "excluding research",
    "without research",
]

def _is_affirmative_followup(prompt: str) -> bool:
    """Check if prompt is a short affirmative/continuation response."""
    prompt_lower = prompt.strip().lower().rstrip("!.,?")
    return prompt_lower in _AFFIRMATIVE_PATTERNS

def _is_negative_followup(prompt: str) -> bool:
    """Check if prompt is a short negative/declining response."""
    prompt_lower = prompt.strip().lower().rstrip("!.,?")
    return prompt_lower in _NEGATIVE_PATTERNS

def _infer_research_filter(prompt: str) -> bool | None:
    """
    Infer an explicit research filter from user text.
    Returns:
      - True for research-only intent
      - False for non-research intent
      - None when no explicit intent is present
    """
    prompt_lower = prompt.lower()
    if any(pattern in prompt_lower for pattern in _NEGATED_RESEARCH_PATTERNS):
        return False
    if "research" in prompt_lower:
        return True
    return None

def _enforce_prompt_filters(
    tool_name: str,
    tool_input: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    """
    Defensive guardrail: if the prompt clearly asks for a filter and the model omits it,
    inject that filter before executing the tool.
    """
    if tool_name not in {"search_jobs", "job_stats"}:
        return dict(tool_input)

    adjusted_input = dict(tool_input)

    research_filter = _infer_research_filter(prompt)
    if research_filter is not None and adjusted_input.get("is_research") is None:
        adjusted_input["is_research"] = research_filter

    return adjusted_input

def _build_followup_args(tool_name: str, tool_args: dict) -> tuple[str, dict]:
    """
    Convert a previous tool call into an expanded follow-up call.
    - job_stats count → search_jobs with same filters (show listings)
    - search_jobs → job_stats grouped by job_function_std (show breakdown)
    """
    if tool_name == "job_stats":
        # User asked for a count → now show the actual listings
        new_args: dict[str, Any] = {}
        for key in (
            "country",
            "is_remote",
            "is_research",
            "job_type_filled",
            "tools",
            "posted_start",
            "posted_end",
        ):
            if key in tool_args and tool_args[key] is not None:
                new_args[key] = tool_args[key]
        new_args["limit"] = 20
        return "search_jobs", new_args

    if tool_name == "search_jobs":
        # User saw listings → now show a statistical breakdown
        new_args = {"metric": "count", "group_by": "job_function_std"}
        for key in (
            "country",
            "is_remote",
            "is_research",
            "job_type_filled",
            "tools",
            "posted_start",
            "posted_end",
        ):
            if key in tool_args and tool_args[key] is not None:
                new_args[key] = tool_args[key]
        return "job_stats", new_args

    if tool_name == "semantic_search_jobs":
        # User saw semantic results → show more results (double top_k, max 20)
        prev_top_k = tool_args.get("top_k", 5)
        new_args = dict(tool_args)
        new_args["top_k"] = min(prev_top_k * 2, 20)
        return "semantic_search_jobs", new_args

    # Fallback: re-run same tool
    return tool_name, dict(tool_args)


def _base_trace_tags(
    *,
    settings: Any,
    conversation_id: str,
    prompt: str,
) -> dict[str, Any]:
    """Build common trace tags for both full MLflow and lite tracing."""
    tags: dict[str, Any] = {
        "conversation_id": conversation_id,
        "prompt_preview": prompt[:120],
    }
    tag_key = settings.mlflow_trace_tag_key.strip()
    tag_value = settings.mlflow_trace_tag_value.strip()
    if tag_key and tag_value:
        tags[tag_key] = tag_value
    return tags

@router.post(
    "/ask",
    response_model=AskResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ask the AI model a question (with tool calling)",
)
async def ask(request: Request, body: AskRequest):
    """
    Agent orchestration loop with confidence gating and per-turn logging.

    Flow:
      1. Check for pending follow-up confirmations (yes/no).
      2. Send the user prompt + tool definitions to Claude.
      3. If Claude responds with tool_use -> execute, evaluate confidence, re-call.
      4. Repeat until Claude returns a final text answer (up to MAX_TOOL_ROUNDS).
      5. Log the full turn (structured + MLflow).
      6. Apply confidence gate (ANSWER / ASK_CLARIFICATION / DECLINE / HANDOFF).
    """
    settings = get_settings()
    default_experiment_name = settings.mlflow_experiment_name
    default_active_model_name = _safe_logged_model_name(
        settings.mlflow_active_model_name,
        settings.app_version,
    )
    override_experiment_name = request.headers.get("x-mlflow-experiment-name", "").strip()
    override_active_model_name = request.headers.get("x-mlflow-active-model-name", "").strip()
    override_prompt_name = request.headers.get("x-mlflow-prompt-name", "").strip()
    override_prompt_version = request.headers.get("x-mlflow-prompt-version", "").strip()
    override_prompt_uri = request.headers.get("x-mlflow-prompt-uri", "").strip()
    routing_override_active = False

    if _mlflow and override_experiment_name:
        try:
            _mlflow.set_experiment(override_experiment_name)
            if override_active_model_name:
                _mlflow.set_active_model(name=override_active_model_name)
            routing_override_active = True
        except Exception as _exc:
            logger.debug("MLflow per-request routing override failed: %s", _exc)

    # ── Initialize turn tracking (must be before MLflow so conversation_id exists) ──
    turn_id = str(uuid.uuid4())
    conversation_id = body.conversation_id or str(uuid.uuid4())
    trace_tags = _base_trace_tags(
        settings=settings,
        conversation_id=conversation_id,
        prompt=body.prompt,
    )
    timer = TurnTimer()
    timer.start()

    # ── MLflow Lite trace fallback (when full SDK is unavailable) ──────
    _lite_client = None
    _lite_trace_events: list[dict[str, Any]] = []

    tracing_context_cm = nullcontext()
    tracing_context_entered = False
    request_headers = dict(request.headers)
    if (
        _set_tracing_context_from_http_request_headers is not None
        and ("traceparent" in request_headers or "Traceparent" in request_headers)
    ):
        try:
            tracing_context_cm = _set_tracing_context_from_http_request_headers(
                request_headers
            )
        except Exception as _exc:
            logger.debug("MLflow distributed trace context setup failed: %s", _exc)

    # Keep the distributed tracing context active for the full request lifetime.
    try:
        tracing_context_cm.__enter__()
        tracing_context_entered = True
    except Exception as _exc:
        logger.debug("MLflow distributed trace context enter failed: %s", _exc)
        tracing_context_cm = nullcontext()
        tracing_context_cm.__enter__()
        tracing_context_entered = True

    # ── MLflow: create a root/child AGENT trace for the full orchestration ───
    _agent_span = None
    _agent_span_ctx = None
    _trace_id = None
    if _mlflow and _SpanType:
        try:
            # If the caller propagated MLflow trace context headers, this span
            # becomes part of the caller trace instead of starting a detached one.
            _agent_span_ctx = _mlflow.start_span(
                name="ask_agent",
                span_type=_SpanType.AGENT,
            )
            _agent_span = _agent_span_ctx.__enter__()
            _agent_span.set_inputs({
                "prompt": body.prompt,
                "conversation_id": conversation_id,
            })
            _trace_id = _agent_span.trace_id
            trace_tag_updates = dict(trace_tags)
            if override_prompt_name and override_prompt_version:
                trace_tag_updates[LINKED_PROMPTS_TAG] = json.dumps(
                    [{"name": override_prompt_name, "version": override_prompt_version}]
                )
            if override_prompt_uri:
                trace_tag_updates["prompt_uri"] = override_prompt_uri
            if override_prompt_version:
                trace_tag_updates["prompt_version"] = override_prompt_version
            _mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": conversation_id,
                },
                tags=trace_tag_updates,
            )
        except Exception as _exc:
            logger.debug("MLflow span creation failed: %s", _exc)
            _agent_span = None
            _agent_span_ctx = None

    # If full MLflow tracing is unavailable, use REST trace fallback.
    if _trace_id is None:
        try:
            from app.services.mlflow_lite import get_lite_client

            _lite_client = get_lite_client()
            if _lite_client:
                _trace_id = _lite_client.start_trace(
                    prompt=body.prompt,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    policy_version=POLICY_VERSION,
                    trace_name="ask_agent_lite",
                )
                if _trace_id:
                    _lite_trace_events.append(
                        {
                            "ts_ms": int(time.time() * 1000),
                            "kind": "trace_start",
                            "payload": {"trace_id": _trace_id},
                        }
                    )
        except Exception as _exc:
            logger.debug("MLflow Lite trace start failed: %s", _exc)

    effective_system_prompt = body.system or JOBLAB_SYSTEM

    turn_record = TurnRecord(
        conversation_id=conversation_id,
        turn_id=turn_id,
        user_prompt=body.prompt,
        prompt_char_count=len(body.prompt),
        policy_version=POLICY_VERSION,
        system_prompt_chars=len(effective_system_prompt),
    )

    def _lite_append_event(kind: str, payload: dict[str, Any]) -> None:
        """Collect compact trace events for MLflow Lite timeline tags."""
        if not (_lite_client and _trace_id):
            return
        try:
            _lite_trace_events.append(
                {
                    "ts_ms": int(time.time() * 1000),
                    "kind": kind,
                    "payload": payload,
                }
            )
        except Exception:
            pass

    def _finalize_turn(
        answer: str,
        usage: dict | None,
        tool_calls: list[dict] | None,
        job_results: list[dict] | None = None,
        gate_outcome: str = "ANSWER",
        gate_confidence: float = 1.0,
        gate_reason: str = "",
        result_type: str = "answer",
        error: str | None = None,
    ) -> AskResponse:
        """Helper: finalize turn record, log, and return response."""
        def _restore_mlflow_route_override() -> None:
            if not (_mlflow and routing_override_active):
                return
            try:
                _mlflow.set_experiment(default_experiment_name)
                _mlflow.set_active_model(name=default_active_model_name)
            except Exception as _exc:
                logger.debug("MLflow per-request routing restore failed: %s", _exc)

        def _close_distributed_trace_context() -> None:
            if not tracing_context_entered:
                return
            try:
                tracing_context_cm.__exit__(None, None, None)
            except Exception as _exc:
                logger.debug("MLflow distributed trace context exit failed: %s", _exc)

        turn_record.total_latency_ms = round(timer.total_ms(), 1)
        turn_record.llm_latency_ms = timer.get("llm")
        turn_record.tool_latency_ms = timer.get("tool")
        turn_record.gate_outcome = gate_outcome
        turn_record.gate_confidence = gate_confidence
        turn_record.gate_reason = gate_reason
        turn_record.result_type = result_type
        turn_record.result_length = len(answer)
        turn_record.error = error
        if usage:
            turn_record.input_tokens = usage.get("input_tokens", 0)
            turn_record.output_tokens = usage.get("output_tokens", 0)
            turn_record.total_tokens = (
                turn_record.input_tokens + turn_record.output_tokens
            )

        # Add final turn state to the full MLflow trace so it is filterable
        # and visible in the trace table/dashboard, not only inside span outputs.
        if _mlflow and _trace_id:
            try:
                _mlflow.update_current_trace(
                    tags={
                        "gate_outcome": gate_outcome,
                        "gate_confidence": f"{gate_confidence:.6f}",
                        "gate_reason": gate_reason[:200],
                        "result_type": result_type,
                        "tool_calls_count": str(len(tool_calls or [])),
                    }
                )
            except Exception as _exc:
                logger.debug("MLflow trace tag update failed: %s", _exc)
            try:
                from mlflow.entities import AssessmentSource, AssessmentSourceType

                assessment_source = AssessmentSource(
                    source_type=AssessmentSourceType.CODE,
                    source_id="confidence_gate",
                )
                _mlflow.log_feedback(
                    trace_id=_trace_id,
                    name="turn_outcome",
                    value=gate_outcome,
                    rationale=gate_reason or f"Turn completed with outcome {gate_outcome}.",
                    source=assessment_source,
                    metadata={"result_type": result_type},
                )
                _mlflow.log_feedback(
                    trace_id=_trace_id,
                    name="gate_confidence",
                    value=round(gate_confidence, 6),
                    rationale=gate_reason or "Confidence assigned by confidence_gate.",
                    source=assessment_source,
                    metadata={"result_type": result_type},
                )
            except Exception as _exc:
                logger.debug("MLflow trace assessment logging failed: %s", _exc)

        # Finalize MLflow Lite trace (if active)
        if _lite_client and _trace_id:
            try:
                _lite_client.end_trace(
                    trace_id=_trace_id,
                    status="ERROR" if error else "OK",
                    answer=answer,
                    usage={
                        "input_tokens": turn_record.input_tokens,
                        "output_tokens": turn_record.output_tokens,
                        "total_tokens": turn_record.total_tokens,
                    },
                    gate_outcome=gate_outcome,
                    gate_confidence=gate_confidence,
                    gate_reason=gate_reason,
                    result_type=result_type,
                    error=error,
                    timeline=_lite_trace_events,
                    tags={
                        "tool_calls_count": len(tool_calls or []),
                        "tool_rounds": turn_record.tool_rounds,
                        "soft_enforcement_retries": turn_record.soft_enforcement_retries,
                    },
                )
            except Exception as _exc:
                logger.debug("MLflow Lite trace end failed: %s", _exc)
        elif _lite_client and not _trace_id:
            try:
                _lite_client.spool_trace_complete(
                    prompt=body.prompt,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    policy_version=POLICY_VERSION,
                    answer=answer,
                    usage={
                        "input_tokens": turn_record.input_tokens,
                        "output_tokens": turn_record.output_tokens,
                        "total_tokens": turn_record.total_tokens,
                    },
                    gate_outcome=gate_outcome,
                    gate_confidence=gate_confidence,
                    gate_reason=gate_reason,
                    result_type=result_type,
                    error=error,
                    timeline=_lite_trace_events,
                    tags={
                        "tool_calls_count": len(tool_calls or []),
                        "tool_rounds": turn_record.tool_rounds,
                        "soft_enforcement_retries": turn_record.soft_enforcement_retries,
                    },
                )
            except Exception as _exc:
                logger.debug("MLflow Lite trace spool failed: %s", _exc)

        # Log turn (structured + MLflow if available)
        log_turn(turn_record)
        update_aggregate_metrics(turn_record)

        # Close the AGENT span with outputs
        if _agent_span and _agent_span_ctx:
            try:
                _agent_span.set_outputs({
                    "answer": answer[:500],
                    "gate_outcome": gate_outcome,
                    "gate_confidence": gate_confidence,
                    "tool_calls_count": len(tool_calls) if tool_calls else 0,
                })
                _agent_span_ctx.__exit__(None, None, None)
            except Exception:
                pass

        _close_distributed_trace_context()
        _restore_mlflow_route_override()

        return AskResponse(
            answer=answer,
            model=settings.bedrock_model_id,
            usage=usage,
            tool_calls=tool_calls or None,
            job_results=job_results or None,
            gate_outcome=gate_outcome,
            result_type=result_type,
            conversation_id=conversation_id,
            turn_id=turn_id,
            trace_id=_trace_id,
        )

# ------------------------------------------------------------------
    # ── Dialogue state: handle affirmative/negative follow-ups ──────────
    pending = get_pending_followup(conversation_id)

    if pending and _is_negative_followup(body.prompt):
        clear_pending_followup(conversation_id)
        return _finalize_turn(
            answer="Alright. Let me know if you'd like additional insights.",
            usage=None,
            tool_calls=None,
            result_type="followup_declined",
        )

    if pending and _is_affirmative_followup(body.prompt):
        clear_pending_followup(conversation_id)
        followup_tool = pending["tool_name"]
        followup_args = pending["tool_args"]

        # Build expanded tool call from the pending context
        exec_tool_name, exec_tool_args = _build_followup_args(followup_tool, followup_args)
        logger.info(
            "Pending follow-up confirmed: %s(%s) → %s(%s)",
            followup_tool, followup_args, exec_tool_name, exec_tool_args,
        )

        executor = TOOL_EXECUTORS.get(exec_tool_name)
        if executor is None:
            return _finalize_turn(
                answer=f"Unknown tool: {exec_tool_name}",
                usage=None,
                tool_calls=None,
                gate_outcome="DECLINE",
                gate_confidence=0.0,
                gate_reason=f"Unknown tool: {exec_tool_name}",
                result_type="error",
                error=f"Unknown tool: {exec_tool_name}",
            )

        tool_error = None
        result_data = None
        with timer.track("tool"):
            try:
                result_data = executor(exec_tool_args)
                result_json = json.dumps(result_data, default=str)
            except Exception as exc:
                logger.exception("Follow-up tool %s failed", exec_tool_name)
                tool_error = str(exc)

        if tool_error:
            return _finalize_turn(
                answer=f"Tool execution failed: {tool_error}",
                usage=None,
                tool_calls=[{"name": exec_tool_name, "input": exec_tool_args}],
                gate_outcome="ASK_CLARIFICATION",
                gate_confidence=0.1,
                gate_reason=f"Follow-up tool execution failed: {tool_error}",
                result_type="error",
                error=tool_error,
            )

        # Confidence gate for follow-up result
        gate = evaluate_confidence(
            exec_tool_name,
            exec_tool_args,
            result_data,
            latency_ms=timer.get("tool"),
        )
        _lite_append_event(
            "tool",
            {
                "name": exec_tool_name,
                "latency_ms": timer.get("tool"),
                "error": tool_error or "",
                "gate_outcome": gate.outcome.value,
                "gate_confidence": gate.confidence,
                "result_count": len(result_data) if isinstance(result_data, list) else 0,
            },
        )
        track_confidence(conversation_id, gate)
        turn_record.tools_called = [{"name": exec_tool_name, "input": exec_tool_args}]

        # Store this as the new last tool
        set_last_tool(conversation_id, exec_tool_name, exec_tool_args)

        # Store mentioned jobs for follow-up context
        mentioned = _extract_jobs_from_results(exec_tool_name, result_data)
        if mentioned:
            set_mentioned_jobs(conversation_id, mentioned)

        # Send tool result to Claude for a natural language summary
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"The user confirmed they want more details. "
                    f"I executed {exec_tool_name} with {json.dumps(exec_tool_args)}.\n\n"
                    f"Results:\n{result_json}\n\n"
                    f"Summarize these results clearly. Do not expose raw JSON."
                ),
            },
        ]
        with timer.track("llm"):
                raw = invoke_claude(
                    messages=messages,
                    system=effective_system_prompt,
                    tools=[],
                )
        usage = get_usage(raw)
        _lite_append_event(
            "llm",
            {
                "latency_ms": timer.get("llm"),
                "stop_reason": raw.get("stopReason", ""),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )
        answer = extract_text(raw)

        # Modify answer based on gate decision
        if gate.outcome == GateOutcome.ASK_CLARIFICATION and gate.suggestion:
            answer = f"{answer}\n\n{gate.suggestion}"

        # Set pending follow-up for the new result too
        set_pending_followup(conversation_id, {
            "type": "expand_previous_query",
            "tool_name": exec_tool_name,
            "tool_args": exec_tool_args,
        })

        return _finalize_turn(
            answer=answer,
            usage=usage,
            tool_calls=[{"name": exec_tool_name, "input": exec_tool_args}],
            job_results=_build_ai_job_results(exec_tool_name, result_data),
            gate_outcome=gate.outcome.value,
            gate_confidence=gate.confidence,
            gate_reason=gate.reason,
        )

    # ── Normal flow ─────────────────────────────────────────────────────

    # Use the dedicated system prompt by default, but allow explicit override
    # for MLflow prompt evaluation/optimization workflows.
    system = effective_system_prompt

    # Retrieve last tool memory for follow-up refinement
    last_tool_name, last_tool_args = get_last_tool(conversation_id)
    mentioned_jobs = get_mentioned_jobs(conversation_id)

    messages = []

    # Build mentioned-jobs context string (if any)
    jobs_context = ""
    if mentioned_jobs:
        jobs_lines = []
        for j in mentioned_jobs[:10]:
            jobs_lines.append(
                f"  - job_id={j.get('job_id')} | {j.get('actual_role', '?')} "
                f"at {j.get('company_name', '?')} | url={j.get('url', 'N/A')} "
                f"| posted={j.get('posted_date', '?')} | country={j.get('country', '?')}"
            )
        jobs_context = (
            "\n[Previously mentioned jobs (most recent first):\n"
            + "\n".join(jobs_lines)
            + "\n]\n"
        )

    # If user is asking about a previously mentioned job, answer directly
    # from memory — no tool call needed.
    is_job_followup = mentioned_jobs and _is_job_detail_followup(body.prompt)
    if is_job_followup:
        hint = (
            f"{jobs_context}\n"
            f"The user is asking about one of the previously mentioned jobs. "
            f"Identify which job they mean and provide its job_id and url. "
            f"You do NOT need to call a tool — the information is in the context above.\n\n"
            f"User: {body.prompt}"
        )
        messages.append({"role": "user", "content": hint})
    # If user input is short and likely a refinement, inject a hint
    elif last_tool_name and len(body.prompt.split()) <= 6:
        is_affirmative = _is_affirmative_followup(body.prompt)
        if is_affirmative:
            hint = (
                f"[Context: The previous tool used was '{last_tool_name}' "
                f"with arguments {json.dumps(last_tool_args)}. "
                f"The user said \"{body.prompt}\" which is an affirmative response. "
                f"They want more details or a breakdown of the previous results. "
                f"You MUST call a tool to provide this. Re-use the same filters "
                f"and add grouping or detail to give a richer answer.]"
            )
        else:
            hint = (
                f"[Context: The previous tool used was '{last_tool_name}' "
                f"with arguments {json.dumps(last_tool_args)}. "
                f"The user is likely refining those filters.]{jobs_context}\n\n"
                f"{body.prompt}"
            )
        messages.append({"role": "user", "content": hint})
    else:
        # For regular queries, append jobs context if available
        prompt_with_context = body.prompt
        if jobs_context:
            prompt_with_context = f"{jobs_context}\n{body.prompt}"
        messages.append({"role": "user", "content": prompt_with_context})

    # A prompt is DB-related if it matches keywords OR is a follow-up to a previous tool call.
    # BUT: if it's a job-detail follow-up with known jobs, we already have the
    # answer in memory — don't enforce tool calling.
    db_related_prompt = (
        not is_job_followup
        and (
            _is_database_related(body.prompt)
            or (last_tool_name is not None and len(body.prompt.split()) <= 6)
        )
    )
    no_tool_retry_count = 0
    has_called_tool = False
    collected_tool_calls: list[dict[str, Any]] = []
    last_gate_decision = None
    last_result_data = None
    last_result_tool_name: str | None = None

    try:
        for _round in range(MAX_TOOL_ROUNDS):
            with timer.track("llm"):
                raw = invoke_claude(
                    messages=messages,
                    system=system,
                    tools=TOOL_DEFINITIONS,
                )
            raw_usage = get_usage(raw)
            _lite_append_event(
                "llm",
                {
                    "round": _round + 1,
                    "latency_ms": timer.get("llm"),
                    "stop_reason": raw.get("stopReason", ""),
                    "input_tokens": raw_usage.get("input_tokens", 0),
                    "output_tokens": raw_usage.get("output_tokens", 0),
                    "total_tokens": raw_usage.get("total_tokens", 0),
                },
            )

            # No tool call -> either enforce or return text answer
            if not has_tool_use(raw):
                if (
                    db_related_prompt
                    and not has_called_tool
                    and no_tool_retry_count < MAX_SOFT_ENFORCEMENT_RETRIES
                ):
                    logger.warning(
                        "No tool call for DB-related prompt, applying soft enforcement retry %d/%d",
                        no_tool_retry_count + 1,
                        MAX_SOFT_ENFORCEMENT_RETRIES,
                    )
                    turn_record.soft_enforcement_retries = no_tool_retry_count + 1
                    messages.append(get_assistant_message(raw))
                    messages.append(
                        {
                            "role": "user",
                            "content": "This question requires database access. You must call an appropriate tool.",
                        }
                    )
                    no_tool_retry_count += 1
                    continue

                if db_related_prompt and not has_called_tool:
                    return _finalize_turn(
                        answer="I could not complete this database request because no tool call was produced.",
                        usage=raw_usage,
                        tool_calls=collected_tool_calls or None,
                        gate_outcome="DECLINE",
                        gate_confidence=0.0,
                        gate_reason="No tool call produced for DB-related prompt after retries",
                        result_type="decline",
                    )

                answer = extract_text(raw)

                # Apply confidence gate on final answer
                gate_outcome = "ANSWER"
                gate_confidence = 0.9
                gate_reason = "Direct text answer"

                if last_gate_decision:
                    gate_outcome = last_gate_decision.outcome.value
                    gate_confidence = last_gate_decision.confidence
                    gate_reason = last_gate_decision.reason

                    # Append clarification suggestion if confidence is low
                    if (
                        last_gate_decision.outcome == GateOutcome.ASK_CLARIFICATION
                        and last_gate_decision.suggestion
                    ):
                        answer = f"{answer}\n\n{last_gate_decision.suggestion}"

                # If tools were called, set pending follow-up for confirmation tracking
                if has_called_tool and collected_tool_calls:
                    last_tc = collected_tool_calls[-1]
                    set_pending_followup(conversation_id, {
                        "type": "expand_previous_query",
                        "tool_name": last_tc["name"],
                        "tool_args": last_tc["input"],
                    })

                return _finalize_turn(
                    answer=answer,
                    usage=raw_usage,
                    tool_calls=collected_tool_calls or None,
                    job_results=_build_ai_job_results(last_result_tool_name, last_result_data),
                    gate_outcome=gate_outcome,
                    gate_confidence=gate_confidence,
                    gate_reason=gate_reason,
                )

            # Tool call(s) -> execute each one
            has_called_tool = True
            turn_record.tool_rounds += 1
            tool_calls = extract_tool_calls(raw)
            logger.info("Claude requested %d tool call(s)", len(tool_calls))

            # Append the full assistant message (Converse format)
            messages.append(get_assistant_message(raw))

            # Execute each tool and build tool_result blocks
            tool_results: list[dict[str, Any]] = []
            for tc in tool_calls:
                tool_name = tc["name"]
                raw_tool_input = tc["input"]
                tool_id = tc["id"]
                tool_input = _enforce_prompt_filters(tool_name, raw_tool_input, body.prompt)

                if tool_input != raw_tool_input:
                    logger.info(
                        "Adjusted tool input from prompt constraints: %s raw=%s adjusted=%s",
                        tool_name,
                        raw_tool_input,
                        tool_input,
                    )

                collected_tool_calls.append({"name": tool_name, "input": tool_input})
                turn_record.tools_called.append({"name": tool_name, "input": tool_input})

                executor = TOOL_EXECUTORS.get(tool_name)
                tool_error = None
                result_data = None

                if executor is None:
                    result_content = json.dumps(
                        {"error": f"Unknown tool: {tool_name}"}
                    )
                    tool_error = f"Unknown tool: {tool_name}"
                else:
                    with timer.track("tool"):
                        try:
                            result_data = executor(tool_input)
                            result_content = json.dumps(result_data, default=str)
                        except Exception as exc:
                            logger.exception("Tool %s failed", tool_name)
                            result_content = json.dumps(
                                {"error": f"Tool execution failed: {exc}"}
                            )
                            tool_error = str(exc)

                # ── Confidence Gate: evaluate tool results ──────────────
                consecutive_low = get_consecutive_low_confidence(conversation_id)
                gate = evaluate_confidence(
                    tool_name,
                    tool_input,
                    result_data,
                    latency_ms=timer.get("tool"),
                    error=tool_error,
                    consecutive_low_confidence=consecutive_low,
                )
                track_confidence(conversation_id, gate)
                last_gate_decision = gate

                logger.info(
                    "Confidence gate: %s (%.2f) — %s",
                    gate.outcome.value,
                    gate.confidence,
                    gate.reason,
                )
                _lite_append_event(
                    "tool",
                    {
                        "round": _round + 1,
                        "name": tool_name,
                        "latency_ms": timer.get("tool"),
                        "error": tool_error or "",
                        "gate_outcome": gate.outcome.value,
                        "gate_confidence": gate.confidence,
                        "result_count": len(result_data) if isinstance(result_data, list) else 0,
                    },
                )

                # ── MLflow: enrich current trace with tool + gate info ──
                if _mlflow:
                    try:
                        span = _mlflow.get_current_active_span()
                        if span:
                            result_count = len(result_data) if isinstance(result_data, list) else 0
                            span.set_attributes({
                                f"tool.{tool_name}.result_count": result_count,
                                f"tool.{tool_name}.latency_ms": timer.get("tool"),
                                f"tool.{tool_name}.error": tool_error or "",
                                f"gate.{tool_name}.outcome": gate.outcome.value,
                                f"gate.{tool_name}.confidence": gate.confidence,
                                f"gate.{tool_name}.reason": gate.reason[:200],
                            })
                    except Exception:
                        pass

                # HANDOFF: stop processing, return escalation message
                if gate.outcome == GateOutcome.HANDOFF:
                    return _finalize_turn(
                        answer=gate.suggestion or "I'm unable to process this request reliably. Please try rephrasing.",
                        usage=raw_usage,
                        tool_calls=collected_tool_calls,
                        gate_outcome="HANDOFF",
                        gate_confidence=gate.confidence,
                        gate_reason=gate.reason,
                        result_type="handoff",
                    )

                # Store last successful tool call in memory
                if executor is not None and tool_error is None:
                    set_last_tool(conversation_id, tool_name, tool_input)
                    last_result_data = result_data
                    last_result_tool_name = tool_name

                    # Store mentioned jobs for follow-up context
                    if tool_name in ("search_jobs", "semantic_search_jobs"):
                        try:
                            mentioned = _extract_jobs_from_results(tool_name, result_data)
                            if mentioned:
                                set_mentioned_jobs(conversation_id, mentioned)
                        except Exception:
                            pass  # non-fatal

                tool_results.append(make_tool_result_block(tool_id, result_content))

            messages.append({"role": "user", "content": tool_results})

        # Exhausted rounds - return whatever text we have
        answer = extract_text(raw)
        gate_outcome = "ANSWER"
        gate_confidence = 0.5
        gate_reason = "Exhausted tool rounds"

        if last_gate_decision:
            gate_outcome = last_gate_decision.outcome.value
            gate_confidence = last_gate_decision.confidence
            gate_reason = last_gate_decision.reason

        # Set pending follow-up if tools were called
        if has_called_tool and collected_tool_calls:
            last_tc = collected_tool_calls[-1]
            set_pending_followup(conversation_id, {
                "type": "expand_previous_query",
                "tool_name": last_tc["name"],
                "tool_args": last_tc["input"],
            })

        return _finalize_turn(
            answer=answer or "I was unable to complete the request within the allowed steps.",
            usage=raw_usage,
            tool_calls=collected_tool_calls or None,
            job_results=_build_ai_job_results(last_result_tool_name, last_result_data),
            gate_outcome=gate_outcome,
            gate_confidence=gate_confidence,
            gate_reason=gate_reason,
        )

    except Exception as exc:
        logger.exception("Bedrock / tool-call loop failed")
        # Log the error turn before re-raising
        _finalize_turn(
            answer="",
            usage=None,
            tool_calls=collected_tool_calls or None,
            gate_outcome="HANDOFF",
            gate_confidence=0.0,
            gate_reason=f"Exception: {exc}",
            result_type="error",
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# ── Metrics Endpoint ───────────────────────────────────────────────────────
@router.get(
    "/metrics",
    summary="Get aggregate AI agent metrics",
    response_model=dict,
)
async def metrics():
    """
    Returns aggregate metrics for the AI agent:
      - total_turns, outcome_counts, tool_usage_counts
      - avg_latency_ms, avg_confidence
      - success_rate, clarification_rate, handoff_rate, error_count
    """
    return get_aggregate_metrics()

# ── MLflow Spool Flush Endpoint ─────────────────────────────────────────────
@router.post(
    "/mlflow/flush-spool",
    response_model=MlflowFlushResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Replay queued MLflow Lite logs from S3",
)
async def flush_mlflow_spool(
    body: MlflowFlushRequest,
    x_mlflow_spool_token: str | None = Header(default=None, alias="x-mlflow-spool-token"),
):
    settings = get_settings()
    required = settings.mlflow_spool_flush_token.strip()
    if required and x_mlflow_spool_token != required:
        raise HTTPException(status_code=401, detail="Invalid spool flush token.")

    try:
        from app.services.mlflow_lite import get_lite_client

        lite = get_lite_client()
        if not lite:
            raise HTTPException(
                status_code=500,
                detail="MLflow Lite client is not available.",
            )

        result = lite.flush_spool(max_items=body.max_items)
        logger.info(
            "MLflow spool flush processed=%d succeeded=%d failed=%d",
            result.get("processed", 0),
            result.get("succeeded", 0),
            result.get("failed", 0),
        )
        return MlflowFlushResponse(
            processed=result.get("processed", 0),
            succeeded=result.get("succeeded", 0),
            failed=result.get("failed", 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("MLflow spool flush failed")
        raise HTTPException(status_code=500, detail=f"Spool flush failed: {exc}") from exc

# ── Feedback Endpoint ──────────────────────────────────────────────────────
@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Submit user feedback (thumbs up/down) for a traced AI turn",
)
async def feedback(body: FeedbackRequest):
    """
    Records user feedback against an MLflow trace.

    Preferred path uses trace_id from AskResponse. If trace_id is not yet
    available (tracking server outage), the frontend can send
    (conversation_id, turn_id) and feedback is queued for later attachment.
    """
    # If trace_id is unavailable (common when tracking server is down),
    # accept offline feedback keyed by (conversation_id, turn_id).
    if not body.trace_id:
        try:
            from app.services.mlflow_lite import get_lite_client

            lite = get_lite_client()
            if not lite:
                raise HTTPException(
                    status_code=501,
                    detail="MLflow is not available — feedback cannot be recorded.",
                )

            ok = lite.log_offline_trace_feedback(
                conversation_id=body.conversation_id or "",
                turn_id=body.turn_id or "",
                thumbs_up=body.thumbs_up,
                comment=body.comment or "",
            )
            if not ok:
                raise HTTPException(
                    status_code=500,
                    detail="Offline feedback queueing failed in MLflow Lite.",
                )

            logger.info(
                "Lite offline feedback accepted — conversation_id=%s turn_id=%s thumbs_up=%s comment=%s",
                body.conversation_id,
                body.turn_id,
                body.thumbs_up,
                bool(body.comment),
            )
            return FeedbackResponse(
                trace_id=None,
                conversation_id=body.conversation_id,
                turn_id=body.turn_id,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to record Lite offline feedback for conversation=%s turn=%s",
                body.conversation_id,
                body.turn_id,
            )
            raise HTTPException(status_code=500, detail=f"Feedback recording failed: {exc}") from exc

    if not _mlflow:
        try:
            from app.services.mlflow_lite import get_lite_client

            lite = get_lite_client()
            if not lite:
                raise HTTPException(
                    status_code=501,
                    detail="MLflow is not available — feedback cannot be recorded.",
                )

            ok = lite.log_trace_feedback(
                trace_id=body.trace_id,
                thumbs_up=body.thumbs_up,
                comment=body.comment or "",
            )
            if not ok:
                raise HTTPException(
                    status_code=500,
                    detail="Feedback recording failed in MLflow Lite.",
                )

            logger.info(
                "Lite feedback recorded — trace_id=%s thumbs_up=%s comment=%s",
                body.trace_id,
                body.thumbs_up,
                bool(body.comment),
            )
            return FeedbackResponse(trace_id=body.trace_id)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to record Lite feedback for trace %s", body.trace_id)
            raise HTTPException(status_code=500, detail=f"Feedback recording failed: {exc}") from exc

    try:
        from mlflow.entities import AssessmentSource, AssessmentSourceType

        # Log thumbs up / down
        _mlflow.log_feedback(
            trace_id=body.trace_id,
            name="user_satisfaction",
            value=body.thumbs_up,
            rationale=body.comment or ("User indicated response was helpful" if body.thumbs_up else "User indicated response was not helpful"),
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="ai_explorer_ui",
            ),
        )

        # If a comment was provided, log it as a separate text feedback
        if body.comment:
            _mlflow.log_feedback(
                trace_id=body.trace_id,
                name="user_comment",
                value=body.comment,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id="ai_explorer_ui",
                ),
            )

        logger.info(
            "Feedback recorded — trace_id=%s thumbs_up=%s comment=%s",
            body.trace_id,
            body.thumbs_up,
            bool(body.comment),
        )

        return FeedbackResponse(trace_id=body.trace_id)

    except Exception as exc:
        logger.exception("Failed to record feedback for trace %s", body.trace_id)
        raise HTTPException(status_code=500, detail=f"Feedback recording failed: {exc}") from exc
