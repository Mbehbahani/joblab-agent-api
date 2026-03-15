"""
MLflow-native scorers and judges for the Joblab agent.

These definitions are designed for the evaluation datasets published in Step 3.
They assume the evaluation predict function returns the full /ai/ask response
shape when possible:

    {
        "answer": "...",
        "tool_calls": [...],
        "trace_id": "tr-...",
        ...
    }

The code scorers remain tolerant of plain-string outputs, but they are most
useful with structured outputs and trace IDs.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import scorer

_TOOL_SPAN_NAME_TO_TOOL_NAME = {
    "execute_search_jobs": "search_jobs",
    "execute_job_stats": "job_stats",
    "execute_semantic_search": "semantic_search_jobs",
}

_FILTER_KEYS = {
    "country",
    "is_remote",
    "is_research",
    "job_level_std",
    "job_function_std",
    "company_industry_std",
    "job_type_filled",
    "platform",
    "posted_start",
    "posted_end",
    "role_keyword",
    "query_text",
    "top_k",
}

_DECLINE_INDICATORS = [
    "can't help",
    "cannot help",
    "can't assist",
    "cannot assist",
    "outside",
    "only help with",
    "job-related",
    "job market",
    "i specialize",
    "i'm designed",
    "let me know if",
    "redirect",
]


def _outputs_dict(outputs: Any) -> dict[str, Any]:
    return outputs if isinstance(outputs, dict) else {}


def _answer_text(outputs: Any) -> str:
    if isinstance(outputs, dict):
        answer = outputs.get("answer")
        if isinstance(answer, str):
            return answer
    if isinstance(outputs, str):
        return outputs
    return str(outputs or "")


def _tool_calls_from_outputs(outputs: Any) -> list[dict[str, Any]]:
    if isinstance(outputs, dict):
        tool_calls = outputs.get("tool_calls")
        if isinstance(tool_calls, list):
            return [call for call in tool_calls if isinstance(call, dict)]
    return []


def _trace_id_from_outputs(outputs: Any) -> str | None:
    if isinstance(outputs, dict):
        trace_id = outputs.get("trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            return trace_id.strip()
    return None


def _load_trace(outputs: Any, trace: Any = None) -> Any | None:
    if trace is not None:
        return trace
    trace_id = _trace_id_from_outputs(outputs)
    if not trace_id:
        return None
    try:
        return mlflow.get_trace(trace_id=trace_id, silent=True)
    except Exception:
        return None


def _tool_calls_from_trace(trace: Any | None) -> list[dict[str, Any]]:
    if trace is None:
        return []
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    calls: list[dict[str, Any]] = []
    for span in tool_spans:
        tool_name = _TOOL_SPAN_NAME_TO_TOOL_NAME.get(span.name, span.name)
        calls.append(
            {
                "name": tool_name,
                "input": span.inputs or {},
                "output": span.outputs or {},
            }
        )
    return calls


def _resolved_tool_calls(outputs: Any, trace: Any = None) -> list[dict[str, Any]]:
    calls = _tool_calls_from_outputs(outputs)
    if calls:
        return calls
    return _tool_calls_from_trace(_load_trace(outputs, trace))


def _primary_tool_name(outputs: Any, trace: Any = None) -> str | None:
    calls = _resolved_tool_calls(outputs, trace)
    if not calls:
        return None
    name = calls[0].get("name")
    return name if isinstance(name, str) else None


def _primary_tool_input(outputs: Any, trace: Any = None) -> dict[str, Any]:
    calls = _resolved_tool_calls(outputs, trace)
    if not calls:
        return {}
    tool_input = calls[0].get("input", {})
    return tool_input if isinstance(tool_input, dict) else {}


def _normalize_str(value: Any) -> str:
    return str(value).strip().lower()


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        expected_norm = _normalize_str(expected)
        actual_norm = _normalize_str(actual)
        return expected_norm in actual_norm or actual_norm in expected_norm
    return expected == actual


def _filter_matches(expected_filters: dict[str, Any], actual_filters: dict[str, Any]) -> int:
    matches = 0
    for key, expected_value in expected_filters.items():
        if key not in actual_filters:
            continue
        if _values_match(expected_value, actual_filters.get(key)):
            matches += 1
    return matches


def build_code_scorers() -> list[Any]:
    @scorer(
        name="joblab_tool_choice_accuracy",
        description="Check whether the agent selected the expected tool.",
    )
    def tool_choice_accuracy(inputs, outputs, expectations, trace=None):
        expected_tool = expectations.get("expected_tool") if expectations else None
        actual_tool = _primary_tool_name(outputs, trace)
        if expected_tool is None:
            score = 1.0 if actual_tool is None else 0.0
            rationale = (
                "Correctly avoided tool usage for a boundary/out-of-scope case."
                if score == 1.0
                else f"Unexpected tool call: {actual_tool}"
            )
            return Feedback(value=score, rationale=rationale)
        score = 1.0 if expected_tool == actual_tool else 0.0
        return Feedback(
            value=score,
            rationale=(
                f"Expected tool {expected_tool} and observed {actual_tool}."
                if score == 0.0
                else f"Observed expected tool {expected_tool}."
            ),
        )

    @scorer(
        name="joblab_filter_recall",
        description="Measure how many required user filters were preserved in the tool call.",
    )
    def filter_recall(inputs, outputs, expectations, trace=None):
        expected_filters = (expectations or {}).get("expected_filters") or {}
        actual_tool = _primary_tool_name(outputs, trace)

        # semantic_search_jobs has no discrete filters — the query text IS the input.
        # Scoring filter preservation doesn't apply here; return N/A (1.0).
        if actual_tool == "semantic_search_jobs":
            return Feedback(value=1.0, rationale="semantic_search_jobs has no discrete filters to recall.")

        actual_filters = _primary_tool_input(outputs, trace)
        if not expected_filters:
            return Feedback(value=1.0, rationale="No explicit filters were required.")
        if not actual_filters:
            return Feedback(value=0.0, rationale="No structured tool inputs were available.")
        matches = _filter_matches(expected_filters, actual_filters)
        score = matches / len(expected_filters)
        return Feedback(
            value=score,
            rationale=f"Matched {matches} of {len(expected_filters)} expected filters.",
        )

    @scorer(
        name="joblab_minimal_filter_compliance",
        description="Penalize extra filters that the user did not request.",
    )
    def minimal_filter_compliance(inputs, outputs, expectations, trace=None):
        expected_filters = (expectations or {}).get("expected_filters") or {}
        actual_filters = _primary_tool_input(outputs, trace)
        actual_filter_keys = {
            key
            for key, value in actual_filters.items()
            if key in _FILTER_KEYS and value is not None
        }
        expected_filter_keys = {key for key in expected_filters if key in _FILTER_KEYS}
        if not actual_filter_keys:
            return Feedback(
                value=1.0 if not expected_filter_keys else 0.0,
                rationale="No filter-bearing tool input was available.",
            )
        score = len(actual_filter_keys & expected_filter_keys) / len(actual_filter_keys)
        extra_filters = sorted(actual_filter_keys - expected_filter_keys)
        return Feedback(
            value=score,
            rationale=(
                "No extra filters were added."
                if not extra_filters
                else f"Extra filters detected: {', '.join(extra_filters)}"
            ),
        )

    @scorer(
        name="joblab_out_of_scope_refusal",
        description="Check that out-of-scope questions are declined without tool usage.",
    )
    def out_of_scope_refusal(inputs, outputs, expectations, trace=None):
        expected_tool = (expectations or {}).get("expected_tool")
        category = (expectations or {}).get("category")
        if expected_tool is not None and category != "out_of_scope":
            return Feedback(
                value=1.0,
                rationale="Case is in-scope; refusal is not expected.",
            )
        answer = _answer_text(outputs).lower()
        tool_name = _primary_tool_name(outputs, trace)
        declined = any(indicator in answer for indicator in _DECLINE_INDICATORS)
        if declined and tool_name is None:
            return Feedback(
                value=1.0,
                rationale="Correctly declined without tool usage.",
            )
        if declined or tool_name is None:
            return Feedback(
                value=0.5,
                rationale="Only one of the expected refusal signals was present.",
            )
        return Feedback(
            value=0.0,
            rationale="The answer neither declined clearly nor avoided tool usage.",
        )

    @scorer(
        name="joblab_required_job_fields",
        description="Check that job-listing answers include job_id and a URL/link.",
    )
    def required_job_fields(inputs, outputs, expectations, trace=None):
        expected_tool = (expectations or {}).get("expected_tool")
        expected_result_mode = (expectations or {}).get("expected_result_mode")
        if expected_tool not in {"search_jobs", "semantic_search_jobs"}:
            return Feedback(
                value=1.0,
                rationale="This case does not require job-list formatting.",
            )
        if expected_result_mode == "no_results":
            return Feedback(
                value=1.0,
                rationale="No-results cases are not required to include job_id or URL.",
            )
        answer = _answer_text(outputs).lower()
        has_job_id = "job_id" in answer or "job id" in answer
        has_url = "http://" in answer or "https://" in answer or "url" in answer
        score = (float(has_job_id) + float(has_url)) / 2.0
        missing = []
        if not has_job_id:
            missing.append("job_id")
        if not has_url:
            missing.append("url")
        return Feedback(
            value=score,
            rationale=(
                "Required job fields are present."
                if not missing
                else f"Missing required job fields: {', '.join(missing)}"
            ),
        )

    @scorer(
        name="joblab_semantic_search_exclusivity",
        description="Ensure semantic-search cases use only semantic_search_jobs.",
    )
    def semantic_search_exclusivity(inputs, outputs, expectations, trace=None):
        expected_tool = (expectations or {}).get("expected_tool")
        if expected_tool != "semantic_search_jobs":
            return Feedback(
                value=1.0,
                rationale="This case is not a semantic-search case.",
            )
        tool_names = [call.get("name") for call in _resolved_tool_calls(outputs, trace)]
        if not tool_names:
            return Feedback(value=0.0, rationale="No tool usage was recorded.")
        unique_tool_names = {name for name in tool_names if isinstance(name, str)}
        score = 1.0 if unique_tool_names == {"semantic_search_jobs"} else 0.0
        return Feedback(
            value=score,
            rationale=(
                "Only semantic_search_jobs was used."
                if score == 1.0
                else f"Observed tool families: {sorted(unique_tool_names)}"
            ),
        )

    return [
        tool_choice_accuracy,
        filter_recall,
        out_of_scope_refusal,
    ]


def build_registrable_scorers() -> list[Any]:
    return []


def build_all_scorers() -> list[Any]:
    return [
        *build_code_scorers(),
        *build_registrable_scorers(),
    ]
