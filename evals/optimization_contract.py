"""
Optimization contract for prompt improvement.

This module turns Step 2 (optimization goals + failure modes) into code that
the evaluation scripts can import. It keeps one source of truth for:

- the failure modes that matter for this agent
- the optimization goals they map to
- how existing dataset categories map to those failure modes
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailureModeDefinition:
    key: str
    tier: int
    prompt_sensitive: bool
    description: str
    prompt_sections: tuple[str, ...]
    scorer_targets: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationGoalDefinition:
    key: str
    description: str
    failure_modes: tuple[str, ...]


FAILURE_MODES: dict[str, FailureModeDefinition] = {
    "wrong_tool_choice": FailureModeDefinition(
        key="wrong_tool_choice",
        tier=1,
        prompt_sensitive=True,
        description="The agent selects the wrong tool or uses a tool for an out-of-scope query.",
        prompt_sections=("database_enforcement", "tool_selection", "semantic_search"),
        scorer_targets=("tool choice correctness",),
    ),
    "missing_required_filters": FailureModeDefinition(
        key="missing_required_filters",
        tier=1,
        prompt_sensitive=True,
        description="The agent omits one or more explicit user filters.",
        prompt_sections=("available_filters", "temporal_rules", "data_policy"),
        scorer_targets=("filter recall",),
    ),
    "added_extra_filters": FailureModeDefinition(
        key="added_extra_filters",
        tier=1,
        prompt_sensitive=True,
        description="The agent adds constraints the user did not ask for.",
        prompt_sections=("minimal_filter",),
        scorer_targets=("no extra filters",),
    ),
    "answered_without_tool": FailureModeDefinition(
        key="answered_without_tool",
        tier=1,
        prompt_sensitive=True,
        description="The agent answers a DB-backed question from memory instead of using a tool.",
        prompt_sections=("database_enforcement", "data_policy"),
        scorer_targets=("tool usage required",),
    ),
    "hallucinated_numbers": FailureModeDefinition(
        key="hallucinated_numbers",
        tier=1,
        prompt_sensitive=True,
        description="The answer includes unsupported numbers or conclusions not grounded in tool output.",
        prompt_sections=("data_policy", "response_style"),
        scorer_targets=("answer faithfulness", "numeric grounding"),
    ),
    "wrong_semantic_search_usage": FailureModeDefinition(
        key="wrong_semantic_search_usage",
        tier=2,
        prompt_sensitive=True,
        description="Semantic-search questions are not routed correctly or are mixed with other tool types.",
        prompt_sections=("semantic_search",),
        scorer_targets=("semantic-search exclusivity", "tool choice correctness"),
    ),
    "weak_trend_explanation": FailureModeDefinition(
        key="weak_trend_explanation",
        tier=2,
        prompt_sensitive=True,
        description="The model receives trend data but explains it weakly or incorrectly.",
        prompt_sections=("response_style",),
        scorer_targets=("trend explanation quality",),
    ),
    "poor_result_formatting": FailureModeDefinition(
        key="poor_result_formatting",
        tier=2,
        prompt_sensitive=True,
        description="Job results are not presented in a clear, structured, scannable way.",
        prompt_sections=("response_style",),
        scorer_targets=("formatting quality",),
    ),
    "missing_job_id_or_url": FailureModeDefinition(
        key="missing_job_id_or_url",
        tier=2,
        prompt_sensitive=True,
        description="Job results omit required identifiers like job_id or URL.",
        prompt_sections=("response_style", "memory_rules"),
        scorer_targets=("job result field presence",),
    ),
    "out_of_scope_failure": FailureModeDefinition(
        key="out_of_scope_failure",
        tier=2,
        prompt_sensitive=True,
        description="The model answers off-topic questions instead of declining and redirecting.",
        prompt_sections=("database_enforcement", "identity"),
        scorer_targets=("out-of-scope refusal correctness",),
    ),
    "poor_followup_memory": FailureModeDefinition(
        key="poor_followup_memory",
        tier=3,
        prompt_sensitive=True,
        description="The agent mishandles follow-up refinements or references to previously mentioned jobs.",
        prompt_sections=("memory_rules", "followup_rules"),
        scorer_targets=("follow-up memory quality",),
    ),
    "incorrect_handoff_or_clarification": FailureModeDefinition(
        key="incorrect_handoff_or_clarification",
        tier=3,
        prompt_sensitive=False,
        description="The agent should clarify or hand off but chooses the wrong outcome.",
        prompt_sections=("followup_rules",),
        scorer_targets=("handoff/clarification quality",),
    ),
}


OPTIMIZATION_GOALS: dict[str, OptimizationGoalDefinition] = {
    "tool_policy": OptimizationGoalDefinition(
        key="tool_policy",
        description="Choose the correct tool and avoid direct-memory answers for DB-backed questions.",
        failure_modes=("wrong_tool_choice", "answered_without_tool", "wrong_semantic_search_usage"),
    ),
    "filter_fidelity": OptimizationGoalDefinition(
        key="filter_fidelity",
        description="Preserve explicit user filters and avoid adding extra constraints.",
        failure_modes=("missing_required_filters", "added_extra_filters"),
    ),
    "grounded_analytics": OptimizationGoalDefinition(
        key="grounded_analytics",
        description="Keep answers grounded in tool outputs and avoid unsupported numeric claims.",
        failure_modes=("hallucinated_numbers", "weak_trend_explanation"),
    ),
    "result_presentation": OptimizationGoalDefinition(
        key="result_presentation",
        description="Present results clearly and include required identifiers like job_id and URL.",
        failure_modes=("poor_result_formatting", "missing_job_id_or_url"),
    ),
    "boundary_handling": OptimizationGoalDefinition(
        key="boundary_handling",
        description="Handle out-of-scope questions and conversation follow-ups correctly.",
        failure_modes=("out_of_scope_failure", "poor_followup_memory", "incorrect_handoff_or_clarification"),
    ),
}


CATEGORY_TO_FAILURE_MODES: dict[str, tuple[str, ...]] = {
    "count_query": ("wrong_tool_choice", "answered_without_tool", "hallucinated_numbers"),
    "count_with_filter": (
        "wrong_tool_choice",
        "missing_required_filters",
        "added_extra_filters",
        "hallucinated_numbers",
    ),
    "count_with_date": (
        "wrong_tool_choice",
        "missing_required_filters",
        "added_extra_filters",
        "hallucinated_numbers",
    ),
    "trend_query": ("wrong_tool_choice", "answered_without_tool", "weak_trend_explanation"),
    "trend_with_filter": (
        "wrong_tool_choice",
        "missing_required_filters",
        "added_extra_filters",
        "weak_trend_explanation",
    ),
    "search_query": ("wrong_tool_choice", "poor_result_formatting", "missing_job_id_or_url"),
    "search_with_date": (
        "wrong_tool_choice",
        "missing_required_filters",
        "poor_result_formatting",
        "missing_job_id_or_url",
    ),
    "search_with_level": (
        "wrong_tool_choice",
        "missing_required_filters",
        "poor_result_formatting",
        "missing_job_id_or_url",
    ),
    "multi_filter": (
        "wrong_tool_choice",
        "missing_required_filters",
        "added_extra_filters",
        "poor_result_formatting",
        "missing_job_id_or_url",
    ),
    "semantic_search": (
        "wrong_tool_choice",
        "wrong_semantic_search_usage",
        "poor_result_formatting",
    ),
    "distribution": ("wrong_tool_choice", "hallucinated_numbers"),
    "comparison": ("wrong_tool_choice", "weak_trend_explanation"),
    "aggregation": ("wrong_tool_choice", "hallucinated_numbers"),
    "cross_dimension": (
        "wrong_tool_choice",
        "missing_required_filters",
        "added_extra_filters",
    ),
    "negated_filter": (
        "wrong_tool_choice",
        "missing_required_filters",
        "added_extra_filters",
        "hallucinated_numbers",
    ),
    "out_of_scope": ("out_of_scope_failure",),
}


DEFAULT_FAILURE_MODES_BY_EXPECTED_TOOL: dict[str, tuple[str, ...]] = {
    "job_stats": ("wrong_tool_choice", "answered_without_tool", "hallucinated_numbers"),
    "search_jobs": ("wrong_tool_choice", "poor_result_formatting", "missing_job_id_or_url"),
    "semantic_search_jobs": ("wrong_tool_choice", "wrong_semantic_search_usage"),
}


def infer_failure_modes(
    *,
    category: str = "",
    expected_tool: str | None = None,
) -> list[str]:
    """Infer relevant failure modes for an eval case from its category and tool."""
    failure_modes: list[str] = []
    seen: set[str] = set()

    for failure_mode in CATEGORY_TO_FAILURE_MODES.get(category, ()):
        if failure_mode not in seen:
            seen.add(failure_mode)
            failure_modes.append(failure_mode)

    if expected_tool is not None:
        for failure_mode in DEFAULT_FAILURE_MODES_BY_EXPECTED_TOOL.get(expected_tool, ()):
            if failure_mode not in seen:
                seen.add(failure_mode)
                failure_modes.append(failure_mode)

    if expected_tool is None and category != "out_of_scope":
        if "out_of_scope_failure" not in seen:
            seen.add("out_of_scope_failure")
            failure_modes.append("out_of_scope_failure")

    return failure_modes


def infer_optimization_goals(failure_modes: list[str]) -> list[str]:
    """Map failure modes to optimization-goal keys."""
    goal_keys: list[str] = []
    seen: set[str] = set()
    failure_mode_set = set(failure_modes)
    for goal in OPTIMIZATION_GOALS.values():
        if failure_mode_set.intersection(goal.failure_modes) and goal.key not in seen:
            seen.add(goal.key)
            goal_keys.append(goal.key)
    return goal_keys


def annotate_toolchoice_dataset(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach failure modes and optimization goals to tool-choice dataset rows."""
    annotated: list[dict[str, Any]] = []
    for row in dataset:
        item = deepcopy(row)
        failure_modes = infer_failure_modes(
            category=str(item.get("category", "")),
            expected_tool=item.get("expected_tool"),
        )
        item["failure_modes"] = failure_modes
        item["optimization_goals"] = infer_optimization_goals(failure_modes)
        annotated.append(item)
    return annotated


def annotate_genai_dataset(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach failure modes and optimization goals to MLflow GenAI eval rows."""
    annotated: list[dict[str, Any]] = []
    for row in dataset:
        item = deepcopy(row)
        expectations = dict(item.get("expectations", {}))
        failure_modes = infer_failure_modes(
            category=str(expectations.get("category", "")),
            expected_tool=expectations.get("expected_tool"),
        )
        expectations["failure_modes"] = failure_modes
        expectations["optimization_goals"] = infer_optimization_goals(failure_modes)
        item["expectations"] = expectations
        annotated.append(item)
    return annotated


def summarize_failure_mode_coverage_from_toolchoice(
    dataset: list[dict[str, Any]],
) -> dict[str, int]:
    """Count how many tool-choice cases cover each failure mode."""
    counts = {key: 0 for key in FAILURE_MODES}
    for row in dataset:
        for failure_mode in row.get("failure_modes", []):
            counts[failure_mode] = counts.get(failure_mode, 0) + 1
    return {key: value for key, value in counts.items() if value > 0}


def summarize_failure_mode_coverage_from_genai(
    dataset: list[dict[str, Any]],
) -> dict[str, int]:
    """Count how many GenAI eval cases cover each failure mode."""
    counts = {key: 0 for key in FAILURE_MODES}
    for row in dataset:
        expectations = row.get("expectations", {})
        for failure_mode in expectations.get("failure_modes", []):
            counts[failure_mode] = counts.get(failure_mode, 0) + 1
    return {key: value for key, value in counts.items() if value > 0}
