#!/usr/bin/env python3
"""
Validate JSONL evaluation datasets before running experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_TOOLS = {"search_jobs", "job_stats", "semantic_search_jobs"}


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _validate_required_filters(case_id: str, filters: Any) -> None:
    if filters is None:
        return
    if not isinstance(filters, dict):
        _fail(f"{case_id}: required_filters must be an object")
    for key, value in filters.items():
        if not isinstance(key, str) or not key.strip():
            _fail(f"{case_id}: required_filters keys must be non-empty strings")
        if isinstance(value, dict):
            valid_matcher = "contains" in value or "any_of" in value
            if not valid_matcher:
                _fail(
                    f"{case_id}: matcher object for '{key}' must include "
                    "'contains' or 'any_of'"
                )


def _validate_case(case: dict[str, Any], line_no: int, seen_ids: set[str]) -> None:
    case_id = str(case.get("id") or f"line-{line_no}")

    if "prompt" not in case or not isinstance(case["prompt"], str) or not case["prompt"].strip():
        _fail(f"{case_id}: prompt is required and must be a non-empty string")

    if "id" in case:
        if case["id"] in seen_ids:
            _fail(f"{case_id}: duplicate id")
        seen_ids.add(case["id"])

    if "expected_primary_tool" in case and case["expected_primary_tool"] is not None:
        v = case["expected_primary_tool"]
        if isinstance(v, str):
            if v not in ALLOWED_TOOLS:
                _fail(f"{case_id}: expected_primary_tool '{v}' is invalid")
        elif isinstance(v, list):
            if not v:
                _fail(f"{case_id}: expected_primary_tool list cannot be empty")
            for item in v:
                if item not in ALLOWED_TOOLS:
                    _fail(f"{case_id}: expected_primary_tool item '{item}' is invalid")
        else:
            _fail(f"{case_id}: expected_primary_tool must be string, list, or null")

    if "expect_any_tool_call" in case and not isinstance(case["expect_any_tool_call"], bool):
        _fail(f"{case_id}: expect_any_tool_call must be boolean")

    if "expected_answer_contains" in case:
        v = case["expected_answer_contains"]
        if isinstance(v, str):
            pass
        elif isinstance(v, list):
            if not all(isinstance(item, str) for item in v):
                _fail(f"{case_id}: expected_answer_contains list must contain only strings")
        else:
            _fail(f"{case_id}: expected_answer_contains must be string or list")

    if "conversation_group" in case and not isinstance(case["conversation_group"], str):
        _fail(f"{case_id}: conversation_group must be a string")

    _validate_required_filters(case_id, case.get("required_filters"))


def validate(path: Path) -> tuple[int, int]:
    seen_ids: set[str] = set()
    total = 0
    grouped = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"Line {line_no}: invalid JSON ({exc})")
            if not isinstance(obj, dict):
                _fail(f"Line {line_no}: JSON object expected")
            _validate_case(obj, line_no, seen_ids)
            total += 1
            if "conversation_group" in obj:
                grouped += 1

    if total == 0:
        _fail("Dataset is empty")
    return total, grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eval dataset JSONL format.")
    parser.add_argument(
        "--dataset",
        default="evals/datasets/ai_tool_eval.release.jsonl",
        help="Dataset path.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent / dataset_path
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    total, grouped = validate(dataset_path)
    print(f"Dataset valid: {dataset_path}")
    print(f"Total cases: {total}")
    print(f"Conversation-group cases: {grouped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

