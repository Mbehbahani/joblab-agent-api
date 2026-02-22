# Baseline Playbook (Deep, Practical)

This playbook is the operational standard for running LLM evals in this repo.

Use it before/after any change to:
- prompt modules,
- routing logic,
- tool schemas,
- tool execution behavior,
- model parameters.

## A. Why this matters

Without a baseline, every "improvement" is subjective.
With a baseline, every change has measurable impact:

- quality up or down,
- latency impact,
- failure modes introduced or removed.

## B. What is measured

The evaluator logs these core metrics:

- `http_success_rate`: API reliability under eval load.
- `primary_tool_accuracy`: how often expected tool was selected.
- `required_filter_exact_rate`: case-level exact match on expected filters.
- `required_filter_coverage`: key-level filter coverage (partial signal).
- `tool_call_expectation_accuracy`: checks "should call a tool / should not call a tool".
- `answer_contains_accuracy`: weak answer correctness check on expected substrings.
- `avg_latency_ms`, `p95_latency_ms`: responsiveness.

## C. Dataset strategy (recommended)

Use two datasets:

1. Smoke dataset (fast, daily)
- 10-20 cases
- used before commits

2. Release dataset (decision-grade)
- 50-100+ cases
- includes conversation-threaded cases
- used for merge/deploy decisions

Current files:
- `evals/datasets/ai_tool_eval.sample.jsonl`
- `evals/datasets/ai_tool_eval.release.jsonl`

## D. Labeling rules

For each case, define only stable expectations:

- `expected_primary_tool`: only when intent is unambiguous
- `required_filters`: only critical filters (country/date/flags/etc.)
- `expect_any_tool_call`: use for follow-up/no-tool behavior checks
- `expected_answer_contains`: use sparingly for policy phrases or entity mentions

Avoid over-labeling brittle details that may vary stylistically.

## E. Run sequence (local)

From `lambda_backend/`:

1. Validate dataset
```powershell
python evals/validate_dataset.py --dataset evals/datasets/ai_tool_eval.release.jsonl
```

2. Start backend
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Run baseline eval
```powershell
python evals/run_ai_offline_eval.py `
  --dataset evals/datasets/ai_tool_eval.release.jsonl `
  --api-base-url http://localhost:8000 `
  --experiment-name joblab_ai_release_eval `
  --run-name baseline_local
```

4. Open MLflow
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## F. Run sequence (Lambda/API Gateway)

```powershell
python evals/run_ai_offline_eval.py `
  --dataset evals/datasets/ai_tool_eval.release.jsonl `
  --api-base-url https://<api-id>.execute-api.us-east-1.amazonaws.com `
  --experiment-name joblab_ai_release_eval `
  --run-name baseline_lambda
```

Compare `baseline_local` vs `baseline_lambda`.

## G. Decision gates (starter)

Set explicit pass/fail gates before coding changes:

- `primary_tool_accuracy >= 0.90`
- `required_filter_exact_rate >= 0.80`
- `tool_call_expectation_accuracy >= 0.90`
- `http_success_rate >= 0.99`
- `p95_latency_ms <= 12000` (adjust to real SLA)

## H. Change workflow standard

For any LLM-related change:

1. Baseline run (save run ID)
2. Apply change
3. Re-run same dataset
4. Compare metrics in MLflow
5. Promote only if gates pass
6. Document regressions if accepted knowingly

## I. Debug playbook for regressions

If `primary_tool_accuracy` drops:
- inspect per-case artifacts where `tool_match=false`
- check system prompt/routing rule changes
- verify tool schema and required fields

If `required_filter_exact_rate` drops:
- inspect cases with missing date/country/flags
- verify prompt-level and code-level filter guards
- ensure model did not over-summarize constraints

If latency jumps:
- inspect p95 outliers
- separate Bedrock time from tool execution time
- check Supabase/RPC response times

## J. Advanced tricks

- Keep "anchor cases" that should never regress (job_id exact lookup, month parsing, no-tool follow-up).
- Add hard negatives (ambiguous prompts) and evaluate `ASK_CLARIFICATION` policy once confidence-gate is added.
- Maintain dataset versions:
  - `ai_tool_eval.release.v1.jsonl`
  - `ai_tool_eval.release.v2.jsonl`
- Store run IDs in PR descriptions for auditable decisions.

## K. What this proves professionally

This baseline loop demonstrates:
- disciplined LLM evaluation,
- reproducible experiment tracking,
- measurable quality governance,
- production-grade change control.

