# MLflow Agent Observability And Evaluation Standard

This document defines the current standard for MLflow tracing, evaluation, prompt registry, AI Gateway usage, and dashboard population in this repository.

Use this as the reference when:

- running baseline evaluation
- debugging missing traces or spans
- checking why Overview tabs are empty
- registering prompts or datasets
- routing judges through AI Gateway
- preparing future optimization or evaluation work

## Scope

This standard covers:

- production agent tracing
- Step 3 baseline evaluation in `MLFLOW_STEPS/step3_run_evaluation.py`
- registered prompts and datasets
- code scorers and LLM judges
- AI Gateway judge routing
- MLflow Overview expectations for Usage, Quality, Tool calls, and Cost

It reflects the current working design in this repo.

## Canonical Components

### Experiments

- Production backend experiment: `joblab-ai-agent-production`
- Learning / evaluation experiment: `Opnew`

### Logged Models

- Production backend default model name: versioned `joblab-ai-agent-v<app_version>`
- Evaluation model name in `Opnew`: `joblab-agent`
- Judge model name in `Opnew`: `myendpoint-judge`

### Prompt Registry

- Prompt name: `joblab-system-prompt`
- Main alias used for evaluation: `production`

### Dataset Registry

- Step 3 evaluation dataset: `opnow-baseline`

### Judge Routing

- Judge endpoint: `gateway:/myendpoint`
- Judge scorer currently used in Step 3: `Correctness`

### Implementation Anchors

- Experiment constants: `MLFLOW_STEPS/pipeline_config.py`
- Production MLflow bootstrap: `app/main.py` -> `mlflow.set_experiment(...)`, `mlflow.set_active_model(...)`
- Step 3 bootstrap: `MLFLOW_STEPS/step3_run_evaluation.py` -> `step3_run_baseline(...)`

## Standard Trace Model

### Production Backend Request

For one user question sent to `/ai/ask`, the desired backend trace shape is:

1. `ask_agent`
2. `BedrockRuntime.converse`
3. `execute_<tool>`
4. `BedrockRuntime.converse`

Important points:

- one question should correspond to one main backend trace
- LLM calls should be child spans, not separate unrelated traces
- tool execution should appear as `SpanType.TOOL`
- token usage and cost should live on LLM spans

Relevant files:

- `app/routers/ai.py`
- `app/services/bedrock.py`
- `app/services/joblab_tools.py`
- `app/main.py`

Implementation anchors:

- root trace/span: `app/routers/ai.py` -> `ask(...)`
- LLM call path: `app/routers/ai.py` -> `invoke_claude(...)`
- Bedrock autolog: `app/main.py` -> `mlflow.bedrock.autolog()`
- TOOL span creation: `app/services/joblab_tools.py` -> `_tool_trace(...)`

### Step 3 Evaluation Request

For one evaluation case in `MLFLOW_STEPS/step3_run_evaluation.py`, the desired trace shape is:

1. `predict_fn`
2. `backend_request`
3. `ask_agent`
4. `BedrockRuntime.converse`
5. `execute_<tool>`
6. `BedrockRuntime.converse`

Meaning:

- `predict_fn` is the evaluation wrapper trace created by MLflow evaluation
- `backend_request` is the explicit client-side child span created in Step 3
- `ask_agent` and all deeper spans come from the backend through distributed trace propagation

This is the standard "one trace per question with all related spans" design.

Implementation anchors:

- evaluation entrypoint: `MLFLOW_STEPS/step3_run_evaluation.py` -> `step3_run_baseline(...)`
- per-case call wrapper: `MLFLOW_STEPS/step3_run_evaluation.py` -> `predict_fn(...)`
- explicit eval child span: `MLFLOW_STEPS/step3_run_evaluation.py` -> `mlflow.start_span(name="backend_request", ...)`
- backend trace join: `app/routers/ai.py` -> `ask(...)` with propagated tracing context

## Why This Structure Matters

### Usage / Cost

Usage and cost dashboards depend on:

- real LLM spans
- token-bearing span attributes
- traces linked to the correct logged model

### Tool Calls

Tool call dashboards depend on:

- real `TOOL` spans
- traces being written into the correct experiment
- traces being attached to the correct logged model

Linked traces from another experiment are useful for debugging, but they are not sufficient for reliable Overview aggregation.

### Quality

Quality depends on:

- MLflow assessments existing on traces
- not only tags or custom metadata

## Prompt Registry Standard

The project standard is:

1. Load the prompt from Prompt Registry, not from a local constant file.
2. Link the prompt to the active model.
3. Link the prompt version to the evaluation run.
4. Propagate prompt provenance onto the trace itself.

### Required Behavior

- Load prompt from `prompts:/joblab-system-prompt@production`
- Use the active model ID when calling `load_prompt(...)`
- Explicitly link prompt version to the run
- Set trace tags:
  - `mlflow.linkedPrompts`
  - `prompt_uri`
  - `prompt_version`

### Why

Without this:

- the prompt text may still be used
- but the `Prompt` column in the trace UI may stay empty
- and the registered prompt page may not show the new evaluation runs clearly

Implementation anchors:

- prompt load: `MLFLOW_STEPS/step3_run_evaluation.py` -> `load_prompt(...)`
- prompt-to-run link: `MLFLOW_STEPS/step3_run_evaluation.py` -> `client.link_prompt_version_to_run(...)`
- prompt trace provenance: `MLFLOW_STEPS/step3_run_evaluation.py` -> `mlflow.update_current_trace(tags={...})`
- backend prompt trace tags: `app/routers/ai.py` -> `update_current_trace(...)`

## Dataset Registry Standard

The project standard is:

- use MLflow registered datasets, not ad hoc local files, for normal evaluation runs

Step 3 dataset loading rule:

- primary path: `get_dataset(name=DATASET_NAME)`
- local JSON fallback is allowed only for dry-run or setup assistance

This keeps dataset lineage stable across evaluations.

Implementation anchors:

- registered dataset load: `MLFLOW_STEPS/step3_run_evaluation.py` -> `get_dataset(name=DATASET_NAME)`
- dry-run local fallback: `MLFLOW_STEPS/step3_run_evaluation.py` -> `DATASET_SOURCE.open(...)`

## Scorers And Judges Standard

### Code Scorers

Current Step 3 code scorers are built from `evals/mlflow_scorers.py`.

These are deterministic, repo-specific evaluators for:

- tool choice
- filter recall
- out-of-scope refusal

### LLM Judge

Current Step 3 LLM judge:

- `Correctness(model="gateway:/myendpoint")`

Standard rule:

- built-in or custom judges may be used
- but judge traffic must go through AI Gateway when we want gateway observability and usage tracking

Implementation anchors:

- code scorer factory: `evals/mlflow_scorers.py` -> `build_code_scorers()`
- Step 3 judge construction: `MLFLOW_STEPS/step3_run_evaluation.py` -> `Correctness(model=JUDGE_MODEL)`

## AI Gateway Standard

The current standard is:

- use `gateway:/myendpoint` for Step 3 judge traffic

This ensures:

- judge calls are routed through AI Gateway
- judge usage is visible through the gateway
- judge traffic is separated cleanly from backend agent traffic

Important distinction:

- backend agent traces come from `/ai/ask`
- judge traces come from the scorer model route through AI Gateway

They are related to the same evaluation workflow, but they are not the same trace source.

Implementation anchors:

- gateway endpoint config: `MLFLOW_STEPS/pipeline_config.py` -> `JUDGE_MODEL`
- judge usage in Step 3: `MLFLOW_STEPS/step3_run_evaluation.py` -> `llm_judge = Correctness(model=JUDGE_MODEL)`

## Evaluation Routing Standard

### Problem This Solves

If Step 3 calls the backend without override routing:

- backend traces may be logged under the backend's default production experiment
- Step 3 runs may only link those traces afterward
- `Opnew` Overview tabs may stay empty even though traces exist elsewhere

### Current Standard

Step 3 sends these headers to `/ai/ask`:

- `x-mlflow-experiment-name`
- `x-mlflow-active-model-name`
- `x-mlflow-prompt-name`
- `x-mlflow-prompt-version`
- `x-mlflow-prompt-uri`

The backend temporarily uses those overrides for that request, then restores its defaults.

### Why

This is what makes evaluation traces land in `Opnew` instead of only in the backend's default experiment.

Implementation anchors:

- header propagation from Step 3: `MLFLOW_STEPS/step3_run_evaluation.py` -> `headers["x-mlflow-experiment-name"] = ...`
- per-request experiment override: `app/routers/ai.py` -> `ask(...)`
- per-request restore of defaults: `app/routers/ai.py` -> `_restore_mlflow_route_override()`

## Operational Metrics Standard

### Step 3 Run-Level Metrics

Step 3 must log at least:

- `eval_input_tokens`
- `eval_output_tokens`
- `eval_total_tokens`
- `mean_eval_latency_ms`
- `median_eval_latency_ms`
- `p95_eval_latency_ms`
- `mean_total_tokens`
- `median_total_tokens`
- `mean_total_cost_usd`
- `median_total_cost_usd`
- `total_cost_usd`

### Backend Detailed Metrics

These live in backend traces and turn logging:

- `total_latency_ms`
- `llm_latency_ms`
- `tool_latency_ms`
- `tool_rounds`
- `soft_enforcement_retries`
- gate outcome and confidence

Standard interpretation:

- Step 3 run metrics are for aggregate evaluation comparison
- backend trace metrics are for per-question debugging and detailed observability

Implementation anchors:

- Step 3 cost computation: `MLFLOW_STEPS/step3_run_evaluation.py` -> `_estimate_total_cost_usd(...)`
- Step 3 operational summaries: `MLFLOW_STEPS/step3_run_evaluation.py` -> `operational_summary`
- backend turn metrics: `app/services/turn_logger.py` -> `TurnRecord`, `log_turn_mlflow(...)`

## Overview Dashboard Expectations

### Usage

Should populate when:

- LLM spans carry token usage
- traces are under the correct experiment/model

### Quality

Should populate when:

- assessments are recorded on traces

Implementation anchors:

- trace outcome assessments: `app/routers/ai.py` -> `_mlflow.log_feedback(...)`
- user feedback assessments / lite fallback: `app/services/mlflow_lite.py` -> `log_trace_feedback(...)`

### Tool calls

Should populate when:

- `TOOL` spans are present
- traces are written into the same experiment being viewed
- traces are under the correct logged model

### Cost

Should populate when:

- token usage exists on LLM spans
- model pricing metadata is available or inferable

## Standard Step 3 Workflow

1. Resolve tracking URI.
2. Set experiment to `Opnew`.
3. Set active model to `joblab-agent`.
4. Load registered prompt from Prompt Registry.
5. Load registered dataset from Dataset Registry.
6. Create evaluation run.
7. For each row:
   - start `backend_request` span
   - propagate trace context to `/ai/ask`
   - route backend trace into `Opnew`
   - route backend trace under `joblab-agent`
   - propagate prompt metadata
8. Evaluate with code scorers and judge scorer.
9. Log operational summaries.
10. Link prompt version to run.
11. Link backend traces to run for direct inspection.

Implementation anchors:

- Step 3 workflow owner: `MLFLOW_STEPS/step3_run_evaluation.py` -> `step3_run_baseline(...)`
- backend trace linking: `MLFLOW_STEPS/step3_run_evaluation.py` -> `client.link_traces_to_run(...)`

## Standard Production Workflow

1. Backend starts with MLflow configured in `app/main.py`.
2. Production traces go to `joblab-ai-agent-production`.
3. Active logged model is the versioned production backend model.
4. `/ai/ask` creates `ask_agent` root trace/span.
5. Bedrock autolog creates LLM spans.
6. Tool decorators create TOOL spans.
7. Feedback writes assessments to traces.

Implementation anchors:

- startup MLflow init: `app/main.py`
- root backend orchestration: `app/routers/ai.py` -> `ask(...)`
- production feedback path: `app/routers/ai.py` -> `feedback(...)`

## Known Failure Modes

### Only `predict_fn` appears in trace breakdown

Meaning:

- distributed tracing did not connect backend spans into the evaluation trace

Typical causes:

- no active child span inside `predict_fn`
- trace context not propagated
- backend did not restore remote trace context correctly

### One trace per span

Meaning:

- the request-wide trace context was detached too early
- later LLM/tool operations started their own root traces

### Overview `Tool calls` is empty

Meaning:

- tool spans exist, but not in the viewed experiment/model
- linked traces from another experiment are not enough for aggregation

### `Prompt` column is empty

Meaning:

- prompt text was used as raw string
- but trace provenance tags or prompt linkage were missing

### Prompt page does not show the new experiment/run

Meaning:

- the prompt was loaded, but not explicitly linked to the run/model in the way MLflow UI expects

### Quality tab is empty

Meaning:

- assessments were not written to traces
- tags alone are not enough

Implementation anchors:

- trace assessments: `app/routers/ai.py` -> `_mlflow.log_feedback(...)`
- lite REST assessments: `app/services/mlflow_lite.py` -> `_create_feedback_assessment(...)`

## Files That Define This Standard

- `app/main.py`
- `app/routers/ai.py`
- `app/services/bedrock.py`
- `app/services/joblab_tools.py`
- `app/services/mlflow_lite.py`
- `app/services/turn_logger.py`
- `MLFLOW_STEPS/pipeline_config.py`
- `MLFLOW_STEPS/step2_register_assets.py`
- `MLFLOW_STEPS/step3_run_evaluation.py`
- `evals/mlflow_scorers.py`

## Practical Rules

- Do not create fake duplicate LLM/tool spans in Step 3 if the backend already emits real spans.
- Do not rely only on cross-experiment trace linking for Overview aggregation.
- Do not pass only raw prompt text if you also care about prompt lineage in MLflow UI.
- Do not treat tags as a substitute for assessments when you want Quality views.
- Do keep one-question-one-trace as the target shape for evaluation and debugging.
- Do keep prompt, dataset, judge, experiment, and model lineage explicit.

## Current Canonical Check

After a correct Step 3 run, you should be able to verify all of the following:

- `Opnew` run exists
- registered prompt version is linked to the run
- registered dataset is used
- trace tree for one evaluation case is structured
- `backend_request` appears under `predict_fn`
- backend `ask_agent` / LLM / TOOL spans are connected or at minimum linked correctly
- judge traffic goes through `myendpoint`
- Overview tabs populate with real data
- prompt provenance is visible on traces and prompt page

If one of those fails, treat it as a standards regression and fix the integration before adding more evaluation logic.
