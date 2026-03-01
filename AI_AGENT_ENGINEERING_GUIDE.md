# AI Agent Engineering Guide — What Was Built & How to Learn From It

## Overview

This guide documents the engineering improvements made to the JobLab AI Agent
(`lambda_backend/app/routers/ai.py`) and teaches you how to replicate this
process for the CV Match tool and future projects.

---

## What Was Wrong Before

The original `ai.py` was a **773-line monolithic file** with these issues:

| Problem | Impact |
|---------|--------|
| 250+ line system prompt mixed into router code | Untestable, can't A/B test prompt variants |
| No confidence evaluation | Agent blindly returns empty/poor results |
| No observability | Can't answer "which tools are used most?" or "what's our error rate?" |
| No evaluation framework | Can't measure if prompt changes improve or regress quality |
| No MLflow tracking | Can't compare experiments, no audit trail |

---

## What Was Built (Architecture)

```
Before:
  ai.py (773 lines, everything mixed together)

After:
  ai.py (refactored orchestrator, ~580 lines, focused on execution flow)
  ├── imports prompt_policy.py    → Composable system prompt (11 testable sections)
  ├── imports confidence_gate.py  → ANSWER/ASK_CLARIFICATION/DECLINE/HANDOFF state machine
  ├── imports turn_logger.py      → Per-turn tracing + MLflow + aggregate metrics
  └── /ai/metrics endpoint        → Live operational dashboard

  evals/
  ├── golden_dataset.json         → 30 test cases (prompt → expected tool + filters)
  └── score_toolchoice.py         → Offline evaluation scorer with MLflow logging
```

### Design Pattern: Policy / Execution / Evaluation Separation

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│   POLICY LAYER     │     │  EXECUTION LAYER   │     │  EVALUATION LAYER  │
│                    │     │                    │     │                    │
│ prompt_policy.py   │────▶│  ai.py (router)    │────▶│ score_toolchoice   │
│ - 11 sections      │     │  - tool loop       │     │ - 30 golden cases  │
│ - versioned        │     │  - confidence gate │     │ - tool accuracy    │
│ - A/B testable     │     │  - turn logging    │     │ - filter recall    │
│                    │     │  - memory          │     │ - filter precision │
└────────────────────┘     └────────────────────┘     └────────────────────┘
```

---

## File-by-File Explanation

### 1. `app/services/prompt_policy.py` — Policy Separation

**Why:** The system prompt is your agent's "brain." Making it monolithic means you can't:
- Test individual rules
- Swap out sections for A/B experiments  
- Track which version of the prompt produced which results

**How it works:**
```python
# Each section is a named constant
SECTION_TOOL_SELECTION = "..."
SECTION_DATA_POLICY = "..."

# Composed via PromptPolicy dataclass
policy = PromptPolicy()                        # full default
policy = PromptPolicy(exclude={"memory_rules"}) # A/B test without memory
system_prompt = policy.build()
```

**Key concept — Policy Version:**
```python
POLICY_VERSION = "1.0.0"  # Bump when you change any section
```
Every turn logs this version. When you change the prompt and re-run evals,
you can compare v1.0.0 vs v1.1.0 in MLflow.

**Interview language:**
> "I separate policy from execution so I can version, test, and A/B-test
> individual prompt sections without touching the orchestration logic."

---

### 2. `app/services/confidence_gate.py` — Confidence State Machine

**Why:** Without confidence gating, the agent blindly returns whatever the tool gives back — even if it's empty, the similarity is terrible, or the tool errored.

**The four outcomes:**

| Outcome | When | What Happens |
|---------|------|-------------|
| `ANSWER` | High confidence result | Normal response |
| `ASK_CLARIFICATION` | Empty results, weak similarity, ambiguous | Append suggestion: "Could you rephrase?" |
| `DECLINE` | Out-of-scope, policy violation | Refuse politely |
| `HANDOFF` | Repeated failures, critical errors | "Let me try a different approach" |

**Gating signals (features used for the decision):**
```python
signals = {
    "result_count": 15,
    "top_similarity": 0.67,    # semantic search only
    "score_margin": 0.12,      # top1 - top2
    "filters_applied": 3,
    "latency_ms": 1200,
    "tool_execution_success": True,
}
```

**How it integrates into ai.py:**
```python
# After every tool execution:
gate = evaluate_confidence(tool_name, tool_input, result_data, latency_ms=...)
track_confidence(conversation_id, gate)

if gate.outcome == GateOutcome.HANDOFF:
    return "I'm unable to process this reliably..."

if gate.outcome == GateOutcome.ASK_CLARIFICATION:
    answer += f"\n\n{gate.suggestion}"
```

**Interview language:**
> "I implemented a confidence-gated automation path with four outcome states.
> The agent evaluates result quality signals — count, similarity score,
> margin, execution success — before deciding whether to answer, ask for
> clarification, or hand off."

---

### 3. `app/services/turn_logger.py` — Observability + MLflow

**Why:** In production, you need to answer:
- "What's our tool selection success rate?"
- "What's the average latency?"
- "How often does the agent ask for clarification?"
- "Which prompts cause errors?"

**What it logs per turn:**

| Category | Fields |
|----------|--------|
| Identity | conversation_id, turn_id |
| Input | user_prompt, prompt_char_count |
| Policy | policy_version, system_prompt_chars |
| Tools | tools_called, tool_rounds, soft_enforcement_retries |
| Confidence | gate_outcome, gate_confidence, gate_reason |
| Timing | total_latency_ms, llm_latency_ms, tool_latency_ms |
| Tokens | input_tokens, output_tokens, total_tokens |
| Result | result_type, result_length, error |

**Two backends:**
1. **Structured logging** — always on, goes to CloudWatch in Lambda
2. **MLflow** — when installed, logs as experiment runs with metrics + params

**Aggregate metrics endpoint (`GET /ai/metrics`):**
```json
{
  "total_turns": 150,
  "outcome_counts": {"ANSWER": 120, "ASK_CLARIFICATION": 25, "DECLINE": 3, "HANDOFF": 2},
  "avg_latency_ms": 2800,
  "avg_confidence": 0.82,
  "success_rate": 0.8,
  "clarification_rate": 0.167,
  "handoff_rate": 0.013,
  "error_count": 2
}
```

**Interview language:**
> "Every agent turn is traced with structured logs and MLflow metrics.
> I track success rate, clarification rate, handoff rate, latency, and
> token usage. This gives me both real-time operational visibility and
> experiment tracking for prompt optimization."

---

### 4. `evals/` — Offline Evaluation Framework

**Why:** You can't improve what you can't measure. The eval framework:
- Establishes a **baseline** score for the current prompt
- Catches **regressions** when you change the prompt
- Scores **tool selection accuracy** and **filter preservation**

**Golden dataset (`golden_dataset.json`):**
30 cases across categories:
- count_query, search_query, trend_query
- semantic_search, multi_filter, out_of_scope
- negated_filter, cross_dimension, aggregation
- Easy / Medium / Hard difficulty levels

**Scoring metrics:**

| Metric | What it Measures |
|--------|-----------------|
| Tool Selection Accuracy | Did the LLM pick the right tool? |
| Filter Recall | Were all expected filters included? |
| Filter Precision | Were unnecessary filters avoided? (minimal filter policy) |
| Overall Score | Harmonic mean of the three |

**Running evaluations:**
```bash
# Dry run — show dataset without LLM calls
python -m evals.score_toolchoice --dry-run

# Run specific cases
python -m evals.score_toolchoice --ids eval_001 eval_006

# Run full eval with MLflow logging
python -m evals.score_toolchoice

# Run without MLflow
python -m evals.score_toolchoice --no-mlflow
```

**Baseline results (v1.0.0):**
```
Tool Selection Accuracy:  100.0%   ← perfect tool choice
Average Filter Recall:     81.8%   ← some filters missed
Average Filter Precision:  85.8%   ← some extra filters added
Average Overall Score:     79.6%   ← room to improve
Average Latency:           2727ms
```

**Interview language:**
> "I maintain a golden evaluation dataset of 30 cases covering different
> query types and difficulty levels. Every prompt change is scored against
> this baseline. I track tool-selection accuracy, filter recall, and
> filter precision — the same metrics used in information retrieval systems."

---

## Baseline Results Interpretation

### What's strong (100% tool accuracy)
The LLM always picks the right tool. The system prompt's tool selection rules
are very effective.

### Where to improve

1. **Semantic search filter recall (37.5%):**
   The LLM rephrases `query_text` differently than the golden labels.
   
   Fix: Make scoring more flexible for `query_text` (semantic similarity
   instead of exact match), or accept that rephrasing is fine.

2. **Some filter precision issues:**
   The LLM occasionally adds a `role_keyword` filter when only `group_by`
   was expected (e.g. "top companies hiring for data science" → adds
   `role_keyword="data science"` which is arguably correct).

3. **Latency (2.7s average):**
   Mostly Claude inference time. Can be improved with prompt compression
   or switching to a faster model for simple queries.

---

## How to Use MLflow

### View the dashboard:
```bash
cd lambda_backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Open http://localhost:5001

### Two experiments:
1. **`joblab-ai-eval`** — Offline evaluation runs (compare prompt versions)
2. **`joblab-ai-agent`** — Per-turn production logs (when agents serve real queries)

### Workflow for prompt optimization:
1. Check current baseline: `python -m evals.score_toolchoice`
2. Change a section in `prompt_policy.py`, bump `POLICY_VERSION`
3. Re-run eval: `python -m evals.score_toolchoice`
4. Compare in MLflow UI: v1.0.0 vs v1.1.0

---

## Your Next Steps (Learning Path)

### Step 1: Understand the code (30 min)
Read these files in order:
1. `app/services/prompt_policy.py` — simplest, just string composition
2. `app/services/confidence_gate.py` — rule-based state machine
3. `app/services/turn_logger.py` — logging + MLflow integration
4. `evals/score_toolchoice.py` — eval runner

### Step 2: Run the eval yourself (15 min)
```bash
cd lambda_backend
..\..\.venv\Scripts\Activate.ps1  # or source ../../.venv/bin/activate

# Dry run to see dataset
python -m evals.score_toolchoice --dry-run

# Run 5 specific cases
python -m evals.score_toolchoice --ids eval_001 eval_002 eval_003 eval_004 eval_005

# Run full eval
python -m evals.score_toolchoice
```

### Step 3: Try improving the prompt (1-2 hours)
1. Open `app/services/prompt_policy.py`
2. Modify `SECTION_TOOL_SELECTION` — e.g. add more examples
3. Bump `POLICY_VERSION` to `"1.1.0"`
4. Run eval: `python -m evals.score_toolchoice`
5. Compare scores in MLflow

### Step 4: Add more eval cases (30 min)
Open `evals/golden_dataset.json` and add cases that cover:
- Edge cases you've seen fail in practice
- Complex multi-filter queries
- Ambiguous prompts (did the agent ask for clarification?)

### Step 5: Apply the same pattern to CV Match (next session)
The same architecture applies to CV Match:
- **Policy separation:** Move CV matching logic into policy configs
- **Confidence gate:** Evaluate similarity scores → ANSWER/ASK/DECLINE
- **Turn logging:** Track embedding latency, match count, reranking quality
- **Eval dataset:** Golden CV → expected top-5 jobs with relevance labels

---

## Professional Terms You Can Now Use

| Term | Where It Appears |
|------|-----------------|
| Policy/execution separation | `prompt_policy.py` vs `ai.py` |
| Confidence-gated automation | `confidence_gate.py` |
| Outcome state machine | ANSWER / ASK / DECLINE / HANDOFF |
| Offline evaluation (eval) | `evals/score_toolchoice.py` |
| Golden dataset | `evals/golden_dataset.json` |
| Tool-selection accuracy | Eval metric |
| Filter recall / precision | IR metrics applied to tool inputs |
| Experiment tracking | MLflow runs |
| Per-turn observability | `turn_logger.py` |
| Structured logging | CloudWatch-compatible JSON logs |

---

## File Layout Summary

```
lambda_backend/
├── app/
│   ├── routers/
│   │   └── ai.py                    ← Refactored: orchestration only
│   └── services/
│       ├── prompt_policy.py         ← NEW: composable system prompt
│       ├── confidence_gate.py       ← NEW: ANSWER/ASK/DECLINE/HANDOFF
│       ├── turn_logger.py           ← NEW: MLflow + structured logging
│       ├── bedrock.py               ← unchanged
│       ├── joblab_tools.py          ← unchanged
│       └── conversation_memory.py   ← unchanged
├── evals/
│   ├── __init__.py
│   ├── golden_dataset.json          ← NEW: 30 test cases
│   ├── score_toolchoice.py          ← NEW: offline evaluation scorer
│   └── results/
│       └── baseline_v1.0.0.json     ← Your baseline scores
├── mlflow.db                        ← MLflow experiment database
└── requirements.txt                 ← Updated: added mlflow
```
