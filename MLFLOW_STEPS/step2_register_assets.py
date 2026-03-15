"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 2 — Register Assets: Prompt, Dataset, Agent Model, Judge Model      ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS DOES (5 sub-steps):

  2a. PROMPT REGISTRATION
      Captures the current system prompt from prompt_policy.py and stores
      it in MLflow's Prompt Registry as a versioned artifact. Attaches
      LLM config (model, temperature, max_tokens) and sets aliases
      ("production", "baseline") for easy reference.

  2b. DATASET REGISTRATION
      Reads evals/optimization_baseline_dataset.json (10 cases), transforms
      to MLflow canonical format (inputs + expectations), and creates an
      MLflow GenAI dataset linked to the experiment.

  2c. AGENT MODEL REGISTRATION  ← NEW
      Uses mlflow.set_active_model() to create a LoggedModel entry for
      your Joblab agent. This links traces to a named GenAI app version
      and logs hyperparameters (model, temperature, prompt) as model params.

  2d. JUDGE MODEL REGISTRATION  ← NEW
      Registers the Correctness LLM judge as a separate LoggedModel.
      This makes the judge visible in the Models tab and lets you track
      judge parameters independently.

  2e. SCORER LISTING
      Lists the 3 code scorers + 1 LLM judge that Step 3 will use.

WHY REGISTER MODELS:
  Without mlflow.set_active_model(), your traces float unattached.
  The Overview dashboard's Token Usage, Cost, and Tool Calls sections
  need traces linked to a LoggedModel to aggregate properly.

WHAT TO CHECK IN MLFLOW UI:
  • Prompt Registry → "joblab-system-prompt" with version + aliases
  • Datasets tab → "opnow-baseline" with 10 records
  • Models tab → "joblab-agent" with params (llm, temperature, etc.)
  • Models tab → "myendpoint-judge" with params (judge model, scorer)

USAGE:
  python MLFLOW_STEPS/step2_register_assets.py
  python MLFLOW_STEPS/step2_register_assets.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ── Make MLFLOW_STEPS/ and project root importable ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_config import (
    AGENT_MODEL_NAME,
    DATASET_NAME,
    DATASET_SOURCE,
    EXPERIMENT_NAME,
    JUDGE_MODEL,
    JUDGE_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    PROJECT_ROOT,
    PROMPT_NAME,
    banner,
    info,
    resolve_tracking_uri,
    warn,
)


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2: MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def step2_register_all(tracking_uri: str, experiment_id: str, dry_run: bool = False) -> None:
    """Register prompt, dataset, agent model, judge model, and list scorers."""
    import mlflow

    banner("STEP 2: Register All Assets")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2a. Prompt
    banner("Step 2a: Register System Prompt", char="─")
    _register_prompt(tracking_uri, dry_run)

    # 2b. Dataset
    banner("Step 2b: Register Evaluation Dataset", char="─")
    _register_dataset(tracking_uri, experiment_id, dry_run)

    # 2c. Agent Model (LoggedModel)
    banner("Step 2c: Register Agent Model (LoggedModel)", char="─")
    _register_agent_model(tracking_uri, dry_run)

    # 2d. Judge Model (LoggedModel)
    banner("Step 2d: Register Judge Model (LoggedModel)", char="─")
    _register_judge_model(tracking_uri, dry_run)

    # 2e. List Scorers
    banner("Step 2e: List Scorers & Judge", char="─")
    _list_scorers(dry_run)


# ═════════════════════════════════════════════════════════════════════════════
#  2a. PROMPT REGISTRATION
# ═════════════════════════════════════════════════════════════════════════════

def _register_prompt(tracking_uri: str, dry_run: bool) -> None:
    """
    Register the current system prompt in MLflow Prompt Registry.

    ┌────────────────────────────────────────────────────────────────────┐
    │ CONCEPT: MLflow Prompt Registry                                    │
    │                                                                    │
    │ The Prompt Registry is an append-only store for system prompts.    │
    │ Each time you register a prompt, MLflow creates a new VERSION.     │
    │ Versions are immutable — you can't edit v5 after it's created.    │
    │                                                                    │
    │ ALIASES let you point a friendly name ("production", "baseline")  │
    │ to a specific version number. You can move aliases freely:        │
    │   "production" → v6    "baseline" → v6                            │
    │                                                                    │
    │ MODEL CONFIG attaches LLM hyperparameters to the prompt version:  │
    │   provider, model_name, temperature, max_tokens                   │
    │                                                                    │
    │ WHY: So your evaluation always knows EXACTLY which prompt + config │
    │ was used. Reproducibility!                                        │
    └────────────────────────────────────────────────────────────────────┘
    """
    from app.services.prompt_policy import DEFAULT_POLICY, POLICY_VERSION, get_system_prompt
    from mlflow.entities.model_registry import PromptModelConfig
    from mlflow.genai import (
        load_prompt,
        register_prompt,
        set_prompt_alias,
        set_prompt_model_config,
        set_prompt_tag,
    )

    template = get_system_prompt(DEFAULT_POLICY)
    sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    print(f"  Prompt name:      {PROMPT_NAME}")
    print(f"  Policy version:   {POLICY_VERSION}")
    print(f"  Template length:  {len(template)} chars")
    print(f"  SHA256:           {sha256[:16]}...")
    print(f"  LLM model:        {LLM_MODEL_NAME}")
    print(f"  Temperature:      {LLM_TEMPERATURE}")
    print(f"  Max tokens:       {LLM_MAX_TOKENS}")

    if dry_run:
        warn("DRY RUN — would register prompt, skipping.")
        return

    # Check if latest version already matches (avoid duplicates)
    latest = None
    try:
        latest = load_prompt(f"prompts:/{PROMPT_NAME}@latest", allow_missing=True)
    except Exception:
        pass

    model_config = PromptModelConfig(
        provider="bedrock",
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    if latest is not None and latest.template == template:
        version = int(latest.version)
        info(f"Prompt unchanged — reusing version {version}")
        set_prompt_model_config(PROMPT_NAME, version, model_config)
    else:
        prompt_version = register_prompt(
            name=PROMPT_NAME,
            template=template,
            commit_message=f"Opnew learning run (policy={POLICY_VERSION}, sha={sha256[:12]})",
            tags={
                "policy_version": POLICY_VERSION,
                "prompt_sha256": sha256,
                "experiment": EXPERIMENT_NAME,
            },
            model_config=model_config,
        )
        version = int(prompt_version.version)
        info(f"Created new prompt version: {version}")

    # Set aliases
    set_prompt_alias(PROMPT_NAME, "production", version)
    info(f"Alias 'production' → version {version}")

    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri)
    try:
        client.get_prompt_version_by_alias(PROMPT_NAME, "baseline")
        info("Alias 'baseline' already exists (not overwriting)")
    except Exception:
        set_prompt_alias(PROMPT_NAME, "baseline", version)
        info(f"Alias 'baseline' → version {version}")

    for key, value in {
        "project": "joblab",
        "component": "system_prompt",
        "agent_type": "tool_calling_jobs_assistant",
    }.items():
        set_prompt_tag(PROMPT_NAME, key, value)

    print()
    print("  📋 WHAT TO CHECK IN MLFLOW UI:")
    print(f"     → Prompt Registry → \"{PROMPT_NAME}\"")
    print(f"     → Version {version} with 'production' alias")
    print(f"     → Model config tab shows: {LLM_MODEL_NAME}")


# ═════════════════════════════════════════════════════════════════════════════
#  2b. DATASET REGISTRATION
# ═════════════════════════════════════════════════════════════════════════════

def _register_dataset(tracking_uri: str, experiment_id: str, dry_run: bool) -> None:
    """
    Register the 10-case optimization baseline dataset.

    ┌────────────────────────────────────────────────────────────────────┐
    │ CONCEPT: MLflow GenAI Datasets                                     │
    │                                                                    │
    │ A dataset in MLflow GenAI is a collection of records, each with:  │
    │   - inputs: {"question": "..."} — what to send to the agent       │
    │   - expectations: {"expected_tool": "...", ...} — ground truth    │
    │   - tags: {"scenario": "...", ...} — for filtering                │
    │                                                                    │
    │ The evaluate() function reads from this dataset, calls predict_fn │
    │ for each record, and scores the output against expectations.      │
    └────────────────────────────────────────────────────────────────────┘
    """
    from collections import Counter

    from mlflow.exceptions import MlflowException
    from mlflow.genai.datasets import (
        create_dataset,
        delete_dataset,
        get_dataset,
        set_dataset_tags,
    )

    with DATASET_SOURCE.open(encoding="utf-8") as f:
        raw_records = json.load(f)

    print(f"  Dataset name:     {DATASET_NAME}")
    print(f"  Source file:      {DATASET_SOURCE.relative_to(PROJECT_ROOT)}")
    print(f"  Record count:     {len(raw_records)}")

    canonical_records = []
    for row in raw_records:
        expected_tool = row.get("expected_tool")
        expectations = {
            key: value
            for key, value in {
                "expected_tool": expected_tool,
                "expected_filters": row.get("expected_filters") or {},
                "expected_response": row.get("expected_response"),
                "scenario": row["scenario"],
                "expected_result_mode": row["expected_result_mode"],
                "optimization_goals": row.get("optimization_goals", []),
                "must_include_fields": row.get("must_include_fields", []),
                "source_case_id": row["id"],
            }.items()
            if value not in (None, "", [])
        }
        tags = {
            "scenario": row["scenario"],
            "expected_tool": expected_tool or "none",
            "expected_result_mode": row["expected_result_mode"],
        }
        canonical_records.append({
            "inputs": {"question": row["prompt"]},
            "expectations": expectations,
            "tags": tags,
        })

    tools = Counter(r["expectations"].get("expected_tool", "none") for r in canonical_records)
    modes = Counter(r["expectations"].get("expected_result_mode", "?") for r in canonical_records)
    print(f"  Tools coverage:   {dict(sorted(tools.items()))}")
    print(f"  Result modes:     {dict(sorted(modes.items()))}")

    if dry_run:
        warn("DRY RUN — would register dataset, skipping.")
        return

    try:
        existing = get_dataset(name=DATASET_NAME)
        delete_dataset(dataset_id=existing.dataset_id)
        info(f"Deleted old dataset: {DATASET_NAME}")
    except MlflowException:
        pass

    dataset = create_dataset(name=DATASET_NAME, experiment_id=experiment_id)
    dataset = dataset.merge_records(canonical_records)
    set_dataset_tags(dataset.dataset_id, {
        "project": "joblab",
        "experiment": EXPERIMENT_NAME,
        "dataset_role": "baseline",
        "record_count": str(len(canonical_records)),
        "description": "10-case learning dataset for Opnew experiment",
    })

    record_count = len(dataset.to_dict().get("records", []))
    info(f"Dataset registered: id={dataset.dataset_id}, records={record_count}")

    print()
    print("  📋 WHAT TO CHECK IN MLFLOW UI:")
    print(f"     → Datasets tab → \"{DATASET_NAME}\"")
    print(f"     → Should show {record_count} records")


# ═════════════════════════════════════════════════════════════════════════════
#  2c. AGENT MODEL REGISTRATION (LoggedModel)
# ═════════════════════════════════════════════════════════════════════════════

def _register_agent_model(tracking_uri: str, dry_run: bool) -> None:
    """
    Register the Joblab agent as an MLflow LoggedModel.

    ┌────────────────────────────────────────────────────────────────────┐
    │ CONCEPT: MLflow LoggedModel (set_active_model)                     │
    │                                                                    │
    │ A LoggedModel represents a version of your GenAI application.     │
    │ It connects:                                                       │
    │   • Traces → so every agent call is linked to this model          │
    │   • Params → hyperparameters (model, temperature, prompt)         │
    │   • Tags   → metadata for filtering                               │
    │                                                                    │
    │ Without a LoggedModel, your traces float unattached and the       │
    │ Overview dashboard shows 0 tokens / $0.00 cost because MLflow    │
    │ doesn't know which model the traces belong to.                    │
    │                                                                    │
    │ mlflow.set_active_model(name="joblab-agent") creates or reuses    │
    │ a LoggedModel. After this call, any traces created in the same    │
    │ experiment will be linked to "joblab-agent".                      │
    │                                                                    │
    │ mlflow.log_model_params({...}) stores hyperparameters on the      │
    │ model so you can compare different versions later.                │
    └────────────────────────────────────────────────────────────────────┘
    """
    import mlflow

    print(f"  Model name:       {AGENT_MODEL_NAME}")
    print(f"  LLM:              {LLM_MODEL_NAME}")
    print(f"  Temperature:      {LLM_TEMPERATURE}")
    print(f"  Max tokens:       {LLM_MAX_TOKENS}")
    print(f"  Prompt:           {PROMPT_NAME}")

    if dry_run:
        warn("DRY RUN — would register agent model, skipping.")
        return

    # set_active_model creates or gets a LoggedModel named "joblab-agent"
    # under the current experiment. Returns an ActiveModel context.
    mlflow.set_active_model(name=AGENT_MODEL_NAME)

    # log_model_params stores key-value pairs on the active model.
    # These appear in the Models tab → Parameters section.
    mlflow.log_model_params({
        "prompt_template": PROMPT_NAME,
        "llm": LLM_MODEL_NAME,
        "provider": "bedrock",
        "temperature": str(LLM_TEMPERATURE),
        "max_tokens": str(LLM_MAX_TOKENS),
        "agent_type": "tool_calling",
        "tools": "search_jobs,job_stats,semantic_search_jobs",
    })

    info(f"Agent model registered: {AGENT_MODEL_NAME}")

    # Clear the active model so we don't accidentally link other things to it
    mlflow.clear_active_model()

    print()
    print("  📋 WHAT TO CHECK IN MLFLOW UI:")
    print(f"     → Models tab → \"{AGENT_MODEL_NAME}\"")
    print("     → Parameters: llm, temperature, max_tokens, tools")
    print("     → After Step 3, traces will be linked to this model")


# ═════════════════════════════════════════════════════════════════════════════
#  2d. JUDGE MODEL REGISTRATION (LoggedModel)
# ═════════════════════════════════════════════════════════════════════════════

def _register_judge_model(tracking_uri: str, dry_run: bool) -> None:
    """
    Register the Correctness judge as an MLflow LoggedModel.

    ┌────────────────────────────────────────────────────────────────────┐
    │ CONCEPT: Judge as a LoggedModel                                    │
    │                                                                    │
    │ The LLM judge (Correctness) is itself a model that:               │
    │   • Reads agent output + expectations                             │
    │   • Calls Bedrock Claude to evaluate quality                      │
    │   • Returns Pass/Fail + rationale                                 │
    │                                                                    │
    │ By registering it as a separate LoggedModel, you can:             │
    │   • Track judge cost independently                                │
    │   • Compare different judge models (e.g., Haiku vs Sonnet)        │
    │   • See judge-related traces in the Models tab                    │
    │                                                                    │
    │ NOTE ON AI GATEWAY:                                               │
    │ The judge model is routed through MLflow AI Gateway using         │
    │ "gateway:/myendpoint". That lets the gateway endpoint own the     │
    │ judge traffic and usage logging instead of bypassing the gateway. │
    └────────────────────────────────────────────────────────────────────┘
    """
    import mlflow

    print(f"  Judge model name: {JUDGE_MODEL_NAME}")
    print(f"  LLM:              {JUDGE_MODEL}")
    print(f"  Scorer:           Correctness (built-in)")

    if dry_run:
        warn("DRY RUN — would register judge model, skipping.")
        return

    mlflow.set_active_model(name=JUDGE_MODEL_NAME)

    mlflow.log_model_params({
        "model": JUDGE_MODEL,
        "provider": "gateway",
        "scorer_type": "builtin",
        "scorer_name": "Correctness",
        "role": "llm_judge",
        "gateway_endpoint": "myendpoint",
        "description": "Evaluates whether agent answers match expected behavior",
    })

    info(f"Judge model registered: {JUDGE_MODEL_NAME}")
    mlflow.clear_active_model()

    print()
    print("  📋 WHAT TO CHECK IN MLFLOW UI:")
    print(f"     → Models tab → \"{JUDGE_MODEL_NAME}\"")
    print("     → Parameters: model, scorer_name, role")


# ═════════════════════════════════════════════════════════════════════════════
#  2e. SCORER LISTING
# ═════════════════════════════════════════════════════════════════════════════

def _list_scorers(dry_run: bool) -> None:
    """List the scorers and judge that will run during evaluation."""
    from evals.mlflow_scorers import build_code_scorers

    code_scorers = build_code_scorers()

    print("  CODE SCORERS (3 — run as Python functions, no LLM cost):")
    for scorer in code_scorers:
        print(f"    • {scorer.name}: {scorer.description}")

    print()
    print("  LLM JUDGE (1 — uses Bedrock Claude to evaluate answer quality):")
    print(f"    • Correctness (model: {JUDGE_MODEL})")
    print("      Checks: Does the answer match expected facts/behavior?")

    print()
    print("  TOTAL: 3 code scorers + 1 LLM judge = 4 scoring signals per case")
    print(f"  COST:  ~10 LLM judge calls × ~0.001 USD each ≈ $0.01 per eval run")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Register assets into Opnew")
    parser.add_argument("--dry-run", action="store_true", help="Preview without registering")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 2 — Register Assets                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    tracking_uri = resolve_tracking_uri()

    # Need the experiment ID for dataset registration
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    step2_register_all(tracking_uri, experiment_id, dry_run=args.dry_run)

    banner("STEP 2 COMPLETE")
    print(f"  Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
