# Cost Tracking Experiment

This folder is isolated from `app/main.py` so you can test model-cost experiments safely.

## Run

```powershell
python -m evals.cost_tracking_experiment.run_cost_experiment
```

## Common variants

```powershell
# Custom experiment name
python -m evals.cost_tracking_experiment.run_cost_experiment --experiment "joblab-cost-test-01"

# Compare multiple Bedrock models
python -m evals.cost_tracking_experiment.run_cost_experiment --models "us.anthropic.claude-3-5-haiku-20241022-v1:0,amazon.nova-pro-v1:0"

# Run Step 6 in a dedicated experiment
python -m evals.cost_tracking_experiment.run_cost_experiment --step6-experiment "04-production-candidate-testing"

# Skip Step 6 if you only want cost comparison
python -m evals.cost_tracking_experiment.run_cost_experiment --skip-step6
```

## What to customize

- Prompt/tasks: edit `get_tasks()` in `run_cost_experiment.py`
- Pricing map: fill `MODEL_PRICING_USD_PER_1K` to log `estimated_cost_usd` and `mlflow.llm.cost` on trace spans
- Default model/region/tracking URI: controlled by CLI args (defaults come from env vars / `.env`)

## Notes

- `mlflow.bedrock.autolog()` captures trace-level token usage and latency.
- `estimated_cost_usd` and trace cost attributes are only logged when pricing is defined in the script.

## Step 6 (Bedrock): Cost Efficiency Breakdown

After Step 5 cost tracking, this script now also logs:

- `input_cost_usd`
- `output_cost_usd`
- `estimated_cost_usd`
- `effective_cost_per_1k_tokens`

And prints a per-model/per-task summary:

```text
model=... | task=... | tokens=... | cost=$... | effective_per_1k=$...
```

This helps compare price efficiency across Bedrock models for your own prompt/task set.

## Step 6 (Bedrock): Tags and Metadata

The script also runs tagged configuration tests in a separate experiment (default `04-production-candidate-testing`):

- Configs: baseline + creative
- Tags: `config_name`, `task`, `stage`, `team`, `version`, `production_candidate`
- Artifact: `config.json` logged per run

This follows the intended pattern:

- Autolog captures model/tokens/latency/I-O
- Manual tags + config capture ownership, stage, and production candidacy metadata
