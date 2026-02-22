# MLflow Evaluation Guide (Offline First)

This folder adds a standards-based offline evaluation loop for `POST /ai/ask`:

1. Run fixed eval prompts
2. Score tool routing + filter preservation + latency
3. Track all runs in MLflow — **locally** or on **DagsHub** (remote)

Use this before/after prompt or orchestration changes so quality is measurable.

## 1) Install eval dependencies

From `lambda_backend/`:

```powershell
python -m pip install -r evals/requirements.txt
```

## 2) Start the backend locally

From `lambda_backend/`:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Run an offline eval

In a second terminal (from `lambda_backend/`):

```powershell
python evals/run_ai_offline_eval.py --api-base-url http://localhost:8000
```

This writes artifacts to:

- `evals/outputs/*_ai_eval_summary.json`
- `evals/outputs/*_ai_eval_results.jsonl`

And logs to local MLflow store:

- `sqlite:///mlflow.db` (recommended local backend)

## 4) Open MLflow UI

From `lambda_backend/`:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

## 5) Remote tracking with DagsHub

DagsHub hosts a free remote MLflow tracking server at
`https://dagshub.com/Mbehbahani/joblab-mlflow.mlflow`.  
Runs logged there are visible to everyone with access to the repo — no local
`mlflow.db` needed.

### One-time setup

**a) Create the DagsHub repo**

1. Go to <https://dagshub.com> and sign in with your GitHub account (`Mbehbahani`).
2. Click **New Repository** → choose **Connect a GitHub repo** or create a blank one
   named `joblab-mlflow`.
3. Copy your **personal access token** from
   <https://dagshub.com/user/settings/tokens>.

**b) Configure credentials locally**

```powershell
# From lambda_backend/
Copy-Item evals/dagshub.env.example evals/dagshub.env
# Then open evals/dagshub.env and replace <your_dagshub_token> with the real token
```

Load env vars before running evals (PowerShell):

```powershell
Get-Content evals/dagshub.env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+?)\s*=\s*(.+)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
```

**c) Install the dagshub package**

```powershell
python -m pip install -r evals/requirements.txt
```

### Running an eval against DagsHub

Add `--dagshub` flag (or set `DAGSHUB_ENABLE=1` in your env):

```powershell
python evals/run_ai_offline_eval.py `
  --api-base-url http://localhost:8000 `
  --experiment-name joblab_ai_release_eval `
  --run-name baseline_dagshub `
  --dagshub
```

### View runs online

```text
https://dagshub.com/Mbehbahani/joblab-mlflow/experiments
```

### Push eval code to DagsHub

```powershell
# Add DagsHub as a remote (first time only)
git remote add dagshub https://dagshub.com/Mbehbahani/joblab-mlflow.git

# Push
git push dagshub main          # or your current branch
```

> **Note:** `evals/dagshub.env` and `mlruns/` are in `.gitignore` and will never
> be committed.  Only the code and dataset files are pushed.

---

## 6) Dataset schema

JSONL file, one case per line.

Fields:

- `id` (optional): stable case id
- `prompt` (required): user query
- `conversation_group` (optional): same group value = same conversation_id across rows (for follow-up tests)
- `expected_primary_tool` (optional): expected first/main tool name
- `required_filters` (optional): key/value filters expected in selected tool input
- `expect_any_tool_call` (optional): boolean expectation for whether a tool should be called
- `expected_answer_contains` (optional): string or list of substrings expected in answer text

Example:

```json
{"id":"stats_country_after_date","prompt":"Count jobs by country posted after 2026-01-01","expected_primary_tool":"job_stats","required_filters":{"group_by":"country","metric":"count","posted_start":"2026-01-01"}}
```

Starter dataset:

- `evals/datasets/ai_tool_eval.sample.jsonl`
- `evals/datasets/ai_tool_eval.release.jsonl`

Matcher notes for `required_filters` values:

- exact: `"country":"Germany"`
- substring: `"country":{"contains":"germany"}`
- alternatives: `"group_by":{"any_of":["country","platform"]}`

## 7) Useful run commands

Custom dataset:

```powershell
python evals/run_ai_offline_eval.py --dataset evals/datasets/my_eval.jsonl --api-base-url http://localhost:8000
```

Validate dataset first:

```powershell
python evals/validate_dataset.py --dataset evals/datasets/ai_tool_eval.release.jsonl
```

Custom experiment/tracking URI:

```powershell
python evals/run_ai_offline_eval.py `
  --api-base-url http://localhost:8000 `
  --experiment-name joblab_router_v2 `
  --tracking-uri sqlite:///mlflow.db
```

Skip MLflow (local scoring only):

```powershell
python evals/run_ai_offline_eval.py --skip-mlflow
```

## 8) What metrics are logged

- `http_success_rate`
- `primary_tool_accuracy`
- `required_filter_exact_rate`
- `required_filter_coverage`
- `tool_call_expectation_accuracy`
- `answer_contains_accuracy`
- `avg_latency_ms`
- `p95_latency_ms`

## 9) Standard workflow (recommended)

1. Baseline run on current code
2. Change prompt/routing/tool logic
3. Re-run same dataset
4. Compare runs in MLflow
5. Accept change only if metrics improve or stay within threshold

## 10) Checking steps and tricks

- Keep a small "smoke" dataset (10-20 cases) and a larger "release" dataset (100+ cases).
- Version datasets in git; changing dataset and code in same PR can hide regressions.
- Keep expected labels strict and deterministic (`expected_primary_tool`, key filters).
- Add hard cases (short follow-ups, ambiguous prompts, temporal constraints).
- Watch latency and accuracy together; avoid quality gains that explode p95 latency.
- Use stable `--experiment-name` per feature branch to compare quickly.

Deep operational guide:

- `evals/BASELINE_PLAYBOOK.md`
