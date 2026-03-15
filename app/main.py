"""
FastAPI application entry-point.
"""

import hashlib
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health, ai, cv_match
from app.services.prompt_policy import DEFAULT_POLICY, get_system_prompt

settings = get_settings()


def _safe_logged_model_name(base_name: str, version: str) -> str:
    safe_version = version.replace(".", "_")
    return f"{base_name}-v{safe_version}"

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ── MLflow Tracing Initialization ───────────────────────────────────────────
# Must happen BEFORE any boto3 clients are created so autolog can patch them.
try:
    import mlflow
    import requests as _health_req

    _log = logging.getLogger(__name__)

    # ── Determine tracking URI (primary vs. direct-DB fallback) ─────────
    _tracking_uri = settings.mlflow_tracking_uri
    _using_fallback = False

    if _tracking_uri.startswith("http"):
        # Probe each candidate server in order:
        #   1. Primary (settings.mlflow_tracking_uri)
        #   2. Local MLflow instance (http://localhost:5001)
        _LOCAL_MLFLOW = "http://localhost:5001"
        _candidates = [_tracking_uri]
        if _tracking_uri.rstrip("/") != _LOCAL_MLFLOW.rstrip("/"):
            _candidates.append(_LOCAL_MLFLOW)

        _reached = False
        for _candidate in _candidates:
            try:
                _resp = _health_req.get(
                    f"{_candidate.rstrip('/')}/health", timeout=5,
                )
                _resp.raise_for_status()
                _tracking_uri = _candidate
                _reached = True
                _log.info("MLflow server is reachable (%s)", _tracking_uri)
                break
            except Exception as _hc_err:
                _log.warning(
                    "MLflow server unreachable (%s): %s", _candidate, _hc_err,
                )

        if not _reached:
            # 3. Direct-DB fallback (MLFLOW_TRACKING_URI_FALLBACK)
            if settings.mlflow_tracking_uri_fallback:
                _tracking_uri = settings.mlflow_tracking_uri_fallback
                _using_fallback = True
                _log.info(
                    "Falling back to direct DB tracking: %s",
                    _tracking_uri.split("@")[-1] if "@" in _tracking_uri else _tracking_uri[:40],
                )
            else:
                # 4. All options exhausted — disable tracing
                _log.warning(
                    "All MLflow servers unreachable and no MLFLOW_TRACKING_URI_FALLBACK set — tracing disabled"
                )
                raise ImportError("skip tracing")  # caught below, cleanly disables tracing

    mlflow.set_tracking_uri(_tracking_uri)

    # When using direct-DB, set the default artifact root so new
    # experiments / runs store artifacts in S3 (not local filesystem).
    if _using_fallback and settings.mlflow_default_artifact_root:
        import os
        os.environ.setdefault(
            "MLFLOW_DEFAULT_ARTIFACT_ROOT",
            settings.mlflow_default_artifact_root,
        )

    experiment = mlflow.set_experiment(settings.mlflow_experiment_name)

    # Link all new traces to a stable LoggedModel representing this agent version.
    try:
        _active_model_name = _safe_logged_model_name(
            settings.mlflow_active_model_name,
            settings.app_version,
        )
        _existing_models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"name = '{_active_model_name}'",
            max_results=1,
            output_format="list",
        )
        if _existing_models:
            _logged_model = _existing_models[0]
        else:
            _logged_model = mlflow.create_external_model(
                name=_active_model_name,
                experiment_id=experiment.experiment_id,
                model_type="agent",
                tags={
                    "app_name": settings.app_name,
                    "app_version": settings.app_version,
                    "agent_kind": "joblab_agent",
                },
            )

        _system_prompt = get_system_prompt()
        _policy_meta = DEFAULT_POLICY.metadata()
        mlflow.set_active_model(model_id=_logged_model.model_id)
        mlflow.log_model_params(
            {
                "app_version": settings.app_version,
                "prompt_policy_version": str(_policy_meta["policy_version"]),
                "prompt_active_sections": ",".join(_policy_meta["active_sections"]),
                "prompt_section_count": str(len(_policy_meta["active_sections"])),
                "prompt_char_count": str(_policy_meta["prompt_char_count"]),
                "prompt_sha256": hashlib.sha256(_system_prompt.encode("utf-8")).hexdigest(),
                "llm_provider": "bedrock",
                "llm_model": settings.bedrock_model_id,
                "temperature": str(settings.bedrock_temperature),
                "max_tokens": str(settings.bedrock_max_tokens),
                "embedding_model": settings.bedrock_embed_model_id,
                "embedding_dimension": str(settings.embed_dimension),
            },
            model_id=_logged_model.model_id,
        )
        _log.info(
            "MLflow active model set — name=%s, model_id=%s",
            _active_model_name,
            _logged_model.model_id,
        )
    except Exception as _model_exc:
        _log.warning("MLflow active model init failed (non-fatal): %s", _model_exc)

    # Auto-trace all Bedrock API calls (converse, invoke_model, etc.).
    # For async trace shipping, set env var MLFLOW_ASYNC_LOGGING=true
    # (handled in .env / Lambda env). lambda_handler.py calls
    # mlflow.flush_async_logging() after each invocation to ensure
    # traces are flushed before Lambda freezes.
    mlflow.bedrock.autolog()

    _log.info(
        "MLflow tracing enabled — tracking_uri=%s, "
        "experiment=%s, bedrock.autolog=ON, fallback=%s",
        _tracking_uri.split("@")[-1] if "@" in _tracking_uri else _tracking_uri,
        settings.mlflow_experiment_name,
        _using_fallback,
    )
except ImportError:
    # MLflow SDK not installed (e.g. Lambda) — try the lightweight REST client
    try:
        from app.services.mlflow_lite import init_lite_client
        _lite = init_lite_client(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )
        if _lite:
            logging.getLogger(__name__).info(
                "MLflow Lite (REST) enabled — %s, experiment=%s",
                settings.mlflow_tracking_uri,
                settings.mlflow_experiment_name,
            )
        else:
            logging.getLogger(__name__).info(
                "MLflow not installed and Lite client unavailable — tracing disabled"
            )
    except Exception as _lite_exc:
        logging.getLogger(__name__).info(
            "MLflow not installed, Lite fallback failed: %s — tracing disabled", _lite_exc
        )
except Exception as exc:
    logging.getLogger(__name__).warning("MLflow init failed (non-fatal): %s", exc)

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(ai.router)
app.include_router(cv_match.router)
