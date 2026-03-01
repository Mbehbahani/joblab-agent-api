"""
FastAPI application entry-point.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health, ai, cv_match

settings = get_settings()

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
        # Probe the tracking server's /health endpoint
        try:
            _resp = _health_req.get(
                f"{_tracking_uri.rstrip('/')}/health", timeout=5,
            )
            _resp.raise_for_status()
            _log.info("MLflow server is reachable (%s)", _tracking_uri)
        except Exception as _hc_err:
            _log.warning(
                "MLflow server unreachable (%s): %s", _tracking_uri, _hc_err,
            )
            if settings.mlflow_tracking_uri_fallback:
                _tracking_uri = settings.mlflow_tracking_uri_fallback
                _using_fallback = True
                _log.info(
                    "Falling back to direct DB tracking: %s",
                    _tracking_uri.split("@")[-1] if "@" in _tracking_uri else _tracking_uri[:40],
                )
            else:
                _log.warning(
                    "MLflow server unreachable and no MLFLOW_TRACKING_URI_FALLBACK set — tracing disabled"
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

    mlflow.set_experiment(settings.mlflow_experiment_name)

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
