"""
Application configuration via environment variables.
Uses pydantic-settings for validation and type safety.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    app_name: str = "LLMBackend"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── AWS / Bedrock ────────────────────────────────────
    # Note: AWS_REGION is automatically provided by Lambda environment
    # For local dev, you can set it or use boto3's automatic detection
    aws_region: str = "us-east-1"  # Default, will be overridden by Lambda's AWS_REGION
    bedrock_model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    bedrock_max_tokens: int = 1024
    bedrock_temperature: float = 0.7
    # ── Bedrock Embeddings ───────────────────────────────────
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    embed_dimension: int = 512
    # ── Supabase ─────────────────────────────────────────
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # ── S3 CV Storage ─────────────────────────────────────
    s3_cv_bucket: str = "joblab-cv-store"

    # ── MLflow Tracking ──────────────────────────────────
    mlflow_tracking_uri: str = "https://mlflow-production-34b0.up.railway.app"
    mlflow_experiment_name: str = "joblab-ai-agent"

    # Direct DB fallback: used when the MLflow REST server is unreachable.
    # Set to a PostgreSQL URI, e.g. postgresql://user:pass@host:port/dbname
    # MLflow writes runs/traces directly to the DB via SQLAlchemy.
    # When the tracking server comes back, data appears automatically.
    mlflow_tracking_uri_fallback: str = ""
    # Artifact root for direct-DB mode (usually S3), e.g. s3://bucket/mlflow
    mlflow_default_artifact_root: str = ""

    # Durable store-and-forward for MLflow Lite when tracking server is down.
    mlflow_spool_enabled: bool = True
    # If empty, defaults to s3_cv_bucket
    mlflow_spool_bucket: str = ""
    # Prefix inside bucket for queued MLflow events
    mlflow_spool_prefix: str = "mlflow-spool"
    # Optional bearer token for manual flush endpoint
    mlflow_spool_flush_token: str = ""
    # Optional auto-flush on each invocation (0 disables)
    mlflow_spool_autoflush_max_items: int = 0

    # ── CORS ─────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
