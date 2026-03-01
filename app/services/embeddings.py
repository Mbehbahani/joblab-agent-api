"""
Bedrock Titan Text Embeddings V2 service.

Provides vector embeddings for semantic search using
Amazon Titan Text Embeddings V2 (512 dimensions).
Reuses the same region and IAM configuration as the main Bedrock service.

MLflow Tracing:
  • Each embedding call is wrapped in an MLflow span
  • Logs model ID, input length, vector dimensions, latency
"""

import json
import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── MLflow availability ─────────────────────────────────────────────────────

_mlflow = None
try:
    import mlflow as _mlflow
except ImportError:
    pass

# ── Singleton client ────────────────────────────────────────────────────────

_embed_client = None


def _get_embed_client():
    """Return a reusable bedrock-runtime client for embeddings (created once)."""
    global _embed_client
    if _embed_client is None:
        settings = get_settings()
        _embed_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_region,
        )
    return _embed_client


def embed_text(text: str) -> list[float]:
    """
    Generate a vector embedding for the given text using
    Bedrock Titan Text Embeddings V2 (512 dims).

    Parameters
    ----------
    text : str
        The text to embed. Should be non-empty.

    Returns
    -------
    list[float]
        A 512-dimensional embedding vector.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    settings = get_settings()
    client = _get_embed_client()

    body: dict[str, Any] = {
        "inputText": text.strip(),
        "dimensions": settings.embed_dimension,
    }

    start = time.time()

    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=settings.bedrock_embed_model_id,
            accept="application/json",
            contentType="application/json",
        )
    except ClientError as exc:
        logger.error("Bedrock embedding invocation failed: %s", exc)
        raise

    response_body = json.loads(response["body"].read())
    embedding: list[float] = response_body["embedding"]

    elapsed = round(time.time() - start, 3)
    logger.info(
        "embed_text  model=%s  dims=%d  chars=%d  time=%.3fs",
        settings.bedrock_embed_model_id,
        len(embedding),
        len(text),
        elapsed,
    )

    # ── MLflow span enrichment ──────────────────────────────────────────
    if _mlflow:
        try:
            span = _mlflow.get_current_active_span()
            if span:
                span.set_attributes({
                    "embedding.model_id": settings.bedrock_embed_model_id,
                    "embedding.input_chars": len(text),
                    "embedding.dimensions": len(embedding),
                    "embedding.latency_ms": round(elapsed * 1000, 1),
                })
        except Exception as e:
            logger.debug("MLflow embedding span enrichment failed (non-fatal): %s", e)

    return embedding
