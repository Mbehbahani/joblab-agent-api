"""
S3 CV Store – persists user CVs and match results in S3.

Each CV is stored as a JSON object at:
    cvs/{cv_id}.json

Schema:
    {
        "cv_id":      "<uuid>",
        "raw_text":   "<full CV text>",
        "embedding":  [<float>, ...],
        "top_matches": null | [<match>, ...]
    }
"""

import json
import logging
import uuid
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Boto3 client (singleton) ────────────────────────────────────────────────

_s3_client = None


def _get_client():
    """Return a reusable boto3 S3 client (created once per Lambda process)."""
    global _s3_client
    if _s3_client is None:
        settings = get_settings()
        _s3_client = boto3.client("s3", region_name=settings.aws_region)
        logger.info("S3 client initialised  region=%s", settings.aws_region)
    return _s3_client


def _bucket() -> str:
    return get_settings().s3_cv_bucket


# ── Public API (mirrors railway_db interface) ───────────────────────────────


def insert_cv(raw_text: str, embedding: list[float]) -> str:
    """
    Save a CV (text + embedding) to S3 and return its UUID.

    Parameters
    ----------
    raw_text : str
        The raw CV text.
    embedding : list[float]
        512-dimensional embedding vector.

    Returns
    -------
    str
        The UUID used as the S3 object key (cvs/{cv_id}.json).
    """
    cv_id = str(uuid.uuid4())
    key = f"cvs/{cv_id}.json"

    payload = {
        "cv_id": cv_id,
        "raw_text": raw_text,
        "embedding": embedding,
        "top_matches": None,
    }

    _get_client().put_object(
        Bucket=_bucket(),
        Key=key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    logger.info("S3: inserted CV  cv_id=%s  chars=%d", cv_id, len(raw_text))
    return cv_id


def update_matches(cv_id: str, matches_json: list[dict[str, Any]]) -> None:
    """
    Write (overwrite) the top_matches field for a CV object in S3.

    Parameters
    ----------
    cv_id : str
        UUID returned by insert_cv.
    matches_json : list[dict]
        Serialisable list of match objects.
    """
    key = f"cvs/{cv_id}.json"
    client = _get_client()
    bucket = _bucket()

    # Fetch existing object to preserve raw_text and embedding
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        payload = json.loads(response["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            logger.warning("S3: CV not found for update  cv_id=%s – creating stub", cv_id)
            payload = {"cv_id": cv_id, "raw_text": "", "embedding": [], "top_matches": None}
        else:
            raise

    payload["top_matches"] = matches_json

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    logger.info("S3: stored matches  cv_id=%s  count=%d", cv_id, len(matches_json))
