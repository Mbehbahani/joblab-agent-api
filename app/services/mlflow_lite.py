"""
MLflow Lite — lightweight REST client for AWS Lambda.

When the full ``mlflow`` SDK is unavailable (too heavy for Lambda's 250 MB
limit), this module provides experiment‑tracking via the MLflow Tracking
Server REST API. It supports:
  - Turn-level run logging (metrics / params / tags)
  - Trace lifecycle logging via trace REST endpoints
  - Trace feedback assessments for lightweight /feedback fallback

Requirements: only ``requests`` (already in the Lambda package).

Usage (auto‑initialised in ``app/main.py``):

    from app.services.mlflow_lite import get_lite_client
    client = get_lite_client()
    if client:
        client.log_turn_async(turn_record.to_dict())
"""

from __future__ import annotations

import json
import logging
import time
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import requests as _http
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Module‑level singleton ─────────────────────────────────────────────────

_client: Optional["MLflowLiteClient"] = None

_MAX_TRACE_METADATA_VALUE = 250
_MAX_TRACE_TAG_KEY = 250
_MAX_TRACE_TAG_VALUE = 4096


def init_lite_client(tracking_uri: str, experiment_name: str) -> Optional["MLflowLiteClient"]:
    """Create and register the singleton client.  Returns ``None`` on failure."""
    global _client
    try:
        _client = MLflowLiteClient(tracking_uri, experiment_name)
        if _client.available:
            return _client
        if _client.spool_ready:
            logger.info(
                "MLflow Lite: tracking server unavailable at init, using S3 spool-only mode"
            )
            return _client
        _client = None
    except Exception as exc:
        logger.warning("MLflow Lite init failed: %s", exc)
        _client = None
    return None


def get_lite_client() -> Optional["MLflowLiteClient"]:
    """Return the singleton, or ``None`` if not initialised."""
    return _client


# ── REST Client ────────────────────────────────────────────────────────────


class MLflowLiteClient:
    """Lightweight MLflow client — uses only the Tracking Server REST API."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.base_url = tracking_uri.rstrip("/")
        self.experiment_name = experiment_name
        self.experiment_id: Optional[str] = None
        self._session = _http.Session()
        self._settings = get_settings()

        self._spool_enabled = False
        self._spool_bucket = ""
        self._spool_prefix = ""
        self._s3 = None
        self._init_spool()

        self._resolve_experiment()

    def _init_spool(self) -> None:
        """Initialize optional S3 spool for durable logging when server is unavailable."""
        try:
            bucket = (self._settings.mlflow_spool_bucket or self._settings.s3_cv_bucket).strip()
            prefix = self._settings.mlflow_spool_prefix.strip().strip("/")
            if not self._settings.mlflow_spool_enabled or not bucket:
                return

            self._s3 = boto3.client("s3", region_name=self._settings.aws_region)
            self._spool_bucket = bucket
            self._spool_prefix = prefix or "mlflow-spool"
            self._spool_enabled = True
            logger.info(
                "MLflow Lite spool enabled  bucket=%s prefix=%s",
                self._spool_bucket,
                self._spool_prefix,
            )
        except Exception as exc:
            logger.warning("MLflow Lite spool init failed: %s", exc)
            self._spool_enabled = False

    # ── helpers ─────────────────────────────────────────────────────────

    def _api(
        self,
        method: str,
        path: str,
        timeout: int = 10,
        **kwargs: Any,
    ) -> Optional[dict]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.request(method, url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp.json() if resp.content else None
        except Exception as exc:
            logger.debug("MLflow Lite %s %s → %s", method, path, exc)
            return None

    @staticmethod
    def _clip(value: Any, max_len: int) -> str:
        text = str(value) if value is not None else ""
        return text if len(text) <= max_len else f"{text[: max_len - 3]}..."

    @staticmethod
    def _safe_json(value: Any, max_len: int) -> str:
        try:
            text = json.dumps(value, default=str, separators=(",", ":"))
        except Exception:
            text = str(value)
        return MLflowLiteClient._clip(text, max_len)

    @staticmethod
    def _normalize_trace_status(status: str) -> str:
        normalized = (status or "").strip().upper()
        if normalized in {"OK", "ERROR", "IN_PROGRESS"}:
            return normalized
        if normalized in {"FINISHED", "SUCCESS", "SUCCEEDED"}:
            return "OK"
        if normalized in {"FAILED", "FAILURE"}:
            return "ERROR"
        return "OK"

    @staticmethod
    def _proto_timestamp_now() -> str:
        """Return an RFC 3339 timestamp string compatible with protobuf JSON."""
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _make_trace_metadata_items(data: dict[str, Any]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for key, value in data.items():
            items.append(
                {
                    "key": MLflowLiteClient._clip(key, _MAX_TRACE_TAG_KEY),
                    "value": MLflowLiteClient._clip(value, _MAX_TRACE_METADATA_VALUE),
                }
            )
        return items

    @staticmethod
    def _make_trace_tag_items(data: dict[str, Any]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for key, value in data.items():
            items.append(
                {
                    "key": MLflowLiteClient._clip(key, _MAX_TRACE_TAG_KEY),
                    "value": MLflowLiteClient._clip(value, _MAX_TRACE_TAG_VALUE),
                }
            )
        return items

    @staticmethod
    def _timeline_to_tags(timeline: list[dict[str, Any]] | None) -> dict[str, str]:
        if not timeline:
            return {}

        tags: dict[str, str] = {"lite.timeline.event_count": str(len(timeline))}
        encoded = MLflowLiteClient._safe_json(timeline, 30000)
        chunk_size = 3900
        chunks = [
            encoded[i : i + chunk_size]
            for i in range(0, len(encoded), chunk_size)
        ]
        tags["lite.timeline.chunks"] = str(len(chunks))
        for i, chunk in enumerate(chunks[:8]):  # Keep bounded and queryable
            tags[f"lite.timeline.{i}"] = chunk
        return tags

    def _spool_key(self, op: str) -> str:
        now = datetime.now(timezone.utc)
        return (
            f"{self._spool_prefix}/"
            f"{now.year:04d}/{now.month:02d}/{now.day:02d}/{now.hour:02d}/"
            f"{int(time.time() * 1000)}-{op}-{uuid.uuid4().hex}.json"
        )

    @staticmethod
    def _safe_key_part(value: str) -> str:
        return (value or "").replace("/", "_").replace("\\", "_")

    def _trace_index_key(self, conversation_id: str, turn_id: str) -> str:
        conv = self._safe_key_part(conversation_id)
        turn = self._safe_key_part(turn_id)
        return f"{self._spool_prefix}-index/{conv}/{turn}.json"

    def _trace_index_put(self, conversation_id: str, turn_id: str, trace_id: str) -> None:
        if not (self._spool_enabled and self._s3 and conversation_id and turn_id and trace_id):
            return
        key = self._trace_index_key(conversation_id, turn_id)
        payload = {
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "trace_id": trace_id,
            "updated_at_ms": int(time.time() * 1000),
        }
        try:
            self._s3.put_object(
                Bucket=self._spool_bucket,
                Key=key,
                Body=json.dumps(payload, default=str),
                ContentType="application/json",
            )
        except Exception as exc:
            logger.warning(
                "MLflow Lite trace index put failed conversation=%s turn=%s err=%s",
                conversation_id,
                turn_id,
                exc,
            )

    def _trace_index_get(self, conversation_id: str, turn_id: str) -> Optional[str]:
        if not (self._spool_enabled and self._s3 and conversation_id and turn_id):
            return None
        key = self._trace_index_key(conversation_id, turn_id)
        try:
            resp = self._s3.get_object(Bucket=self._spool_bucket, Key=key)
            body = resp["Body"].read()
            data = json.loads(body)
            trace_id = str(data.get("trace_id", "")).strip()
            return trace_id or None
        except ClientError:
            return None
        except Exception as exc:
            logger.warning(
                "MLflow Lite trace index get failed conversation=%s turn=%s err=%s",
                conversation_id,
                turn_id,
                exc,
            )
            return None

    def _spool_put(self, op: str, payload: dict[str, Any]) -> bool:
        if not self._spool_enabled or not self._s3:
            return False
        envelope = {
            "v": 1,
            "op": op,
            "created_at_ms": int(time.time() * 1000),
            "payload": payload,
        }
        key = self._spool_key(op)
        try:
            self._s3.put_object(
                Bucket=self._spool_bucket,
                Key=key,
                Body=json.dumps(envelope, default=str),
                ContentType="application/json",
            )
            logger.warning("MLflow Lite: spooled op=%s key=%s", op, key)
            return True
        except Exception as exc:
            logger.warning("MLflow Lite spool put failed op=%s: %s", op, exc)
            return False

    def _spool_list(self, max_items: int) -> list[str]:
        if not self._spool_enabled or not self._s3:
            return []
        keys: list[str] = []
        token: Optional[str] = None

        while len(keys) < max_items:
            kwargs: dict[str, Any] = {
                "Bucket": self._spool_bucket,
                "Prefix": f"{self._spool_prefix}/",
                "MaxKeys": min(1000, max_items - len(keys)),
            }
            if token:
                kwargs["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj.get("Key", "")
                if key.endswith(".json"):
                    keys.append(key)
                    if len(keys) >= max_items:
                        break
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")

        # Sort lexicographically (keys are time-prefixed)
        keys.sort()
        return keys[:max_items]

    def _spool_get(self, key: str) -> Optional[dict[str, Any]]:
        if not self._spool_enabled or not self._s3:
            return None
        try:
            resp = self._s3.get_object(Bucket=self._spool_bucket, Key=key)
            body = resp["Body"].read()
            return json.loads(body)
        except ClientError:
            return None
        except Exception as exc:
            logger.warning("MLflow Lite spool get failed key=%s err=%s", key, exc)
            return None

    def _spool_delete(self, key: str) -> None:
        if not self._spool_enabled or not self._s3:
            return
        try:
            self._s3.delete_object(Bucket=self._spool_bucket, Key=key)
        except Exception as exc:
            logger.warning("MLflow Lite spool delete failed key=%s err=%s", key, exc)

    def _resolve_experiment(self) -> None:
        data = self._api(
            "GET",
            "/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": self.experiment_name},
        )
        if data and "experiment" in data:
            self.experiment_id = data["experiment"]["experiment_id"]
            logger.info(
                "MLflow Lite: experiment '%s' → id=%s",
                self.experiment_name,
                self.experiment_id,
            )
            return

        data = self._api(
            "POST",
            "/api/2.0/mlflow/experiments/create",
            json={"name": self.experiment_name},
        )
        if data:
            self.experiment_id = str(data.get("experiment_id", ""))
            logger.info(
                "MLflow Lite: created experiment '%s' → id=%s",
                self.experiment_name,
                self.experiment_id,
            )

    @property
    def available(self) -> bool:
        return self.experiment_id is not None

    @property
    def spool_ready(self) -> bool:
        return self._spool_enabled

    def _ensure_experiment(self) -> bool:
        if self.experiment_id is not None:
            return True
        self._resolve_experiment()
        return self.experiment_id is not None

    # ── public API ──────────────────────────────────────────────────────

    def start_trace(
        self,
        *,
        prompt: str,
        conversation_id: str,
        turn_id: str,
        policy_version: str = "",
        trace_name: str = "ask_agent_lite",
        metadata: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        spool_on_failure: bool = True,
    ) -> Optional[str]:
        """Start a trace using MLflow's v2 trace endpoint and return trace_id."""
        if not self.available and not self._ensure_experiment():
            return None
        if not self.experiment_id:
            return None

        now_ms = int(time.time() * 1000)
        meta: dict[str, Any] = {
            "mlflow.traceInputs": self._safe_json(
                {
                    "prompt": self._clip(prompt, 800),
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                },
                _MAX_TRACE_METADATA_VALUE,
            ),
            "mlflow.trace.session": conversation_id,
        }
        if metadata:
            meta.update(metadata)

        trace_tags: dict[str, Any] = {
            "mlflow.traceName": trace_name,
            "source": "lambda-lite",
            "conversation_id": conversation_id,
            "turn_id": turn_id,
        }
        if policy_version:
            trace_tags["policy_version"] = policy_version
        if tags:
            trace_tags.update(tags)

        payload = {
            "experiment_id": str(self.experiment_id),
            "timestamp_ms": now_ms,
            "request_metadata": self._make_trace_metadata_items(meta),
            "tags": self._make_trace_tag_items(trace_tags),
        }

        result = self._api("POST", "/api/2.0/mlflow/traces", json=payload, timeout=8)
        if not result:
            return None
        trace_info = result.get("trace_info") or {}
        trace_id = trace_info.get("request_id")
        if trace_id:
            logger.info("MLflow Lite: started trace %s", trace_id[:8])
            self._trace_index_put(conversation_id, turn_id, trace_id)
        return trace_id

    def set_trace_tag(
        self,
        trace_id: str,
        key: str,
        value: Any,
        *,
        spool_on_failure: bool = True,
    ) -> bool:
        """Set a single trace tag via REST (best effort)."""
        if not trace_id:
            return False
        result = self._api(
            "PATCH",
            f"/api/2.0/mlflow/traces/{trace_id}/tags",
            json={
                "key": self._clip(key, _MAX_TRACE_TAG_KEY),
                "value": self._clip(value, _MAX_TRACE_TAG_VALUE),
            },
            timeout=6,
        )
        ok = result is not None
        if not ok and spool_on_failure:
            self._spool_put(
                "trace_tag",
                {
                    "trace_id": trace_id,
                    "key": str(key),
                    "value": str(value),
                },
            )
        return ok

    def set_trace_tags(self, trace_id: str, tags: dict[str, Any]) -> None:
        """Set multiple trace tags via repeated best-effort calls."""
        if not tags or not trace_id:
            return
        for key, value in tags.items():
            self.set_trace_tag(trace_id, key, value)

    def _create_feedback_assessment(
        self,
        *,
        trace_id: str,
        name: str,
        value: Any,
        rationale: str = "",
        source_type: str = "HUMAN",
        source_id: str = "ai_explorer_ui",
    ) -> bool:
        """Create an MLflow trace assessment via the v3 REST API."""
        if not trace_id:
            return False

        now = self._proto_timestamp_now()
        payload: dict[str, Any] = {
            "assessment": {
                "assessment_name": name,
                "trace_id": trace_id,
                "source": {
                    "source_type": source_type,
                    "source_id": source_id,
                },
                "create_time": now,
                "last_update_time": now,
                "feedback": {"value": value},
                "valid": True,
            }
        }
        if rationale:
            payload["assessment"]["rationale"] = rationale[:2000]

        result = self._api(
            "POST",
            f"/api/3.0/mlflow/traces/{trace_id}/assessments",
            json=payload,
            timeout=8,
        )
        return result is not None

    def end_trace(
        self,
        *,
        trace_id: str,
        status: str = "OK",
        answer: str = "",
        usage: dict[str, Any] | None = None,
        gate_outcome: str = "",
        gate_confidence: float | None = None,
        gate_reason: str = "",
        result_type: str = "",
        error: str | None = None,
        timeline: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        spool_on_failure: bool = True,
    ) -> bool:
        """End a trace using MLflow's v2 trace endpoint."""
        if not trace_id:
            return False

        now_ms = int(time.time() * 1000)
        resolved_status = "ERROR" if error else self._normalize_trace_status(status)

        meta: dict[str, Any] = {
            "mlflow.traceOutputs": self._safe_json(
                {"answer": self._clip(answer, 800), "result_type": result_type},
                _MAX_TRACE_METADATA_VALUE,
            ),
        }
        if usage:
            meta["mlflow.trace.tokenUsage"] = self._safe_json(
                {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                _MAX_TRACE_METADATA_VALUE,
            )
        if metadata:
            meta.update(metadata)

        trace_tags: dict[str, Any] = {
            "gate_outcome": gate_outcome,
            "gate_confidence": gate_confidence if gate_confidence is not None else "",
            "gate_reason": self._clip(gate_reason, 1200),
            "result_type": result_type,
            "error": self._clip(error or "", 1200),
        }
        trace_tags.update(self._timeline_to_tags(timeline))
        if tags:
            trace_tags.update(tags)

        payload = {
            "request_id": trace_id,
            "timestamp_ms": now_ms,
            "status": resolved_status,
            "request_metadata": self._make_trace_metadata_items(meta),
            "tags": self._make_trace_tag_items(trace_tags),
        }
        result = self._api(
            "PATCH",
            f"/api/2.0/mlflow/traces/{trace_id}",
            json=payload,
            timeout=8,
        )
        if result:
            logger.info(
                "MLflow Lite: ended trace %s status=%s",
                trace_id[:8],
                resolved_status,
            )
            return True
        if spool_on_failure:
            self._spool_put(
                "trace_end",
                {
                    "trace_id": trace_id,
                    "status": status,
                    "answer": answer,
                    "usage": usage or {},
                    "gate_outcome": gate_outcome,
                    "gate_confidence": gate_confidence,
                    "gate_reason": gate_reason,
                    "result_type": result_type,
                    "error": error or "",
                    "timeline": timeline or [],
                    "metadata": metadata or {},
                    "tags": tags or {},
                },
            )
        return False

    def spool_trace_complete(
        self,
        *,
        prompt: str,
        conversation_id: str,
        turn_id: str,
        policy_version: str = "",
        answer: str = "",
        usage: dict[str, Any] | None = None,
        gate_outcome: str = "",
        gate_confidence: float | None = None,
        gate_reason: str = "",
        result_type: str = "",
        error: str | None = None,
        timeline: list[dict[str, Any]] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> bool:
        """Queue a full trace payload when start/end couldn't be sent immediately."""
        return self._spool_put(
            "trace_complete",
            {
                "prompt": prompt,
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "policy_version": policy_version,
                "answer": answer,
                "usage": usage or {},
                "gate_outcome": gate_outcome,
                "gate_confidence": gate_confidence,
                "gate_reason": gate_reason,
                "result_type": result_type,
                "error": error or "",
                "timeline": timeline or [],
                "tags": tags or {},
            },
        )

    def flush_spool(self, max_items: int = 100) -> dict[str, int]:
        """
        Replay queued MLflow events from S3.

        Returns counters: processed/succeeded/failed.
        """
        summary = {"processed": 0, "succeeded": 0, "failed": 0}
        if not self._spool_enabled:
            return summary

        keys = self._spool_list(max(1, max_items))
        for key in keys:
            summary["processed"] += 1
            envelope = self._spool_get(key)
            if not envelope:
                summary["failed"] += 1
                continue

            op = envelope.get("op")
            payload = envelope.get("payload", {})
            ok = False

            try:
                if op == "turn":
                    ok = self._log_turn(payload, spool_on_failure=False)
                elif op == "trace_complete":
                    trace_id = self.start_trace(
                        prompt=str(payload.get("prompt", "")),
                        conversation_id=str(payload.get("conversation_id", "")),
                        turn_id=str(payload.get("turn_id", "")),
                        policy_version=str(payload.get("policy_version", "")),
                        trace_name="ask_agent_lite",
                        spool_on_failure=False,
                    )
                    if trace_id:
                        ok = self.end_trace(
                            trace_id=trace_id,
                            status="ERROR" if payload.get("error") else "OK",
                            answer=str(payload.get("answer", "")),
                            usage=payload.get("usage", {}),
                            gate_outcome=str(payload.get("gate_outcome", "")),
                            gate_confidence=payload.get("gate_confidence"),
                            gate_reason=str(payload.get("gate_reason", "")),
                            result_type=str(payload.get("result_type", "")),
                            error=str(payload.get("error", "")) or None,
                            timeline=payload.get("timeline", []),
                            tags=payload.get("tags", {}),
                            spool_on_failure=False,
                        )
                elif op == "trace_end":
                    ok = self.end_trace(
                        trace_id=str(payload.get("trace_id", "")),
                        status=str(payload.get("status", "OK")),
                        answer=str(payload.get("answer", "")),
                        usage=payload.get("usage", {}),
                        gate_outcome=str(payload.get("gate_outcome", "")),
                        gate_confidence=payload.get("gate_confidence"),
                        gate_reason=str(payload.get("gate_reason", "")),
                        result_type=str(payload.get("result_type", "")),
                        error=str(payload.get("error", "")) or None,
                        timeline=payload.get("timeline", []),
                        metadata=payload.get("metadata", {}),
                        tags=payload.get("tags", {}),
                        spool_on_failure=False,
                    )
                elif op == "trace_tag":
                    ok = self.set_trace_tag(
                        trace_id=str(payload.get("trace_id", "")),
                        key=str(payload.get("key", "")),
                        value=str(payload.get("value", "")),
                        spool_on_failure=False,
                    )
                elif op == "trace_feedback":
                    ok = self.log_trace_feedback(
                        trace_id=str(payload.get("trace_id", "")),
                        thumbs_up=bool(payload.get("thumbs_up")),
                        comment=str(payload.get("comment", "")),
                        spool_on_failure=False,
                    )
                elif op == "trace_feedback_offline":
                    ok = self.log_offline_trace_feedback(
                        conversation_id=str(payload.get("conversation_id", "")),
                        turn_id=str(payload.get("turn_id", "")),
                        thumbs_up=bool(payload.get("thumbs_up")),
                        comment=str(payload.get("comment", "")),
                        spool_on_failure=False,
                    )
            except Exception as exc:
                logger.warning("MLflow Lite spool replay failed key=%s err=%s", key, exc)
                ok = False

            if ok:
                summary["succeeded"] += 1
                self._spool_delete(key)
            else:
                summary["failed"] += 1

        return summary

    def log_turn_async(self, turn_data: dict) -> None:
        """Fire‑and‑forget: log an agent turn as an MLflow run."""
        if not self.available and not self._ensure_experiment():
            self._spool_put("turn", dict(turn_data))
            return
        t = threading.Thread(
            target=self._log_turn,
            args=(dict(turn_data),),
            daemon=True,
        )
        t.start()

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for any in-flight background threads (best-effort)."""
        # threading.enumerate() includes daemon threads; we just give
        # them a moment to finish before Lambda freezes.
        deadline = time.time() + timeout
        for t in threading.enumerate():
            if t.name.startswith("Thread") and t.daemon and t.is_alive():
                remaining = deadline - time.time()
                if remaining > 0:
                    t.join(timeout=remaining)

    # ── internal ────────────────────────────────────────────────────────

    def _log_turn(self, d: dict, spool_on_failure: bool = True) -> bool:  # noqa: C901
        try:
            if not self.available and not self._ensure_experiment():
                if spool_on_failure:
                    self._spool_put("turn", d)
                return False

            now_ms = int(time.time() * 1000)
            turn_id = str(d.get("turn_id", ""))
            conv_id = str(d.get("conversation_id", ""))
            latency = float(d.get("total_latency_ms", 0))

            # ── create run ──────────────────────────────────────────
            result = self._api("POST", "/api/2.0/mlflow/runs/create", json={
                "experiment_id": self.experiment_id,
                "start_time": int(now_ms - latency),
                "run_name": f"turn-{turn_id[:8]}",
                "tags": [
                    {"key": "mlflow.runName", "value": f"turn-{turn_id[:8]}"},
                    {"key": "conversation_id", "value": conv_id},
                    {"key": "turn_id", "value": turn_id},
                    {"key": "source", "value": "lambda"},
                    {"key": "gate_outcome",
                     "value": str(d.get("gate_outcome", ""))},
                    {"key": "prompt_preview",
                     "value": str(d.get("user_prompt", ""))[:120]},
                ],
            })
            if not result or "run" not in result:
                logger.warning("MLflow Lite: create-run failed")
                if spool_on_failure:
                    self._spool_put("turn", d)
                return False

            run_id: str = result["run"]["info"]["run_id"]

            # ── metrics ─────────────────────────────────────────────
            metrics: list[dict] = []
            for key in (
                "total_latency_ms", "llm_latency_ms", "tool_latency_ms",
                "input_tokens", "output_tokens", "total_tokens",
                "gate_confidence", "tool_rounds", "soft_enforcement_retries",
                "prompt_char_count", "result_length",
            ):
                val = d.get(key)
                if val is not None:
                    try:
                        metrics.append({
                            "key": key,
                            "value": float(val),
                            "timestamp": now_ms,
                            "step": 0,
                        })
                    except (TypeError, ValueError):
                        pass

            # ── params ──────────────────────────────────────────────
            params: list[dict] = []
            for src, mx in (
                ("policy_version", 250),
                ("gate_outcome", 250),
                ("gate_reason", 250),
                ("result_type", 250),
                ("conversation_id", 36),
                ("turn_id", 250),
            ):
                val = d.get(src)
                if val is not None:
                    params.append({"key": src, "value": str(val)[:mx]})

            tools = d.get("tools_called", [])
            tool_names = ",".join(
                (t.get("name", "?") if isinstance(t, dict) else str(t))
                for t in tools
            ) or "none"
            params.append({"key": "tools_called", "value": tool_names})

            error = d.get("error")
            if error:
                params.append({"key": "error", "value": str(error)[:500]})

            # ── log batch ───────────────────────────────────────────
            if metrics or params:
                batch_result = self._api("POST", "/api/2.0/mlflow/runs/log-batch", json={
                    "run_id": run_id,
                    "metrics": metrics,
                    "params": params,
                })
                if batch_result is None:
                    logger.warning("MLflow Lite: log-batch failed for run %s", run_id[:8])
                    if spool_on_failure:
                        self._spool_put("turn", d)
                    return False

            # ── end run ─────────────────────────────────────────────
            status = "FAILED" if error else "FINISHED"
            update_result = self._api("POST", "/api/2.0/mlflow/runs/update", json={
                "run_id": run_id,
                "status": status,
                "end_time": now_ms,
            })
            if update_result is None:
                logger.warning("MLflow Lite: run update failed for run %s", run_id[:8])
                if spool_on_failure:
                    self._spool_put("turn", d)
                return False

            logger.info(
                "MLflow Lite: logged run %s for turn %s",
                run_id[:8], turn_id[:8],
            )
            return True
        except Exception as exc:
            logger.warning("MLflow Lite: _log_turn error: %s", exc)
            if spool_on_failure:
                self._spool_put("turn", d)
            return False

    # ── feedback ────────────────────────────────────────────────────────

    def log_trace_feedback(
        self,
        trace_id: str,
        thumbs_up: bool,
        comment: str = "",
        *,
        spool_on_failure: bool = True,
    ) -> bool:
        """Attach feedback assessments to a trace so MLflow quality views can use them."""
        if not trace_id:
            return False
        rationale = (
            comment[:1000]
            if comment
            else (
                "User indicated response was helpful"
                if thumbs_up
                else "User indicated response was not helpful"
            )
        )

        ok = self._create_feedback_assessment(
            trace_id=trace_id,
            name="user_satisfaction",
            value=thumbs_up,
            rationale=rationale,
        )
        if ok and comment:
            self._create_feedback_assessment(
                trace_id=trace_id,
                name="user_comment",
                value=comment[:1000],
            )

        if ok:
            self.set_trace_tag(
                trace_id,
                "feedback.thumbs_up",
                str(thumbs_up).lower(),
                spool_on_failure=False,
            )
            if comment:
                self.set_trace_tag(
                    trace_id,
                    "feedback.comment",
                    comment[:1000],
                    spool_on_failure=False,
                )
            return True

        if spool_on_failure:
            self._spool_put(
                "trace_feedback",
                {
                    "trace_id": trace_id,
                    "thumbs_up": thumbs_up,
                    "comment": comment[:1000],
                },
            )
        return False

    def log_offline_trace_feedback(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        thumbs_up: bool,
        comment: str = "",
        spool_on_failure: bool = True,
    ) -> bool:
        """
        Record feedback when trace_id is not available yet.
        Resolves trace_id via S3 trace-index once the trace is replayed.
        """
        if not conversation_id or not turn_id:
            return False

        trace_id = self._trace_index_get(conversation_id, turn_id)
        if trace_id:
            return self.log_trace_feedback(
                trace_id=trace_id,
                thumbs_up=thumbs_up,
                comment=comment,
                spool_on_failure=spool_on_failure,
            )

        if spool_on_failure:
            return self._spool_put(
                "trace_feedback_offline",
                {
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "thumbs_up": thumbs_up,
                    "comment": comment[:1000],
                },
            )
        return False

    def log_feedback(self, run_id: str, thumbs_up: bool, comment: str = "") -> None:
        """Attach user feedback as tags on an existing run."""
        if not self.available:
            return
        self._api("POST", "/api/2.0/mlflow/runs/set-tag", json={
            "run_id": run_id,
            "key": "feedback.thumbs_up",
            "value": str(thumbs_up).lower(),
        })
        if comment:
            self._api("POST", "/api/2.0/mlflow/runs/set-tag", json={
                "run_id": run_id,
                "key": "feedback.comment",
                "value": comment[:500],
            })
