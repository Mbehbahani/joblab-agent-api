"""
Pydantic models for request / response validation.
"""

from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    """Body for the /ai/ask endpoint."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User question to send to the AI model.",
    )
    system: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional system prompt to guide the model.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for session memory. Auto-generated if omitted.",
    )


# ── Responses ───────────────────────────────────────────────────────────────


class AskResponse(BaseModel):
    """Successful AI response."""

    answer: str
    model: str
    usage: dict | None = None
    tool_calls: list[dict] | None = None
    conversation_id: str | None = None
    trace_id: str | None = Field(
        default=None,
        description="MLflow trace ID for this turn. Used by the client to submit feedback.",
    )


class FeedbackRequest(BaseModel):
    """Body for the /ai/feedback endpoint."""

    trace_id: str = Field(
        ...,
        description="MLflow trace ID returned in the AskResponse.",
    )
    thumbs_up: bool = Field(
        ...,
        description="True = positive feedback, False = negative.",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional free-text comment from the user.",
    )


class FeedbackResponse(BaseModel):
    """Response confirming feedback was recorded."""

    status: str = "ok"
    trace_id: str


class MlflowFlushRequest(BaseModel):
    """Body for manual MLflow spool flush."""

    max_items: int = Field(
        default=100,
        ge=1,
        le=5000,
        description="Maximum queued events to replay from S3 in this call.",
    )


class MlflowFlushResponse(BaseModel):
    """Result of spool flush replay."""

    status: str = "ok"
    processed: int
    succeeded: int
    failed: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    detail: str
