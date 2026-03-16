"""
Pydantic models for request / response validation.
"""

from pydantic import BaseModel, Field, model_validator


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
        max_length=20000,
        description="Optional system prompt override used for evaluation and optimization.",
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
    job_results: list[dict] | None = None
    gate_outcome: str | None = None
    result_type: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = Field(
        default=None,
        description="Server-generated turn ID for this answer. Used for offline feedback fallback.",
    )
    trace_id: str | None = Field(
        default=None,
        description="MLflow trace ID for this turn. Used by the client to submit feedback.",
    )


class FeedbackRequest(BaseModel):
    """Body for the /ai/feedback endpoint."""

    trace_id: str | None = Field(
        default=None,
        description="MLflow trace ID returned in the AskResponse when available.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID fallback when trace_id is not available yet.",
    )
    turn_id: str | None = Field(
        default=None,
        description="Turn ID fallback when trace_id is not available yet.",
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

    @model_validator(mode="after")
    def _validate_locator(self) -> "FeedbackRequest":
        if self.trace_id:
            return self
        if self.conversation_id and self.turn_id:
            return self
        raise ValueError(
            "Provide trace_id, or provide both conversation_id and turn_id."
        )


class FeedbackResponse(BaseModel):
    """Response confirming feedback was recorded."""

    status: str = "ok"
    trace_id: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None


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
