"""
Pydantic v2 models for agent trace data.

An AgentTrace captures a single agent invocation as an ordered sequence of
AgentSteps.  Each step may include a ToolCall (with labelled expected values
for offline / labelled-eval datasets) alongside free-form input/output text.

Design notes
------------
- All fields use strict Pydantic v2 validation.
- expected_tool / expected_args are optional so the same schema works for both
  production traces (unlabelled) and curated eval sets (labelled).
- Timestamps are stored as ISO-8601 strings so the model serialises cleanly
  to/from JSONL without custom encoders.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    """A single tool invocation inside one agent step."""

    name: str = Field(..., description="Name of the tool that was called.")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool.")
    success: bool = Field(..., description="Whether the tool call completed without error.")
    error_type: str | None = Field(
        default=None,
        description=(
            "Categorised error when success=False. "
            "One of: schema_mismatch, timeout, api_error, other."
        ),
    )

    @model_validator(mode="after")
    def error_type_only_on_failure(self) -> ToolCall:
        if self.success and self.error_type is not None:
            raise ValueError("error_type must be None when success=True")
        return self


class AgentStep(BaseModel):
    """One reasoning / action step within an agent trace."""

    step_id: int = Field(..., ge=0, description="Zero-based index of this step in the trace.")
    input: str = Field(..., description="Input text or observation fed into this step.")
    tool_call: ToolCall | None = Field(
        default=None, description="Tool invocation, if this step called a tool."
    )
    tool_result: str | None = Field(
        default=None, description="Raw string result returned by the tool."
    )
    output: str = Field(..., description="The agent's output / reasoning for this step.")
    timestamp: str = Field(..., description="ISO-8601 timestamp when this step was executed.")

    # Labelled-eval fields — present in curated datasets, absent in raw production traces
    expected_tool: str | None = Field(
        default=None,
        description="Ground-truth tool name expected at this decision point.",
    )
    expected_args: dict[str, Any] | None = Field(
        default=None,
        description="Ground-truth args expected for the tool call.",
    )


class AgentTrace(BaseModel):
    """
    A complete agent invocation: ordered sequence of steps plus metadata.

    Args:
        trace_id:    Unique identifier for this trace.
        steps:       Ordered list of AgentStep objects (at least one required).
        final_output: The agent's final answer / response to the user.
        metadata:    Optional bag of extra fields (session_id, user_id, etc.).
    """

    trace_id: str = Field(..., description="Unique identifier for this trace.")
    steps: list[AgentStep] = Field(..., min_length=1, description="Ordered steps; at least one.")
    final_output: str = Field(..., description="Final answer produced by the agent.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def steps_are_ordered(self) -> AgentTrace:
        for i, step in enumerate(self.steps):
            if step.step_id != i:
                raise ValueError(
                    f"Step at position {i} has step_id={step.step_id}; expected {i}. "
                    "Steps must be ordered and zero-indexed."
                )
        return self

    @property
    def num_steps(self) -> int:
        """Number of steps in this trace."""
        return len(self.steps)

    @property
    def tool_steps(self) -> list[AgentStep]:
        """Steps that include a tool call."""
        return [s for s in self.steps if s.tool_call is not None]

    @property
    def labelled_tool_steps(self) -> list[AgentStep]:
        """Tool steps that also have an expected_tool label."""
        return [s for s in self.tool_steps if s.expected_tool is not None]
