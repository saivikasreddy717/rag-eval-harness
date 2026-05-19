"""
Mock trace generators for unit tests and example data.

All generators produce deterministic, realistic-looking AgentTrace objects
without any network calls.  Use these in tests and to generate the
examples/agent_traces_sample.jsonl file.
"""

from __future__ import annotations

from rag_eval.agent_eval.trace import AgentStep, AgentTrace, ToolCall

# ---------------------------------------------------------------------------
# Single-step traces
# ---------------------------------------------------------------------------


def make_single_step_trace(trace_id: str = "t-001", success: bool = True) -> AgentTrace:
    """One-step trace: a direct answer with no tool call."""
    return AgentTrace(
        trace_id=trace_id,
        steps=[
            AgentStep(
                step_id=0,
                input="What is the capital of France?",
                tool_call=None,
                tool_result=None,
                output="The capital of France is Paris.",
                timestamp="2026-05-18T10:00:00Z",
            )
        ],
        final_output="The capital of France is Paris.",
        metadata={"source": "fixture"},
    )


def make_single_step_tool_trace(trace_id: str = "t-002", success: bool = True) -> AgentTrace:
    """One-step trace with a tool call."""
    return AgentTrace(
        trace_id=trace_id,
        steps=[
            AgentStep(
                step_id=0,
                input="Search for recent LLM papers.",
                tool_call=ToolCall(
                    name="web_search",
                    args={"query": "recent LLM papers 2026"},
                    success=success,
                    error_type=None if success else "api_error",
                ),
                tool_result='["Paper A", "Paper B"]' if success else None,
                output="Found 2 recent papers." if success else "Search failed.",
                timestamp="2026-05-18T10:01:00Z",
                expected_tool="web_search",
                expected_args={"query": "recent LLM papers 2026"},
            )
        ],
        final_output="Found 2 recent papers." if success else "Search failed.",
        metadata={"source": "fixture"},
    )


# ---------------------------------------------------------------------------
# Multi-step traces
# ---------------------------------------------------------------------------


def make_multi_step_trace(
    trace_id: str = "t-003",
    num_steps: int = 4,
    all_success: bool = True,
) -> AgentTrace:
    """Multi-step trace with alternating search → retrieve → summarise steps."""
    steps = []
    tools = ["web_search", "document_retriever", "summarizer", "answer_generator"]
    for i in range(num_steps):
        tool_name = tools[i % len(tools)]
        success = all_success or (i != 1)  # step 1 fails when not all_success
        steps.append(
            AgentStep(
                step_id=i,
                input=f"Step {i} input: processing query about RAG systems.",
                tool_call=ToolCall(
                    name=tool_name,
                    args={"param": f"value_{i}"},
                    success=success,
                    error_type=None if success else "timeout",
                ),
                tool_result=f"Result from {tool_name}" if success else None,
                output=f"Step {i} reasoning: obtained data, proceeding.",
                timestamp=f"2026-05-18T10:0{i}:00Z",
                expected_tool=tool_name,
                expected_args={"param": f"value_{i}"},
            )
        )
    return AgentTrace(
        trace_id=trace_id,
        steps=steps,
        final_output="RAG systems combine retrieval with generation for accurate answers.",
        metadata={"source": "fixture", "num_steps": num_steps},
    )


def make_wrong_tool_trace(trace_id: str = "t-004") -> AgentTrace:
    """Trace where the agent picks the wrong tool (for tool selection accuracy tests)."""
    return AgentTrace(
        trace_id=trace_id,
        steps=[
            AgentStep(
                step_id=0,
                input="Retrieve the document about vector databases.",
                tool_call=ToolCall(
                    name="web_search",  # wrong: should have used document_retriever
                    args={"query": "vector databases"},
                    success=True,
                    error_type=None,
                ),
                tool_result="Some web results.",
                output="Found some results via web search.",
                timestamp="2026-05-18T10:05:00Z",
                expected_tool="document_retriever",  # ground truth
                expected_args={"doc_id": "vdb-overview"},
            )
        ],
        final_output="Found some results via web search.",
        metadata={"source": "fixture"},
    )


def make_schema_mismatch_trace(trace_id: str = "t-005") -> AgentTrace:
    """Trace with a schema_mismatch tool failure."""
    return AgentTrace(
        trace_id=trace_id,
        steps=[
            AgentStep(
                step_id=0,
                input="Call the payment API.",
                tool_call=ToolCall(
                    name="payment_api",
                    args={"amount": "not-a-number"},  # wrong type
                    success=False,
                    error_type="schema_mismatch",
                ),
                tool_result=None,
                output="Tool call failed due to invalid argument type.",
                timestamp="2026-05-18T10:06:00Z",
                expected_tool="payment_api",
                expected_args={"amount": 99.99},
            )
        ],
        final_output="Payment processing failed.",
        metadata={"source": "fixture"},
    )


def make_incoherent_trace(trace_id: str = "t-006") -> AgentTrace:
    """Multi-step trace where later steps don't build on earlier ones (low coherence)."""
    return AgentTrace(
        trace_id=trace_id,
        steps=[
            AgentStep(
                step_id=0,
                input="Find the population of Tokyo.",
                tool_call=ToolCall(
                    name="web_search",
                    args={"query": "population of Tokyo"},
                    success=True,
                    error_type=None,
                ),
                tool_result="Tokyo population: ~14 million (city), ~37 million (metro).",
                output="Retrieved Tokyo population data.",
                timestamp="2026-05-18T10:07:00Z",
                expected_tool="web_search",
            ),
            AgentStep(
                step_id=1,
                input="Next step.",
                tool_call=ToolCall(
                    name="calculator",
                    args={"expression": "2 + 2"},  # completely unrelated
                    success=True,
                    error_type=None,
                ),
                tool_result="4",
                output="Calculated 2+2=4.",  # ignores Tokyo data entirely
                timestamp="2026-05-18T10:08:00Z",
            ),
            AgentStep(
                step_id=2,
                input="Continue.",
                tool_call=None,
                tool_result=None,
                output="The answer is Paris.",  # hallucinated, ignores all prior steps
                timestamp="2026-05-18T10:09:00Z",
            ),
        ],
        final_output="The answer is Paris.",
        metadata={"source": "fixture"},
    )


def make_coherent_multi_step_trace(trace_id: str = "t-007") -> AgentTrace:
    """Well-structured 4-step trace with clear reasoning chain (high coherence)."""
    return AgentTrace(
        trace_id=trace_id,
        steps=[
            AgentStep(
                step_id=0,
                input="What companies were founded by Elon Musk?",
                tool_call=ToolCall(
                    name="web_search",
                    args={"query": "companies founded by Elon Musk"},
                    success=True,
                    error_type=None,
                ),
                tool_result="Tesla, SpaceX, Neuralink, The Boring Company, xAI.",
                output="Search returned a list of companies.",
                timestamp="2026-05-18T10:10:00Z",
                expected_tool="web_search",
            ),
            AgentStep(
                step_id=1,
                input="Retrieve details about SpaceX.",
                tool_call=ToolCall(
                    name="document_retriever",
                    args={"doc_id": "spacex-overview"},
                    success=True,
                    error_type=None,
                ),
                tool_result="SpaceX was founded in 2002 with the mission of reducing space travel costs.",
                output="Retrieved SpaceX founding details.",
                timestamp="2026-05-18T10:11:00Z",
                expected_tool="document_retriever",
            ),
            AgentStep(
                step_id=2,
                input="Retrieve details about Tesla.",
                tool_call=ToolCall(
                    name="document_retriever",
                    args={"doc_id": "tesla-overview"},
                    success=True,
                    error_type=None,
                ),
                tool_result="Tesla was co-founded by Elon Musk in 2004, focusing on EVs.",
                output="Retrieved Tesla founding details.",
                timestamp="2026-05-18T10:12:00Z",
                expected_tool="document_retriever",
            ),
            AgentStep(
                step_id=3,
                input="Summarise findings.",
                tool_call=ToolCall(
                    name="summarizer",
                    args={"text": "SpaceX (2002), Tesla (2004), Neuralink, Boring Company, xAI."},
                    success=True,
                    error_type=None,
                ),
                tool_result="Elon Musk founded SpaceX in 2002 and co-founded Tesla in 2004.",
                output="Summary complete.",
                timestamp="2026-05-18T10:13:00Z",
                expected_tool="summarizer",
            ),
        ],
        final_output=(
            "Elon Musk founded SpaceX (2002), co-founded Tesla (2004), "
            "and also started Neuralink, The Boring Company, and xAI."
        ),
        metadata={"source": "fixture"},
    )


def make_empty_tool_args_trace(trace_id: str = "t-008") -> AgentTrace:
    """Trace with a successful tool call that has no args (edge case)."""
    return AgentTrace(
        trace_id=trace_id,
        steps=[
            AgentStep(
                step_id=0,
                input="Get current timestamp.",
                tool_call=ToolCall(
                    name="get_timestamp",
                    args={},
                    success=True,
                    error_type=None,
                ),
                tool_result="2026-05-18T10:14:00Z",
                output="Current timestamp retrieved.",
                timestamp="2026-05-18T10:14:00Z",
                expected_tool="get_timestamp",
                expected_args={},
            )
        ],
        final_output="The current timestamp is 2026-05-18T10:14:00Z.",
        metadata={"source": "fixture"},
    )


# ---------------------------------------------------------------------------
# Convenience collection
# ---------------------------------------------------------------------------


def all_sample_traces() -> list[AgentTrace]:
    """Return all 8 fixture traces in a stable order."""
    return [
        make_single_step_trace("t-001"),
        make_single_step_tool_trace("t-002", success=True),
        make_multi_step_trace("t-003", num_steps=4, all_success=True),
        make_wrong_tool_trace("t-004"),
        make_schema_mismatch_trace("t-005"),
        make_incoherent_trace("t-006"),
        make_coherent_multi_step_trace("t-007"),
        make_empty_tool_args_trace("t-008"),
    ]
