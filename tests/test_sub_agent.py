"""
Unit tests for SubAgent and AgentExecutor classes.

Tests cover:
- SubAgent tool schema building
- Agent execution loop with tool calls
- AgentExecutor parallel execution
- Results conversion to standard format
"""

import pytest
import respx
import json
from httpx import Response
from datetime import datetime

from execution.sub_agent import SubAgent, AgentExecutionResult, ToolCall, load_tools_config
from execution.agent_executor import AgentExecutor, TestPromptConfig, AgentTestRunSummary
from execution.tool_executor import ToolExecutor
from execution.config_schema import ToolConfig, ToolParameter


# ============================================================================
# SubAgent Tests
# ============================================================================

class TestSubAgent:
    """Tests for SubAgent class."""

    @pytest.fixture
    def tools_config(self):
        """Create test tool configurations."""
        return [
            ToolConfig(
                name="test_search",
                description="Search for information",
                parameters={
                    "query": ToolParameter(type="string", required=True, source="model"),
                    "session_id": ToolParameter(type="string", required=True, source="sub_agent"),
                },
                endpoint={
                    "url": "https://api.test.com/search",
                    "method": "POST",
                    "body_template": '{"query": "{{query}}", "session_id": "{{session_id}}"}',
                },
            ),
            ToolConfig(
                name="get_data",
                description="Get data from database",
                parameters={
                    "id": ToolParameter(type="string", required=True, source="model"),
                },
                endpoint={
                    "url": "https://api.test.com/data",
                    "method": "GET",
                },
            ),
        ]

    @pytest.fixture
    def tool_executor(self, tools_config):
        """Create a tool executor with test config."""
        return ToolExecutor(tools_config)

    @pytest.fixture
    def sub_agent(self, tools_config, tool_executor):
        """Create a SubAgent for testing."""
        return SubAgent(
            model_id="test/model",
            system_prompt="You are a helpful assistant.",
            tools_config=tools_config,
            tool_executor=tool_executor,
            model_pricing={"input_per_1m": 1.0, "output_per_1m": 2.0},
            api_key="test-api-key",
            max_iterations=5,
            timeout=30.0,
        )

    def test_tools_schema_building(self, sub_agent):
        """Test that tools are correctly converted to OpenRouter format."""
        schema = sub_agent.tools_schema
        assert len(schema) == 2

        # Check first tool
        search_tool = next(t for t in schema if t["function"]["name"] == "test_search")
        assert search_tool["type"] == "function"
        assert "query" in search_tool["function"]["parameters"]["properties"]
        # session_id should NOT be in schema (source=sub_agent)
        assert "session_id" not in search_tool["function"]["parameters"]["properties"]

    def test_cost_calculation(self, sub_agent):
        """Test cost calculation."""
        cost = sub_agent.calculate_cost(1000, 500)
        expected = (1000 * 1.0 + 500 * 2.0) / 1_000_000
        assert cost == expected

    @pytest.mark.asyncio
    async def test_execute_no_tools(self, sub_agent):
        """Test execution without tool calls."""
        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "Hello! I'm here to help."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 10},
                    },
                )
            )

            result = await sub_agent.execute(
                user_prompt="Hello!",
                prompt_id="test_1",
            )

            assert result.success
            assert result.final_response == "Hello! I'm here to help."
            assert result.num_tool_calls == 0
            assert result.num_iterations == 1

    @pytest.mark.asyncio
    async def test_execute_with_tool_call(self, sub_agent):
        """Test execution with tool call."""
        call_count = [0]

        def make_response(request):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call - model wants to use a tool
                return Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "function": {
                                                "name": "test_search",
                                                "arguments": '{"query": "test query"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 15},
                    },
                )
            else:
                # Second call - model responds with tool result
                return Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "I found the answer: test result"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                    },
                )

        async with respx.mock(base_url="https://openrouter.ai") as openrouter_mock:
            openrouter_mock.post("/api/v1/chat/completions").mock(side_effect=make_response)

            # Mock the tool execution
            async with respx.mock(base_url="https://api.test.com") as tool_mock:
                tool_mock.post("/search").mock(
                    return_value=Response(200, json={"result": "test result"})
                )

                result = await sub_agent.execute(
                    user_prompt="Search for something",
                    prompt_id="test_2",
                )

                assert result.success
                assert "test result" in result.final_response
                assert result.num_tool_calls == 1
                assert result.tools_used == ["test_search"]
                assert result.num_iterations == 2

    @pytest.mark.asyncio
    async def test_execute_max_iterations(self, sub_agent):
        """Test that execution stops at max iterations."""
        # Always return tool calls to force max iterations
        def always_tool_call(request):
            return Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {
                                            "name": "test_search",
                                            "arguments": '{"query": "loop"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 10},
                },
            )

        async with respx.mock(base_url="https://openrouter.ai") as openrouter_mock:
            openrouter_mock.post("/api/v1/chat/completions").mock(side_effect=always_tool_call)

            async with respx.mock(base_url="https://api.test.com") as tool_mock:
                tool_mock.post("/search").mock(
                    return_value=Response(200, json={"result": "loop"})
                )

                result = await sub_agent.execute(
                    user_prompt="Keep looping",
                    prompt_id="test_3",
                )

                assert not result.success
                assert result.max_iterations_reached
                assert result.num_iterations == 5  # max_iterations


# ============================================================================
# AgentExecutor Tests
# ============================================================================

class TestAgentExecutor:
    """Tests for AgentExecutor class."""

    @pytest.fixture
    def executor_setup(self, tmp_path):
        """Set up executor with test configs."""
        # Create models config
        models_config = """
api:
  base_url: "https://openrouter.ai/api/v1"

models:
  - id: "test/model-a"
    name: "Test Model A"
    provider: test
    context_length: 4096
    pricing:
      input_per_1m: 1.00
      output_per_1m: 2.00
    tier: budget

  - id: "test/model-b"
    name: "Test Model B"
    provider: test
    context_length: 8192
    pricing:
      input_per_1m: 2.00
      output_per_1m: 4.00
    tier: balanced
"""
        models_file = tmp_path / "models.yaml"
        models_file.write_text(models_config)

        # Create tools config
        tools_config = """
tools:
  - name: test_tool
    description: A test tool
    parameters:
      query:
        type: string
        required: true
        source: model
    endpoint:
      url: "https://api.test.com/test"
      method: POST
      body_template: '{"query": "{{query}}"}'
"""
        tools_file = tmp_path / "tools.yaml"
        tools_file.write_text(tools_config)

        executor = AgentExecutor(
            models_config_path=str(models_file),
            tools_config_path=str(tools_file),
            api_key="test-api-key",
            max_concurrent=5,
        )

        return executor

    def test_get_all_model_ids(self, executor_setup):
        """Test getting all model IDs."""
        executor = executor_setup
        models = executor.get_all_model_ids()
        assert "test/model-a" in models
        assert "test/model-b" in models

    def test_get_models_by_tier(self, executor_setup):
        """Test getting models by tier."""
        executor = executor_setup
        budget = executor.get_models_by_tier("budget")
        assert "test/model-a" in budget
        assert "test/model-b" not in budget

    @pytest.mark.asyncio
    async def test_execute_single(self, executor_setup):
        """Test single agent execution."""
        executor = executor_setup

        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "Test response"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                    },
                )
            )

            result = await executor.execute_single(
                model_id="test/model-a",
                system_prompt="You are helpful.",
                user_prompt="Hello",
                prompt_id="test_1",
            )

            assert result.success
            assert result.final_response == "Test response"

    @pytest.mark.asyncio
    async def test_execute_parallel(self, executor_setup):
        """Test parallel agent execution."""
        executor = executor_setup

        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "Parallel response"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 25, "completion_tokens": 12},
                    },
                )
            )

            prompts = [
                TestPromptConfig(id="p1", content="Prompt 1"),
                TestPromptConfig(id="p2", content="Prompt 2"),
            ]

            summary = await executor.execute_parallel(
                system_prompt="You are helpful.",
                test_prompts=prompts,
                model_ids=["test/model-a", "test/model-b"],
            )

            assert len(summary.results) == 2
            # 2 prompts × 2 models = 4 results total
            total_results = sum(len(r.results) for r in summary.results)
            assert total_results == 4

            # Check model stats
            assert "test/model-a" in summary.model_stats
            assert "test/model-b" in summary.model_stats

    @pytest.mark.asyncio
    async def test_phase1_screening(self, executor_setup):
        """Test Phase 1 screening."""
        executor = executor_setup

        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "Screening response"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 15},
                    },
                )
            )

            prompts = [
                TestPromptConfig(id="p1", content="Prompt 1"),
                TestPromptConfig(id="p2", content="Prompt 2"),
                TestPromptConfig(id="p3", content="Prompt 3"),
                TestPromptConfig(id="p4", content="Prompt 4"),
            ]

            summary = await executor.run_phase1_screening(
                system_prompt="You are helpful.",
                test_prompts=prompts,
                num_prompts=2,  # Only first 2
            )

            assert summary.phase == "screening"
            assert len(summary.results) == 2  # Only 2 prompts used


# ============================================================================
# ToolCall Record Tests
# ============================================================================

class TestToolCallRecord:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test ToolCall creation."""
        tc = ToolCall(
            tool_name="test_tool",
            parameters={"query": "test"},
            result={"data": "result"},
            latency_ms=150.0,
            success=True,
        )
        assert tc.tool_name == "test_tool"
        assert tc.success
        assert tc.latency_ms == 150.0

    def test_tool_call_with_error(self):
        """Test ToolCall with error."""
        tc = ToolCall(
            tool_name="test_tool",
            parameters={"query": "test"},
            result=None,
            latency_ms=50.0,
            success=False,
            error="Connection failed",
        )
        assert not tc.success
        assert tc.error == "Connection failed"


# ============================================================================
# AgentExecutionResult Tests
# ============================================================================

class TestAgentExecutionResult:
    """Tests for AgentExecutionResult dataclass."""

    def test_success_property(self):
        """Test success property."""
        result = AgentExecutionResult(
            model_id="test/model",
            prompt_id="p1",
            session_id="sess1",
            final_response="Response",
            finish_reason="stop",
            total_latency_ms=500.0,
            llm_latency_ms=300.0,
            tool_latency_ms=200.0,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        assert result.success

        result_with_error = AgentExecutionResult(
            model_id="test/model",
            prompt_id="p1",
            session_id="sess1",
            final_response="",
            finish_reason="error",
            total_latency_ms=100.0,
            llm_latency_ms=100.0,
            tool_latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error="Timeout",
        )
        assert not result_with_error.success

    def test_to_dict(self):
        """Test serialization to dict."""
        result = AgentExecutionResult(
            model_id="test/model",
            prompt_id="p1",
            session_id="sess1",
            final_response="Test",
            finish_reason="stop",
            total_latency_ms=200.0,
            llm_latency_ms=150.0,
            tool_latency_ms=50.0,
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.0005,
            tool_calls=[
                ToolCall(
                    tool_name="tool1",
                    parameters={"a": "b"},
                    result="result",
                    latency_ms=50.0,
                    success=True,
                )
            ],
            num_tool_calls=1,
            tools_used=["tool1"],
        )

        data = result.to_dict()
        assert data["model_id"] == "test/model"
        assert data["num_tool_calls"] == 1
        assert len(data["tool_calls"]) == 1
