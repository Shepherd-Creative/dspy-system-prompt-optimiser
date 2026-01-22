"""
Unit tests for the Prompt Testing Infrastructure.

Tests cover:
- Data models (schema)
- PromptTester class with mocked API responses
- Cost calculation
- Error handling
- SimpleLLMEvaluator (without DeepEval dependency)
"""

import pytest
import respx
from httpx import Response
from datetime import datetime
import asyncio

from execution.prompt_tester_schema import (
    TestPrompt,
    ModelResponse,
    PromptTestResult,
    TestRunSummary,
    PromptTestInput,
    EvaluationCriteria,
    ModelRanking,
)
from execution.prompt_tester import PromptTester
from execution.deepeval_evaluator import SimpleLLMEvaluator, OpenRouterLLM


# ============================================================================
# Schema Tests
# ============================================================================

class TestSchemaModels:
    """Tests for data model classes."""

    def test_test_prompt_creation(self):
        prompt = TestPrompt(id="test1", content="Hello")
        assert prompt.id == "test1"
        assert prompt.content == "Hello"
        assert prompt.category == "general"

    def test_test_prompt_auto_id(self):
        prompt = TestPrompt(id="", content="Hello")
        assert len(prompt.id) == 8  # Auto-generated UUID prefix

    def test_model_response_success(self):
        response = ModelResponse(
            model_id="test/model",
            prompt_id="p1",
            content="Hello!",
            finish_reason="stop",
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        )
        assert response.success is True
        assert response.total_tokens == 15

    def test_model_response_with_error(self):
        response = ModelResponse(
            model_id="test/model",
            prompt_id="p1",
            content="",
            finish_reason="error",
            latency_ms=50.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error="Connection failed",
        )
        assert response.success is False

    def test_prompt_test_result_aggregation(self):
        responses = [
            ModelResponse(
                model_id="model_a",
                prompt_id="p1",
                content="Fast",
                finish_reason="stop",
                latency_ms=50.0,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            ),
            ModelResponse(
                model_id="model_b",
                prompt_id="p1",
                content="Slow but cheap",
                finish_reason="stop",
                latency_ms=200.0,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.0005,
            ),
        ]
        result = PromptTestResult(prompt_id="p1", prompt_content="Test", responses=responses)

        assert result.fastest_model == "model_a"
        assert result.cheapest_model == "model_b"
        assert result.total_cost_usd == 0.0015

    def test_model_ranking_composite_score(self):
        ranking = ModelRanking(
            model_id="test/model",
            avg_quality_score=0.8,
            avg_latency_ms=1000.0,
            avg_cost_usd=0.001,
            success_rate=1.0,
            total_responses=5,
        )
        score = ranking.calculate_composite()
        assert 0.0 <= score <= 1.0
        assert ranking.composite_score == score

    def test_test_run_summary_rankings(self):
        summary = TestRunSummary(
            run_id="test123",
            system_prompt="You are helpful.",
            phase="screening",
            started_at=datetime.now(),
            results=[
                PromptTestResult(
                    prompt_id="p1",
                    prompt_content="Test",
                    responses=[
                        ModelResponse(
                            model_id="model_a",
                            prompt_id="p1",
                            content="Response A",
                            finish_reason="stop",
                            latency_ms=100.0,
                            input_tokens=10,
                            output_tokens=10,
                            cost_usd=0.001,
                            eval_scores={"quality": 0.9},
                        ),
                        ModelResponse(
                            model_id="model_b",
                            prompt_id="p1",
                            content="Response B",
                            finish_reason="stop",
                            latency_ms=200.0,
                            input_tokens=10,
                            output_tokens=10,
                            cost_usd=0.002,
                            eval_scores={"quality": 0.7},
                        ),
                    ],
                ),
            ],
        )
        rankings = summary.calculate_rankings()
        assert len(rankings) == 2
        # Model A should rank higher (better quality)
        assert rankings[0].model_id == "model_a"


# ============================================================================
# PromptTester Tests
# ============================================================================

class TestPromptTester:
    """Tests for PromptTester class with mocked API."""

    @pytest.fixture
    def tester(self, tmp_path):
        """Create a PromptTester with a test config."""
        config_content = """
api:
  base_url: "https://openrouter.ai/api/v1"
  env_key: "OPENROUTER_API_KEY"

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
      input_per_1m: 3.00
      output_per_1m: 6.00
    tier: balanced
"""
        config_file = tmp_path / "models.yaml"
        config_file.write_text(config_content)

        return PromptTester(
            models_config_path=str(config_file),
            api_key="test-api-key",
            timeout=30.0,
            max_concurrent=5,
        )

    def test_get_all_model_ids(self, tester):
        models = tester.get_all_model_ids()
        assert "test/model-a" in models
        assert "test/model-b" in models
        assert len(models) == 2

    def test_get_models_by_tier(self, tester):
        budget = tester.get_models_by_tier("budget")
        assert "test/model-a" in budget
        assert "test/model-b" not in budget

        balanced = tester.get_models_by_tier("balanced")
        assert "test/model-b" in balanced

    def test_cost_calculation(self, tester):
        # Model A: $1/1M input, $2/1M output
        cost = tester.calculate_cost("test/model-a", 1000, 500)
        expected = (1000 * 1.0 + 500 * 2.0) / 1_000_000
        assert cost == expected

    def test_cost_calculation_unknown_model(self, tester):
        cost = tester.calculate_cost("unknown/model", 1000, 500)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_single_prompt(self, tester):
        """Test single prompt execution with mocked API."""
        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "Hello! I'm a test model."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                    },
                )
            )

            prompt = TestPrompt(id="test1", content="Hello, who are you?")
            response = await tester.test_prompt(
                system_prompt="You are helpful.",
                test_prompt=prompt,
                model_id="test/model-a",
            )

            assert response.success
            assert response.content == "Hello! I'm a test model."
            assert response.finish_reason == "stop"
            assert response.input_tokens == 50
            assert response.output_tokens == 20
            assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_single_prompt_error(self, tester):
        """Test error handling for failed API request."""
        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(500, text="Internal Server Error")
            )

            prompt = TestPrompt(id="test1", content="Hello")
            response = await tester.test_prompt(
                system_prompt="You are helpful.",
                test_prompt=prompt,
                model_id="test/model-a",
            )

            assert not response.success
            assert "500" in response.error
            assert response.content == ""

    @pytest.mark.asyncio
    async def test_parallel_execution(self, tester):
        """Test parallel execution across multiple models."""
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
                        "usage": {"prompt_tokens": 30, "completion_tokens": 10},
                    },
                )
            )

            input_config = PromptTestInput(
                system_prompt="You are helpful.",
                test_prompts=[
                    TestPrompt(id="p1", content="Prompt 1"),
                    TestPrompt(id="p2", content="Prompt 2"),
                ],
                model_ids=["test/model-a", "test/model-b"],
            )

            summary = await tester.test_prompts_parallel(input_config)

            assert summary.run_id
            assert summary.phase == "testing"
            assert len(summary.results) == 2
            # 2 prompts × 2 models = 4 responses total
            total_responses = sum(len(r.responses) for r in summary.results)
            assert total_responses == 4

    @pytest.mark.asyncio
    async def test_phase1_screening(self, tester):
        """Test Phase 1 screening with subset of prompts."""
        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "Response"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                    },
                )
            )

            test_prompts = [
                TestPrompt(id="p1", content="Prompt 1"),
                TestPrompt(id="p2", content="Prompt 2"),
                TestPrompt(id="p3", content="Prompt 3"),
                TestPrompt(id="p4", content="Prompt 4"),
                TestPrompt(id="p5", content="Prompt 5"),
            ]

            summary = await tester.run_phase1_screening(
                system_prompt="You are helpful.",
                test_prompts=test_prompts,
                num_prompts=3,  # Only use first 3
            )

            assert summary.phase == "screening"
            assert len(summary.results) == 3  # Only 3 prompts used


# ============================================================================
# Evaluator Tests
# ============================================================================

class TestSimpleLLMEvaluator:
    """Tests for SimpleLLMEvaluator with mocked API."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with test API key."""
        return SimpleLLMEvaluator(
            judge_model_id="test/judge-model",
            api_key="test-api-key",
            max_concurrent=2,
        )

    @pytest.mark.asyncio
    async def test_evaluate_response(self, evaluator):
        """Test single response evaluation."""
        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": "SCORE: 0.85\nREASON: The response follows instructions well."
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )

            criteria = EvaluationCriteria(
                name="Test Criteria",
                criteria="Is this good?",
                threshold=0.7,
            )

            score, reason = await evaluator.evaluate_response(
                user_input="Hello",
                model_output="Hi there!",
                criteria=criteria,
            )

            assert score == 0.85
            assert "follows instructions" in reason

    @pytest.mark.asyncio
    async def test_evaluate_model_response(self, evaluator):
        """Test evaluating a ModelResponse object."""
        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": "SCORE: 0.9\nREASON: Excellent response."
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )

            response = ModelResponse(
                model_id="test/model",
                prompt_id="p1",
                content="This is my response.",
                finish_reason="stop",
                latency_ms=100.0,
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.001,
            )

            criteria_list = [
                EvaluationCriteria(name="Quality", criteria="Is it good?", threshold=0.7),
            ]

            evaluated = await evaluator.evaluate_model_response(
                response=response,
                user_input="What's up?",
                criteria_list=criteria_list,
            )

            assert evaluated.eval_scores is not None
            assert "Quality" in evaluated.eval_scores
            assert evaluated.eval_scores["Quality"] == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_failed_response(self, evaluator):
        """Test that failed responses are skipped."""
        response = ModelResponse(
            model_id="test/model",
            prompt_id="p1",
            content="",
            finish_reason="error",
            latency_ms=50.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error="Connection failed",
        )

        criteria_list = [
            EvaluationCriteria(name="Quality", criteria="Is it good?", threshold=0.7),
        ]

        evaluated = await evaluator.evaluate_model_response(
            response=response,
            user_input="Hello",
            criteria_list=criteria_list,
        )

        # Failed responses should not have eval scores
        assert evaluated.eval_scores is None


# ============================================================================
# OpenRouterLLM Tests
# ============================================================================

class TestOpenRouterLLM:
    """Tests for OpenRouterLLM wrapper."""

    def test_initialization(self):
        llm = OpenRouterLLM(model_id="test/model", api_key="test-key")
        assert llm.get_model_name() == "test/model"
        assert llm.load_model() is llm

    def test_initialization_no_key(self):
        # Clear env var if set
        import os
        original = os.environ.get("OPENROUTER_API_KEY")
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
            OpenRouterLLM(model_id="test/model")

        # Restore
        if original:
            os.environ["OPENROUTER_API_KEY"] = original

    @pytest.mark.asyncio
    async def test_async_generate(self):
        """Test async generation."""
        llm = OpenRouterLLM(model_id="test/model", api_key="test-key")

        async with respx.mock(base_url="https://openrouter.ai") as mock:
            mock.post("/api/v1/chat/completions").mock(
                return_value=Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "Generated text"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )

            result = await llm.a_generate("Test prompt")
            assert result == "Generated text"


# ============================================================================
# Integration Tests (still mocked, but tests full flow)
# ============================================================================

class TestIntegration:
    """Integration tests for the full workflow."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Set up tester and evaluator."""
        config_content = """
api:
  base_url: "https://openrouter.ai/api/v1"

models:
  - id: "test/fast-model"
    name: "Fast Model"
    provider: test
    context_length: 4096
    pricing:
      input_per_1m: 0.50
      output_per_1m: 1.00
    tier: budget

  - id: "test/quality-model"
    name: "Quality Model"
    provider: test
    context_length: 8192
    pricing:
      input_per_1m: 2.00
      output_per_1m: 4.00
    tier: balanced
"""
        config_file = tmp_path / "models.yaml"
        config_file.write_text(config_content)

        tester = PromptTester(
            models_config_path=str(config_file),
            api_key="test-api-key",
        )
        evaluator = SimpleLLMEvaluator(
            judge_model_id="test/judge",
            api_key="test-api-key",
        )
        return tester, evaluator

    @pytest.mark.asyncio
    async def test_full_screening_workflow(self, setup):
        """Test complete Phase 1 screening workflow."""
        tester, evaluator = setup

        async with respx.mock(base_url="https://openrouter.ai") as mock:
            # Mock model responses
            call_count = [0]

            def model_response_handler(request):
                call_count[0] += 1
                return Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": f"Response {call_count[0]}"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 15},
                    },
                )

            mock.post("/api/v1/chat/completions").mock(side_effect=model_response_handler)

            # Run screening
            test_prompts = [
                TestPrompt(id="t1", content="Hello"),
                TestPrompt(id="t2", content="What is 2+2?"),
            ]

            summary = await tester.run_phase1_screening(
                system_prompt="You are helpful.",
                test_prompts=test_prompts,
                num_prompts=2,
            )

            # Verify structure
            assert summary.phase == "screening"
            assert len(summary.results) == 2
            assert summary.total_cost_usd > 0

            # Evaluate (need more mocked calls)
            summary = await evaluator.evaluate_test_run(
                summary,
                criteria_list=[
                    EvaluationCriteria(
                        name="Quality",
                        criteria="Is it good?",
                        threshold=0.7,
                    )
                ],
            )

            # Rankings should be calculated
            assert len(summary.model_rankings) == 2
