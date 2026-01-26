"""
Core PromptTester class for testing system prompts across multiple LLM models.

Uses OpenRouter API for unified access to models from Anthropic, Google, xAI, DeepSeek.
Supports parallel execution with configurable concurrency limits.
"""

import asyncio
import os
import time
import uuid
import yaml
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .prompt_tester_schema import (
    TestPrompt,
    ModelResponse,
    PromptTestResult,
    TestRunSummary,
    PromptTestInput,
    EvaluationCriteria,
    ToolCallRecord,
)
from .agent_executor import AgentExecutor, TestPromptConfig, AgentTestRunSummary
from .sub_agent import AgentExecutionResult

load_dotenv()


class PromptTester:
    """
    Test system prompts across multiple LLM models via OpenRouter API.

    Supports:
    - Parallel execution with configurable concurrency
    - Phase 1 screening (subset of prompts, all models)
    - Phase 3 deep testing (all prompts, selected models)
    - Cost calculation based on model pricing
    - Integration with DeepEval for quality scoring
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        models_config_path: str = "config/models.yaml",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_concurrent: int = 10,
    ):
        """
        Initialize the PromptTester.

        Args:
            models_config_path: Path to models.yaml configuration file
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent API requests
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment or parameters")

        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Load models configuration
        self.models_config = self._load_models_config(models_config_path)
        self.models = {m["id"]: m for m in self.models_config.get("models", [])}

    def _load_models_config(self, config_path: str) -> Dict[str, Any]:
        """Load models configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Models config not found: {config_path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def get_all_model_ids(self) -> List[str]:
        """Return list of all configured model IDs."""
        return list(self.models.keys())

    def get_models_by_tier(self, tier: str) -> List[str]:
        """Return model IDs for a specific tier (budget, balanced, premium)."""
        return [
            m["id"] for m in self.models_config.get("models", [])
            if m.get("tier") == tier
        ]

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost in USD for a model request.

        Args:
            model_id: The model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        model = self.models.get(model_id)
        if not model:
            return 0.0

        pricing = model.get("pricing", {})
        input_rate = pricing.get("input_per_1m", 0.0)
        output_rate = pricing.get("output_per_1m", 0.0)

        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        return cost

    async def test_prompt(
        self,
        system_prompt: str,
        test_prompt: TestPrompt,
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """
        Test a single prompt against a single model.

        Args:
            system_prompt: The system prompt to test
            test_prompt: The test prompt to send
            model_id: The model to test
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            ModelResponse with content, metrics, and any errors
        """
        start_time = time.time()

        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.OPENROUTER_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "X-Title": "Prompt Optimser Agent",
                        },
                        json={
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": test_prompt.content},
                            ],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )

                    latency_ms = (time.time() - start_time) * 1000

                    if response.status_code != 200:
                        error_detail = response.text
                        return ModelResponse(
                            model_id=model_id,
                            prompt_id=test_prompt.id,
                            content="",
                            finish_reason="error",
                            latency_ms=latency_ms,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=0.0,
                            error=f"HTTP {response.status_code}: {error_detail}",
                        )

                    data = response.json()
                    choices = data.get("choices", [])
                    usage = data.get("usage", {})

                    if not choices:
                        return ModelResponse(
                            model_id=model_id,
                            prompt_id=test_prompt.id,
                            content="",
                            finish_reason="error",
                            latency_ms=latency_ms,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=0.0,
                            error="No choices in response",
                        )

                    content = choices[0].get("message", {}).get("content", "")
                    finish_reason = choices[0].get("finish_reason", "unknown")
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

                    cost = self.calculate_cost(model_id, input_tokens, output_tokens)

                    return ModelResponse(
                        model_id=model_id,
                        prompt_id=test_prompt.id,
                        content=content,
                        finish_reason=finish_reason,
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost,
                    )

            except httpx.TimeoutException:
                latency_ms = (time.time() - start_time) * 1000
                return ModelResponse(
                    model_id=model_id,
                    prompt_id=test_prompt.id,
                    content="",
                    finish_reason="error",
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    error=f"Request timed out after {self.timeout}s",
                )
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                return ModelResponse(
                    model_id=model_id,
                    prompt_id=test_prompt.id,
                    content="",
                    finish_reason="error",
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    error=str(e),
                )

    async def test_prompts_parallel(
        self,
        input_config: PromptTestInput,
        phase: str = "testing",
    ) -> TestRunSummary:
        """
        Run all test prompts against all specified models in parallel.

        Args:
            input_config: PromptTestInput with system prompt, test prompts, and optional model list
            phase: Phase name for the test run

        Returns:
            TestRunSummary with all results aggregated
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.now()

        model_ids = input_config.model_ids or self.get_all_model_ids()

        # Create all tasks
        tasks = []
        task_map = {}  # (prompt_id, model_id) -> task index

        for prompt in input_config.test_prompts:
            for model_id in model_ids:
                task = self.test_prompt(
                    system_prompt=input_config.system_prompt,
                    test_prompt=prompt,
                    model_id=model_id,
                    max_tokens=input_config.max_tokens,
                    temperature=input_config.temperature,
                )
                task_map[(prompt.id, model_id)] = len(tasks)
                tasks.append(task)

        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results by prompt
        results_by_prompt: Dict[str, PromptTestResult] = {}
        for prompt in input_config.test_prompts:
            results_by_prompt[prompt.id] = PromptTestResult(
                prompt_id=prompt.id,
                prompt_content=prompt.content,
            )

        for (prompt_id, model_id), idx in task_map.items():
            response = responses[idx]
            if isinstance(response, Exception):
                response = ModelResponse(
                    model_id=model_id,
                    prompt_id=prompt_id,
                    content="",
                    finish_reason="error",
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    error=str(response),
                )
            results_by_prompt[prompt_id].responses.append(response)

        summary = TestRunSummary(
            run_id=run_id,
            system_prompt=input_config.system_prompt,
            phase=phase,
            started_at=started_at,
            completed_at=datetime.now(),
            results=list(results_by_prompt.values()),
            evaluation_criteria=input_config.evaluation_criteria,
        )

        return summary

    async def run_phase1_screening(
        self,
        system_prompt: str,
        test_prompts: List[TestPrompt],
        num_prompts: int = 3,
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None,
    ) -> TestRunSummary:
        """
        Run Phase 1: Initial screening with first N prompts across ALL models.

        Args:
            system_prompt: The system prompt to test
            test_prompts: Full list of test prompts
            num_prompts: Number of prompts to use for screening (default: 3)
            evaluation_criteria: Optional custom evaluation criteria

        Returns:
            TestRunSummary with screening results
        """
        screening_prompts = test_prompts[:num_prompts]

        input_config = PromptTestInput(
            system_prompt=system_prompt,
            test_prompts=screening_prompts,
            model_ids=None,  # All models
            evaluation_criteria=evaluation_criteria or [],
        )

        summary = await self.test_prompts_parallel(input_config, phase="screening")
        summary.calculate_rankings()

        return summary

    async def run_phase3_deep_testing(
        self,
        system_prompt: str,
        test_prompts: List[TestPrompt],
        model_ids: List[str],
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None,
    ) -> TestRunSummary:
        """
        Run Phase 3: Deep testing with ALL prompts on selected models.

        Args:
            system_prompt: The system prompt to test
            test_prompts: Full list of test prompts
            model_ids: Selected model IDs from Phase 2
            evaluation_criteria: Optional custom evaluation criteria

        Returns:
            TestRunSummary with deep testing results
        """
        input_config = PromptTestInput(
            system_prompt=system_prompt,
            test_prompts=test_prompts,
            model_ids=model_ids,
            evaluation_criteria=evaluation_criteria or [],
        )

        summary = await self.test_prompts_parallel(input_config, phase="deep_testing")
        summary.calculate_rankings()

        return summary

    def format_summary_report(self, summary: TestRunSummary) -> str:
        """
        Format a TestRunSummary as a readable report.

        Args:
            summary: The test run summary to format

        Returns:
            Formatted string report
        """
        lines = [
            f"# Prompt Test Report",
            f"",
            f"**Run ID:** {summary.run_id}",
            f"**Phase:** {summary.phase}",
            f"**Started:** {summary.started_at.isoformat()}",
            f"**Duration:** {summary.total_duration_ms:.0f}ms" if summary.total_duration_ms else "",
            f"**Total Cost:** ${summary.total_cost_usd:.4f}",
            f"**Models Tested:** {len(summary.models_tested)}",
            f"**Prompts Tested:** {summary.prompts_tested}",
            f"",
            f"## Model Rankings",
            f"",
        ]

        if summary.model_rankings:
            lines.append("| Rank | Model | Quality | Latency | Tool Calls | Cost | Reliability | Composite |")
            lines.append("|------|-------|---------|---------|------------|------|-------------|-----------|")
            for i, ranking in enumerate(summary.model_rankings[:10], 1):
                lines.append(
                    f"| {i} | {ranking.model_id} | "
                    f"{ranking.avg_quality_score:.2f} | "
                    f"{ranking.avg_latency_ms:.0f}ms | "
                    f"{ranking.avg_tool_calls:.1f} | "
                    f"${ranking.avg_cost_usd:.4f} | "
                    f"{ranking.success_rate:.0%} | "
                    f"{ranking.composite_score:.3f} |"
                )
        else:
            lines.append("_No rankings available (run evaluation first)_")

        lines.extend([
            f"",
            f"## Top Models",
            f"",
        ])

        top_models = summary.get_top_models(4)
        if top_models:
            for i, model_id in enumerate(top_models, 1):
                lines.append(f"{i}. {model_id}")
        else:
            lines.append("_No top models identified_")

        return "\n".join(lines)

    # =========================================================================
    # Agent-Based Testing (with tool support)
    # =========================================================================

    def _convert_agent_result_to_model_response(
        self,
        agent_result: AgentExecutionResult,
    ) -> ModelResponse:
        """Convert AgentExecutionResult to ModelResponse for evaluation compatibility."""
        tool_call_records = [
            ToolCallRecord(
                tool_name=tc.tool_name,
                parameters=tc.parameters,
                result_summary=str(tc.result)[:500],
                latency_ms=tc.latency_ms,
                success=tc.success,
                error=tc.error,
            )
            for tc in agent_result.tool_calls
        ]

        return ModelResponse(
            model_id=agent_result.model_id,
            prompt_id=agent_result.prompt_id,
            content=agent_result.final_response,
            finish_reason=agent_result.finish_reason,
            latency_ms=agent_result.total_latency_ms,
            input_tokens=agent_result.input_tokens,
            output_tokens=agent_result.output_tokens,
            cost_usd=agent_result.cost_usd,
            error=agent_result.error,
            tool_calls=tool_call_records,
            num_tool_calls=agent_result.num_tool_calls,
            tools_used=agent_result.tools_used,
            llm_latency_ms=agent_result.llm_latency_ms,
            tool_latency_ms=agent_result.tool_latency_ms,
            num_iterations=agent_result.num_iterations,
        )

    def _convert_agent_summary_to_test_summary(
        self,
        agent_summary: AgentTestRunSummary,
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None,
    ) -> TestRunSummary:
        """Convert AgentTestRunSummary to TestRunSummary for evaluation compatibility."""
        results = []
        for agent_result in agent_summary.results:
            prompt_result = PromptTestResult(
                prompt_id=agent_result.prompt_id,
                prompt_content=agent_result.prompt_content,
                responses=[
                    self._convert_agent_result_to_model_response(r)
                    for r in agent_result.results
                ],
            )
            results.append(prompt_result)

        return TestRunSummary(
            run_id=agent_summary.run_id,
            system_prompt=agent_summary.system_prompt,
            phase=agent_summary.phase,
            started_at=agent_summary.started_at,
            completed_at=agent_summary.completed_at,
            results=results,
            evaluation_criteria=evaluation_criteria or [],
        )

    async def run_agent_phase1_screening(
        self,
        system_prompt: str,
        test_prompts: List[TestPrompt],
        tools_config_path: str = "config/tools.yaml",
        num_prompts: int = 3,
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None,
        max_iterations: int = 10,
    ) -> TestRunSummary:
        """
        Run Phase 1 screening using sub-agents with tool support.

        Each model runs as a full agent that can call tools defined in tools.yaml.

        Args:
            system_prompt: The system prompt to test
            test_prompts: Full list of test prompts
            tools_config_path: Path to tools.yaml
            num_prompts: Number of prompts to use for screening
            evaluation_criteria: Optional custom evaluation criteria
            max_iterations: Maximum agent loop iterations per execution

        Returns:
            TestRunSummary with screening results including tool call metrics
        """
        # Convert TestPrompt to TestPromptConfig
        agent_prompts = [
            TestPromptConfig(
                id=p.id,
                content=p.content,
                category=p.category,
                expected_behavior=p.expected_behavior,
            )
            for p in test_prompts[:num_prompts]
        ]

        # Create executor with tools
        executor = AgentExecutor(
            models_config_path="config/models.yaml",
            tools_config_path=tools_config_path,
            api_key=self.api_key,
            max_concurrent=self.max_concurrent,
            max_iterations=max_iterations,
            timeout=self.timeout,
        )

        # Run screening
        agent_summary = await executor.run_phase1_screening(
            system_prompt=system_prompt,
            test_prompts=agent_prompts,
            num_prompts=num_prompts,
        )

        # Convert to standard format
        summary = self._convert_agent_summary_to_test_summary(
            agent_summary, evaluation_criteria
        )
        summary.calculate_rankings()

        return summary

    async def run_agent_phase3_deep_testing(
        self,
        system_prompt: str,
        test_prompts: List[TestPrompt],
        model_ids: List[str],
        tools_config_path: str = "config/tools.yaml",
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None,
        max_iterations: int = 10,
    ) -> TestRunSummary:
        """
        Run Phase 3 deep testing using sub-agents with tool support.

        Args:
            system_prompt: The system prompt to test
            test_prompts: Full list of test prompts
            model_ids: Selected model IDs from Phase 2
            tools_config_path: Path to tools.yaml
            evaluation_criteria: Optional custom evaluation criteria
            max_iterations: Maximum agent loop iterations per execution

        Returns:
            TestRunSummary with deep testing results including tool call metrics
        """
        # Convert TestPrompt to TestPromptConfig
        agent_prompts = [
            TestPromptConfig(
                id=p.id,
                content=p.content,
                category=p.category,
                expected_behavior=p.expected_behavior,
            )
            for p in test_prompts
        ]

        # Create executor with tools
        executor = AgentExecutor(
            models_config_path="config/models.yaml",
            tools_config_path=tools_config_path,
            api_key=self.api_key,
            max_concurrent=self.max_concurrent,
            max_iterations=max_iterations,
            timeout=self.timeout,
        )

        # Run deep testing
        agent_summary = await executor.run_phase3_deep_testing(
            system_prompt=system_prompt,
            test_prompts=agent_prompts,
            model_ids=model_ids,
        )

        # Convert to standard format
        summary = self._convert_agent_summary_to_test_summary(
            agent_summary, evaluation_criteria
        )
        summary.calculate_rankings()

        return summary
