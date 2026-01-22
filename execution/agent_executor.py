"""
AgentExecutor for running multiple SubAgents in parallel.

Orchestrates the execution of sub-agents across different models,
collecting results and metrics for comparison.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

import yaml

from .sub_agent import SubAgent, AgentExecutionResult, load_tools_config
from .tool_executor import ToolExecutor
from .config_schema import ToolConfig


@dataclass
class TestPromptConfig:
    """Configuration for a test prompt."""
    id: str
    content: str
    category: str = "general"
    expected_behavior: Optional[str] = None


@dataclass
class AgentTestResult:
    """Results for a single prompt across multiple agents."""
    prompt_id: str
    prompt_content: str
    results: List[AgentExecutionResult] = field(default_factory=list)

    @property
    def successful_results(self) -> List[AgentExecutionResult]:
        return [r for r in self.results if r.success]

    @property
    def fastest_model(self) -> Optional[str]:
        successful = self.successful_results
        if not successful:
            return None
        return min(successful, key=lambda r: r.total_latency_ms).model_id

    @property
    def cheapest_model(self) -> Optional[str]:
        successful = self.successful_results
        if not successful:
            return None
        return min(successful, key=lambda r: r.cost_usd).model_id

    @property
    def fewest_tool_calls_model(self) -> Optional[str]:
        """Model that completed with fewest tool calls."""
        successful = self.successful_results
        if not successful:
            return None
        return min(successful, key=lambda r: r.num_tool_calls).model_id

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)


@dataclass
class AgentTestRunSummary:
    """Complete summary of a test run across multiple prompts and models."""
    run_id: str
    system_prompt: str
    phase: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: List[AgentTestResult] = field(default_factory=list)
    model_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.total_cost_usd for r in self.results)

    @property
    def total_duration_ms(self) -> Optional[float]:
        if not self.completed_at:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    @property
    def models_tested(self) -> List[str]:
        models = set()
        for result in self.results:
            for agent_result in result.results:
                models.add(agent_result.model_id)
        return sorted(models)

    def calculate_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate aggregate statistics per model."""
        model_data: Dict[str, List[AgentExecutionResult]] = {}

        for result in self.results:
            for agent_result in result.results:
                if agent_result.model_id not in model_data:
                    model_data[agent_result.model_id] = []
                model_data[agent_result.model_id].append(agent_result)

        stats = {}
        for model_id, results in model_data.items():
            successful = [r for r in results if r.success]

            stats[model_id] = {
                "total_runs": len(results),
                "successful_runs": len(successful),
                "success_rate": len(successful) / len(results) if results else 0,
                "avg_latency_ms": sum(r.total_latency_ms for r in successful) / len(successful) if successful else 0,
                "avg_llm_latency_ms": sum(r.llm_latency_ms for r in successful) / len(successful) if successful else 0,
                "avg_tool_latency_ms": sum(r.tool_latency_ms for r in successful) / len(successful) if successful else 0,
                "avg_tool_calls": sum(r.num_tool_calls for r in successful) / len(successful) if successful else 0,
                "total_tool_calls": sum(r.num_tool_calls for r in results),
                "avg_cost_usd": sum(r.cost_usd for r in successful) / len(successful) if successful else 0,
                "total_cost_usd": sum(r.cost_usd for r in results),
                "avg_input_tokens": sum(r.input_tokens for r in successful) / len(successful) if successful else 0,
                "avg_output_tokens": sum(r.output_tokens for r in successful) / len(successful) if successful else 0,
                "tools_used": list(set(
                    tool for r in results for tool in r.tools_used
                )),
            }

        self.model_stats = stats
        return stats

    def get_ranked_models(self, weight_quality: float = 0.4, weight_latency: float = 0.25,
                          weight_cost: float = 0.2, weight_reliability: float = 0.15) -> List[str]:
        """
        Rank models by composite score.

        Note: Quality score must be added via DeepEval evaluation before ranking is meaningful.
        Without evaluation, this ranks by latency/cost/reliability only.
        """
        if not self.model_stats:
            self.calculate_model_stats()

        # Find max values for normalization
        max_latency = max(s["avg_latency_ms"] for s in self.model_stats.values()) or 1
        max_cost = max(s["avg_cost_usd"] for s in self.model_stats.values()) or 1

        scored_models = []
        for model_id, stats in self.model_stats.items():
            # Normalize (higher is better)
            latency_score = 1 - (stats["avg_latency_ms"] / max_latency)
            cost_score = 1 - (stats["avg_cost_usd"] / max_cost)
            reliability_score = stats["success_rate"]

            # Quality score placeholder (0.5 until DeepEval adds scores)
            quality_score = stats.get("avg_quality_score", 0.5)

            composite = (
                quality_score * weight_quality +
                latency_score * weight_latency +
                cost_score * weight_cost +
                reliability_score * weight_reliability
            )

            scored_models.append((model_id, composite))

        scored_models.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in scored_models]


class AgentExecutor:
    """
    Orchestrates parallel execution of SubAgents across multiple models.
    """

    def __init__(
        self,
        models_config_path: str = "config/models.yaml",
        tools_config_path: str = "config/tools.yaml",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        max_iterations: int = 10,
        timeout: float = 120.0,
    ):
        """
        Initialize the AgentExecutor.

        Args:
            models_config_path: Path to models.yaml
            tools_config_path: Path to tools.yaml
            api_key: OpenRouter API key
            max_concurrent: Maximum concurrent agent executions
            max_iterations: Maximum iterations per agent
            timeout: Timeout for each agent execution
        """
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.max_iterations = max_iterations
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Load configurations
        self.models_config = self._load_models_config(models_config_path)
        self.models = {m["id"]: m for m in self.models_config.get("models", [])}
        self.tools_config = load_tools_config(tools_config_path)

        # Initialize tool executor
        tool_configs = [
            ToolConfig(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
                endpoint=t.endpoint,
            )
            for t in self.tools_config
        ]
        self.tool_executor = ToolExecutor(tool_configs)

    def _load_models_config(self, config_path: str) -> Dict[str, Any]:
        """Load models configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Models config not found: {config_path}")
        with open(path) as f:
            return yaml.safe_load(f)

    def get_all_model_ids(self) -> List[str]:
        """Return all configured model IDs."""
        return list(self.models.keys())

    def get_models_by_tier(self, tier: str) -> List[str]:
        """Return model IDs for a specific tier."""
        return [
            m["id"] for m in self.models_config.get("models", [])
            if m.get("tier") == tier
        ]

    def _create_sub_agent(self, model_id: str, system_prompt: str) -> SubAgent:
        """Create a SubAgent for a specific model."""
        model_config = self.models.get(model_id, {})
        pricing = model_config.get("pricing", {"input_per_1m": 0, "output_per_1m": 0})

        return SubAgent(
            model_id=model_id,
            system_prompt=system_prompt,
            tools_config=self.tools_config,
            tool_executor=self.tool_executor,
            model_pricing=pricing,
            api_key=self.api_key,
            max_iterations=self.max_iterations,
            timeout=self.timeout,
        )

    async def execute_single(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        prompt_id: str,
    ) -> AgentExecutionResult:
        """Execute a single prompt on a single model."""
        async with self._semaphore:
            agent = self._create_sub_agent(model_id, system_prompt)
            return await agent.execute(user_prompt, prompt_id)

    async def execute_parallel(
        self,
        system_prompt: str,
        test_prompts: List[TestPromptConfig],
        model_ids: Optional[List[str]] = None,
        phase: str = "testing",
    ) -> AgentTestRunSummary:
        """
        Execute test prompts across multiple models in parallel.

        Args:
            system_prompt: System prompt for all agents
            test_prompts: List of test prompts
            model_ids: Models to test (None = all models)
            phase: Phase name for tracking

        Returns:
            AgentTestRunSummary with all results
        """
        run_id = uuid.uuid4().hex[:8]
        started_at = datetime.now()
        model_ids = model_ids or self.get_all_model_ids()

        # Create all tasks
        tasks = []
        task_map = {}  # (prompt_id, model_id) -> task index

        for prompt in test_prompts:
            for model_id in model_ids:
                task = self.execute_single(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_prompt=prompt.content,
                    prompt_id=prompt.id,
                )
                task_map[(prompt.id, model_id)] = len(tasks)
                tasks.append(task)

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results by prompt
        results_by_prompt: Dict[str, AgentTestResult] = {}
        for prompt in test_prompts:
            results_by_prompt[prompt.id] = AgentTestResult(
                prompt_id=prompt.id,
                prompt_content=prompt.content,
            )

        for (prompt_id, model_id), idx in task_map.items():
            result = results[idx]
            if isinstance(result, Exception):
                # Create error result
                result = AgentExecutionResult(
                    model_id=model_id,
                    prompt_id=prompt_id,
                    session_id="error",
                    final_response="",
                    finish_reason="error",
                    total_latency_ms=0,
                    llm_latency_ms=0,
                    tool_latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0,
                    error=str(result),
                )
            results_by_prompt[prompt_id].results.append(result)

        summary = AgentTestRunSummary(
            run_id=run_id,
            system_prompt=system_prompt,
            phase=phase,
            started_at=started_at,
            completed_at=datetime.now(),
            results=list(results_by_prompt.values()),
        )
        summary.calculate_model_stats()

        return summary

    async def run_phase1_screening(
        self,
        system_prompt: str,
        test_prompts: List[TestPromptConfig],
        num_prompts: int = 3,
    ) -> AgentTestRunSummary:
        """
        Run Phase 1: Initial screening with first N prompts across ALL models.
        """
        screening_prompts = test_prompts[:num_prompts]
        return await self.execute_parallel(
            system_prompt=system_prompt,
            test_prompts=screening_prompts,
            model_ids=None,  # All models
            phase="screening",
        )

    async def run_phase3_deep_testing(
        self,
        system_prompt: str,
        test_prompts: List[TestPromptConfig],
        model_ids: List[str],
    ) -> AgentTestRunSummary:
        """
        Run Phase 3: Deep testing with ALL prompts on selected models.
        """
        return await self.execute_parallel(
            system_prompt=system_prompt,
            test_prompts=test_prompts,
            model_ids=model_ids,
            phase="deep_testing",
        )

    def format_results_table(self, summary: AgentTestRunSummary) -> str:
        """Format results as a markdown table."""
        lines = [
            f"# Agent Test Results",
            f"",
            f"**Run ID:** {summary.run_id}",
            f"**Phase:** {summary.phase}",
            f"**Duration:** {summary.total_duration_ms:.0f}ms" if summary.total_duration_ms else "",
            f"**Total Cost:** ${summary.total_cost_usd:.4f}",
            f"**Models Tested:** {len(summary.models_tested)}",
            f"**Prompts Tested:** {len(summary.results)}",
            f"",
            f"## Model Performance",
            f"",
            f"| Model | Success | Avg Latency | Avg Tool Calls | Avg Cost | Tools Used |",
            f"|-------|---------|-------------|----------------|----------|------------|",
        ]

        for model_id in sorted(summary.model_stats.keys()):
            stats = summary.model_stats[model_id]
            lines.append(
                f"| {model_id} | "
                f"{stats['success_rate']:.0%} | "
                f"{stats['avg_latency_ms']:.0f}ms | "
                f"{stats['avg_tool_calls']:.1f} | "
                f"${stats['avg_cost_usd']:.4f} | "
                f"{', '.join(stats['tools_used'][:3])}{'...' if len(stats['tools_used']) > 3 else ''} |"
            )

        lines.extend([
            f"",
            f"## Detailed Results",
            f"",
        ])

        for result in summary.results[:5]:  # First 5 prompts
            lines.append(f"### Prompt: {result.prompt_content[:50]}...")
            lines.append("")
            for agent_result in result.results:
                status = "OK" if agent_result.success else "ERR"
                lines.append(
                    f"- **{agent_result.model_id}** [{status}]: "
                    f"{agent_result.total_latency_ms:.0f}ms, "
                    f"{agent_result.num_tool_calls} tool calls, "
                    f"${agent_result.cost_usd:.4f}"
                )
                if agent_result.tool_calls:
                    tools_str = " → ".join(tc.tool_name for tc in agent_result.tool_calls[:3])
                    lines.append(f"  - Tools: {tools_str}")
            lines.append("")

        return "\n".join(lines)
