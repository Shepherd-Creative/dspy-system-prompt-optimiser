"""
Data models for the Prompt Testing Infrastructure.

Defines Pydantic models for test inputs, model responses, and aggregated results
used throughout the 3-phase prompt optimization workflow.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class TestPrompt:
    """A single test prompt to evaluate against models."""
    id: str
    content: str
    category: str = "general"
    expected_behavior: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class EvaluationCriteria:
    """Custom evaluation criteria for G-Eval scoring."""
    name: str
    criteria: str
    threshold: float = 0.7


@dataclass
class ToolCallRecord:
    """Record of a single tool call made by an agent."""
    tool_name: str
    parameters: Dict[str, Any]
    result_summary: str  # Truncated result for storage
    latency_ms: float
    success: bool
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    relevance_score: Optional[float] = None
    relevance_reason: Optional[str] = None


@dataclass
class ModelResponse:
    """Response from a single model for a single prompt."""
    model_id: str
    prompt_id: str
    content: str
    finish_reason: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None
    # DeepEval scores (populated after evaluation)
    eval_scores: Optional[Dict[str, float]] = None
    eval_reasons: Optional[Dict[str, str]] = None
    # Tool call tracking (for agent-based execution)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    num_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    llm_latency_ms: float = 0.0  # Time spent on LLM calls only
    tool_latency_ms: float = 0.0  # Time spent on tool execution
    num_iterations: int = 1  # Agent loop iterations

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class PromptTestResult:
    """Aggregated results for a single prompt across all tested models."""
    prompt_id: str
    prompt_content: str
    responses: List[ModelResponse] = field(default_factory=list)

    @property
    def successful_responses(self) -> List[ModelResponse]:
        return [r for r in self.responses if r.success]

    @property
    def fastest_model(self) -> Optional[str]:
        successful = self.successful_responses
        if not successful:
            return None
        return min(successful, key=lambda r: r.latency_ms).model_id

    @property
    def cheapest_model(self) -> Optional[str]:
        successful = self.successful_responses
        if not successful:
            return None
        return min(successful, key=lambda r: r.cost_usd).model_id

    @property
    def highest_quality_model(self) -> Optional[str]:
        """Returns model with highest average eval score."""
        successful = [r for r in self.successful_responses if r.eval_scores]
        if not successful:
            return None

        def avg_score(r: ModelResponse) -> float:
            scores = r.eval_scores.values()
            return sum(scores) / len(scores) if scores else 0

        return max(successful, key=avg_score).model_id

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.responses)


@dataclass
class ModelRanking:
    """Ranking data for a single model."""
    model_id: str
    avg_quality_score: float
    avg_latency_ms: float
    avg_cost_usd: float
    success_rate: float
    total_responses: int
    # Tool-related metrics
    avg_tool_calls: float = 0.0
    total_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    avg_llm_latency_ms: float = 0.0
    avg_tool_latency_ms: float = 0.0
    # Weighted composite score (quality 40%, latency 25%, cost 20%, reliability 15%)
    composite_score: float = 0.0

    def calculate_composite(
        self,
        quality_weight: float = 0.4,
        latency_weight: float = 0.25,
        cost_weight: float = 0.2,
        reliability_weight: float = 0.15,
        max_latency_ms: float = 30000.0,
        max_cost_usd: float = 0.01
    ) -> float:
        """Calculate weighted composite score (higher is better)."""
        # Normalize scores (all should be 0-1, higher is better)
        quality_norm = self.avg_quality_score  # Already 0-1
        latency_norm = 1.0 - min(self.avg_latency_ms / max_latency_ms, 1.0)
        cost_norm = 1.0 - min(self.avg_cost_usd / max_cost_usd, 1.0)
        reliability_norm = self.success_rate

        self.composite_score = (
            quality_norm * quality_weight +
            latency_norm * latency_weight +
            cost_norm * cost_weight +
            reliability_norm * reliability_weight
        )
        return self.composite_score


@dataclass
class TestRunSummary:
    """Complete summary of a test run across multiple prompts and models."""
    run_id: str
    system_prompt: str
    phase: str  # "screening", "deep_testing"
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: List[PromptTestResult] = field(default_factory=list)
    model_rankings: List[ModelRanking] = field(default_factory=list)
    evaluation_criteria: List[EvaluationCriteria] = field(default_factory=list)

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
            for response in result.responses:
                models.add(response.model_id)
        return sorted(models)

    @property
    def prompts_tested(self) -> int:
        return len(self.results)

    def get_top_models(self, n: int = 4) -> List[str]:
        """Return top N models by composite score."""
        if not self.model_rankings:
            return []
        sorted_rankings = sorted(
            self.model_rankings,
            key=lambda r: r.composite_score,
            reverse=True
        )
        return [r.model_id for r in sorted_rankings[:n]]

    def calculate_rankings(self) -> List[ModelRanking]:
        """Aggregate responses by model and calculate rankings."""
        # Group responses by model
        model_responses: Dict[str, List[ModelResponse]] = {}
        for result in self.results:
            for response in result.responses:
                if response.model_id not in model_responses:
                    model_responses[response.model_id] = []
                model_responses[response.model_id].append(response)

        rankings = []
        for model_id, responses in model_responses.items():
            successful = [r for r in responses if r.success]

            # Calculate average quality score
            quality_scores = []
            for r in successful:
                if r.eval_scores:
                    avg = sum(r.eval_scores.values()) / len(r.eval_scores)
                    quality_scores.append(avg)
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

            # Calculate other averages
            avg_latency = sum(r.latency_ms for r in successful) / len(successful) if successful else 0.0
            avg_cost = sum(r.cost_usd for r in successful) / len(successful) if successful else 0.0
            success_rate = len(successful) / len(responses) if responses else 0.0

            # Calculate tool-related metrics
            avg_tool_calls = sum(r.num_tool_calls for r in successful) / len(successful) if successful else 0.0
            total_tool_calls = sum(r.num_tool_calls for r in responses)
            tools_used = list(set(tool for r in responses for tool in r.tools_used))
            avg_llm_latency = sum(r.llm_latency_ms for r in successful) / len(successful) if successful else 0.0
            avg_tool_latency = sum(r.tool_latency_ms for r in successful) / len(successful) if successful else 0.0

            ranking = ModelRanking(
                model_id=model_id,
                avg_quality_score=avg_quality,
                avg_latency_ms=avg_latency,
                avg_cost_usd=avg_cost,
                success_rate=success_rate,
                total_responses=len(responses),
                avg_tool_calls=avg_tool_calls,
                total_tool_calls=total_tool_calls,
                tools_used=tools_used,
                avg_llm_latency_ms=avg_llm_latency,
                avg_tool_latency_ms=avg_tool_latency,
            )
            ranking.calculate_composite()
            rankings.append(ranking)

        self.model_rankings = sorted(rankings, key=lambda r: r.composite_score, reverse=True)
        return self.model_rankings


@dataclass
class PromptTestInput:
    """Input configuration for a prompt test run."""
    system_prompt: str
    test_prompts: List[TestPrompt]
    model_ids: Optional[List[str]] = None  # None = use all models from config
    evaluation_criteria: List[EvaluationCriteria] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.7

    def __post_init__(self):
        # Add default evaluation criteria if none provided
        if not self.evaluation_criteria:
            self.evaluation_criteria = [
                EvaluationCriteria(
                    name="Instruction Following",
                    criteria="How well does the response follow the system prompt instructions and constraints?",
                    threshold=0.7
                ),
                EvaluationCriteria(
                    name="Response Quality",
                    criteria="Is the response helpful, accurate, well-structured, and appropriate for the task?",
                    threshold=0.7
                )
            ]
