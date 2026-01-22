"""
DeepEval integration for objective quality scoring of model responses.

Uses G-Eval (LLM-as-a-judge) with a budget model via OpenRouter API to evaluate
responses from all other models, keeping evaluation cost low while maintaining
quality assessment.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any

import httpx
from dotenv import load_dotenv

from .prompt_tester_schema import (
    ModelResponse,
    TestRunSummary,
    EvaluationCriteria,
)

load_dotenv()


# Try to import DeepEval - gracefully handle if not installed
try:
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object  # Fallback for type hints


class OpenRouterLLM(DeepEvalBaseLLM if DEEPEVAL_AVAILABLE else object):
    """
    Custom LLM implementation that routes through OpenRouter API.

    Used as the judge model for DeepEval G-Eval evaluations.
    Defaults to claude-haiku-4.5 for cost efficiency (~$1/$5 per 1M tokens).
    """

    def __init__(
        self,
        model_id: str = "anthropic/claude-haiku-4.5",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the OpenRouter LLM wrapper.

        Args:
            model_id: The model to use for evaluation (default: claude-haiku-4.5)
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout
        self._base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model_id

    def load_model(self) -> "OpenRouterLLM":
        """Load model (no-op for API-based models)."""
        return self

    def generate(self, prompt: str) -> str:
        """
        Synchronously generate a response.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response text
        """
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "DSPy Prompt Optimizer - Evaluator",
            },
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic for evaluation
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response text
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "DSPy Prompt Optimizer - Evaluator",
                },
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class DeepEvalEvaluator:
    """
    Evaluator that uses DeepEval G-Eval metrics for objective quality scoring.

    Uses LLM-as-a-judge approach with custom evaluation criteria.
    """

    def __init__(
        self,
        judge_model_id: str = "anthropic/claude-haiku-4.5",
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize the DeepEval evaluator.

        Args:
            judge_model_id: Model to use as judge (default: claude-haiku-4.5)
            api_key: OpenRouter API key
            max_concurrent: Maximum concurrent evaluation calls
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "DeepEval is not installed. Run: pip install deepeval"
            )

        self.judge_model = OpenRouterLLM(model_id=judge_model_id, api_key=api_key)
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def create_metric(self, criteria: EvaluationCriteria) -> "GEval":
        """
        Create a G-Eval metric from evaluation criteria.

        Args:
            criteria: The evaluation criteria to use

        Returns:
            Configured GEval metric
        """
        return GEval(
            name=criteria.name,
            criteria=criteria.criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=self.judge_model,
            threshold=criteria.threshold,
        )

    def evaluate_response(
        self,
        user_input: str,
        model_output: str,
        criteria: EvaluationCriteria,
    ) -> tuple[float, str]:
        """
        Evaluate a single response against one criterion.

        Args:
            user_input: The original user prompt
            model_output: The model's response
            criteria: The evaluation criteria

        Returns:
            Tuple of (score, reason)
        """
        metric = self.create_metric(criteria)
        test_case = LLMTestCase(
            input=user_input,
            actual_output=model_output,
        )
        metric.measure(test_case)
        return metric.score, metric.reason

    async def evaluate_response_async(
        self,
        user_input: str,
        model_output: str,
        criteria: EvaluationCriteria,
    ) -> tuple[float, str]:
        """
        Asynchronously evaluate a single response against one criterion.

        Args:
            user_input: The original user prompt
            model_output: The model's response
            criteria: The evaluation criteria

        Returns:
            Tuple of (score, reason)
        """
        async with self._semaphore:
            # G-Eval doesn't have native async, so run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.evaluate_response,
                user_input,
                model_output,
                criteria,
            )

    async def evaluate_model_response(
        self,
        response: ModelResponse,
        user_input: str,
        criteria_list: List[EvaluationCriteria],
    ) -> ModelResponse:
        """
        Evaluate a model response against all criteria.

        Args:
            response: The ModelResponse to evaluate
            user_input: The original user prompt
            criteria_list: List of evaluation criteria

        Returns:
            ModelResponse with eval_scores and eval_reasons populated
        """
        if not response.success or not response.content:
            return response

        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}

        tasks = [
            self.evaluate_response_async(user_input, response.content, criteria)
            for criteria in criteria_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for criteria, result in zip(criteria_list, results):
            if isinstance(result, Exception):
                scores[criteria.name] = 0.0
                reasons[criteria.name] = f"Evaluation error: {result}"
            else:
                score, reason = result
                scores[criteria.name] = score if score is not None else 0.0
                reasons[criteria.name] = reason or ""

        response.eval_scores = scores
        response.eval_reasons = reasons
        return response

    async def evaluate_test_run(
        self,
        summary: TestRunSummary,
        criteria_list: Optional[List[EvaluationCriteria]] = None,
    ) -> TestRunSummary:
        """
        Evaluate all responses in a test run.

        Args:
            summary: The TestRunSummary to evaluate
            criteria_list: Optional custom criteria (defaults to summary's criteria)

        Returns:
            TestRunSummary with all responses evaluated
        """
        criteria = criteria_list or summary.evaluation_criteria
        if not criteria:
            # Default criteria
            criteria = [
                EvaluationCriteria(
                    name="Instruction Following",
                    criteria="How well does the response follow the system prompt instructions?",
                    threshold=0.7,
                ),
                EvaluationCriteria(
                    name="Response Quality",
                    criteria="Is the response helpful, accurate, and well-structured?",
                    threshold=0.7,
                ),
            ]

        tasks = []
        for result in summary.results:
            for response in result.responses:
                if response.success and response.content:
                    task = self.evaluate_model_response(
                        response=response,
                        user_input=result.prompt_content,
                        criteria_list=criteria,
                    )
                    tasks.append(task)

        await asyncio.gather(*tasks)

        # Recalculate rankings with evaluation scores
        summary.calculate_rankings()

        return summary


class SimpleLLMEvaluator:
    """
    Fallback evaluator that uses direct LLM calls without DeepEval dependency.

    Useful when DeepEval is not installed or for simpler evaluation needs.
    """

    def __init__(
        self,
        judge_model_id: str = "anthropic/claude-haiku-4.5",
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize the simple evaluator.

        Args:
            judge_model_id: Model to use as judge
            api_key: OpenRouter API key
            max_concurrent: Maximum concurrent evaluation calls
        """
        self.judge_model = OpenRouterLLM(model_id=judge_model_id, api_key=api_key)
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_response(
        self,
        user_input: str,
        model_output: str,
        criteria: EvaluationCriteria,
    ) -> tuple[float, str]:
        """
        Evaluate a response using direct LLM prompting.

        Args:
            user_input: The original user prompt
            model_output: The model's response
            criteria: The evaluation criteria

        Returns:
            Tuple of (score, reason)
        """
        prompt = f"""You are an expert evaluator. Evaluate the following response based on the given criteria.

CRITERIA: {criteria.name}
{criteria.criteria}

USER INPUT:
{user_input}

MODEL RESPONSE:
{model_output}

Provide your evaluation in the following format:
SCORE: [0.0 to 1.0]
REASON: [Brief explanation]

Be objective and strict in your evaluation."""

        async with self._semaphore:
            try:
                result = await self.judge_model.a_generate(prompt)

                # Parse score from response
                score = 0.5  # Default
                reason = result

                if "SCORE:" in result:
                    score_line = result.split("SCORE:")[1].split("\n")[0]
                    try:
                        score = float(score_line.strip())
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                    except ValueError:
                        pass

                if "REASON:" in result:
                    reason = result.split("REASON:")[1].strip()

                return score, reason
            except Exception as e:
                return 0.0, f"Evaluation error: {e}"

    async def evaluate_model_response(
        self,
        response: ModelResponse,
        user_input: str,
        criteria_list: List[EvaluationCriteria],
    ) -> ModelResponse:
        """
        Evaluate a model response against all criteria.

        Args:
            response: The ModelResponse to evaluate
            user_input: The original user prompt
            criteria_list: List of evaluation criteria

        Returns:
            ModelResponse with eval_scores and eval_reasons populated
        """
        if not response.success or not response.content:
            return response

        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}

        tasks = [
            self.evaluate_response(user_input, response.content, criteria)
            for criteria in criteria_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for criteria, result in zip(criteria_list, results):
            if isinstance(result, Exception):
                scores[criteria.name] = 0.0
                reasons[criteria.name] = f"Evaluation error: {result}"
            else:
                score, reason = result
                scores[criteria.name] = score
                reasons[criteria.name] = reason

        response.eval_scores = scores
        response.eval_reasons = reasons
        return response

    async def evaluate_test_run(
        self,
        summary: TestRunSummary,
        criteria_list: Optional[List[EvaluationCriteria]] = None,
    ) -> TestRunSummary:
        """
        Evaluate all responses in a test run.

        Args:
            summary: The TestRunSummary to evaluate
            criteria_list: Optional custom criteria (defaults to summary's criteria)

        Returns:
            TestRunSummary with all responses evaluated
        """
        criteria = criteria_list or summary.evaluation_criteria
        if not criteria:
            criteria = [
                EvaluationCriteria(
                    name="Instruction Following",
                    criteria="How well does the response follow the system prompt instructions?",
                    threshold=0.7,
                ),
                EvaluationCriteria(
                    name="Response Quality",
                    criteria="Is the response helpful, accurate, and well-structured?",
                    threshold=0.7,
                ),
            ]

        tasks = []
        for result in summary.results:
            for response in result.responses:
                if response.success and response.content:
                    task = self.evaluate_model_response(
                        response=response,
                        user_input=result.prompt_content,
                        criteria_list=criteria,
                    )
                    tasks.append(task)

        await asyncio.gather(*tasks)
        summary.calculate_rankings()

        return summary


def get_evaluator(
    use_deepeval: bool = True,
    judge_model_id: str = "anthropic/claude-haiku-4.5",
    api_key: Optional[str] = None,
    max_concurrent: int = 5,
):
    """
    Factory function to get the appropriate evaluator.

    Args:
        use_deepeval: Whether to use DeepEval (if available)
        judge_model_id: Model to use as judge
        api_key: OpenRouter API key
        max_concurrent: Maximum concurrent evaluation calls

    Returns:
        Either DeepEvalEvaluator or SimpleLLMEvaluator
    """
    if use_deepeval and DEEPEVAL_AVAILABLE:
        return DeepEvalEvaluator(
            judge_model_id=judge_model_id,
            api_key=api_key,
            max_concurrent=max_concurrent,
        )
    else:
        return SimpleLLMEvaluator(
            judge_model_id=judge_model_id,
            api_key=api_key,
            max_concurrent=max_concurrent,
        )
