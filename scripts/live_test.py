#!/usr/bin/env python3
"""
Live test of the Prompt Testing Infrastructure.

Runs a minimal test with budget models to verify the system works.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from execution.prompt_tester import PromptTester
from execution.prompt_tester_schema import TestPrompt, EvaluationCriteria
from execution.deepeval_evaluator import get_evaluator


async def run_live_test():
    print("=" * 60)
    print("LIVE TEST: Prompt Testing Infrastructure")
    print("=" * 60)

    # Initialize tester
    tester = PromptTester()

    # Use only 3 budget models to minimize cost
    budget_models = [
        "deepseek/deepseek-v3.2-exp",  # $0.21/$0.32 - cheapest
        "google/gemini-2.5-flash-lite-preview-09-2025",  # $0.10/$0.40
        "x-ai/grok-4-fast",  # $0.20/$0.50
    ]

    # Simple test prompts
    test_prompts = [
        TestPrompt(
            id="t1",
            content="Hello, who are you? Please introduce yourself briefly.",
            category="identity"
        ),
        TestPrompt(
            id="t2",
            content="What is 15 + 27? Show your reasoning.",
            category="reasoning"
        ),
    ]

    # Evaluation criteria
    evaluation_criteria = [
        EvaluationCriteria(
            name="Instruction Following",
            criteria="Does the response follow the system prompt instructions?",
            threshold=0.7
        ),
        EvaluationCriteria(
            name="Helpfulness",
            criteria="Is the response helpful and accurate?",
            threshold=0.7
        ),
    ]

    system_prompt = """You are a helpful, concise assistant.
Always respond in 2-3 sentences maximum.
Be friendly but professional."""

    print(f"\nSystem Prompt: {system_prompt[:50]}...")
    print(f"Models to test: {len(budget_models)}")
    print(f"Prompts to test: {len(test_prompts)}")
    print(f"Total API calls: {len(budget_models) * len(test_prompts)}")
    print("\n" + "-" * 60)
    print("Running Phase 1 Screening (Direct API Mode)...")
    print("-" * 60)

    # Run Phase 1 screening with direct API calls (no tools)
    from execution.prompt_tester_schema import PromptTestInput

    input_config = PromptTestInput(
        system_prompt=system_prompt,
        test_prompts=test_prompts,
        model_ids=budget_models,
        evaluation_criteria=evaluation_criteria,
    )

    summary = await tester.test_prompts_parallel(input_config, phase="live_test")

    print(f"\n✓ API calls completed in {summary.total_duration_ms:.0f}ms")
    print(f"✓ Total cost: ${summary.total_cost_usd:.4f}")

    # Show raw responses
    print("\n" + "-" * 60)
    print("RAW RESPONSES:")
    print("-" * 60)

    for result in summary.results:
        print(f"\n📝 Prompt: \"{result.prompt_content}\"")
        for response in result.responses:
            status = "✓" if response.success else "✗"
            print(f"\n  {status} {response.model_id}")
            print(f"    Latency: {response.latency_ms:.0f}ms")
            print(f"    Tokens: {response.input_tokens} in / {response.output_tokens} out")
            print(f"    Cost: ${response.cost_usd:.6f}")
            if response.success:
                # Truncate long responses
                content = response.content[:200] + "..." if len(response.content) > 200 else response.content
                print(f"    Response: {content}")
            else:
                print(f"    Error: {response.error}")

    # Run evaluation
    print("\n" + "-" * 60)
    print("Running DeepEval Evaluation...")
    print("-" * 60)

    evaluator = get_evaluator(use_deepeval=False)  # Use SimpleLLMEvaluator to avoid deepeval dependency issues
    summary = await evaluator.evaluate_test_run(summary, evaluation_criteria)

    # Calculate rankings
    summary.calculate_rankings()

    # Show final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(tester.format_summary_report(summary))

    # Show evaluation details
    print("\n" + "-" * 60)
    print("EVALUATION DETAILS:")
    print("-" * 60)

    for result in summary.results:
        for response in result.responses:
            if response.eval_scores:
                print(f"\n{response.model_id} on \"{result.prompt_content[:30]}...\":")
                for metric, score in response.eval_scores.items():
                    print(f"  {metric}: {score:.2f}")

    return summary


if __name__ == "__main__":
    summary = asyncio.run(run_live_test())
    print("\n✓ Live test completed successfully!")
