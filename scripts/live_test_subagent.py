#!/usr/bin/env python3
"""
Live test of Sub-Agent execution with real tools.

Runs sub-agents that can call RAG tools to answer questions.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from execution.prompt_tester import PromptTester
from execution.prompt_tester_schema import TestPrompt, EvaluationCriteria
from execution.deepeval_evaluator import get_evaluator


async def run_subagent_test():
    print("=" * 70)
    print("LIVE TEST: Sub-Agent Execution with Tools")
    print("=" * 70)

    # Initialize tester
    tester = PromptTester()

    # Use only 2 budget models to minimize cost
    budget_models = [
        "deepseek/deepseek-v3.2-exp",  # $0.21/$0.32 - cheapest, good at tool use
        "google/gemini-2.5-flash",      # $0.30/$2.50 - good balance
    ]

    # Test prompts that should trigger tool usage
    test_prompts = [
        TestPrompt(
            id="rag1",
            content="What AI projects or tools has Pierre built? Search the knowledge base.",
            category="rag_retrieval"
        ),
        TestPrompt(
            id="rag2",
            content="What do you know about RAG systems and vector databases?",
            category="knowledge_query"
        ),
    ]

    # Evaluation criteria including tool usage
    evaluation_criteria = [
        EvaluationCriteria(
            name="Instruction Following",
            criteria="Does the response follow the system prompt instructions?",
            threshold=0.7
        ),
        EvaluationCriteria(
            name="Tool Usage",
            criteria="Did the agent appropriately use tools to find information?",
            threshold=0.7
        ),
        EvaluationCriteria(
            name="Response Quality",
            criteria="Is the final response helpful, accurate, and well-sourced?",
            threshold=0.7
        ),
    ]

    system_prompt = """You are a RAG (Retrieval Augmented Generation) assistant with access to a knowledge base.

Your job is to answer questions by searching the knowledge base using the available tools.

Available tools:
- dynamic_hybrid_search: Search the knowledge base using hybrid search (semantic + keyword)
- query_knowledge_graph: Query entity relationships in the knowledge graph

Instructions:
1. When asked a question, use the appropriate search tool to find relevant information
2. Synthesize the search results into a clear, helpful answer
3. If the search returns no results, say so honestly
4. Keep responses concise (2-3 sentences for simple questions)"""

    print(f"\nSystem Prompt Preview:")
    print("-" * 40)
    print(system_prompt[:200] + "...")
    print("-" * 40)
    print(f"\nModels to test: {budget_models}")
    print(f"Prompts to test: {len(test_prompts)}")
    print(f"Total sub-agent executions: {len(budget_models) * len(test_prompts)}")
    print(f"\nTools available: dynamic_hybrid_search, query_knowledge_graph")

    print("\n" + "=" * 70)
    print("Running Sub-Agent Phase 1 Screening...")
    print("=" * 70)

    try:
        # Run Phase 1 screening with sub-agents
        summary = await tester.run_agent_phase1_screening(
            system_prompt=system_prompt,
            test_prompts=test_prompts,
            tools_config_path="config/tools.yaml",
            num_prompts=2,
            evaluation_criteria=evaluation_criteria,
            max_iterations=5,  # Limit iterations to control cost
        )

        print(f"\n✓ Sub-agent executions completed in {summary.total_duration_ms:.0f}ms")
        print(f"✓ Total cost: ${summary.total_cost_usd:.4f}")

        # Show detailed results
        print("\n" + "=" * 70)
        print("DETAILED RESULTS:")
        print("=" * 70)

        for result in summary.results:
            print(f"\n📝 Prompt: \"{result.prompt_content[:60]}...\"")
            print("-" * 60)

            for response in result.responses:
                status = "✓" if response.success else "✗"
                print(f"\n  {status} {response.model_id}")
                print(f"    Total Latency: {response.latency_ms:.0f}ms")
                print(f"    LLM Latency: {response.llm_latency_ms:.0f}ms")
                print(f"    Tool Latency: {response.tool_latency_ms:.0f}ms")
                print(f"    Tokens: {response.input_tokens} in / {response.output_tokens} out")
                print(f"    Cost: ${response.cost_usd:.6f}")
                print(f"    Tool Calls: {response.num_tool_calls}")
                print(f"    Tools Used: {response.tools_used or 'None'}")
                print(f"    Iterations: {response.num_iterations}")

                if response.tool_calls:
                    print(f"    Tool Call Details:")
                    for tc in response.tool_calls[:3]:  # Show first 3
                        tc_status = "✓" if tc.success else "✗"
                        print(f"      {tc_status} {tc.tool_name} ({tc.latency_ms:.0f}ms)")
                        # Show truncated params
                        params_str = str(tc.parameters)[:50]
                        print(f"        Params: {params_str}...")

                if response.success:
                    content = response.content[:300] + "..." if len(response.content) > 300 else response.content
                    print(f"    Response: {content}")
                else:
                    print(f"    Error: {response.error}")

        # Run evaluation
        print("\n" + "=" * 70)
        print("Running Evaluation...")
        print("=" * 70)

        evaluator = get_evaluator(use_deepeval=False)
        summary = await evaluator.evaluate_test_run(summary, evaluation_criteria)
        summary.calculate_rankings()

        # Show final report
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70)
        print(tester.format_summary_report(summary))

        # Show evaluation scores
        print("\n" + "-" * 70)
        print("EVALUATION SCORES:")
        print("-" * 70)

        for result in summary.results:
            for response in result.responses:
                if response.eval_scores:
                    print(f"\n{response.model_id}:")
                    print(f"  Prompt: \"{result.prompt_content[:40]}...\"")
                    for metric, score in response.eval_scores.items():
                        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                        print(f"  {metric}: [{bar}] {score:.2f}")

        return summary

    except Exception as e:
        print(f"\n✗ Error during sub-agent test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    summary = asyncio.run(run_subagent_test())
    if summary:
        print("\n" + "=" * 70)
        print("✓ Sub-agent live test completed successfully!")
        print("=" * 70)
    else:
        print("\n✗ Sub-agent test failed")
