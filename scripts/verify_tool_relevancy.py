import asyncio
import os
from unittest.mock import AsyncMock, patch
from execution.prompt_tester_schema import ModelResponse, ToolCallRecord, EvaluationCriteria
from execution.deepeval_evaluator import SimpleLLMEvaluator

async def verify_relevancy():
    print("Verifying Tool Relevancy Logic...\n")

    # 1. Mock Judge Response
    # We mock the LLM call to return a specific formatted "Judge" response
    mock_judge_response = """
    SCORE: 0.9
    REASON: The tool call 'search_docs' with query 'metrics' is highly relevant because the user explicitly asked about metrics. The intent aligns with the goal.
    """

    with patch('execution.deepeval_evaluator.OpenRouterLLM') as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.a_generate = AsyncMock(return_value=mock_judge_response)
        
        # 2. Setup Evaluator
        evaluator = SimpleLLMEvaluator(api_key="mock_key")
        # Manually attach mock because SimpleLLMEvaluator creates its own instance
        evaluator.judge_model = mock_instance

        # 3. Create Test Data
        tool_call = ToolCallRecord(
            tool_name="search_docs",
            parameters={"query": "metrics"},
            result_summary="Found stuff",
            latency_ms=100,
            success=True,
            relevance_score=None # Should be updated
        )

        response = ModelResponse(
            model_id="test-model",
            prompt_id="test",
            content="Done.",
            finish_reason="stop",
            latency_ms=100,
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.0,
            tool_calls=[tool_call]
        )

        # 4. Run Evaluation
        print("Evaluating Model Response...")
        await evaluator.evaluate_model_response(
            response, 
            user_input="Tell me about metrics", 
            criteria_list=[] # We only care about tool eval here
        )

        # 5. Verify Results
        tc = response.tool_calls[0]
        print("-" * 50)
        print(f"Tool Name: {tc.tool_name}")
        print(f"Relevance Score: {tc.relevance_score}")
        print(f"Relevance Reason: {tc.relevance_reason}")
        
        if tc.relevance_score == 0.9:
            print("\n✅ SUCCESS: Metric captured correctly.")
        else:
            print(f"\n❌ FAILURE: Expected 0.9, got {tc.relevance_score}")

if __name__ == "__main__":
    asyncio.run(verify_relevancy())
