import asyncio
import uuid
from datetime import datetime
from execution.prompt_tester_schema import ToolCallRecord, ModelResponse, PromptTestResult, TestRunSummary, ModelRanking
from execution.sub_agent import AgentExecutionResult

async def verify_metrics():
    print("Verifying Metric Capture Capabilities...\n")

    # 1. Simulate a Tool Call Record (now with tokens)
    tool_call = ToolCallRecord(
        tool_name="dynamic_hybrid_search",
        parameters={"query": "test query"},
        result_summary="Found 3 documents...",
        latency_ms=150.0,
        success=True,
        input_tokens=150, # Simulated result size (e.g. 600 chars)
        output_tokens=45, # Simulated cost of generating this tool call
        relevance_score=0.9,
        relevance_reason="Tool was highly relevant to the user query about documents."
    )

    # 2. Simulate a Model Response (Agent Execution Result)
    response = ModelResponse(
        model_id="test-model",
        prompt_id="test-prompt-1",
        content="Here are the documents you asked for...",
        finish_reason="stop",
        latency_ms=1200.0,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0001,
        tool_calls=[tool_call],
        num_tool_calls=1,
        tools_used=["dynamic_hybrid_search"],
        llm_latency_ms=1000.0,
        tool_latency_ms=150.0,
        num_iterations=1,
        eval_scores={"Helpfulness": 0.95},
        eval_reasons={"Helpfulness": "Response was accurate."}
    )

    # 3. Aggregate into Result
    result = PromptTestResult(
        prompt_id="test-prompt-1",
        prompt_content="Find documents about X",
        responses=[response]
    )

    # 4. Verify Metrics
    print("MATCHING METRICS TO REQUIREMENTS:")
    print("-" * 50)
    
    print(f"1. Quality of response: {response.eval_scores} (Captured)")
    print(f"2. Time taken to completion: {response.latency_ms}ms (Captured)")
    print(f"3. Total Tokens used: {response.total_tokens} (Captured)")
    print(f"4. Tools called: {response.tools_used} (Captured)")
    
    # 5. Relevancy
    print(f"5. Relevancy of tool call: Score={tool_call.relevance_score}, Reason='{tool_call.relevance_reason}' (Captured in schema, needs Eval Impl)")
    
    print(f"6. Time taken per tool call: {tool_call.latency_ms}ms (Captured)")
    print(f"7. Tokens used per tool call: Output={tool_call.output_tokens} (Generation), Input={tool_call.input_tokens} (Result Volume) (Captured)")
    print(f"8. Number of times each tool was called: {response.num_tool_calls} (Captured)")

    print("\nVERIFICATION COMPLETE.")

if __name__ == "__main__":
    asyncio.run(verify_metrics())
