
import asyncio
import os
from dotenv import load_dotenv
from execution.sub_agent import SubAgent, load_tools_config
from execution.tool_executor import ToolExecutor

load_dotenv()

SYSTEM_PROMPT = """
# IDENTITY & PURPOSE
You are Pierre Gallet's digital twin... (truncated for brevity, using full prompt from user request)
"""

# Minimal mock of tool config since we don't have the actual tools.yaml for this specific agent
# We'll rely on the existing config/tools.yaml available in the project
async def run_test():
    try:
        tools_config = load_tools_config()
    except FileNotFoundError:
        print("Warning: config/tools.yaml not found, running without tools")
        tools_config = []

    tool_executor = ToolExecutor(tools_config)
    
    # Use a dummy key if env var not set to prevent crash during init
    api_key = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-dummy")

    agent = SubAgent(
        model_id='anthropic/claude-3-5-sonnet',
        system_prompt=SYSTEM_PROMPT,
        tools_config=tools_config,
        tool_executor=tool_executor,
        model_pricing={'input_per_1m': 3.0, 'output_per_1m': 15.0},
        api_key=api_key 
    )
    
    print("Running Agent Execution...")
    result = await agent.execute(
        user_prompt="You say you understand 'separation of concerns' from factory work. That's a programming principle. Explain it technically, not with manufacturing metaphors.",
        prompt_id='test-1'
    )
    
    print("-" * 50)
    print(f"Final Response Snippet: {result.final_response[:200]}...")
    print(f"Finish Reason: {result.finish_reason}")
    if result.error:
        print(f"Error Details: {result.error}")
    print(f"Total Latency: {result.total_latency_ms:.0f}ms")
    print(f"Total Cost: ${result.cost_usd:.6f}")
    
    if result.tool_calls:
        print(f"\nTool Calls: {len(result.tool_calls)}")
        for tc in result.tool_calls:
            print(f" - Tool: {tc.tool_name}")
            print(f"   Latency: {tc.latency_ms:.0f}ms")
            print(f"   Generation Cost (Output Tokens): {tc.output_tokens}")
            print(f"   Result Volume (Input Tokens): {tc.input_tokens}")

if __name__ == '__main__':
    asyncio.run(run_test())
