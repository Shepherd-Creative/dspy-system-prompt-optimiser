import pytest
import respx
from httpx import Response
from execution.tool_executor import ToolExecutor
from execution.config_schema import ToolConfig, ToolParameter

@pytest.fixture
def mock_tool_config():
    return ToolConfig(
        name="test_tool",
        description="A test tool",
        parameters={
            "param1": ToolParameter(type="string", required=True),
            "param2": ToolParameter(type="number", default=42)
        },
        endpoint={
            "url": "https://api.example.com/test",
            "method": "POST",
            "body_template": '{"p1": "{{param1}}", "p2": {{param2}}}'
        }
    )

@pytest.mark.asyncio
async def test_tool_executor_http(mock_tool_config):
    executor = ToolExecutor([mock_tool_config])
    
    async with respx.mock(base_url="https://api.example.com") as respx_mock:
        route = respx_mock.post("/test").mock(return_value=Response(200, json={"status": "ok"}))
        
        result = await executor.execute("test_tool", {"param1": "value1"})
        
        assert result.content == {"status": "ok"}
        assert result.metrics.latency_ms > 0
        assert not result.metrics.error
        
        # Verify request body rendering
        last_request = route.calls.last.request
        assert last_request.content == b'{"p1": "value1", "p2": 42}'

@pytest.mark.asyncio
async def test_tool_executor_missing_param(mock_tool_config):
    executor = ToolExecutor([mock_tool_config])
    result = await executor.execute("test_tool", {})  # Missing param1
    
    assert result.metrics.error
    assert "Missing required parameter" in str(result.content)
