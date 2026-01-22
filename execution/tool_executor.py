import asyncio
import json
import os
import re
import time
import httpx
import asyncpg
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from .tool_schema import ToolDefinition
from .config_schema import ToolConfig

load_dotenv()

@dataclass
class ToolMetrics:
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: bool = False

@dataclass
class ToolResult:
    content: Any
    metrics: ToolMetrics

def substitute_env_vars(text: str) -> str:
    """Substitute ${VAR_NAME} patterns with environment variable values."""
    if not text:
        return text

    def replace_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))  # Keep original if not found

    return re.sub(r'\$\{([^}]+)\}', replace_var, text)


class ToolExecutor:
    def __init__(self, tool_configs: list[ToolConfig], pg_pool: asyncpg.Pool = None):
        self.tools = {t.name: ToolDefinition(t) for t in tool_configs}
        self.pg_pool = pg_pool

    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given parameters."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]
        
        # Validate parameters
        validated_params = tool.validate_parameters(parameters)

        start_time = time.time()
        # Simple token counting approximation (can be improved with tiktoken)
        input_tokens = len(json.dumps(validated_params)) // 4 

        try:
            if tool.endpoint.type == "postgres":
                result = await self._execute_postgres(tool, validated_params)
            else:
                # Default to HTTP
                result = await self._execute_http(tool, validated_params)
            
            error = False
        except Exception as e:
            result = {"error": str(e)}
            error = True

        latency_ms = (time.time() - start_time) * 1000
        output_tokens = len(json.dumps(result)) // 4

        return ToolResult(
            content=result,
            metrics=ToolMetrics(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=error
            )
        )

    async def _execute_http(self, tool: ToolDefinition, params: Dict[str, Any]) -> Any:
        try:
            body = tool.render_body(params)
            input_data = json.loads(body) if body else params
        except json.JSONDecodeError:
            # Fallback if body is not JSON (e.g. form data or raw string)
            input_data = params

        # Substitute environment variables in URL and headers
        url = substitute_env_vars(tool.endpoint.url)
        headers = {
            k: substitute_env_vars(v) for k, v in (tool.endpoint.headers or {}).items()
        }

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=tool.endpoint.method,
                url=url,
                headers=headers,
                json=input_data,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    async def _execute_postgres(self, tool: ToolDefinition, params: Dict[str, Any]) -> Any:
        if not self.pg_pool:
            raise ValueError("Postgres pool not initialized")

        if tool.endpoint.operation == "execute_query":
            # Direct SQL execution (careful with injection, but this is the requirement)
            query = params.get("sql_query")
            if not query:
                raise ValueError("Missing sql_query parameter")
            
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]

        elif tool.endpoint.operation == "select":
            # Structured select
            table = tool.endpoint.table
            filters = tool.endpoint.filter
            # Simple construction logic - in real app would need more robust query builder
            # This is a placeholder for the logic
            return f"Mock Postgres SELECT execution on {table}"
        
        return None
