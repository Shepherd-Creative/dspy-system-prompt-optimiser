"""
SubAgent class for tool-calling LLM agents.

Each SubAgent wraps an LLM model with:
- System prompt injection
- Tools from config/tools.yaml converted to function calling format
- Agent loop execution (model → tool calls → tool execution → response)
- Tool call tracking for metrics
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

import httpx
import yaml
from dotenv import load_dotenv

from .tool_executor import ToolExecutor
from .config_schema import ToolConfig, ToolParameter

load_dotenv()


@dataclass
class ToolCall:
    """Record of a single tool call made by the agent."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    latency_ms: float
    success: bool
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentExecutionResult:
    """Complete result of an agent execution."""
    model_id: str
    prompt_id: str
    session_id: str

    # Final output
    final_response: str
    finish_reason: str

    # Metrics
    total_latency_ms: float
    llm_latency_ms: float  # Time spent waiting for LLM responses
    tool_latency_ms: float  # Time spent executing tools

    # Token usage
    input_tokens: int
    output_tokens: int
    cost_usd: float

    # Tool tracking
    tool_calls: List[ToolCall] = field(default_factory=list)
    num_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)

    # Iteration tracking
    num_iterations: int = 0
    max_iterations_reached: bool = False

    # Error handling
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "session_id": self.session_id,
            "final_response": self.final_response,
            "finish_reason": self.finish_reason,
            "total_latency_ms": self.total_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "tool_latency_ms": self.tool_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "num_tool_calls": self.num_tool_calls,
            "tools_used": self.tools_used,
            "num_iterations": self.num_iterations,
            "error": self.error,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "parameters": tc.parameters,
                    "result": str(tc.result)[:500],  # Truncate large results
                    "latency_ms": tc.latency_ms,
                    "success": tc.success,
                }
                for tc in self.tool_calls
            ],
        }


class SubAgent:
    """
    A sub-agent that wraps an LLM model with tool-calling capabilities.

    Uses OpenRouter's function calling API to enable tool use across
    different model providers.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_id: str,
        system_prompt: str,
        tools_config: List[ToolConfig],
        tool_executor: ToolExecutor,
        model_pricing: Dict[str, float],
        api_key: Optional[str] = None,
        max_iterations: int = 10,
        timeout: float = 120.0,
    ):
        """
        Initialize a SubAgent.

        Args:
            model_id: OpenRouter model identifier
            system_prompt: System prompt to inject
            tools_config: List of ToolConfig objects from tools.yaml
            tool_executor: ToolExecutor instance for executing tool calls
            model_pricing: Dict with 'input_per_1m' and 'output_per_1m' keys
            api_key: OpenRouter API key (defaults to env)
            max_iterations: Maximum agent loop iterations
            timeout: Request timeout in seconds
        """
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.tools_config = tools_config
        self.tool_executor = tool_executor
        self.model_pricing = model_pricing
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.max_iterations = max_iterations
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        # Convert tools to OpenRouter function calling format
        self.tools_schema = self._build_tools_schema()

    def _build_tools_schema(self) -> List[Dict[str, Any]]:
        """Convert ToolConfig objects to OpenRouter function calling schema."""
        tools = []
        for tool in self.tools_config:
            # Build parameters schema
            properties = {}
            required = []

            for param_name, param_config in tool.parameters.items():
                # Skip parameters that come from sub_agent or env (not from model)
                if param_config.source in ("sub_agent", "env", "prompt_optimiser"):
                    continue

                prop = {"type": param_config.type}
                if param_config.description:
                    prop["description"] = param_config.description
                if param_config.default is not None:
                    prop["default"] = param_config.default

                properties[param_name] = prop

                if param_config.required and param_config.default is None:
                    required.append(param_name)

            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description.strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            tools.append(tool_schema)

        return tools

    def _inject_context_params(
        self,
        tool_name: str,
        params: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, Any]:
        """Inject sub_agent and env sourced parameters."""
        tool_config = next(
            (t for t in self.tools_config if t.name == tool_name), None
        )
        if not tool_config:
            return params

        enriched = dict(params)
        for param_name, param_config in tool_config.parameters.items():
            if param_config.source == "sub_agent":
                enriched[param_name] = session_id
            elif param_config.source == "env":
                env_var = getattr(param_config, 'env_var', param_name.upper())
                enriched[param_name] = os.getenv(env_var, "")

        return enriched

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        input_rate = self.model_pricing.get("input_per_1m", 0)
        output_rate = self.model_pricing.get("output_per_1m", 0)
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    async def execute(
        self,
        user_prompt: str,
        prompt_id: str,
        session_id: Optional[str] = None,
    ) -> AgentExecutionResult:
        """
        Execute the agent loop for a given user prompt.

        Args:
            user_prompt: The user's prompt/question
            prompt_id: Identifier for this prompt
            session_id: Optional session ID (auto-generated if not provided)

        Returns:
            AgentExecutionResult with response, metrics, and tool call history
        """
        session_id = session_id or f"{self.model_id.replace('/', '_')}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tool_calls_history: List[ToolCall] = []
        total_input_tokens = 0
        total_output_tokens = 0
        llm_latency_ms = 0
        tool_latency_ms = 0
        iteration = 0

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                while iteration < self.max_iterations:
                    iteration += 1
                    llm_start = time.time()

                    # Make LLM request
                    response = await client.post(
                        f"{self.OPENROUTER_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "X-Title": "Prompt Optimser Agent - SubAgent",
                        },
                        json={
                            "model": self.model_id,
                            "messages": messages,
                            "tools": self.tools_schema if self.tools_schema else None,
                            "tool_choice": "auto" if self.tools_schema else None,
                        },
                    )

                    llm_latency_ms += (time.time() - llm_start) * 1000

                    if response.status_code != 200:
                        return AgentExecutionResult(
                            model_id=self.model_id,
                            prompt_id=prompt_id,
                            session_id=session_id,
                            final_response="",
                            finish_reason="error",
                            total_latency_ms=(time.time() - start_time) * 1000,
                            llm_latency_ms=llm_latency_ms,
                            tool_latency_ms=tool_latency_ms,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            cost_usd=self.calculate_cost(total_input_tokens, total_output_tokens),
                            tool_calls=tool_calls_history,
                            num_tool_calls=len(tool_calls_history),
                            tools_used=list(set(tc.tool_name for tc in tool_calls_history)),
                            num_iterations=iteration,
                            error=f"HTTP {response.status_code}: {response.text}",
                        )

                    data = response.json()
                    usage = data.get("usage", {})
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)

                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    finish_reason = choice.get("finish_reason", "")

                    # Check if model wants to call tools
                    tool_calls = message.get("tool_calls", [])

                    if tool_calls:
                        # Add assistant message with tool calls to history
                        messages.append(message)

                        # Estimate tokens per tool call by splitting the generation cost
                        # This is an approximation since we can't get exact tokens per tool call from API
                        current_output_tokens = usage.get("completion_tokens", 0)
                        tokens_per_tool = current_output_tokens // len(tool_calls) if tool_calls else 0

                        # Execute each tool call
                        for tool_call in tool_calls:
                            tool_start = time.time()
                            func = tool_call.get("function", {})
                            tool_name = func.get("name", "")

                            try:
                                tool_params = json.loads(func.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                tool_params = {}

                            # Inject context parameters
                            enriched_params = self._inject_context_params(
                                tool_name, tool_params, session_id
                            )

                            # Execute tool
                            try:
                                result = await self.tool_executor.execute(
                                    tool_name, enriched_params
                                )
                                tool_result = result.content
                                tool_success = not result.metrics.error
                                tool_error = None
                            except Exception as e:
                                tool_result = {"error": str(e)}
                                tool_success = False
                                tool_error = str(e)

                            tool_call_latency = (time.time() - tool_start) * 1000
                            tool_latency_ms += tool_call_latency

                            # Estimate tokens for the tool result (approx 4 chars per token)
                            # This helps validat tool efficiency (Latency vs Data Volume)
                            result_str = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                            result_tokens = len(result_str) // 4

                            # Record tool call
                            tool_calls_history.append(ToolCall(
                                tool_name=tool_name,
                                parameters=enriched_params,
                                result=tool_result,
                                latency_ms=tool_call_latency,
                                success=tool_success,
                                error=tool_error,
                                output_tokens=tokens_per_tool,  # Cost to ask (LLM generation)
                                input_tokens=result_tokens,     # Cost to process result (Tool Output)
                            ))

                            # Add tool result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", ""),
                                "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result,
                            })

                        # Continue loop to get model's response with tool results
                        continue

                    else:
                        # No tool calls - model is done
                        final_content = message.get("content", "")

                        return AgentExecutionResult(
                            model_id=self.model_id,
                            prompt_id=prompt_id,
                            session_id=session_id,
                            final_response=final_content,
                            finish_reason=finish_reason,
                            total_latency_ms=(time.time() - start_time) * 1000,
                            llm_latency_ms=llm_latency_ms,
                            tool_latency_ms=tool_latency_ms,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            cost_usd=self.calculate_cost(total_input_tokens, total_output_tokens),
                            tool_calls=tool_calls_history,
                            num_tool_calls=len(tool_calls_history),
                            tools_used=list(set(tc.tool_name for tc in tool_calls_history)),
                            num_iterations=iteration,
                        )

                # Max iterations reached
                return AgentExecutionResult(
                    model_id=self.model_id,
                    prompt_id=prompt_id,
                    session_id=session_id,
                    final_response="",
                    finish_reason="max_iterations",
                    total_latency_ms=(time.time() - start_time) * 1000,
                    llm_latency_ms=llm_latency_ms,
                    tool_latency_ms=tool_latency_ms,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cost_usd=self.calculate_cost(total_input_tokens, total_output_tokens),
                    tool_calls=tool_calls_history,
                    num_tool_calls=len(tool_calls_history),
                    tools_used=list(set(tc.tool_name for tc in tool_calls_history)),
                    num_iterations=iteration,
                    max_iterations_reached=True,
                    error="Maximum iterations reached without completion",
                )

        except httpx.TimeoutException:
            return AgentExecutionResult(
                model_id=self.model_id,
                prompt_id=prompt_id,
                session_id=session_id,
                final_response="",
                finish_reason="error",
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=llm_latency_ms,
                tool_latency_ms=tool_latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cost_usd=self.calculate_cost(total_input_tokens, total_output_tokens),
                tool_calls=tool_calls_history,
                num_tool_calls=len(tool_calls_history),
                tools_used=list(set(tc.tool_name for tc in tool_calls_history)),
                num_iterations=iteration,
                error=f"Request timed out after {self.timeout}s",
            )
        except Exception as e:
            return AgentExecutionResult(
                model_id=self.model_id,
                prompt_id=prompt_id,
                session_id=session_id,
                final_response="",
                finish_reason="error",
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=llm_latency_ms,
                tool_latency_ms=tool_latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cost_usd=self.calculate_cost(total_input_tokens, total_output_tokens),
                tool_calls=tool_calls_history,
                num_tool_calls=len(tool_calls_history),
                tools_used=list(set(tc.tool_name for tc in tool_calls_history)),
                num_iterations=iteration,
                error=str(e),
            )


def load_tools_config(tools_config_path: str = "config/tools.yaml") -> List[ToolConfig]:
    """Load tools configuration from YAML file."""
    with open(tools_config_path) as f:
        config = yaml.safe_load(f)

    tools = []
    for tool_data in config.get("tools", []):
        parameters = {}
        for param_name, param_data in tool_data.get("parameters", {}).items():
            parameters[param_name] = ToolParameter(**param_data)

        tools.append(ToolConfig(
            name=tool_data["name"],
            description=tool_data["description"],
            parameters=parameters,
            endpoint=tool_data.get("endpoint"),
        ))

    return tools
