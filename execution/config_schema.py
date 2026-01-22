from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
import yaml

class ToolEndpoint(BaseModel):
    url: Optional[str] = None
    method: Optional[str] = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    body_template: Optional[str] = None
    type: Optional[str] = None  # http, postgres, supabase
    connection: Optional[str] = None
    operation: Optional[str] = None
    table: Optional[str] = None
    filter: Optional[str] = None

class ToolParameter(BaseModel):
    type: str
    required: bool = False
    description: Optional[str] = None
    default: Optional[Any] = None
    fixed_value: Optional[Any] = None
    source: Optional[str] = None  # model, context, fixed

class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ToolParameter]
    endpoint: Optional[Union[ToolEndpoint, str, Dict[str, Any]]] = None

    @field_validator('endpoint', mode='before')
    def parse_endpoint(cls, v):
        if isinstance(v, str):
            return ToolEndpoint(url=v)
        if isinstance(v, dict):
            return ToolEndpoint(**v)
        return v

class TestPrompt(BaseModel):
    id: str
    prompt: str
    category: Optional[str] = "general"
    expected_intent: Optional[str] = None
    expected_tool: Optional[str] = None
    # We use a permissive Dict for weights/params to allow flexible testing
    expected_weights: Optional[Dict[str, Union[str, float]]] = None
    expected_workflow: Optional[List[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    expected_behaviour: Optional[str] = None
    evaluation: Optional[List[str]] = None

class ExecutionSettings(BaseModel):
    parallel: bool = True
    timeout_seconds: int = 120
    session_id_format: str = "{test_id}_{model}_{timestamp}"

class OptimizerConfig(BaseModel):
    system_prompt: str
    tools: List[ToolConfig]
    test_prompts: List[TestPrompt]
    models_to_test: List[str]
    chairman_model: Optional[str] = None
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
