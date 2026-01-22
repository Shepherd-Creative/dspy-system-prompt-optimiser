from typing import Dict, Any, Optional, List
from jinja2 import Template
from .config_schema import ToolConfig, ToolParameter

class ToolDefinition:
    def __init__(self, config: ToolConfig):
        self.name = config.name
        self.description = config.description
        self.parameters = config.parameters
        self.endpoint = config.endpoint

    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that provided parameters match schema."""
        validated = {}
        
        # Check for required parameters
        for name, schema in self.parameters.items():
            if schema.required and name not in params:
                 # Check if default is available
                if schema.default is not None:
                    validated[name] = schema.default
                else:
                    raise ValueError(f"Missing required parameter: {name}")
            elif name in params:
                validated[name] = params[name]
            elif schema.default is not None:
                validated[name] = schema.default

        # Add fixed values
        for name, schema in self.parameters.items():
            if schema.fixed_value is not None:
                validated[name] = schema.fixed_value

        return validated

    def render_body(self, params: Dict[str, Any]) -> str:
        """Render request body template with parameters."""
        if not self.endpoint or not self.endpoint.body_template:
            return ""
        
        template = Template(self.endpoint.body_template)
        return template.render(**params)
