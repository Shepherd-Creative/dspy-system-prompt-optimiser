import yaml
import os
import re
from typing import Any, Dict
from .config_schema import OptimizerConfig

class ConfigLoader:
    def __init__(self):
        self.env_pattern = re.compile(r'\$\{(\w+)\}')

    def _replace_env_vars(self, content: str) -> str:
        """Replace ${VAR} with environment variable values."""
        def replace(match):
            env_var = match.group(1)
            return os.getenv(env_var, f"${{{env_var}}}")
        return self.env_pattern.sub(replace, content)

    def load_config(self, config_path: str) -> OptimizerConfig:
        """Load and validate configuration from a YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            raw_content = f.read()

        # Replace environment variables
        processed_content = self._replace_env_vars(raw_content)
        
        # Parse YAML
        try:
            config_dict = yaml.safe_load(processed_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

        # Validate with Pydantic
        try:
            return OptimizerConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

if __name__ == "__main__":
    # verification
    loader = ConfigLoader()
    print("ConfigLoader initialized.")
