"""Simple model client factory for Autogen team."""

import os
from typing import Dict, Any

from autogen_ext.models.openai import OpenAIChatCompletionClient


def create_model_client(model_config: Dict[str, Any]) -> OpenAIChatCompletionClient:
    """
    Create a model client based on configuration.

    Args:
        model_config: Dictionary with model configuration including:
            - model: Model name/identifier
            - temperature: Sampling temperature
            - provider: 'openai' or 'openrouter'
            - model_family: Model family identifier

    Returns:
        Configured OpenAIChatCompletionClient
    """
    # Extract needed values from config
    model = model_config.get("model", "gpt-4o-mini")
    temperature = model_config.get("temperature", 0.2)
    provider = model_config.get("provider", "openai")
    vision = model_config.get("vision", "True") == "True"
    function_calling = model_config.get("function_calling", "True") == "True"
    json_output = model_config.get("json_output", "True") == "True"
    require_full_parameter_support = model_config.get(
        "require_full_parameter_support", False
    )
    middle_out = model_config.get("middle_out", False)

    # Create base client args
    client_args = {
        "model": model,
        "temperature": temperature,
        "model_info": {
            "family": model_config.get("model_family", "unknown"),
            "vision": vision,
            "function_calling": function_calling,
            "json_output": json_output,
        },
    }

    # Adjust for provider
    if provider == "openrouter":
        client_args["base_url"] = "https://openrouter.ai/api/v1"
        client_args["api_key"] = os.environ["OPENROUTER_API_KEY"]

        # Initialize additional parameters if needed
        if "additional_parameters" not in client_args:
            client_args["additional_parameters"] = {}

        # Configure provider preferences
        if require_full_parameter_support:
            if "provider" not in client_args["additional_parameters"]:
                client_args["additional_parameters"]["provider"] = {}
            client_args["additional_parameters"]["provider"]["require_parameters"] = (
                True
            )

        # Add middle-out transform if enabled
        if middle_out:
            client_args["additional_parameters"]["transforms"] = ["middle-out"]

    # Create and return the client
    return OpenAIChatCompletionClient(**client_args)
