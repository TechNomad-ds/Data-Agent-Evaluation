"""
Model pricing configurations for OpenRouter API.
All prices are per million tokens for research-grade precision.
"""
import logging

logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "anthropic/claude-3-7-sonnet-20250219": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "anthropic/claude-3-5-sonnet-20241022": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "anthropic/claude-3-5-haiku-20241022": {
        "prompt_tokens_cost_per_million": 1.00,
        "completion_tokens_cost_per_million": 5.00,
    },
    "anthropic/claude-sonnet-4": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "anthropic/claude-opus-4.1": {
        "prompt_tokens_cost_per_million": 15.00,
        "completion_tokens_cost_per_million": 75.00,
    },
    "mistral-large-latest": {
        "prompt_tokens_cost_per_million": 2.00,
        "completion_tokens_cost_per_million": 6.00,
    },
    "codestral-latest": {
        "prompt_tokens_cost_per_million": 1.00,
        "completion_tokens_cost_per_million": 3.00,
    },
    "palmyra-fin": {
        "prompt_tokens_cost_per_million": 5.00,
        "completion_tokens_cost_per_million": 25.00,
    },
    "gpt-4o": {
        "prompt_tokens_cost_per_million": 2.50,
        "completion_tokens_cost_per_million": 10.00,
    },
    "openai/o3-mini": {
        "prompt_tokens_cost_per_million": 2.00,
        "completion_tokens_cost_per_million": 8.00,
    },
    "openai/o4-mini": {
        "prompt_tokens_cost_per_million": 1.50,
        "completion_tokens_cost_per_million": 6.00,
    },
    "openai/gpt-4.1": {
        "prompt_tokens_cost_per_million": 30.00,
        "completion_tokens_cost_per_million": 60.00,
    },
    "openai/gpt-4.1-mini": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 12.00,
    },
    "openai/gpt-4.1-nano": {
        "prompt_tokens_cost_per_million": 0.15,
        "completion_tokens_cost_per_million": 0.60,
    },
    "x-ai/grok-3": {
        "prompt_tokens_cost_per_million": 2.00,
        "completion_tokens_cost_per_million": 10.00,
    },
    "x-ai/grok-3-mini": {
        "prompt_tokens_cost_per_million": 0.20,
        "completion_tokens_cost_per_million": 1.00,
    },
    "google/gemini-2.5-pro-preview-05-06": {
        "prompt_tokens_cost_per_million": 1.25,
        "completion_tokens_cost_per_million": 5.00,
        "completion_tokens_cost_per_million_thinking": 10.00,
    },
    "google/gemini-2.5-flash-preview-04-17": {
        "prompt_tokens_cost_per_million": 0.075,
        "completion_tokens_cost_per_million": 0.30,
        "completion_tokens_cost_per_million_thinking": 0.60,
    },
    "deepseek/deepseek-chat-v3.1": {
        "prompt_tokens_cost_per_million": 0.14,
        "completion_tokens_cost_per_million": 0.28,
    },
    "meta-llama/llama-4-maverick": {
        "prompt_tokens_cost_per_million": 0.18,
        "completion_tokens_cost_per_million": 0.18,
    },
    "meta-llama/llama-4-scout": {
        "prompt_tokens_cost_per_million": 0.18,
        "completion_tokens_cost_per_million": 0.18,
    },
    "meta-llama/llama-guard-4-12b": {
        "prompt_tokens_cost_per_million": 0.20,
        "completion_tokens_cost_per_million": 0.20,
    },
    "meta-llama/llama-3.3-8b-instruct:free": {
        "prompt_tokens_cost_per_million": 0.00,
        "completion_tokens_cost_per_million": 0.00,
    },
    "qwen/qwen3-32b": {
        "prompt_tokens_cost_per_million": 0.70,
        "completion_tokens_cost_per_million": 2.10,
    },
    "moonshotai/kimi-k2": {
        "prompt_tokens_cost_per_million": 2.00,
        "completion_tokens_cost_per_million": 8.00,
    },
    "openai/gpt-oss-20b": {
        "prompt_tokens_cost_per_million": 0.04,
        "completion_tokens_cost_per_million": 0.15,
    },
    "openai/gpt-oss-120b": {
        "prompt_tokens_cost_per_million": 0.0072,
        "completion_tokens_cost_per_million": 0.28,
    },
    "x-ai/grok-4": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "openai/gpt-5": {
        "prompt_tokens_cost_per_million": 1.25,
        "completion_tokens_cost_per_million": 10.00,
    },
    "openai/gpt-5-nano": {
        "prompt_tokens_cost_per_million": 0.40,
        "completion_tokens_cost_per_million": 0.05,
    },
    "openai/gpt-5-mini": {
        "prompt_tokens_cost_per_million": 0.25,
        "completion_tokens_cost_per_million": 2.00,
    },
}


def get_pricing(model_type: str, model_id: str) -> dict:
    """
    Retrieves pricing information for a specific model.
    Research-grade precision: raises ValueError if model pricing is not found.
    
    Args:
        model_type: The type of the model (for logging, not used in lookup)
        model_id: The exact model ID to look up in pricing table
        
    Returns:
        Dictionary containing pricing information
        
    Raises:
        ValueError: If model_id is not found in pricing table
    """
    if model_id not in MODEL_PRICING:
        available_models = list(MODEL_PRICING.keys())
        raise ValueError(
            f"No pricing found for model '{model_id}' (type: {model_type}). "
            f"Available models: {available_models}"
        )
    
    pricing = MODEL_PRICING[model_id]
    logger.debug(f"Retrieved pricing for {model_type} model '{model_id}': {pricing}")
    return pricing