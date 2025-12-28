"""
LLM Clients package for interacting with various Language Model APIs.
"""

from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient

AnthropicClient = OpenRouterClient
from .gemini_client import GeminiClient
from .writer_client import WriterClient
from .groq_client import GroqClient
from .xai_client import XAIClient
from .mistral_client import MistralClient
from .bedrock_client import BedrockClient
from .sagemaker_client import SageMakerClient
def get_llm_response(prompt: str, model_config: dict, is_json_response_expected: bool = False):
    """Routes all LLM requests to OpenRouterClient with retry logic."""
    import time
    import random
    import logging
    from .. import config
    
    logger = logging.getLogger(__name__)
    model_type = model_config.get("type")
    model_id = model_config.get("model_id")
    parameters = model_config.get("parameters", {}).copy()
    config_id = model_config.get("config_id", model_id)
    max_retries = getattr(config, 'DEFAULT_MAX_RETRIES', 3)
    
    if not config.OPENROUTER_API_KEY:
        logger.error(f"Missing OpenRouter API key for model {config_id}.")
        return {"error_message": "Missing OpenRouter API key", "response_time": 0}
    
    logger.info(f"Using OpenRouterClient for {model_type} model {config_id} (retry-enabled)")
    
    def _calculate_backoff_time(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff time with jitter."""
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.1, delay + jitter)
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        start_time = time.time()
        
        try:
            client = OpenRouterClient(api_key=config.OPENROUTER_API_KEY, model_id=model_id)
            client_response = client.get_response(
                prompt=prompt,
                parameters=parameters,
                is_json_response_expected=is_json_response_expected
            )
            
            if "error_message" in client_response:
                error_msg = client_response["error_message"]
                mock_error = Exception(error_msg)
                if attempt < max_retries and client._is_retryable_error(mock_error):
                    wait_time = _calculate_backoff_time(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {config_id}, retrying in {wait_time:.2f}s: {error_msg}")
                    time.sleep(wait_time)
                    continue
            
            return client_response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            last_error = e
            
            try:
                client = OpenRouterClient(api_key=config.OPENROUTER_API_KEY, model_id=model_id)
                if attempt < max_retries and client._is_retryable_error(e):
                    wait_time = _calculate_backoff_time(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {config_id}, retrying in {wait_time:.2f}s: {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Non-retryable error or max retries reached for {config_id}: {str(e)}")
                    break
            except:
                logger.error(f"Failed to create client for retry check for {config_id}: {str(e)}")
                break
    
    elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
    error_message = f"Failed after {max_retries + 1} attempts: {str(last_error) if last_error else 'Unknown error'}"
    logger.error(f"OpenRouterClient final failure for {config_id}: {error_message}")
    return {"error_message": error_message, "response_time": elapsed_time}

__all__ = [
    'BaseLLMClient',
    'OpenAIClient',
    'OpenRouterClient', 'AnthropicClient', 
    'GeminiClient',
    'WriterClient',
    'GroqClient',
    'XAIClient',
    'MistralClient',
    'BedrockClient',
    'SageMakerClient',
'get_llm_response'
]