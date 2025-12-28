"""
Utility functions for LLM clients including retry logic and client factory.
"""
import time
import random
import logging
from typing import Dict, Optional, Any

from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient
from .gemini_client import GeminiClient
from .writer_client import WriterClient
from .groq_client import GroqClient
from .xai_client import XAIClient
from .mistral_client import MistralClient
from .bedrock_client import BedrockClient
from .sagemaker_client import SageMakerClient

logger = logging.getLogger(__name__)

CLIENT_MAP = {
    'openai': OpenAIClient,
    'anthropic': OpenRouterClient,
    'gemini': GeminiClient,
    'google': GeminiClient,
    'writer': WriterClient,
    'openrouter': OpenRouterClient,
    'groq': GroqClient,
    'xai': XAIClient,
    'mistral': MistralClient,
    'bedrock': BedrockClient,
    'sagemaker': SageMakerClient,
}


def create_client(model_type: str, api_key: str, model_id: str, **kwargs):
    """Create an LLM client based on model type."""
    if model_type not in CLIENT_MAP:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(CLIENT_MAP.keys())}")
    
    client_class = CLIENT_MAP[model_type]
    
    if model_type in ['bedrock', 'sagemaker']:
        region = kwargs.get('region', 'us-east-1')
        return client_class(api_key, model_id, region)
    else:
        return client_class(api_key, model_id)


def get_llm_response(
    model_type: str,
    api_key: str,
    model_id: str,
    prompt: str,
    parameters: Optional[Dict[str, Any]] = None,
    is_json_response_expected: bool = False,
    max_retries: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Get response from LLM with retry logic.
    
    Args:
        model_type: Type of model (openai, anthropic, gemini, etc.)
        api_key: API key for the service
        model_id: Model identifier
        prompt: The prompt to send
        parameters: Model parameters
        is_json_response_expected: Whether to expect JSON response
        max_retries: Maximum number of retries
        **kwargs: Additional arguments for client creation
        
    Returns:
        Dictionary containing response data or error information
    """
    if parameters is None:
        parameters = {}
    
    try:
        client = create_client(model_type, api_key, model_id, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create client for {model_type}: {e}")
        return {
            "error_message": f"Failed to create client: {str(e)}",
            "details": {"type": type(e).__name__, "message": str(e)}
        }
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            response = client.get_response(prompt, parameters, is_json_response_expected)
            
            if "error_message" in response:
                error_msg = response["error_message"]
                mock_error = Exception(error_msg)
                if attempt < max_retries and client._is_retryable_error(mock_error):
                    wait_time = _calculate_backoff_time(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {model_type}, retrying in {wait_time:.2f}s: {error_msg}")
                    time.sleep(wait_time)
                    continue
            
            return response
            
        except Exception as e:
            last_error = e
            
            if attempt < max_retries and client._is_retryable_error(e):
                wait_time = _calculate_backoff_time(attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {model_type}, retrying in {wait_time:.2f}s: {str(e)}")
                time.sleep(wait_time)
            else:
                logger.error(f"Non-retryable error or max retries reached for {model_type}: {str(e)}")
                break
    
    error_details = {
        "type": type(last_error).__name__ if last_error else "Unknown",
        "message": str(last_error) if last_error else "Unknown error"
    }
    
    if hasattr(last_error, 'response'):
        try:
            if hasattr(last_error.response, 'status_code'):
                error_details["status_code"] = last_error.response.status_code
            if hasattr(last_error.response, 'text'):
                error_details["response_body"] = last_error.response.text[:500]
        except:
            pass
    
    return {
        "error_message": f"Failed after {max_retries + 1} attempts: {str(last_error) if last_error else 'Unknown error'}",
        "details": error_details
    }


def _calculate_backoff_time(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff time with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return max(0.1, delay + jitter)


def validate_api_key(api_key: str, service_name: str) -> bool:
    """Validate API key format."""
    if not api_key or not isinstance(api_key, str):
        logger.error(f"Invalid API key for {service_name}: API key must be a non-empty string")
        return False
    
    if len(api_key.strip()) < 10:
        logger.error(f"Invalid API key for {service_name}: API key appears to be too short")
        return False
    
    return True