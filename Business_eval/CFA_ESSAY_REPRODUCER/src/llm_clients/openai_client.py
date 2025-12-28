"""
OpenAI client implementation.
"""
import logging
from typing import Dict, Optional, Tuple, Any
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError as OpenAITimeoutError

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        self.client = OpenAI(api_key=api_key)
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to OpenAI."""
        openai_params = parameters.copy()
        
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            **openai_params
        )
    
    def _extract_response_text(self, api_response: Any) -> str:
        """Extract response text from OpenAI response."""
        if api_response.choices and api_response.choices[0].message:
            return api_response.choices[0].message.content.strip()
        return ""
    
    def _extract_token_counts(self, api_response: Any) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from OpenAI response."""
        if hasattr(api_response, 'usage') and api_response.usage:
            return api_response.usage.prompt_tokens, api_response.usage.completion_tokens
        logger.warning(f"Token usage data not found in OpenAI response for {self.model_id}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if OpenAI error is retryable."""
        if isinstance(error, (RateLimitError, APIConnectionError, OpenAITimeoutError)):
            return True
        if isinstance(error, APIError) and error.status_code is not None and error.status_code >= 500:
            return True
        return False