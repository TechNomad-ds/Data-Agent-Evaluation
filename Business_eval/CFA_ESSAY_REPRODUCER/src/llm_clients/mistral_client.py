"""
Mistral client implementation.
"""
import logging
from typing import Dict, Optional, Tuple, Any
from mistralai import Mistral

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class MistralClient(BaseLLMClient):
    """Client for Mistral API."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        self.client = Mistral(api_key=api_key)
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to Mistral."""
        return self.client.chat.complete(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            **parameters
        )
    
    def _extract_response_text(self, api_response: Any) -> str:
        """Extract response text from Mistral response."""
        if hasattr(api_response, 'choices') and api_response.choices:
            choice = api_response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content.strip()
        return ""
    
    def _extract_token_counts(self, api_response: Any) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from Mistral response."""
        if hasattr(api_response, 'usage') and api_response.usage:
            input_tokens = getattr(api_response.usage, 'prompt_tokens', None)
            output_tokens = getattr(api_response.usage, 'completion_tokens', None)
            return input_tokens, output_tokens
        
        logger.warning(f"Token usage data not found in Mistral response for {self.model_id}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if Mistral error is retryable."""
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['rate limit', 'quota', '429', '500', '502', '503', '504']):
            return True
        
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'network']):
            return True
        
        return False