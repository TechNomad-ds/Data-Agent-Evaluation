"""
Writer client implementation.
"""
import json
import logging
import requests
from typing import Dict, Optional, Tuple, Any

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class WriterClient(BaseLLMClient):
    """Client for Writer API."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        self.base_url = "https://api.writer.com/v1/chat/completions"
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to Writer."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            **parameters
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _extract_response_text(self, api_response: Dict[str, Any]) -> str:
        """Extract response text from Writer response."""
        if 'choices' in api_response and api_response['choices']:
            choice = api_response['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content']
                
                finish_reason = choice.get('finish_reason', '')
                if finish_reason == 'length':
                    logger.warning(f"Writer response was truncated due to length limit for model {self.model_id}")
                    try:
                        json.loads(content)
                    except json.JSONDecodeError:
                        if content.strip().endswith(','):
                            content = content.strip()[:-1]
                        if not content.strip().endswith('}'):
                            content += '}'
                
                return content.strip()
        return ""
    
    def _extract_token_counts(self, api_response: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from Writer response."""
        if 'usage' in api_response:
            usage = api_response['usage']
            input_tokens = usage.get('prompt_tokens')
            output_tokens = usage.get('completion_tokens')
            
            if input_tokens is None or output_tokens is None:
                logger.warning(f"Incomplete token usage data from Writer API for {self.model_id}, using tiktoken estimation")
                
                if 'choices' in api_response and api_response['choices']:
                    response_text = self._extract_response_text(api_response)
                    if input_tokens is None:
                        input_tokens = None
                    if output_tokens is None:
                        output_tokens = self._estimate_tokens_tiktoken(response_text)
            
            return input_tokens, output_tokens
        
        logger.warning(f"Token usage data not found in Writer response for {self.model_id}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if Writer error is retryable."""
        if isinstance(error, requests.exceptions.RequestException):
            if hasattr(error, 'response') and error.response is not None:
                status_code = error.response.status_code
                if status_code in [429, 500, 502, 503, 504]:
                    return True
            if isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                return True
        return False