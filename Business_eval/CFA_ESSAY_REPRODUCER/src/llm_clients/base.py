"""
Base class for all LLM clients.
"""
import time
import json
import logging
import re
import tiktoken
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for all LLM clients."""
    
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
    
    @abstractmethod
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make the actual API call to the LLM service."""
        pass
    
    @abstractmethod
    def _extract_response_text(self, api_response: Any) -> str:
        """Extract the response text from the API response."""
        pass
    
    @abstractmethod
    def _extract_token_counts(self, api_response: Any) -> Tuple[Optional[int], Optional[int]]:
        """Extract input and output token counts from the API response."""
        pass
    
    @abstractmethod
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        pass
    
    def get_response(self, prompt: str, parameters: Dict[str, Any], 
                    is_json_response_expected: bool = False) -> Dict[str, Any]:
        """
        Get response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            parameters: Model parameters
            is_json_response_expected: Whether to expect JSON response
            
        Returns:
            Dictionary containing response data
        """
        start_time = time.time()
        
        try:
            api_response = self._make_api_call(prompt, parameters)
            response_text = self._extract_response_text(api_response)
            input_tokens, output_tokens = self._extract_token_counts(api_response)
            
            if is_json_response_expected:
                parsed_content = self._parse_json_response(response_text)
            else:
                parsed_content = response_text
            
            elapsed_time = time.time() - start_time
            
            return {
                "response_content": parsed_content,
                "raw_response_text": response_text,
                "response_time": elapsed_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            
            return {
                "error_message": str(e),
                "response_time": elapsed_time,
                "details": {"type": type(e).__name__, "message": str(e)}
            }
    
    def _parse_json_response(self, response_text: str) -> Any:
        """Parse JSON response from text."""
        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({.*?})|(\[.*?\])", 
                                 response_text, re.DOTALL)
            if json_match:
                json_str = next(g for g in json_match.groups() if g is not None)
                return json.loads(json_str)
            else:
                return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return "X"
    
    def _estimate_tokens_tiktoken(self, text: str, encoding_name: str = "cl100k_base") -> Optional[int]:
        """Estimate token count using tiktoken."""
        if not text:
            return 0
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Could not estimate tokens using tiktoken: {e}")
            return None
    
    def _get_tokens_from_headers(self, headers: Dict[str, str], 
                                input_key: str, output_key: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from response headers."""
        input_val_str = headers.get(input_key)
        output_val_str = headers.get(output_key)
        
        input_tokens = None
        if input_val_str and input_val_str.isdigit():
            input_tokens = int(input_val_str)
            
        output_tokens = None
        if output_val_str and output_val_str.isdigit():
            output_tokens = int(output_val_str)
            
        return input_tokens, output_tokens