"""
Gemini client implementation.
"""
import logging
from typing import Dict, Optional, Tuple, Any
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse, BlockedPromptException, StopCandidateException

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        genai.configure(
            api_key=api_key,
            client_options={'api_endpoint': 'http://123.129.219.111:3000'}  
        )
        self.model = genai.GenerativeModel(model_id)
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to Gemini."""
        gemini_params = parameters.copy()
        
        generation_config = {}
        if 'temperature' in gemini_params:
            generation_config['temperature'] = gemini_params.pop('temperature')
        if 'top_p' in gemini_params:
            generation_config['top_p'] = gemini_params.pop('top_p')
        if 'max_output_tokens' in gemini_params:
            generation_config['max_output_tokens'] = gemini_params.pop('max_output_tokens')
        
        thinking_config = None
        if 'flash' in self.model_id.lower():
            try:
                from google.generativeai.types import ThinkingConfig
                thinking_config = ThinkingConfig(include_thinking=True)
            except ImportError:
                logger.warning("ThinkingConfig not available, proceeding without thinking config")
        
        try:
            from google.generativeai.types import GenerateContentConfig
            config = GenerateContentConfig(
                generation_config=generation_config,
                thinking_config=thinking_config
            )
        except (ImportError, TypeError):
            config = generation_config
        
        return self.model.generate_content(prompt, generation_config=config)
    
    def _extract_response_text(self, api_response: GenerateContentResponse) -> str:
        """Extract response text from Gemini response."""
        try:
            if api_response.text:
                return api_response.text.strip()
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not extract text from Gemini response: {e}")
            
            if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback:
                if hasattr(api_response.prompt_feedback, 'block_reason'):
                    return f"Content blocked: {api_response.prompt_feedback.block_reason}"
            
            if hasattr(api_response, 'candidates') and api_response.candidates:
                candidate = api_response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        return candidate.content.parts[0].text.strip()
        
        return ""
    
    def _extract_token_counts(self, api_response: GenerateContentResponse) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from Gemini response."""
        if hasattr(api_response, 'usage_metadata') and api_response.usage_metadata:
            usage = api_response.usage_metadata
            
            input_tokens = getattr(usage, 'prompt_token_count', None)
            output_tokens = getattr(usage, 'candidates_token_count', None)
            
            thinking_tokens = getattr(usage, 'cached_content_token_count', None)
            if thinking_tokens and output_tokens:
                output_tokens += thinking_tokens
            
            if not input_tokens and not output_tokens:
                total_tokens = getattr(usage, 'total_token_count', None)
                if total_tokens:
                    input_tokens = total_tokens // 2
                    output_tokens = total_tokens - input_tokens
            
            return input_tokens, output_tokens
        
        logger.warning(f"Token usage data not found in Gemini response for {self.model_id}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if Gemini error is retryable."""
        if isinstance(error, (BlockedPromptException, StopCandidateException)):
            return False
        
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['rate limit', 'quota', '429', '500', '502', '503', '504']):
            return True
        
        return False