"""
OpenRouter client implementation using OpenAI SDK.
"""
import logging
from typing import Dict, Optional, Tuple, Any
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError

from .base import BaseLLMClient
from .. import config as _global_config

logger = logging.getLogger(__name__)

class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API using OpenAI SDK."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        self.client = OpenAI(
            base_url="http://123.129.219.111:3000/v1",
            api_key= "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )
        
        self.extra_headers = {
            "HTTP-Referer": _global_config.YOUR_SITE_URL,
            "X-Title": _global_config.YOUR_APP_NAME
        }
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to OpenRouter using OpenAI SDK."""
        openai_params = parameters.copy()
        
        stream = openai_params.pop("stream", False)
        
        if stream:
            logger.info(f"Using streaming for {self.model_id} as explicitly requested")
            return self._make_streaming_call(prompt, openai_params)
        
        try:
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=self.extra_headers,
                **openai_params
            )
        except Exception as e:
            error_msg = str(e).lower()
            
            if "expecting value" in error_msg or "invalid json" in error_msg or "json" in error_msg:
                logger.error(f"JSON parsing error for {self.model_id}: {e}")
                logger.error(f"This usually indicates OpenRouter returned non-JSON content (HTML error page, etc.)")
                
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    logger.error(f"Raw response content: {e.response.text[:500]}...")
                elif hasattr(e, 'body'):
                    logger.error(f"Error body: {e.body}")
                
                raise Exception(f"OpenRouter API returned invalid JSON for {self.model_id}. "
                               f"This is likely a temporary API issue or rate limiting. "
                               f"Original error: {e}")
            else:
                raise
    
    def _make_streaming_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make streaming API call using OpenAI SDK."""
        openai_params = parameters.copy()
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_headers=self.extra_headers,
                **openai_params
            )
            
            full_content = ""
            final_response = None
            usage_info = None
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and choice.delta and choice.delta.content:
                        full_content += choice.delta.content
                    if choice.finish_reason:
                        final_response = chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = chunk.usage
            
            if final_response:
                class Message:
                    def __init__(self, content):
                        self.content = content
                        self.role = "assistant"
                
                class Choice:
                    def __init__(self, message):
                        self.message = message
                        self.finish_reason = "stop"
                        self.index = 0
                
                class CompletionResponse:
                    def __init__(self, content, usage, model_id):
                        self.choices = [Choice(Message(content))]
                        self.usage = usage
                        self.id = getattr(final_response, 'id', 'streaming-response')
                        self.object = "chat.completion"
                        self.model = model_id
                
                return CompletionResponse(full_content, usage_info, self.model_id)
            
            class MinimalResponse:
                def __init__(self, content):
                    self.choices = [type('Choice', (), {
                        'message': type('Message', (), {'content': content})()
                    })()]
                    self.usage = None
            
            return MinimalResponse(full_content)
                
        except Exception as e:
            logger.warning(f"Streaming failed for {self.model_id}, falling back to non-streaming: {e}")
            try:
                return self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    extra_headers=self.extra_headers,
                    **openai_params
                )
            except Exception as fallback_e:
                error_msg = str(fallback_e).lower()
                
                if "expecting value" in error_msg or "invalid json" in error_msg or "json" in error_msg:
                    logger.error(f"JSON parsing error in fallback for {self.model_id}: {fallback_e}")
                    logger.error(f"This usually indicates OpenRouter returned non-JSON content (HTML error page, etc.)")
                    
                    if hasattr(fallback_e, 'response') and hasattr(fallback_e.response, 'text'):
                        logger.error(f"Raw response content: {fallback_e.response.text[:500]}...")
                    elif hasattr(fallback_e, 'body'):
                        logger.error(f"Error body: {fallback_e.body}")
                    
                    raise Exception(f"OpenRouter API returned invalid JSON for {self.model_id} (in streaming fallback). "
                                   f"This is likely a temporary API issue or rate limiting. "
                                   f"Original error: {fallback_e}")
                else:
                    raise
    
    def _extract_response_text(self, api_response: Any) -> str:
        """Extract response text from OpenRouter response."""
        if api_response.choices and len(api_response.choices) > 0:
            message = api_response.choices[0].message
            if message and message.content:
                return message.content.strip()
        return ""
    
    def _extract_token_counts(self, api_response: Any) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from OpenRouter response."""
        if hasattr(api_response, 'usage') and api_response.usage:
            return api_response.usage.prompt_tokens, api_response.usage.completion_tokens
        logger.warning(f"Token usage data not found in OpenRouter response for {self.model_id}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if OpenRouter error is retryable."""
        if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIError):
            if hasattr(error, 'status_code') and error.status_code is not None and error.status_code >= 500:
                return True
        
        error_msg = str(error).lower()
        if "expecting value" in error_msg or "invalid json" in error_msg:
            logger.info(f"JSON parsing error detected for {self.model_id}, marking as retryable")
            return True
        
        if "openrouter api returned invalid json" in error_msg:
            return True
            
        return False