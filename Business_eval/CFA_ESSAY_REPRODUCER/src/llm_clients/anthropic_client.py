"""
Anthropic client implementation.
"""
import logging
from typing import Dict, Optional, Tuple, Any
from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError, APITimeoutError as AnthropicTimeoutError, APIStatusError

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic API."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        self.client = Anthropic(
            api_key=api_key,
            base_url="http://123.129.219.111:3000/v1"  # 注意这里通常需要加上 /v1
        )
        self.thinking_models = ['claude-opus-4-1', 'claude-opus-4-1-20250805']
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to Anthropic."""
        anthropic_params = parameters.copy()
        
        if self._is_thinking_model() and self._should_use_streaming(anthropic_params):
            print(f"[DEBUG] Using streaming for {self.model_id} with max_tokens={anthropic_params.get('max_tokens', 'N/A')}")
            logger.info(f"Using streaming for {self.model_id} to handle extended thinking")
            return self._make_streaming_call(prompt, anthropic_params)
        else:
            print(f"[DEBUG] Using non-streaming for {self.model_id}. Is thinking model: {self._is_thinking_model()}, Should use streaming: {self._should_use_streaming(anthropic_params)}")
        
        return self.client.messages.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            **anthropic_params
        )
    
    def _is_thinking_model(self) -> bool:
        """Check if the current model is a thinking model that may require extended processing."""
        return any(thinking_model in self.model_id.lower() for thinking_model in self.thinking_models)
    
    def _should_use_streaming(self, parameters: Dict[str, Any]) -> bool:
        """Determine if streaming should be used based on parameters."""
        max_tokens = parameters.get('max_tokens', 0)
        return max_tokens > 2000
    
    def _make_streaming_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make streaming API call and collect the full response."""
        anthropic_params = parameters.copy()
        
        try:
            print(f"[DEBUG] Starting streaming call for {self.model_id}")
            stream = self.client.messages.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **anthropic_params
            )
            
            full_content = ""
            message_obj = None
            
            for chunk in stream:
                print(f"[DEBUG] Streaming chunk type: {chunk.type}")
                logger.debug(f"Streaming chunk type: {chunk.type}")
                if chunk.type == "message_start":
                    message_obj = chunk.message
                    print(f"[DEBUG] Message start received")
                    logger.debug(f"Message start: {message_obj}")
                elif chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        full_content += chunk.delta.text
                        print(f"[DEBUG] Added text_delta chunk, total length: {len(full_content)}")
                        logger.debug(f"Added text_delta chunk, total length: {len(full_content)}")
                    elif hasattr(chunk.delta, 'thinking'):
                        full_content += chunk.delta.thinking
                        print(f"[DEBUG] Added thinking_delta chunk, total length: {len(full_content)}")
                        logger.debug(f"Added thinking_delta chunk, total length: {len(full_content)}")
                    else:
                        print(f"[DEBUG] Unknown delta type in content_block_delta: {chunk.delta}")
                        logger.debug(f"Unknown delta type in content_block_delta: {chunk.delta}")
                elif chunk.type == "message_delta":
                    if hasattr(chunk, 'usage') and message_obj:
                        message_obj.usage = chunk.usage
                        print(f"[DEBUG] Updated usage")
                        logger.debug(f"Updated usage: {message_obj.usage}")
            
            print(f"[DEBUG] Streaming complete. Full content length: {len(full_content)}")
            logger.info(f"Streaming complete. Full content length: {len(full_content)}")
            logger.debug(f"Full content preview: {full_content[:200]}...")
            
            if message_obj:
                if message_obj.content and len(message_obj.content) > 0:
                    message_obj.content[0].text = full_content
                    print(f"[DEBUG] Updated message_obj content length: {len(message_obj.content[0].text)}")
                    logger.debug(f"Updated message_obj content: {message_obj.content[0].text[:200]}...")
                else:
                    print(f"[DEBUG] Creating new content block with text")
                    class ContentBlock:
                        def __init__(self, text):
                            self.text = text
                            self.type = "text"
                    
                    message_obj.content = [ContentBlock(full_content)]
                    print(f"[DEBUG] Created new content block with length: {len(full_content)}")
                    
                return message_obj
            else:
                print(f"[DEBUG] No message_obj found, using fallback StreamingResponse")
                logger.warning("No message_obj found, using fallback StreamingResponse")
                class StreamingResponse:
                    def __init__(self, content_text):
                        self.content = [type('Content', (), {'text': content_text})()]
                        self.usage = None
                
                return StreamingResponse(full_content)
                
        except Exception as e:
            print(f"[DEBUG] Streaming failed: {e}")
            logger.warning(f"Streaming failed for {self.model_id}, falling back to non-streaming: {e}")
            return self.client.messages.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **anthropic_params
            )
    
    def _extract_response_text(self, api_response: Any) -> str:
        """Extract response text from Anthropic response."""
        if api_response.content and len(api_response.content) > 0:
            return api_response.content[0].text.strip()
        return ""
    
    def _extract_token_counts(self, api_response: Any) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from Anthropic response."""
        if hasattr(api_response, 'usage') and api_response.usage:
            return api_response.usage.input_tokens, api_response.usage.output_tokens
        logger.warning(f"Token usage data not found in Anthropic response for {self.model_id}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if Anthropic error is retryable."""
        if isinstance(error, (RateLimitError, APIConnectionError, AnthropicTimeoutError)):
            return True
        if isinstance(error, APIStatusError) and error.status_code is not None and error.status_code >= 500:
            return True
        if isinstance(error, APIError):
            return True
        return False