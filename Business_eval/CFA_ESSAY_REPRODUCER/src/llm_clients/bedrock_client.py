"""
Bedrock client implementation.
"""
import json
import logging
import boto3
from typing import Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError, NoCredentialsError

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock API."""
    
    def __init__(self, api_key: str, model_id: str, region: str = "us-east-1"):
        super().__init__(api_key, model_id)
        self.region = region
        
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if not credentials:
                raise NoCredentialsError()
        except Exception as e:
            logger.error(f"AWS credentials not found or invalid: {e}")
            raise
        
        self.client = boto3.client('bedrock-runtime', region_name=region)
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to Bedrock."""
        if 'anthropic' in self.model_id.lower():
            body = self._construct_anthropic_body(prompt, parameters)
        elif 'mistral' in self.model_id.lower():
            body = self._construct_mistral_body(prompt, parameters)
        elif 'meta' in self.model_id.lower():
            body = self._construct_meta_body(prompt, parameters)
        else:
            body = self._construct_anthropic_body(prompt, parameters)
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        
        response_body['_headers'] = response.get('ResponseMetadata', {}).get('HTTPHeaders', {})
        
        return response_body
    
    def _construct_anthropic_body(self, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Construct request body for Anthropic models on Bedrock."""
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31"
        }
        
        if 'max_tokens' in parameters:
            body['max_tokens'] = parameters['max_tokens']
        if 'temperature' in parameters:
            body['temperature'] = parameters['temperature']
        if 'top_p' in parameters:
            body['top_p'] = parameters['top_p']
        
        return body
    
    def _construct_mistral_body(self, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Construct request body for Mistral models on Bedrock."""
        body = {
            "prompt": prompt
        }
        
        if 'max_tokens' in parameters:
            body['max_tokens'] = parameters['max_tokens']
        if 'temperature' in parameters:
            body['temperature'] = parameters['temperature']
        if 'top_p' in parameters:
            body['top_p'] = parameters['top_p']
        
        return body
    
    def _construct_meta_body(self, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Construct request body for Meta models on Bedrock."""
        body = {
            "prompt": prompt
        }
        
        if 'max_gen_len' in parameters:
            body['max_gen_len'] = parameters['max_gen_len']
        elif 'max_tokens' in parameters:
            body['max_gen_len'] = parameters['max_tokens']
        if 'temperature' in parameters:
            body['temperature'] = parameters['temperature']
        if 'top_p' in parameters:
            body['top_p'] = parameters['top_p']
        
        return body
    
    def _extract_response_text(self, api_response: Dict[str, Any]) -> str:
        """Extract response text from Bedrock response."""
        if 'content' in api_response and isinstance(api_response['content'], list):
            if api_response['content'] and 'text' in api_response['content'][0]:
                return api_response['content'][0]['text'].strip()
        elif 'outputs' in api_response and isinstance(api_response['outputs'], list):
            if api_response['outputs'] and 'text' in api_response['outputs'][0]:
                return api_response['outputs'][0]['text'].strip()
        elif 'generation' in api_response:
            return api_response['generation'].strip()
        elif 'completion' in api_response:
            return api_response['completion'].strip()
        
        return ""
    
    def _extract_token_counts(self, api_response: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from Bedrock response."""
        if 'usage' in api_response:
            usage = api_response['usage']
            input_tokens = usage.get('input_tokens') or usage.get('prompt_tokens')
            output_tokens = usage.get('output_tokens') or usage.get('completion_tokens')
            if input_tokens is not None and output_tokens is not None:
                return input_tokens, output_tokens
        
        headers = api_response.get('_headers', {})
        if headers:
            input_tokens, output_tokens = self._get_tokens_from_headers(
                headers, 'x-amzn-bedrock-input-token-count', 'x-amzn-bedrock-output-token-count'
            )
            if input_tokens is not None and output_tokens is not None:
                return input_tokens, output_tokens
        
        logger.warning(f"Token usage data not found in Bedrock response for {self.model_id}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if Bedrock error is retryable."""
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            if error_code in ['ThrottlingException', 'ServiceUnavailableException', 'InternalServerException']:
                return True
            
            status_code = error.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if status_code and status_code >= 500:
                return True
        
        return False