"""
SageMaker client implementation.
"""
import json
import logging
import boto3
from typing import Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError, NoCredentialsError

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class SageMakerClient(BaseLLMClient):
    """Client for AWS SageMaker endpoints."""
    
    def __init__(self, api_key: str, model_id: str, region: str = "us-east-1"):
        super().__init__(api_key, model_id)
        self.region = region
        self.endpoint_name = model_id
        
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if not credentials:
                raise NoCredentialsError()
        except Exception as e:
            logger.error(f"AWS credentials not found or invalid: {e}")
            raise
        
        self.client = boto3.client('sagemaker-runtime', region_name=region)
    
    def _make_api_call(self, prompt: str, parameters: Dict[str, Any]) -> Any:
        """Make API call to SageMaker endpoint."""
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }
        
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        response_body = json.loads(response['Body'].read().decode())
        return response_body
    
    def _extract_response_text(self, api_response: Dict[str, Any]) -> str:
        """Extract response text from SageMaker response."""
        if 'generated_text' in api_response:
            return api_response['generated_text'].strip()
        elif 'outputs' in api_response and isinstance(api_response['outputs'], list):
            if api_response['outputs'] and 'generated_text' in api_response['outputs'][0]:
                return api_response['outputs'][0]['generated_text'].strip()
        elif 'predictions' in api_response and isinstance(api_response['predictions'], list):
            if api_response['predictions']:
                return str(api_response['predictions'][0]).strip()
        elif isinstance(api_response, list) and api_response:
            return str(api_response[0]).strip()
        elif isinstance(api_response, str):
            return api_response.strip()
        
        return str(api_response).strip()
    
    def _extract_token_counts(self, api_response: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from SageMaker response."""
        logger.info(f"Token counts not reliably available for SageMaker endpoint {self.endpoint_name}")
        return None, None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if SageMaker error is retryable."""
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            if error_code in ['ThrottlingException', 'ServiceUnavailableException', 'InternalServerException']:
                return True
            
            status_code = error.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if status_code and status_code >= 500:
                return True
        
        return False