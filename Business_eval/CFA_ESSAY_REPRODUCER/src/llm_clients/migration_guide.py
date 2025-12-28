"""
Migration guide for updating imports from the old llm_clients.py to the new modular structure.

OLD IMPORT PATTERNS:
from llm_clients import get_llm_response
from llm_clients import get_openai_response, get_anthropic_response, etc.

NEW IMPORT PATTERNS:
from llm_clients import get_llm_response
from llm_clients import OpenAIClient, AnthropicClient, etc.

USAGE CHANGES:
Old way:
    response = get_openai_response(api_key, model_id, prompt, parameters)

New way:
    response = get_llm_response('openai', api_key, model_id, prompt, parameters)

Or using client directly:
    client = OpenAIClient(api_key, model_id)
    response = client.get_response(prompt, parameters)

FUNCTION MAPPING:
- get_openai_response() -> get_llm_response('openai', ...)
- get_anthropic_response() -> get_llm_response('anthropic', ...)
- get_gemini_response() -> get_llm_response('gemini', ...)
- get_writer_response() -> get_llm_response('writer', ...)
- get_groq_response() -> get_llm_response('groq', ...)
- get_xai_response() -> get_llm_response('xai', ...)
- get_mistral_response() -> get_llm_response('mistral', ...)
- get_bedrock_response() -> get_llm_response('bedrock', ...)
- get_sagemaker_response() -> get_llm_response('sagemaker', ...)
"""

def migrate_function_calls():
    """
    Example of how to migrate from old function calls to new ones.
    This is for reference only - you'll need to update your actual code.
    """
    
    
    from llm_clients import get_llm_response
    
    response = get_llm_response('openai', api_key, model_id, prompt, parameters)
    
    response = get_llm_response('anthropic', api_key, model_id, prompt, parameters)
    
    response = get_llm_response('gemini', api_key, model_id, prompt, parameters)
    
    response = get_llm_response('writer', api_key, model_id, prompt, parameters)
    
    response = get_llm_response('groq', api_key, model_id, prompt, parameters)
    
    response = get_llm_response('xai', api_key, model_id, prompt, parameters)
    
    response = get_llm_response('mistral', api_key, model_id, prompt, parameters)
    
    response = get_llm_response('bedrock', api_key, model_id, prompt, parameters, region='us-east-1')
    
    response = get_llm_response('sagemaker', api_key, endpoint_name, prompt, parameters, region='us-east-1')


def migrate_direct_client_usage():
    """
    Example of using clients directly for more control.
    """
    from llm_clients import OpenAIClient, AnthropicClient, GeminiClient
    
    openai_client = OpenAIClient(api_key, model_id)
    anthropic_client = AnthropicClient(api_key, model_id)
    gemini_client = GeminiClient(api_key, model_id)
    
    openai_response = openai_client.get_response(prompt, parameters)
    anthropic_response = anthropic_client.get_response(prompt, parameters)
    gemini_response = gemini_client.get_response(prompt, parameters)


if __name__ == "__main__":
    print(__doc__)