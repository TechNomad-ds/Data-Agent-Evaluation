"""
Model configurations tailored for Chain-of-Thought (CoT) prompting strategies.
"""

ALL_MODEL_CONFIGS_COT = [    
    {
        "config_id": "gemini-2.5-pro-cot",
        "type": "openrouter",
        "model_id": "google/gemini-2.5-pro-preview-05-06",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 65536
        }
    },
    {
        "config_id": "gemini-2.5-flash-cot",
        "type": "openrouter",
        "model_id": "google/gemini-2.5-flash-preview-04-17",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 65536,
            "thinking_budget": 24576
        }
    },
    {
        "config_id": "claude-3.7-sonnet-cot",
        "type": "openrouter",
        "model_id": "anthropic/claude-3-7-sonnet-20250219",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 64000
        }
    },
    {
        "config_id": "claude-3.5-sonnet-cot",
        "type": "openrouter",
        "model_id": "anthropic/claude-3-5-sonnet-20241022",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "claude-3.5-haiku-cot",
        "type": "openrouter",
        "model_id": "anthropic/claude-3-5-haiku-20241022",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "mistral-large-official-cot",
        "type": "openrouter",
        "model_id": "mistral-large-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 4096 
        }
    },
    {
        "config_id": "codestral-latest-cot",
        "type": "openrouter",
        "model_id": "codestral-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 32768 
        }
    },
    {
        "config_id": "gpt-4o-cot",
        "type": "openrouter",
        "model_id": "openai/gpt-4o",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5
            
        }
    },
    {
        "config_id": "gpt-o3-cot",
        "type": "openrouter",
        "model_id": "openai/o3", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-o4-mini-cot",
        "type": "openrouter",
        "model_id": "openai/o4-mini", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1-cot",
        "type": "openrouter",
        "model_id": "openai/gpt-4.1",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
            
        }
    },
    {
        "config_id": "gpt-4.1-mini-cot",
        "type": "openrouter",
        "model_id": "openai/gpt-4.1-mini",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
            
        }
    },
    {
        "config_id": "gpt-4.1-nano-cot",
        "type": "openrouter",
        "model_id": "openai/gpt-4.1-nano",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "grok-3-mini-beta-cot-high-effort",
        "type": "openrouter",
        "model_id": "x-ai/grok-3-mini",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "high"
        }
    },
    {
        "config_id": "grok-3-mini-beta-cot-low-effort",
        "type": "openrouter",
        "model_id": "x-ai/grok-3-mini",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "low"
        }
    },
    {
        "config_id": "grok-3-cot",
        "type": "openrouter",
        "model_id": "x-ai/grok-3",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 128000
        }
    },
    {
        "config_id": "claude-opus-4.1-cot",
        "type": "openrouter",
        "model_id": "anthropic/claude-opus-4.1",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "max_tokens": 32000
        }
    },
    {
        "config_id": "qwen3-32b-cot",
        "type": "openrouter",
        "model_id": "qwen/qwen3-32b",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 40960
        }
    },
    {
        "config_id": "kimi-k2-cot",
        "type": "openrouter",
        "model_id": "moonshotai/kimi-k2",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },

    
    {
        "config_id": "palmyra-fin-cot",
        "type": "openrouter",
        "model_id": "palmyra-fin",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5
        }
    },
    {
        "config_id": "deepseek-r1-cot",
        "type": "openrouter",
        "model_id": "deepseek/deepseek-chat-v3.1", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-4-maverick",
        "type": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-4-scout",
        "type": "openrouter",
        "model_id": "meta-llama/llama-4-scout",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-guard-4",
        "type": "openrouter",
        "model_id": "meta-llama/llama-guard-4-12b",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 128
        }
    },
    {
        "config_id": "groq-llama3.3-70b-cot",
        "type": "openrouter",
        "model_id": "meta-llama/llama-3.3-8b-instruct:free",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-llama3.1-8b-instant-cot",
        "type": "openrouter",
        "model_id": "meta-llama/llama-3.3-8b-instruct:free",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "gpt-oss-20b-cot",
        "type": "openrouter",
        "model_id": "openai/gpt-oss-20b",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 65536
        }
    },
    {
        "config_id": "gpt-oss-120b-cot",
        "type": "openrouter",
        "model_id": "openai/gpt-oss-120b",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 65536
        }
    },
    {
        "config_id": "grok-4-cot",
        "type": "openrouter",
        "model_id": "x-ai/grok-4",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 128000
        }
    }
] 