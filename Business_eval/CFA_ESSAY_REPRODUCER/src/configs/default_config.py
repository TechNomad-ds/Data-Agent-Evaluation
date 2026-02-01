"""
Default model configurations for general essay generation strategies.
"""

ALL_MODEL_CONFIGS = [
    
    {
    "config_id": "claude-3.7-sonnet",
    "type": "openrouter",
    "model_id": "anthropic/claude-3-7-sonnet-20250219",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.999,
        "top_k": 250,
        "max_tokens": 64000
    }
    },
    {
        "config_id": "claude-3.5-sonnet",
        "type": "openrouter",
        "model_id": "anthropic/claude-3-5-sonnet-20241022",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "claude-3.5-haiku",
        "type": "openrouter",
        "model_id": "anthropic/claude-3-5-haiku-20241022",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "claude-sonnet-4",
        "type": "openrouter",
        "model_id": "anthropic/claude-sonnet-4",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "max_tokens": 64000
        }
    },
    {
        "config_id": "mistral-large-official",
        "type": "openrouter",
        "model_id": "mistral-large-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096
        }
    },
    {
        "config_id": "codestral-latest-official",
        "type": "openrouter",
        "model_id": "codestral-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "palmyra-fin-default",
        "type": "openrouter",
        "model_id": "palmyra-fin",
        "parameters": {
            "temperature": 0.0,
            "max_tokens": 4096
        }
    },
    {
        "config_id": "gpt-4o",
        "type": "openrouter",
        "model_id": "gpt-4o",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 8192
        }
    },
    
    {
        "config_id": "o3-mini",
        "type": "openrouter",
        "model_id": "openai/o3-mini",
        "parameters": {
            "temperature": 1.0,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "o4-mini",
        "type": "openrouter",
        "model_id": "openai/o4-mini",
        "parameters": {
            "temperature": 1.0,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "gpt-4.1",
        "type": "openrouter",
        "model_id": "gpt-4.1",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-mini",
        "type": "openrouter",
        "model_id": "openai/gpt-4.1-mini",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-nano",
        "type": "openrouter",
        "model_id": "openai/gpt-4.1-nano",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    
    {
        "config_id": "grok-3",
        "type": "openrouter",
        "model_id": "x-ai/grok-3",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 16384
        }
    },
    {
        "config_id": "grok-3-mini-beta-high-effort",
        "type": "openrouter",
        "model_id": "x-ai/grok-3-mini",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "high",
            "max_tokens": 8192
        }
    },
    {
        "config_id": "grok-3-mini-beta-low-effort",
        "type": "openrouter",
        "model_id": "x-ai/grok-3-mini",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "low",
            "max_tokens": 8192
        }
    },
    {
    "config_id": "gemini-2.5-pro",
    "type": "openrouter",
    "model_id": "google/gemini-2.5-pro-preview-05-06",
    "parameters": {
        "top_p": 0.95,
        "top_k": 64,
        "thinking_budget": 24576,
    }
    },
    {
    "config_id": "gemini-2.5-flash",
    "type": "openrouter",
    "model_id": "google/gemini-2.5-flash-preview-04-17",
    "parameters": {
        "top_p": 0.95,
        "top_k": 64,
        "thinking_budget": 24576,
    }
    },
    {
        "config_id": "deepseek-r1",
        "type": "openrouter",
        "model_id": "deepseek/deepseek-chat-v3.1",
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
        "config_id": "groq-llama3.3-70b",
        "type": "openrouter",
        "model_id": "meta-llama/llama-3.3-8b-instruct:free",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-llama3.1-8b-instant",
        "type": "openrouter",
        "model_id": "meta-llama/llama-3.3-8b-instruct:free",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32000
        }
    },
    {
        "config_id": "claude-opus-4.1",
        "type": "openrouter",
        "model_id": "anthropic/claude-opus-4.1",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "max_tokens": 40960
        }
    },
    {
        "config_id": "qwen3-32b",
        "type": "openrouter",
        "model_id": "qwen/qwen3-32b",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "kimi-k2",
        "type": "openrouter",
        "model_id": "moonshotai/kimi-k2",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-oss-20b",
        "type": "openrouter",
        "model_id": "openai/gpt-oss-20b",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 65536
        }
    },
    {
        "config_id": "gpt-oss-120b",
        "type": "openrouter", 
        "model_id": "openai/gpt-oss-120b",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 65536
        }
    },
    {
        "config_id": "grok-4",
        "type": "openrouter",
        "model_id": "x-ai/grok-4",
        "parameters": {
            "temperature": 0.1
        }
    },
    {
        "config_id": "gpt-5",
        "type": "openrouter",
        "model_id": "openai/gpt-5",
        "parameters": {
            "max_tokens": 128000
        }
    },
    {
        "config_id": "gpt-5-nano",
        "type": "openrouter",
        "model_id": "openai/gpt-5-nano",
        "parameters": {
            "max_tokens": 128000
        }
    },
    {
        "config_id": "gpt-5-mini",
        "type": "openrouter",
        "model_id": "openai/gpt-5-mini",
        "parameters": {
            "max_tokens": 128000
        }
    },
] 