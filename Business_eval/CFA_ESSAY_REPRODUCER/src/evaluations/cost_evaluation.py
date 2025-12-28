import logging
from typing import List, Dict, Any, Optional
from ..configs.model_pricing import get_pricing

logger = logging.getLogger(__name__)

def calculate_model_cost(results_data: List[Dict[str, Any]], model_config_item: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the estimated cost for API calls based on token usage.
    Since all models now go through OpenRouter, pricing is standardized.

    Args:
        results_data: A list of dictionaries, where each dictionary contains
                      results from a single LLM call, potentially including
                      'input_tokens' and 'output_tokens'.
        model_config_item: The configuration dictionary for the model run,
                           used to check model type and ID.

    Returns:
        A dictionary containing the total estimated cost, e.g., {'total_cost': 0.123}.
        Returns {'total_cost': 0.0} if pricing is missing or token data is unavailable.
    """
    total_cost = 0.0
    model_type = model_config_item.get("type")
    model_id = model_config_item.get("model_id", "")
    config_id_used = model_config_item.get("config_id", model_id)

    if not model_id:
         logger.error("Missing model_id from model_config_item for cost calculation.")
         return {"total_cost": 0.0}
         
    try:
        pricing = get_pricing(model_type, model_id)
    except ValueError as e:
        logger.warning(f"Skipping cost calculation for run using model ID '{model_id}' (type: {model_type}): {e}")
        return {"total_cost": 0.0}

    price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
    price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    
    model_parameters = model_config_item.get("parameters", {})
    if model_type == "gemini" and model_parameters.get("thinking_budget", 0) > 0:
        thinking_price = pricing.get("completion_tokens_cost_per_million_thinking")
        if thinking_price:
            price_per_output_token = thinking_price / 1_000_000
            logger.info(f"Using thinking-specific output token pricing for Gemini model {model_id}.")

    if price_per_input_token == 0.0 and price_per_output_token == 0.0:
        logger.warning(f"Both input and output token prices are zero for model '{model_id}' (type: {model_type}). Cost will be zero.")
        return {"total_cost": 0.0}
    
    num_items_missing_tokens = 0
    for item in results_data:
        input_tokens = item.get('input_tokens')
        output_tokens = item.get('output_tokens')

        if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
            item_cost = (input_tokens * price_per_input_token) + (output_tokens * price_per_output_token)
            total_cost += item_cost
        else:
            num_items_missing_tokens += 1
            

    if num_items_missing_tokens > 0 and len(results_data) > 0: 
        logger.warning(f"Cost calculation for {model_id} (type: {model_type}) might be incomplete. Missing token data for {num_items_missing_tokens}/{len(results_data)} items.")

    logger.info(f"Estimated cost for {model_type} model {model_id}: ${total_cost:.6f}")
    return {"total_cost": total_cost} 


def calculate_total_cost_from_aggregated_tokens(
    total_input_tokens: Optional[int],
    total_output_tokens: Optional[int],
    model_id_for_pricing: str,
    model_type_for_pricing: str,
    model_parameters_for_pricing: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Calculates the estimated total cost based on aggregated input and output token counts.
    Since all models now go through OpenRouter, pricing is standardized.

    Args:
        total_input_tokens: Total number of input tokens for the run.
        total_output_tokens: Total number of output tokens for the run.
        model_id_for_pricing: The specific model ID to use for fetching pricing.
        model_type_for_pricing: The type of the model (e.g., 'openai', 'anthropic').
        model_parameters_for_pricing: Optional model parameters for pricing calculation.

    Returns:
        A dictionary containing the total estimated cost, e.g., {'total_cost': 0.123}.
        Returns {'total_cost': 0.0} if pricing is missing or token data is incomplete.
    """
    if total_input_tokens is None or total_output_tokens is None:
        logger.warning(f"Cannot calculate cost for {model_id_for_pricing}: Missing aggregated token counts.")
        return {"total_cost": 0.0}

    try:
        pricing = get_pricing(model_type_for_pricing, model_id_for_pricing)
    except ValueError as e:
        logger.warning(f"Skipping cost calculation for model ID '{model_id_for_pricing}' (type: {model_type_for_pricing}): {e}")
        return {"total_cost": 0.0}

    price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
    price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    
    parameters = model_parameters_for_pricing if model_parameters_for_pricing is not None else {}
    if model_type_for_pricing == "gemini" and parameters.get("thinking_budget", 0) > 0:
        thinking_price = pricing.get("completion_tokens_cost_per_million_thinking")
        if thinking_price:
            price_per_output_token = thinking_price / 1_000_000
            logger.info(f"Using thinking-specific output token pricing for Gemini model {model_id_for_pricing} (Aggregated).")

    if price_per_input_token == 0.0 and price_per_output_token == 0.0:
        logger.warning(f"Both input and output token prices are zero for model '{model_id_for_pricing}' (type: {model_type_for_pricing}). Cost will be zero.")
    
    cost = (total_input_tokens * price_per_input_token) + (total_output_tokens * price_per_output_token)
    
    logger.info(f"Calculated total cost for {model_type_for_pricing} model {model_id_for_pricing} (Aggregated): ${cost:.6f}")
    return {"total_cost": cost}