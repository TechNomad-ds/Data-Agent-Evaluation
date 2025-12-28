"""
Script to process JSON files for new models and generate comprehensive performance metrics.
This script reads the evaluated_results JSON files for specific new models and calculates
performance metrics that match the format expected by the main analysis script.

New models to process: gpt-5, gpt-5-nano, gpt-5-mini, grok-4, kimi-k2, qwen3-32b, 
gpt-oss-20b, gpt-oss-120b, claude-opus-4.1
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NEW_MODELS = [
    "gpt-5", "gpt-5-nano", "gpt-5-mini", "grok-4", "kimi-k2", 
    "qwen3-32b", "gpt-oss-20b", "gpt-oss-120b", "claude-opus-4.1"
]

STRATEGY_MAPPING = {
    "default_essay": "Default Essay (Single Pass)",
    "self_consistency_essay_n3": "Self-Consistency Essay (N=3 samples)",
    "self_consistency_essay_n5": "Self-Consistency Essay (N=5 samples)",
    "self_discover_essay": "Self-Discover Essay"
}

def extract_strategy_from_filename(filename):
    """Extract strategy name from JSON filename."""
    if "self_consistency_essay_n3" in filename:
        return "Self-Consistency Essay (N=3 samples)"
    elif "self_consistency_essay_n5" in filename:
        return "Self-Consistency Essay (N=5 samples)"
    elif "self_discover_essay" in filename:
        return "Self-Discover Essay"
    elif "default_essay" in filename:
        return "Default Essay (Single Pass)"
    else:
        return "Unknown Strategy"

def calculate_metrics_from_json(json_data):
    """Calculate comprehensive metrics from JSON data."""
    results = []
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    for entry in json_data:
        model_id = entry.get("config_id", "unknown")
        strategy = extract_strategy_from_filename(str(entry.get("strategy", "")))
        
        if model_id not in NEW_MODELS:
            continue
            
        cosine_sim = entry.get("cosine_similarity", np.nan)
        self_grade = entry.get("self_grade_score", 0)
        rouge_precision = entry.get("rouge_l_precision", np.nan)
        rouge_recall = entry.get("rouge_l_recall", np.nan)
        rouge_f1 = entry.get("rouge_l_f1measure", np.nan)
        
        latency_ms = entry.get("response_time", 0) * 1000 if entry.get("response_time") else 0
        input_tokens = entry.get("input_tokens", 0) if entry.get("input_tokens") is not None else 0
        output_tokens = entry.get("output_tokens", 0) if entry.get("output_tokens") is not None else 0
        
        api_cost = calculate_api_cost(model_id, input_tokens, output_tokens)
        
        answer_length = len(entry.get("cleaned_llm_answer", ""))
        
        error = entry.get("error", "")
        if error == "null" or not error:
            error = ""
            
        result = {
            "model": model_id,
            "strategy": strategy,
            "run_timestamp": current_time,
            "question_id": entry.get("position_in_file", "unknown"),
            "cosine_similarity": cosine_sim,
            "self_grade_score": self_grade,
            "rouge_l_precision": rouge_precision,
            "rouge_l_recall": rouge_recall,
            "rouge_l_f1measure": rouge_f1,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "api_cost": api_cost,
            "answer_length": answer_length,
            "error": error
        }
        
        results.append(result)
    
    return results

def calculate_api_cost(model_id, input_tokens, output_tokens):
    """Calculate API cost based on model and token usage."""
    pricing = {
        "gpt-5": {"input": 0.00001, "output": 0.00003},
        "gpt-5-nano": {"input": 0.000005, "output": 0.000015},
        "gpt-5-mini": {"input": 0.0000075, "output": 0.0000225},
        "grok-4": {"input": 0.00001, "output": 0.00003},
        "kimi-k2": {"input": 0.000008, "output": 0.000024},
        "qwen3-32b": {"input": 0.000006, "output": 0.000018},
        "gpt-oss-20b": {"input": 0.000004, "output": 0.000012},
        "gpt-oss-120b": {"input": 0.00002, "output": 0.00006},
        "claude-opus-4.1": {"input": 0.000015, "output": 0.000075}
    }
    
    if model_id in pricing:
        input_cost = input_tokens * pricing[model_id]["input"]
        output_cost = output_tokens * pricing[model_id]["output"]
        return input_cost + output_cost
    else:
        return 0.0

def find_json_files(base_path, models):
    """Find all relevant JSON files for the specified models."""
    json_files = []
    base_path = Path(base_path)
    
    search_patterns = [
        "*/evaluated_results_*__*.json"
    ]
    
    for pattern in search_patterns:
        files = list(base_path.glob(pattern))
        for file in files:
            filename = file.name
            for model in models:
                if f"_{model}__" in filename or f"_{model}_" in filename:
                    json_files.append(file)
                    break
    
    return list(set(json_files))

def load_json_file(filepath):
    """Load and parse JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {filepath}: {e}")
        return None

def calculate_model_strategy_performance(df):
    """Calculate comprehensive performance metrics for each model-strategy combination."""
    if df.empty:
        return pd.DataFrame()
    
    df['successful_run'] = df['error'].isna() | (df['error'] == '') | (df['error'] == 'nan')
    successful_df = df[df['successful_run']].copy()
    
    if successful_df.empty:
        logging.warning("No successful runs found in the processed data.")
        return pd.DataFrame()
    
    results = []
    
    for (model, strategy), group in successful_df.groupby(['model', 'strategy']):
        cosine_mean = group['cosine_similarity'].mean()
        cosine_std = group['cosine_similarity'].std()
        cosine_count = group['cosine_similarity'].count()
        
        rouge_mean = group['rouge_l_f1measure'].mean()
        rouge_std = group['rouge_l_f1measure'].std()
        rouge_count = group['rouge_l_f1measure'].count()
        
        self_grade_sum = group['self_grade_score'].sum()
        self_grade_mean = group['self_grade_score'].mean()
        self_grade_std = group['self_grade_score'].std()
        self_grade_count = group['self_grade_score'].count()
        
        latency_mean = group['latency_ms'].mean()
        latency_std = group['latency_ms'].std()
        
        api_cost_mean = group['api_cost'].mean()
        api_cost_std = group['api_cost'].std()
        api_cost_sum = group['api_cost'].sum()
        
        input_tokens_mean = group['input_tokens'].mean()
        output_tokens_mean = group['output_tokens'].mean()
        
        answer_length_mean = group['answer_length'].mean()
        
        total_runs = len(df[(df['model'] == model) & (df['strategy'] == strategy)])
        success_rate = len(group) / total_runs if total_runs > 0 else 0
        
        result = {
            'model': model,
            'strategy': strategy,
            'total_runs': total_runs,
            'successful_runs': len(group),
            'success_rate': success_rate,
            
            'cosine_similarity_mean': cosine_mean,
            'cosine_similarity_std': cosine_std,
            'cosine_similarity_count': cosine_count,
            
            'rouge_l_f1measure_mean': rouge_mean,
            'rouge_l_f1measure_std': rouge_std,
            'rouge_l_f1measure_count': rouge_count,
            
            'self_grade_score_sum': self_grade_sum,
            'self_grade_score_mean': self_grade_mean,
            'self_grade_score_std': self_grade_std,
            'self_grade_score_count': self_grade_count,
            'self_grade_score_normalized': (self_grade_sum / 149) * 100,
            
            'latency_ms_mean': latency_mean,
            'latency_ms_std': latency_std,
            
            'api_cost_mean': api_cost_mean,
            'api_cost_std': api_cost_std,
            'api_cost_sum': api_cost_sum,
            
            'input_tokens_mean': input_tokens_mean,
            'output_tokens_mean': output_tokens_mean,
            'answer_length_mean': answer_length_mean
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Process JSON files for new models and generate performance metrics")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), 
                       help="Path to results directory containing JSON files")
    parser.add_argument("--output_file", type=Path, default=Path("new_models_performance.csv"),
                       help="Output CSV file path")
    parser.add_argument("--raw_output_file", type=Path, default=Path("new_models_raw_data.csv"),
                       help="Output CSV file for raw data (matching main CSV format)")
    
    args = parser.parse_args()
    
    logging.info(f"Processing new models: {', '.join(NEW_MODELS)}")
    
    json_files = find_json_files(args.results_dir, NEW_MODELS)
    logging.info(f"Found {len(json_files)} JSON files to process")
    
    if not json_files:
        logging.error("No JSON files found for the specified models")
        return
    
    all_results = []
    
    for json_file in json_files:
        logging.info(f"Processing: {json_file}")
        json_data = load_json_file(json_file)
        
        if json_data is None:
            continue
            
        file_results = calculate_metrics_from_json(json_data)
        all_results.extend(file_results)
    
    if not all_results:
        logging.error("No results extracted from JSON files")
        return
    
    df_raw = pd.DataFrame(all_results)
    logging.info(f"Processed {len(df_raw)} total entries")
    
    df_raw.to_csv(args.raw_output_file, index=False)
    logging.info(f"Saved raw data to: {args.raw_output_file}")
    
    df_performance = calculate_model_strategy_performance(df_raw)
    
    if df_performance.empty:
        logging.error("No performance metrics calculated")
        return
    
    df_performance.to_csv(args.output_file, index=False)
    logging.info(f"Saved performance metrics to: {args.output_file}")
    
    print("\nProcessing Summary:")
    print(f"Total entries processed: {len(df_raw)}")
    print(f"Unique models found: {df_raw['model'].nunique()}")
    print(f"Unique strategies found: {df_raw['strategy'].nunique()}")
    print(f"Model-strategy combinations: {len(df_performance)}")
    
    print("\nModels processed:")
    for model in sorted(df_raw['model'].unique()):
        count = len(df_raw[df_raw['model'] == model])
        success_count = len(df_raw[(df_raw['model'] == model) & (df_raw['error'] == '')])
        print(f"  {model}: {count} total entries ({success_count} successful)")
    
    print("\nStrategies found:")
    for strategy in sorted(df_raw['strategy'].unique()):
        count = len(df_raw[df_raw['strategy'] == strategy])
        print(f"  {strategy}: {count} entries")

if __name__ == "__main__":
    main()