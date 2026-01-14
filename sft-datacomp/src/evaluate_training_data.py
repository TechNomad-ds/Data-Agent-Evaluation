#!/usr/bin/env python3
"""
Evaluation script for training data construction task.
This script validates the quality and format of the generated training data.
"""

import argparse
import json
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import sys

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


def validate_jsonl_format(jsonl_path: str):
    """
    Validate that the JSONL file has correct format.
    Returns: (is_valid, error_message, num_examples)
    """
    if not os.path.exists(jsonl_path):
        return False, f"File does not exist: {jsonl_path}", 0
    
    num_examples = 0
    required_fields = {'instruction', 'response'}
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        return False, f"Line {line_num}: Expected JSON object, got {type(data).__name__}", num_examples
                    
                    # Check required fields
                    if not required_fields.issubset(data.keys()):
                        missing = required_fields - set(data.keys())
                        return False, f"Line {line_num}: Missing required fields: {missing}", num_examples
                    
                    # Check field types
                    if not isinstance(data['instruction'], str):
                        return False, f"Line {line_num}: 'instruction' must be a string", num_examples
                    if not isinstance(data['response'], str):
                        return False, f"Line {line_num}: 'response' must be a string", num_examples
                    
                    # Check non-empty
                    if not data['instruction'].strip():
                        return False, f"Line {line_num}: 'instruction' cannot be empty", num_examples
                    if not data['response'].strip():
                        return False, f"Line {line_num}: 'response' cannot be empty", num_examples
                    
                    num_examples += 1
                except json.JSONDecodeError as e:
                    return False, f"Line {line_num}: Invalid JSON: {str(e)}", num_examples
        
        if num_examples == 0:
            return False, "JSONL file is empty or contains no valid examples", 0
        
        return True, "", num_examples
    except Exception as e:
        return False, f"Error reading file: {str(e)}", num_examples


def analyze_training_data(jsonl_path: str) -> Dict[str, Any]:
    """
    Analyze the training data for quality metrics.
    """
    metrics = {
        'total_examples': 0,
        'avg_instruction_length': 0.0,
        'avg_response_length': 0.0,
        'min_instruction_length': float('inf'),
        'max_instruction_length': 0,
        'min_response_length': float('inf'),
        'max_response_length': 0,
    }
    
    total_instruction_length = 0
    total_response_length = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                instruction_len = len(data['instruction'])
                response_len = len(data['response'])
                
                metrics['total_examples'] += 1
                total_instruction_length += instruction_len
                total_response_length += response_len
                
                metrics['min_instruction_length'] = min(metrics['min_instruction_length'], instruction_len)
                metrics['max_instruction_length'] = max(metrics['max_instruction_length'], instruction_len)
                metrics['min_response_length'] = min(metrics['min_response_length'], response_len)
                metrics['max_response_length'] = max(metrics['max_response_length'], response_len)
        
        if metrics['total_examples'] > 0:
            metrics['avg_instruction_length'] = total_instruction_length / metrics['total_examples']
            metrics['avg_response_length'] = total_response_length / metrics['total_examples']
        
        if metrics['min_instruction_length'] == float('inf'):
            metrics['min_instruction_length'] = 0
        if metrics['min_response_length'] == float('inf'):
            metrics['min_response_length'] = 0
    
    except Exception as e:
        print(f"Error analyzing data: {e}", file=sys.stderr)
    
    return metrics


def find_training_data_files(task_dir: str) -> List[str]:
    """
    Find all JSONL files in the training_data directory.
    """
    training_data_dir = os.path.join(task_dir, 'training_data')
    jsonl_files = []
    
    if os.path.exists(training_data_dir):
        for root, dirs, files in os.walk(training_data_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
    
    return sorted(jsonl_files)


def _normalize_text(s: str) -> str:
    """
    Normalize text for dedup:
    - lower
    - collapse whitespace
    - remove some punctuation noise
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"'`]", "", s)
    s = re.sub(r"[，。,\.!?！？；;：:\(\)\[\]\{\}<>《》]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def compute_dedup_metrics(jsonl_path: str,
                          near_threshold: int = 95,
                          max_bucket_size: int = 80,
                          max_pairs_per_bucket: int = 4000) -> Dict[str, Any]:
    """
    Compute dedup metrics:
    - exact dup rates (pair/instruction/response)
    - near-dup estimated rate using bucketing + rapidfuzz if available

    Notes:
    - near-dup needs rapidfuzz; if unavailable, we only output exact metrics.
    - Bucketing avoids O(n^2) explosion; still provides a useful estimate.
    """
    pairs = []
    insts = []
    resps = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inst = obj["instruction"]
            resp = obj["response"]

            n_inst = _normalize_text(inst)
            n_resp = _normalize_text(resp)
            n_pair = f"i:{n_inst}\nr:{n_resp}"

            insts.append(_sha1(n_inst))
            resps.append(_sha1(n_resp))
            pairs.append(_sha1(n_pair))

    n = len(pairs)
    if n == 0:
        return {
            "dedup_total_examples": 0,
            "dedup_exact_pair_unique": 0,
            "dedup_exact_pair_dup_rate": 0.0,
            "dedup_exact_instruction_unique": 0,
            "dedup_exact_instruction_dup_rate": 0.0,
            "dedup_exact_response_unique": 0,
            "dedup_exact_response_dup_rate": 0.0,
            "dedup_near_pair_dup_rate_est": None,
            "dedup_near_threshold": near_threshold,
            "dedup_near_available": bool(fuzz),
        }

    exact_pair_unique = len(set(pairs))
    exact_inst_unique = len(set(insts))
    exact_resp_unique = len(set(resps))

    exact_pair_dup_rate = 1.0 - (exact_pair_unique / n)
    exact_inst_dup_rate = 1.0 - (exact_inst_unique / n)
    exact_resp_dup_rate = 1.0 - (exact_resp_unique / n)

    result = {
        "dedup_total_examples": n,
        "dedup_exact_pair_unique": exact_pair_unique,
        "dedup_exact_pair_dup_rate": exact_pair_dup_rate,
        "dedup_exact_instruction_unique": exact_inst_unique,
        "dedup_exact_instruction_dup_rate": exact_inst_dup_rate,
        "dedup_exact_response_unique": exact_resp_unique,
        "dedup_exact_response_dup_rate": exact_resp_dup_rate,
        "dedup_near_threshold": near_threshold,
        "dedup_near_available": bool(fuzz),
    }

    # ---- near-dup estimate ----
    if fuzz is None:
        result["dedup_near_pair_dup_rate_est"] = None
        result["dedup_near_reason"] = "rapidfuzz not available"
        return result

    # We need actual normalized strings again for similarity comparisons
    norm_pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_inst = _normalize_text(obj["instruction"])
            n_resp = _normalize_text(obj["response"])
            norm_pairs.append(f"i:{n_inst}\nr:{n_resp}")

    # Bucketing by prefix hash (first 80 chars) to find likely near-duplicates
    buckets: Dict[str, List[int]] = {}
    for idx, p in enumerate(norm_pairs):
        key_seed = p[:80]  # prefix
        key = _sha1(key_seed)[:8]
        buckets.setdefault(key, []).append(idx)

    # Compare within buckets (capped)
    dup_flags = [False] * n
    comparisons = 0
    near_dups_found = 0

    for _, idxs in buckets.items():
        if len(idxs) <= 1:
            continue
        # limit bucket size
        if len(idxs) > max_bucket_size:
            idxs = idxs[:max_bucket_size]

        # pairwise within bucket, but cap comparisons per bucket
        local_comp = 0
        for i in range(len(idxs)):
            if local_comp >= max_pairs_per_bucket:
                break
            a = idxs[i]
            if dup_flags[a]:
                continue
            for j in range(i + 1, len(idxs)):
                if local_comp >= max_pairs_per_bucket:
                    break
                b = idxs[j]
                if dup_flags[b]:
                    continue
                # fuzz ratio on entire pair text
                score = fuzz.ratio(norm_pairs[a], norm_pairs[b])
                comparisons += 1
                local_comp += 1
                if score >= near_threshold:
                    # mark b as near-dup of a
                    dup_flags[b] = True
                    near_dups_found += 1

    # near dup rate estimate among compared-candidate space:
    # Here we treat "flagged as near-dup" / total examples as an estimate.
    # It may undercount if duplicates land in different buckets.
    near_dup_rate_est = near_dups_found / n

    result["dedup_near_pair_dup_rate_est"] = near_dup_rate_est
    result["dedup_near_comparisons"] = comparisons
    result["dedup_near_dups_flagged"] = near_dups_found
    result["dedup_near_bucket_count"] = len(buckets)

    return result





def main():
    parser = argparse.ArgumentParser(description='Evaluate training data construction task')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the training data directory (task directory)')
    parser.add_argument('--templates-dir', type=str, default='',
                       help='Templates directory (not used for this task)')
    parser.add_argument('--limit', type=int, default=-1,
                       help='Limit number of examples to validate (not used, validates all)')
    parser.add_argument('--json-output-file', type=str, required=True,
                       help='Path to output JSON metrics file')
    
    args = parser.parse_args()
    
    # For this task, model-path is actually the task directory
    task_dir = args.model_path
    
    # Find all JSONL files in training_data directory
    jsonl_files = find_training_data_files(task_dir)
    
    if not jsonl_files:
        # Check if training_data directory exists but is empty
        training_data_dir = os.path.join(task_dir, 'training_data')
        if not os.path.exists(training_data_dir):
            error_msg = "training_data directory not found"
        else:
            error_msg = "No JSONL files found in training_data directory"
        
        result = {
            'valid': False,
            'error': error_msg,
            'files_found': 0,
            'total_examples': 0,
            'metrics': {}
        }
        
        with open(args.json_output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"ERROR: {error_msg}")
        sys.exit(1)
    
    # Validate all found JSONL files
    all_valid = True
    total_examples = 0
    file_results = []
    
    for jsonl_file in jsonl_files:
        is_valid, error_msg, num_examples = validate_jsonl_format(jsonl_file)
        rel_path = os.path.relpath(jsonl_file, task_dir)
        
        file_metrics = analyze_training_data(jsonl_file) if is_valid else {}
        
        file_result = {
            'file': rel_path,
            'valid': is_valid,
            'num_examples': num_examples,
            'metrics': file_metrics
        }
        
        if error_msg:
            file_result['error'] = error_msg
        
        file_results.append(file_result)
        
        if is_valid:
            total_examples += num_examples
            print(f"✓ {rel_path}: {num_examples} valid examples")
            if file_metrics:
                print(f"  Avg instruction length: {file_metrics['avg_instruction_length']:.1f}")
                print(f"  Avg response length: {file_metrics['avg_response_length']:.1f}")
        else:
            all_valid = False
            print(f"✗ {rel_path}: {error_msg}")
    
    # Overall metrics
    overall_metrics = {
        'files_validated': len(jsonl_files),
        'files_valid': sum(1 for f in file_results if f['valid']),
        'total_examples': total_examples
    }
    
    # Find main file (prefer train.jsonl, otherwise largest file)
    main_file = None
    main_file_path = os.path.join(task_dir, 'training_data', 'train.jsonl')
    if os.path.exists(main_file_path):
        main_file = main_file_path
    else:
        # Use file with most examples
        valid_files = [f for f in file_results if f['valid']]
        if valid_files:
            main_file_result = max(valid_files, key=lambda x: x['num_examples'])
            main_file = os.path.join(task_dir, main_file_result['file'])
    
    if main_file and os.path.exists(main_file):
        overall_metrics.update(analyze_training_data(main_file))

        # Add dedup metrics (exact + near-dup estimate)
        try:
            dedup_metrics = compute_dedup_metrics(main_file, near_threshold=95)
            overall_metrics.update(dedup_metrics)
        except Exception as e:
            overall_metrics["dedup_error"] = str(e)

    
    result = {
        'valid': all_valid,
        'total_examples': total_examples,
        'files': file_results,
        'metrics': overall_metrics
    }
    
    # Save results
    with open(args.json_output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Total files: {len(jsonl_files)}")
    print(f"Valid files: {overall_metrics['files_valid']}")
    print(f"Total examples: {total_examples}")
    print(f"{'='*60}")
    
    if not all_valid:
        sys.exit(1)
    
    print("✓ All training data files are valid!")


if __name__ == '__main__':
    main()
