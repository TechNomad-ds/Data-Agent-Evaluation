import json
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="MedCaseReasoning 评分脚本")
    parser.add_argument("--input_path", type=str, default="./results/medcase_results.jsonl", help="推理结果文件路径")
    parser.add_argument("--output_path", type=str, default="./results/medcase_evaluated.jsonl", help="评分保存路径")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="担任裁判的模型(建议用强模型)")
    parser.add_argument("--api_key", type=str, default="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    parser.add_argument("--base_url", type=str, default="http://123.129.219.111:3000/v1")
    parser.add_argument("--threads", type=int, default=5)
    return parser.parse_args()

# --- 论文中的 Prompt 模板 ---

# Prompt 7: 诊断准确率裁判
ACCURACY_PROMPT = "Is our predicted diagnosis correct (y/n)?\nPredicted diagnosis: {pred}, True diagnosis: {true}\nAnswer [y/n]."


REASONING_MATCH_PROMPT = """Analyze if the model's reasoning covers the following clinician's point.
Clinician's point: {gold_point}
Model's reasoning: {model_reasoning}
Does the model mention or imply this specific point? Answer [y/n]."""

args = parse_args()
client = OpenAI(api_key=args.api_key, base_url=args.base_url)

def extract_tag(text, tag):
    """提取 <think> 或 <answer> 标签内容"""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def get_llm_decision(prompt):
    """调用 LLM 获取 y/n 判断"""
    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        ans = response.choices[0].message.content.lower()
        return 'y' in ans
    except:
        return False

def evaluate_item(item):
    llm_output = item.get("llm_response", "")
    pred_diag = extract_tag(llm_output, "answer")
    model_think = extract_tag(llm_output, "think")
    true_diag = item.get("final_diagnosis", "")
    gold_reasoning = item.get("diagnostic_reasoning", "")

    # 1. 评测准确率 (Accuracy)
    acc_prompt = ACCURACY_PROMPT.format(pred=pred_diag, true=true_diag)
    is_correct = get_llm_decision(acc_prompt)

    # 2. 评测推理召回率 (Reasoning Recall)
    # 将金标准推理按数字编号拆分成列表
    gold_points = re.split(r'\d+\.', gold_reasoning)
    gold_points = [p.strip() for p in gold_points if p.strip()]
    
    hits = 0
    if gold_points:
        for point in gold_points:
            match_prompt = REASONING_MATCH_PROMPT.format(gold_point=point, model_reasoning=model_think)
            if get_llm_decision(match_prompt):
                hits += 1
        recall = hits / len(gold_points)
    else:
        recall = 0.0

    return {
        "pmcid": item.get("pmcid"),
        "is_correct": is_correct,
        "recall": recall,
        "num_points": len(gold_points)
    }

def main():
    if not os.path.exists(args.input_path):
        print(f"找不到推理文件: {args.input_path}")
        return

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"正在使用 {args.model_name} 评估 {len(data)} 条数据...")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(tqdm(executor.map(evaluate_item, data), total=len(data)))

    # 统计最终结果
    total = len(results)
    avg_accuracy = sum(1 for r in results if r['is_correct']) / total
    avg_recall = sum(r['recall'] for r in results) / total

    print("\n" + "="*30)
    print(f"评测报告 ({args.model_name} 裁判)")
    print("-" * 30)
    print(f"样本总量: {total}")
    print(f"诊断准确率 (Accuracy): {avg_accuracy:.2%}")
    print(f"推理召回率 (Reasoning Recall): {avg_recall:.2%}")
    print("="*30)

    # 保存明细
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()