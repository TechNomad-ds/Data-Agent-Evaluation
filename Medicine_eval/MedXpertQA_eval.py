import os
import json
import re
import subprocess
import datetime
import argparse
from pathlib import Path
from collections import defaultdict

# --- 路径自动校准 ---
BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "MedXpertQA" / "eval"

# 测评模式定义
EVAL_MODES = [
    ("zero_shot", "ao"),
    ("zero_shot", "cot")
]

def clean_r1_answer(text):
    """同步 eval.ipynb 中的 R1 答案提取逻辑"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # 1. 剔除思考过程
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 寻找最后的选项字母 (A/B/C/D)
    # 匹配模式：选项前可能有空格、换行或"The answer is"
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1].upper()
    return ""

def calculate_detailed_metrics(result_file):
    if not result_file.exists():
        return "N/A"
    
    # 统计字典
    stats = {
        "Overall": {"correct": 0, "total": 0},
        "Reasoning": {"correct": 0, "total": 0},
        "Understanding": {"correct": 0, "total": 0}
    }
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # 1. 提取预测值（处理列表情况）
                pred_raw = data.get("prediction", "")
                if isinstance(pred_raw, list) and len(pred_raw) > 0:
                    pred = str(pred_raw[0]).upper() # 拿到 "F"
                else:
                    pred = str(pred_raw).upper()

                # 2. 提取标准答案
                gold = data.get("label", "")
                if isinstance(gold, list) and len(gold) > 0:
                    gold = str(gold[0]).upper()
                else:
                    gold = str(gold).upper()
                
                # 3. 获取题目维度 (如果字段不存在，从 ID 尝试判断)
                q_type = data.get("question_type")
                if not q_type:
                    # 某些版本根据 ID 前缀判断，这里默认归类到 Overall
                    q_type = "Other"

                # 4. 判定对错 (优先使用文件自带的 'correct' 字段，更准)
                is_correct = data.get("correct")
                if is_correct is None: # 如果没这个字段，就手动比对
                    is_correct = (pred == gold and gold != "")

                # 5. 累加统计
                stats["Overall"]["total"] += 1
                if is_correct: stats["Overall"]["correct"] += 1
                
                if q_type in stats:
                    stats[q_type]["total"] += 1
                    if is_correct: stats[q_type]["correct"] += 1
        
        # 格式化输出
        res_parts = []
        for cat in ["Overall", "Reasoning", "Understanding"]:
            s = stats[cat]
            if s["total"] > 0:
                acc = (s["correct"] / s["total"]) * 100
                res_parts.append(f"{cat}: {acc:.2f}%({s['correct']}/{s['total']})")
        return " | ".join(res_parts) if res_parts else "No valid data found"

    except Exception as e:
        return f"Error: {str(e)}"
    
def run_experiment(args):
    main_py = EVAL_DIR / "main.py"
    if not main_py.exists():
        print(f"❌ 路径错误: 找不到 {main_py}")
        return

    summary_report = []

    for method, p_type in EVAL_MODES:
        mode_name = f"{method}_{p_type}"
        print(f"\n🚀 [运行模式] {mode_name}")
        
        cmd = [
            "python3", "main.py",
            "--model", args.model,
            "--dataset", args.dataset,
            "--task", args.task,
            "--method", method,
            "--prompting-type", p_type,
            "--output-dir", args.output_dir,
            "--num-threads", str(args.threads),
            "--temperature", "0.1",
            "--max-samples", str(args.max_samples),
            
        ]

        # 执行
        process = subprocess.Popen(cmd, cwd=str(EVAL_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            if "Completed" in line or "INFO" in line:
                print(f"  {line.strip()}")
        process.wait()

        # 匹配路径 (使用你提供的实际路径逻辑)
        result_file = (
            EVAL_DIR / "outputs" / args.output_dir / args.model / 
            args.dataset / method / p_type / f"{args.dataset}_{args.task}_output.jsonl"
        )
        
        # 获取多维度评分
        metrics_str = calculate_detailed_metrics(result_file)
        summary_report.append((mode_name, metrics_str or "N/A"))
        print(f"🏁 {mode_name} 结果: {metrics_str}")

    # 打印最终总表
    print("\n" + "="*100)
    print(f"📊 MedXpertQA 多维度评测报告 (模型: {args.model})")
    print("-" * 100)
    print(f"{'测评模式':<20} | {'各维度准确率 (Accuracy / Correct / Total)':<60}")
    print("-" * 100)
    for mode, scores in summary_report:
        print(f"{mode:<20} | {scores}")
    print("="*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-r1")
    parser.add_argument("--dataset", default="medxpertqa_sampled")
    parser.add_argument("--task", default="text")
    parser.add_argument("--output_dir", default="dev")
    parser.add_argument("--threads", default=10, type=int)
    parser.add_argument("--max-samples", default=-1, type=int)
    args = parser.parse_args()
    
    run_experiment(args)