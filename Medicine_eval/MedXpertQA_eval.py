#!/usr/bin/env python3
"""
MedXpertQA 调用脚本
在 `MedXpertQA/eval` 目录下调用 `main.py`，接受常用参数并保存日志。
"""
import argparse
import subprocess
from pathlib import Path
import datetime

EVAL_DIR = Path("MedXpertQA") / "eval"
OUTPUT_DIR = Path("MedXpertQA_results")


def run_main(model, dataset, task, output_dir, method, prompting_type, temperature):
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"run_{model}_{dataset}_{task}_{timestamp}.log"

    cmd = [
        "python",
        "main.py",
        "--model", model,
        "--dataset", dataset,
        "--task", task,
        "--output-dir", output_dir,
        "--method", method,
        "--prompting-type", prompting_type,
        "--temperature", str(temperature),
    ]

    print(f"Running: {' '.join(cmd)} in {EVAL_DIR}")
    # 强制只做 text 子集测试：如果传入 mm 则覆盖为 text
    if isinstance(task, str) and (',' in task or task.strip().lower() != 'text'):
        # 如果包含多个 task，或不是 'text'，我们只运行 text
        if 'text' in task:
            task = 'text'
        else:
            print(f"⚠️ 任务类型 '{task}' 被检测为非文本，已强制改为 'text'（只跑文本子集）")
            task = 'text'

    with open(log_file, 'w', encoding='utf-8') as f:
        proc = subprocess.run(cmd, cwd=str(EVAL_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(proc.stdout)
        if proc.returncode != 0:
            print(f"process exited with {proc.returncode}; see {log_file}")
        else:
            print(f"Finished successfully. log: {log_file}")


def main():
    parser = argparse.ArgumentParser(description='MedXpertQA 调用脚本')
    parser.add_argument('--model', type=str, default='deepseek-r1', help='模型名或逗号分隔模型列表')
    parser.add_argument('--dataset', type=str, default='medxpertqa', help='数据集名称')
    parser.add_argument('--task', type=str, default='text', help='任务类型: text 或 mm')
    parser.add_argument('--output_dir', type=str, default='dev', help='输出目录标识')
    parser.add_argument('--method', type=str, default='zero_shot', help='inference method')
    parser.add_argument('--prompting_type', type=str, default='cot', help='prompting type: cot/ao/few_shot')
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    run_main(args.model, args.dataset, args.task, args.output_dir, args.method, args.prompting_type, args.temperature)


if __name__ == '__main__':
    main()
