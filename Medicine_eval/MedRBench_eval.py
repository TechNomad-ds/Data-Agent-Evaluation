#!/usr/bin/env python3
"""
MedRBench 自动化调用脚本
运行推理脚本（位于 src/Inference）并运行评估脚本（位于 src/Evaluation）。
"""
import subprocess
import argparse
from pathlib import Path
import time

PROJECT_DIR = Path("MedRBench")
INFERENCE_DIR = PROJECT_DIR / "src" / "Inference"
EVAL_DIR = PROJECT_DIR / "src" / "Evaluation"
RESULT_DIR = Path("MedRBench_results")


def run_cmd(cmd, cwd, log_path=None):
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    try:
        res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("STDOUT:\n")
                f.write(res.stdout or "")
                f.write("\n\nSTDERR:\n")
                f.write(res.stderr or "")
                f.write(f"\n\nRETURN CODE: {res.returncode}\n")
        if res.returncode != 0:
            print(f"Command exited with {res.returncode}")
        return res.returncode == 0
    except Exception as e:
        print(f"Exception running command: {e}")
        return False


def run_inference(scripts, test_sleep=1):
    RESULT_DIR.mkdir(exist_ok=True)
    success = True
    for script in scripts:
        script_path = INFERENCE_DIR / script
        if not script_path.exists():
            print(f"Missing inference script: {script_path}")
            success = False
            continue

        log = RESULT_DIR / f"inference_{script}.log"
        ok = run_cmd(["python", str(script_path)], INFERENCE_DIR, log)
        success = success and ok
        time.sleep(test_sleep)

    return success


def run_evaluations(scripts, test_sleep=0.5):
    success = True
    for script in scripts:
        script_path = EVAL_DIR / script
        if not script_path.exists():
            print(f"Missing eval script: {script_path}")
            success = False
            continue

        log = RESULT_DIR / f"eval_{script}.log"
        ok = run_cmd(["python", str(script_path)], EVAL_DIR, log)
        success = success and ok
        time.sleep(test_sleep)

    return success


def main():
    parser = argparse.ArgumentParser(description="MedRBench 调用脚本")
    parser.add_argument("--inference", action="store_true", help="运行推理脚本")
    parser.add_argument("--evaluate", action="store_true", help="运行评估脚本")
    parser.add_argument("--all", action="store_true", help="推理 + 评估 都运行")
    args = parser.parse_args()

    # 默认行为：all
    if not (args.inference or args.evaluate or args.all):
        args.all = True

    if args.all or args.inference:
        print("\n🔎 开始运行 MedRBench 推理脚本")
        inference_list = ["oracle_diagnose.py"]
        # 如果需要，可扩展为运行其它 inference 脚本
        run_inference(inference_list)

    if args.all or args.evaluate:
        print("\n📊 开始运行 MedRBench 评估脚本")
        eval_list = [
            "oracle_diagnose_accuracy.py",
            "oracle_diagnose_reasoning.py",
            "treatment_final_accuracy.py",
            "treatment_reasoning.py",
        ]
        run_evaluations(eval_list)

    print("\n✅ MedRBench 调用完成。结果见目录:", RESULT_DIR)


if __name__ == '__main__':
    main()
