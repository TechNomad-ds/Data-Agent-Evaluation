#!/usr/bin/env python3
"""
MedRBench 推理专用调用脚本（仅运行 src/Inference 下的推理脚本）
"""
import argparse
import time
import subprocess
from pathlib import Path

PROJECT_DIR = Path("MedRBench")
INFERENCE_DIR = PROJECT_DIR / "src" / "Inference"
RESULT_DIR = Path("MedRBench_inference_results")


def run(cmd, cwd):
    print(f"Running {' '.join(cmd)} in {cwd}")
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        print(r.stdout[:1000])
        if r.returncode != 0:
            print(f"Command failed: {r.stderr[:500]}")
        return r.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, default='oracle_diagnose.py', help='要运行的推理脚本（文件名）')
    parser.add_argument('--all', action='store_true', help='运行所有内置推理脚本')
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)

    available = [p.name for p in INFERENCE_DIR.glob('*.py')]
    if args.all:
        targets = available
    else:
        if args.script not in available:
            print(f"脚本 {args.script} 不存在，可用脚本: {available}")
            return
        targets = [args.script]

    for s in targets:
        print(f"\n== 执行: {s} ==")
        ok = run(['python', s], INFERENCE_DIR)
        if not ok:
            print(f"脚本 {s} 执行失败，继续下一个")
        time.sleep(1)

    print('\nFinished. 输出目录: ', RESULT_DIR)


if __name__ == '__main__':
    main()
