#!/bin/bash

# --- 配置区 ---
# 自动生成带时间戳的日志文件名
LOG_FILE="business_eval_$(date +%Y%m%d_%H%M%S).log"
LIB_STDC="/opt/conda/envs/data-agent/lib/libstdc++.so.6"

# 开启全量日志捕获：屏幕显示的同时记录到文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo "开始 Business_eval 全项目自动化测试"
echo "启动时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "==============================================="

# 1. FinCDM
echo ">>> [1/5] 运行 FinCDM..."
python FinCDM/src/api-eval.py

# 2. FinEval-KR
echo ">>> [2/5] 运行 FinEval-KR (Genera -> Eval -> Stastic)..."
python FinEval-KR/genera_a.py && \
python FinEval-KR/eval.py && \
python FinEval-KR/stastic.py

# 3. XFinbench
echo ">>> [3/5] 运行 XFinbench..."
# 赋予执行权限以防万一
chmod +x XFinBench/run.sh
bash XFinBench/run.sh

# 4. CFA_ESSAY_REPRODUCER
echo ">>> [4/5] 运行 CFA_ESSAY_REPRODUCER..."
LD_PRELOAD=$LIB_STDC python3 -m CFA_ESSAY_REPRODUCER.src.main.py

# 5. PIXIU
echo ">>> [5/5] 运行 PIXIU..."
# 设置环境变量
export HF_TOKEN="XXXXXXXXXXXXXXXXXXXXXX"

# 执行评测命令
LD_PRELOAD=$LIB_STDC python PIXIU/src/eval.py \
    --model gpt-4 \
    --tasks "flare_fpb,flare_fiqasa" \
    --batch_size 1 \
    --limit 5 \
    --output_path "results_PIXIU/my_model_test.json"

echo "==============================================="
echo "所有商业评测任务已完成！"
echo "完成时间: $(date)"
echo "==============================================="