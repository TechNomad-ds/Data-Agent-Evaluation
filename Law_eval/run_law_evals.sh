#!/bin/bash

# --- 环境与路径配置 ---
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# 自动生成带时间戳的日志文件名
LOG_FILE="$OUTPUT_DIR/law_eval_$(date +%Y%m%d_%H%M%S).log"

# 开启全量日志捕获：屏幕显示的同时记录到日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo "开始 Law_eval 法律领域自动化评测"
echo "启动时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "==============================================="

# 1. LEXam
echo ">>> [1/3] 运行 LEXam..."
cd LEXam
# 1.1 运行基础评估
python eval.py

# 1.2 运行自定义评分 (customized_judge)
echo "    > 正在进行内容评分 (customized_judge)..."
python customized_judge.py \
    --input_file ./results/result_oq_gpt-4o-mini.csv \
    --output_file ./results/result_oq_graded.csv

# 1.3 运行选择题评估 (evaluation)
echo "    > 正在进行选择题统计 (evaluation)..."
python evaluation.py \
    --input_file ./results/result_mcq_gpt-4o-mini.csv \
    --response_field gpt-4o-mini_answer \
    --task_type mcq_letters

cd ..

# 2. LegalEval-Q (含后台进程处理)
echo ">>> [2/3] 运行 LegalEval-Q..."
cd LegalEval-Q

echo "    > 启动后端请求服务 (evaluator_request.py)..."
# 在后台启动后端程序，并将输出也记入日志
python evaluator_request.py &
BACKEND_PID=$!

# 等待几秒确保后端启动完毕
sleep 5

echo "    > 运行评测程序 (eval.py)..."
python eval.py

# 评测完成后，关闭后台进程
echo "    > 正在关闭后端服务 (PID: $BACKEND_PID)..."
kill $BACKEND_PID

cd ..

# 3. legalbench
echo ">>> [3/3] 运行 legalbench..."
cd legalbench
python eval.py
cd ..

echo "==============================================="
echo "所有法律评测任务已完成！"
echo "完成时间: $(date)"
echo "日志查看: $LOG_FILE"
echo "==============================================="