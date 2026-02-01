#!/bin/bash

# --- 配置区 ---
LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S).log"
LD_PRELOAD_PATH="/opt/conda/envs/data-agent/lib/libstdc++.so.6"

# 使用 exec 和 tee 将所有输出同步记录到日志文件
# &> 表示合并标准输出和错误输出， | tee -a 表示同时输出到屏幕和追加到文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo "开始全项目自动化测试 | 时间: $(date)"
echo "日志将保存至: $LOG_FILE"
echo "==============================================="

# 1. DiagnosisArena
echo ">>> [1/5] 运行 DiagnosisArena..."
python DiagnosisArena_eval.py || echo "⚠️ DiagnosisArena 运行出错"

# 2. MedCaseReasoning
echo ">>> [2/5] 运行 MedCaseReasoning (Inference & Eval)..."
python MedCaseReasoning/inferce.py && python MedCaseReasoning/eval.py || echo "⚠️ MedCaseReasoning 运行出错"

# 3. MedXpertQA
echo ">>> [3/5] 运行 MedXpertQA..."
LD_PRELOAD=$LD_PRELOAD_PATH python MedXpertQA_eval.py || echo "⚠️ MedXpertQA 运行出错"

# 4. HELM
echo ">>> [4/5] 运行 HELM 评测..."
LD_PRELOAD=$LD_PRELOAD_PATH helm-run \
    -c my_run.conf \
    --suite medical_test_v1 \
    --max-eval-instances 5 \
    --output-path ./helm/benchmark_output || echo "⚠️ HELM 运行出错"

# 5. MedRBench
echo ">>> [5/5] 运行 MedRBench 系列任务..."

# 推理
python MedRBench/src/Inference/free_turn.py
python MedRBench/src/Inference/one_turn.py
python MedRBench/src/Inference/oracle_diagnose.py
python MedRBench/src/Inference/oracle_treatment_planning.py

# 文件整理
echo ">>> 整理 MedRBench 推理结果文件..."
TARGET_DIR="MedRBench/data/InferenceResults"
mkdir -p "$TARGET_DIR"
cp -rf MedRBench/src/Inference/1_turn_my_test_model "$TARGET_DIR/"
cp -rf MedRBench/src/Inference/free_turn_my_test_model "$TARGET_DIR/"
cp -f MedRBench/src/Inference/oracle_diagnosis.json "$TARGET_DIR/"
cp -f MedRBench/src/Inference/oracle_treatment.json "$TARGET_DIR/"

# 评估
echo ">>> 运行 MedRBench 最终评估脚本..."
python MedRBench/src/Evaluation/1turn_json_genera.py
python MedRBench/src/Evaluation/free_turn_json_genera.py

eval_scripts=(
    "1turn_diagnose_accuracy.py"
    "1turn_reasoning.py"
    "free_turn_diagnose_accuracy.py"
    "free_turn_assessment_recommendation.py"
    "oracle_diagnose_accuracy.py"
    "oracle_diagnose_reasoning.py"
    "treatment_final_accuracy.py"
    "treatment_reasoning.py"
)

for script in "${eval_scripts[@]}"; do
    echo "正在执行评估: $script"
    python "MedRBench/src/Evaluation/$script"
done

echo "==============================================="
echo "所有自动化测试任务已完成！"
echo "完整日志已记录在: $LOG_FILE"
echo "==============================================="