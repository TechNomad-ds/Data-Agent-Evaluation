#!/bin/bash

# 1. 设置项目绝对路径，确保 python 能找到 evaluate/main.py
export PROJECT_PATH=$(pwd)

# 2. 定义你要测试的模型名称
# 注意：这个名称应与你在 load_models.py 中判断逻辑匹配
MODEL_NAME="gpt-4o" 

# 3. 自动遍历 3 种任务类型 (涵盖 TU, TR, FF, SP, NM 所有金融能力)
for TASK in bool mcq calcu
do
    # 4. 自动遍历 2 种推理方式 (DA = 直接回答, CoT = 思维链)
    for REASON in DA CoT
    do
        echo "================================================"
        echo "正在评测：任务[$TASK] | 模式[$REASON] | 模型[$MODEL_NAME]"
        echo "================================================"
        
        # 运行主程序
        # --retri_type no 表示不提供参考书（无检索）
        python evaluate/main.py \
            --dataset XFinBench \
            --task $TASK \
            --model $MODEL_NAME \
            --reason_type $REASON \
            --retri_type no \
            --sys_msg On
            
        echo -e "完成！结果已保存。\n"
    done
done

echo "恭喜！全能力双模评测已全部结束。"
echo "请前往目录: ${PROJECT_PATH}/evaluate/results/ 查看 CSV 结果文件。"