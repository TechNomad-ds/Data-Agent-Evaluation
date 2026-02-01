# import os, json

# SCORE_DIR = "./scores"
# # 初始化统计字典
# stats = {k: {"sum": 0, "total": 0} for k in ["Accuracy", "Knowledge_Score", "Reasoning_Score", "CS_Memory", "CS_Understand", "CS_Apply", "CS_Analyze", "CS_Evaluate"]}

# for file in os.listdir(SCORE_DIR):
#     for line in open(os.path.join(SCORE_DIR, file), 'r', encoding='utf-8'):
#         item = json.loads(line)
#         scores = item['llm_scores']
#         per_step_str = item['per_step']
        
#         # 1. 基础三项统计
#         for key in ["Accuracy", "Knowledge_Score", "Reasoning_Score"]:
#             stats[key]["sum"] += scores[key]
#             stats[key]["total"] += 1
            
#         # 2. 认知五项统计 (只有在题目 per_step 涉及该项时才计入分母)
#         cog_mapping = {
#             "记忆": "CS_Memory", "理解": "CS_Understand", "应用": "CS_Apply", 
#             "分析": "CS_Analyze", "评价": "CS_Evaluate"
#         }
#         for cn_name, en_name in cog_mapping.items():
#             if cn_name in per_step_str:
#                 stats[en_name]["sum"] += scores[en_name]
#                 stats[en_name]["total"] += 1

# print("\n--- FinEval-KR 项目复现最终得分 (归一化) ---")
# for metric, data in stats.items():
#     avg = data["sum"] / data["total"] if data["total"] > 0 else 0
#     print(f"{metric:16}: {avg:.4f}  (样本数: {data['total']})")
import os
import json

# 配置目录
SCORE_DIR = "./scores"

# 初始化统计字典
# sum: 分数累加, total: 分母(有效样本数)
stats = {k: {"sum": 0, "total": 0} for k in [
    "Accuracy", "Knowledge_Score", "Reasoning_Score", 
    "CS_Memory", "CS_Understand", "CS_Apply", "CS_Analyze", "CS_Evaluate"
]}

# 认知标签映射
cog_mapping = {
    "记忆": "CS_Memory", 
    "理解": "CS_Understand", 
    "应用": "CS_Apply", 
    "分析": "CS_Analyze", 
    "评价": "CS_Evaluate"
}

if not os.path.exists(SCORE_DIR):
    print(f"错误: 找不到目录 {SCORE_DIR}")
    exit()

# 遍历打分后的文件
files = [f for f in os.listdir(SCORE_DIR) if f.endswith(".json") or f.endswith(".jsonl")]

for file in files:
    file_path = os.path.join(SCORE_DIR, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
                # 统一使用 eval.py 中保存的键名 'eight_scores'
                scores = item.get('eight_scores')
                
                if not scores:
                    continue

                per_step_str = str(item.get('per_step', ""))

                # 1. 基础三项统计 (Accuracy, KS, RS)
                # 这三项通常以全量题目为分母
                for key in ["Accuracy", "Knowledge_Score", "Reasoning_Score"]:
                    if key in scores:
                        stats[key]["sum"] += scores[key]
                        stats[key]["total"] += 1
                
                # 2. 认知五项统计 (CS系列)
                # 只有在题目原始标签 per_step 涉及该项时，才计入该项的分母
                for cn_name, en_name in cog_mapping.items():
                    if cn_name in per_step_str:
                        stats[en_name]["sum"] += scores.get(en_name, 0)
                        stats[en_name]["total"] += 1
                        
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

# 输出最终结果
print("\n" + "="*60)
print(f"{'FinEval-KR 归一化得分统计结果':^60}")
print("="*60)
print(f"{'能力维度 (Metric)':<20} | {'得分 (Score)':<10} | {'样本总数 (Total)'}")
print("-" * 60)

for metric, data in stats.items():
    # 归一化计算：总分 / 涉及该能力的题目总数
    avg = data["sum"] / data["total"] if data["total"] > 0 else 0
    print(f"{metric:<20} | {avg:.4f}     | {data['total']}")

print("="*60)