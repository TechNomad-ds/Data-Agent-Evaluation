import os, json, openai
from tqdm import tqdm

ANS_DIR = "./answers"
SCORE_DIR = "./scores"
os.makedirs(SCORE_DIR, exist_ok=True)

# 建议使用性能更强的模型作为裁判 (如 gpt-4o 或 qwen2.5-72b)
judge_client = openai.OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 
    base_url="http://123.129.219.111:3000/v1"
)

def get_judge_prompt(full_item):
    """
    完善后的深度提示词，包含 8 个维度的详细定义
    """
    return f"""你是一位精通金融知识与逻辑分析的资深评测专家。你需要根据 FinEval-KR 评测框架，对一个大语言模型（LLM）的回答进行全方位审计。

### 1. 输入数据汇总
请仔细阅读以下包含题目元数据和模型生成的回答：
{json.dumps(full_item, ensure_ascii=False, indent=2)}

### 2. 评测指标定义 (8个维度)
请基于以上输入，判定以下 8 个指标（回答正确给 1，错误给 0）：

- **Accuracy (准确率)**: 最终答案的数值、结论以及核心推导逻辑是否与 `gt`（标准答案）完全一致。
- **Knowledge_Score (知识得分)**: 模型是否展现了题目 `point` 中要求的金融背景知识、公式或制度准则。
    * 判定准则：即使计算错误，只要模型准确写出了正确的公式或概念，此项也给 1。只有当模型完全不知道相关知识或写错核心公式时给 0。
- **Reasoning_Score (推理得分)**: 在排除了纯粹的知识缺失后，模型的逻辑推导、因果链条、步骤拆解是否严谨。
    * 判定准则：若模型知道公式但算错了，或逻辑断裂，此项给 0。
- **CS_Memory (认知层级1-记忆)**: 对应 `per_step` 中标注为“记忆”的步骤。模型是否准确提取了事实性信息？
- **CS_Understand (认知层级2-理解)**: 对应 `per_step` 中标注为“理解”的步骤。模型是否正确解释了金融现象或概念？
- **CS_Apply (认知层级3-应用)**: 对应 `per_step` 中标注为“应用”的步骤。模型能否将公式/准则代入具体业务场景进行计算/操作？
- **CS_Analyze (认知层级4-分析)**: 对应 `per_step` 中标注为“分析”的步骤。模型能否识别多因素间的复杂关系或进行多步拆解？
- **CS_Evaluate (认知层级5-评价)**: 对应 `per_step` 中标注为“评价”的步骤。模型能否基于计算结果做出最终的风险判定或决策建议？

### 3. 判定逻辑约束 (重要)
- 若 `Accuracy` 为 1，则通常意味着其他所有维度均为 1（前提是该维度在 `per_step` 中存在）。
- 若 `per_step` 中没有明确标注某个认知层级（例如没有“评价”步），则该项对应的 CS 分数默认给 0。
- 请通过对比 `model_answer` 与 `gt` 的步骤，精准定位模型是在哪一步出错，从而判定是知识问题还是推理问题。

### 4. 输出格式要求
请严格返回以下 JSON 格式，不要包含任何解释文字：
{{
  "Accuracy": 0,
  "Knowledge_Score": 0,
  "Reasoning_Score": 0,
  "CS_Memory": 0,
  "CS_Understand": 0,
  "CS_Apply": 0,
  "CS_Analyze": 0,
  "CS_Evaluate": 0
}}"""

# 遍历 answers 文件夹
for file in os.listdir(ANS_DIR):
    if not file.endswith(".jsonl"): continue
    
    scored_results = []
    for line in open(os.path.join(ANS_DIR, file), 'r', encoding='utf-8'):
        item = json.loads(line)
        
        # 调用打分模型
        try:
            response = judge_client.chat.completions.create(
                model="gpt-4o", # 裁判模型
                messages=[{"role": "user", "content": get_judge_prompt(item)}],
                response_format={"type": "json_object"} # 强制 JSON 输出
            )
            scores = json.loads(response.choices[0].message.content)
            item['eight_scores'] = scores
        except Exception as e:
            print(f"Error scoring item: {e}")
            item['eight_scores'] = {k: 0 for k in ["Accuracy", "Knowledge_Score", "Reasoning_Score", "CS_Memory", "CS_Understand", "CS_Apply", "CS_Analyze", "CS_Evaluate"]}
            
        scored_results.append(item)

    # 保存打分结果文件
    with open(os.path.join(SCORE_DIR, file), 'w', encoding='utf-8') as f:
        for it in scored_results:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")