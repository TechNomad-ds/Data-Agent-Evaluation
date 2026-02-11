import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import argparse

# ==========================================
# 1. 全局配置 (请根据实际情况修改)
# ==========================================
API_BASE_URL = "http://123.129.219.111:3000/v1"
API_KEY = "sk-ptwvi0XgJUL6JyjyMS9faFUSDbw5vxC1hX0EqQvxeUUG0Kv7"
JUDGE_MODEL_NAME = "gpt-4o"  # 裁判模型，通常使用更强的模型

GENE_ARGS = {
    "temperature": 0,  # 裁判评分建议设为0以保证一致性
    "max_tokens": 4096
}

SKIP_MODELS = ['gold', 'prompt', 'question', 'course', 'gold_answer']

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

JUDGE_SYSTEM = "Act as a Judge specializing in the evaluation of Swiss law schools exams. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning."
JUDGE_PROMPT = """Goal:
Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.

Context:
You will be provided with a response (labeled: Model's Answer) to a law school exam question (labeled: Question) and a reference answer (labeled: Reference Answer). 

Return format:
    After reviewing the response:
    1. Explanation: Briefly explain your reasoning regarding how the response conforms to or deviates from the reference answer. 
    2. Constructive feedback: Additionally, provide neutral, constructive feedback and corrections in the style of a university professor.
    3. Correctness score: Assign a final correctness score on a scale from 0.0 to 1.0 (in increments of 0.1). This score should reflect the extent to which the response satisfies the reference answer, where 
        - 1.0 = complete fulfillment (100%) 
        - lower scores reflect proportionate shortfalls (e.g. 0.5 = 50% fulfillment). 
	    - strictly follow the format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". 

Warnings:
    - In some cases, the reference answer may include only keywords or factual elements to be examined, along with (+), (-) or (+/-). Respect these indications when determining correctness:
        - (+) means the element must be affirmed.
        - (–) means the element must be denied.
        - (-/+) indicates that arguments in either direction are acceptable if legally sound.
    - Deviations or additional elements not found in the reference answer should generally be penalized unless you are certain they are legally correct and relevant. Assume the reference answer includes all information necessary for a perfect response.
    - The reference answer may contain citations (e.g., from books or law review articles), which the response does not need to replicate. However, statutes should be cited precisely, specifying Abs., Ziff., or lit. whenever applicable.
    - If the reference answer includes separate sub-points, use these for proportional scoring guidance (e.g., addressing 2 out of 4 sub-points correctly equals approximately a 0.5 score).
Judge the below case, give the brief reasoning process and the final grade.


Question:
```{question_fact}```

Reference Answer:
```{ref_answer}```

Model's Answer:
```[{model_answer}]```

Your Judgment:
"""


def get_judge_response(messages):
    """发送同步请求给裁判模型"""
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=messages,
            **GENE_ARGS
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Judge API: {e}")
        return f"Error: [[0.0]]\nReason: {str(e)}"

def extract_score(text):
    """从裁判的回应中提取 [[score]]"""
    search = re.search(r"\[\[(\d\.?\d?)\]\]", text)
    if search:
        try:
            score = float(search.group(1))
            return min(max(score, 0.0), 1.0) # 限制在 0-1 之间
        except:
            return 0.0
    return 0.0

# ==========================================
# 4. 主评测逻辑
# ==========================================
def main(args):
    # 1. 读取结果文件
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    if args.sample > 0:
        df = df.head(args.sample)

    # 2. 识别需要被评价的模型答案列
    model_answer_columns = []
    for col in df.columns:
        if col.endswith('_answer') and col not in SKIP_MODELS:
            model_answer_columns.append(col)
    
    if not model_answer_columns:
        print("No model answer columns (ending with _answer) found to judge.")
        return

    # 3. 遍历每个模型的列进行评估
    for col in model_answer_columns:
        model_name = col.replace('_answer', '')
        print(f"\n>>> Judging performance for model: {model_name}")
        
        judgments = []
        scores = []
        
        # 逐行评估
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Progress ({model_name})"):
            question = row['question']
            ref = row['gold_answer']
            ans = row[col]
            
            # 处理思考模型（如果是 R1/o1 可能带有 think 标签）
            if isinstance(ans, str) and '</think>' in ans:
                ans = ans.split('</think>')[-1].strip()
            
            # 构造提示词
            prompt_content = JUDGE_PROMPT.format(
                question_fact=question, 
                ref_answer=ref, 
                model_answer=ans
            )
            
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt_content}
            ]
            
            # 获取裁判评价
            judge_res = get_judge_response(messages)
            score = extract_score(judge_res)
            
            judgments.append(judge_res)
            scores.append(score * 100) # 转换为百分制

        # 4. 将评价结果存入 DataFrame
        df[f'{model_name}_judge_content'] = judgments
        df[f'{model_name}_score'] = scores

        # 5. 计算统计指标 (Mean & Bootstrap Variance)
        score_array = np.array(scores)
        mean_score = np.mean(score_array)
        
        # Bootstrap 计算稳定性
        n = len(score_array)
        boot_means = [np.mean(np.random.choice(score_array, n, replace=True)) for _ in range(1000)]
        variance = np.var(boot_means)
        
        print(f"Results for {model_name}:")
        print(f"  - Mean Score: {mean_score:.2f}")
        print(f"  - Bootstrap Variance: {variance:.4f}")

    # 6. 保存最终评价文件
    df.to_csv(args.output_file, index=False, encoding='utf-8')
    print(f"\nGrading complete. Results saved to: {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Judge for Legal Questions")
    parser.add_argument("--input_file", type=str, required=True, help="Result CSV file from litellm_eval")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file with judge scores")
    parser.add_argument("--sample", type=int, default=-1, help="Number of samples to judge (default -1 for all)")
    
    args = parser.parse_args()
    main(args)







