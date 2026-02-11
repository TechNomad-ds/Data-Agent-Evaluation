import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import datasets
import pandas as pd
import re
from tqdm import tqdm
from openai import OpenAI

# 设置镜像站（解决网络问题）


# 导入 LegalBench 本地模块
from tasks import TASKS
from utils import generate_prompts
from evaluation import evaluate

# ================= 配置区 =================
API_KEY = "sk-xxxxxxxxxxxxx"
BASE_URL = "http://123.129.219.111:3000/v1" 
MODEL_NAME = "deepseek-r1"


TARGET_TASKS = [
    "abercrombie",            # 商标法分类 (基础)
    "hearsay",                # 证据法-传闻证据判断 (高难逻辑)
    "personal_jurisdiction",  # 民事诉讼法-管辖权 (法律程序)
    "ucc_v_common_law",       # 法律适用判断 (合同冲突)
    "contract_qa"             # 合同条款提取 (理解力)
]

SAMPLE_LIMIT = 3  # 每个任务测试的样本数
# ==========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def clean_r1_output(text):
    """提取模型输出中的核心答案，去除思维链和多余前缀"""
    import re
    # 1. 移除 <think> 标签及其内容
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 统一转小写并移除首尾空格/标点
    cleaned = cleaned.strip().lower()
    
    # 3. 关键改进：移除常见的引导词前缀 (如 "answer: ", "the answer is: ")
    cleaned = re.sub(r'^(answer|the answer is|result|prediction)[:\s]+', '', cleaned)
    
    # 4. 针对不同任务的关键词捕获逻辑
    # 针对 Yes/No 任务（contract_qa, hearsay 等）
    if cleaned.startswith("yes"): return "Yes"
    if cleaned.startswith("no"): return "No"
    
    # 针对 Abercrombie 任务
    for label in ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]:
        if label in cleaned: 
            return label
            
    # 针对 UCC 任务
    if "ucc" in cleaned: return "UCC"
    if "common law" in cleaned: return "Common Law"
    
    # 如果都没匹配上，返回首字母大写的处理结果（符合 LegalBench 答案格式）
    return cleaned.capitalize()

def get_llm_response(prompt, task_name):
    """调用 LLM 并获取清晰的回答"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"You are a legal expert. Task: {task_name}. Provide a concise answer. If it is a Yes/No question, answer only 'Yes' or 'No'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        raw_content = response.choices[0].message.content
        return clean_r1_output(raw_content)
    except Exception as e:
        print(f"API 请求失败: {e}")
        return ""

def run_benchmark():
    summary_results = {}
    import re # 确保导入正则

    for task_name in TARGET_TASKS:
        print(f"\n" + "="*50)
        print(f"正在测评任务: {task_name}")
        
        try:
            # 1. 加载数据
            dataset = datasets.load_dataset("nguha/legalbench", task_name, trust_remote_code=True)
            test_df = dataset["test"].to_pandas().head(SAMPLE_LIMIT)
            
            # 2. 加载任务对应的 Prompt 模板
            template_path = f"tasks/{task_name}/base_prompt.txt"
            if os.path.exists(template_path):
                with open(template_path, "r") as f:
                    prompt_template = f.read()
            else:
                # 备用模板（如果本地文件缺失）
                prompt_template = "Context: {{text}}\nQuestion: {{question}}\nAnswer:"

            # 3. 生成 Prompts
            prompts = generate_prompts(prompt_template, test_df)
            
            # 4. 执行推理
            generations = []
            print(f"正在调用 {MODEL_NAME} 推断...")
            for i, p in enumerate(tqdm(prompts)):
                res = get_llm_response(p, task_name)
                generations.append(res)
                # 实时打印前两个案例的对比，方便观察
                if i < 2:
                    print(f"\n[案例 {i+1}]")
                    print(f"预测结果: {res}")
                    print(f"标准答案: {test_df['answer'].iloc[i]}")

            # 5. 评估得分
            answers = test_df["answer"].tolist()
            score = evaluate(task_name, generations, answers)
            summary_results[task_name] = score
            print(f"\n任务 [{task_name}] 测评得分: {score:.4f}")
            
        except Exception as e:
            print(f"任务 {task_name} 运行中出错: {e}")
            summary_results[task_name] = "Error"

    # ================= 最终汇总 =================
    print("\n" + "█"*50)
    print("      LegalBench 代表性任务测评汇总表")
    print("-" * 50)
    print(f"{'任务名称':<25} | {'得分':<10}")
    print("-" * 50)
    for task, score in summary_results.items():
        score_display = f"{score:.4f}" if isinstance(score, float) else score
        print(f"{task:<25} | {score_display:<10}")
    print("█"*50)

if __name__ == "__main__":
    run_benchmark()