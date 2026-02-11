import os
import json
import requests
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================
# 1. 待测模型配置 (OpenAI 兼容接口)
API_KEY = "sk-xxxxxxxxxxxxx"
BASE_URL = "http://123.129.219.111:3000/v1" 
MODEL_NAME = "gpt-4o" 

# 2. 测评后端配置
EVAL_URL = "http://localhost:4399/evaluate"

# 3. 数据路径配置
DATA_PATH = "./data/evaluate_data/random_sampled_query_evaluation_600.json"
SAVE_PATH = f"./data/result/eval_{MODEL_NAME}.json"

# 4. 全局测试开关：设为数字(如 3 或 5)则仅测少量样本，设为 None 则全量测试
SAMPLE_SIZE = 3 
# ===========================================

def request_llm_answer(query, client, model):
    """调用待测模型获取法律回答"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的律师，请回答用户的法律问题。"},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n[Error] 模型调用失败: {e}")
        return None

def request_evaluation(query, content):
    """调用本地评估后端进行打分"""
    data = {"query": query, "content": content}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(EVAL_URL, json=data, headers=headers, timeout=600) # 增加超时保护
        if response.status_code == 200:
            res = response.json()
            return {
                "reasoning": res.get('reasoning', ""),
                "ans": res.get('ans', ""),
                "score": res.get('score', 0)
            }
    except Exception as e:
        print(f"\n[Error] 测评服务请求失败: {e}")
    return None

def main():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    if not os.path.exists(DATA_PATH):
        print(f"未找到测试集文件: {DATA_PATH}")
        return
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # --- 核心修改：根据 SAMPLE_SIZE 截取测试数据 ---
    if SAMPLE_SIZE is not None:
        print(f"注意：当前开启了抽样模式，仅测试前 {SAMPLE_SIZE} 条数据。")
        test_data = test_data[:SAMPLE_SIZE]
    # -------------------------------------------

    results = []
    print(f"开始测评模型: {MODEL_NAME}，待处理条数: {len(test_data)}")

    for item in tqdm(test_data):
        query = item['query']
        
        # 1. 获取模型回答
        content = request_llm_answer(query, client, MODEL_NAME)
        if not content: continue
        
        # 2. 获取评估结果
        eval_res = request_evaluation(query, content)
        if not eval_res: continue
        
        # 3. 收集结果
        results.append({
            "query": query,
            "content": content,
            "reasoning": eval_res['reasoning'],
            "ans": eval_res['ans'],
            "score": eval_res['score']
        })

    # 4. 保存结果
    if not results:
        print("未产生任何有效测评结果，请检查服务状态。")
        return

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    final_output = {
        MODEL_NAME: {
            "query": [r['query'] for r in results],
            "content": [r['content'] for r in results],
            "reasoning": [r['reasoning'] for r in results],
            "ans": [r['ans'] for r in results],
            "score": [r['score'] for r in results]
        }
    }

    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    
    avg_score = sum([r['score'] for r in results]) / len(results)
    print(f"\n测评完成！结果已保存至: {SAVE_PATH}")
    print(f"本次测试样本数: {len(results)}")
    print(f"模型平均分: {avg_score:.2f}")

if __name__ == "__main__":
    main()