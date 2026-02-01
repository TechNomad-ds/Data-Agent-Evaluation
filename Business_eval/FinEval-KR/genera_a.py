import os, json, openai
from tqdm import tqdm

DATA_DIR = "./data"
ANS_DIR = "./answers"
os.makedirs(ANS_DIR, exist_ok=True)
client = openai.OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", base_url="http://123.129.219.111:3000/v1")

def get_generation_prompt(question):
    """
    为被测模型构建的系统化提示词，强制要求逻辑链条
    """
    return [
        {"role": "system", "content": "你是一位专业的金融分析师。请针对用户提出的金融问题进行深入分析，要求逻辑严谨、步骤清晰。"},
        {"role": "user", "content": f"""请阅读以下金融问题并给出详细的解答。

【题目内容】：
{question}

【回答要求】：
1. **分步推理**：请将解题过程拆分为明确的步骤（如步骤一、步骤二...）。
2. **展示公式**：如果涉及数学计算，请务必列出所使用的金融公式。
3. **结论明确**：在最后给出一个清晰的“最终答案”。
4. **语言专业**：使用标准的金融术语。

请开始你的推理和回答："""}
    ]

for file in os.listdir(DATA_DIR):
    if not file.endswith(".jsonl"):
        continue
    
    with open(os.path.join(DATA_DIR, file), 'r', encoding='utf-8') as f_in, \
         open(os.path.join(ANS_DIR, file), 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Generating Answers for {file}"):
            item = json.loads(line)
            
            # 使用优化后的提示词调用模型
            try:
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=get_generation_prompt(item['instruction']),
                    temperature=0.1, # 降低随机性，保证推理稳定性
                    max_tokens=2048
                )
                item['model_answer'] = res.choices[0].message.content
            except Exception as e:
                print(f"Error generating answer: {e}")
                item['model_answer'] = "Generation Failed"
                
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")