import re
import json
import time
import random
import concurrent.futures
import numpy as np
import requests
from tqdm import tqdm
from openai import OpenAI

# Configuration Constants
# Change these to your own API keys and model URLs
QWQ_URL = 'https://api.siliconflow.cn/v1/'
QWQ_API_KEY = 'YOUR_API_KEY'

GEMINI_URL = 'https://aigptapi.com/v1/'
GEMINI_API_KEY = 'YOUR_API_KEY'

DEEPSEEK_R1_URL = "http://10.17.3.65:1025/v1/chat/completions"

O1_API_KEY_LIST = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
]

MY_MODEL_URL = 'http://123.129.219.111:3000/v1'
MY_MODEL_API_KEY = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
MY_MODEL_NAME = 'deepseek-r1'

DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"
DATA_PATH = '../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json'
PROMPT_TEMPLATE_PATH = './instructions/oracle_diagnose.txt'

# =====================
# MODEL API INTERFACES
# =====================

def create_qwq_client():
    """Create and return OpenAI client for QwQ model"""
    return OpenAI(
        base_url=QWQ_URL,
        api_key=QWQ_API_KEY
    )

def query_qwq_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the QwQ model and handle potential rate limit errors"""
    client = create_qwq_client()
    while True:
        try:
            response = client.chat.completions.create(
                model="Qwen/QwQ-32B-Preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying QwQ model: {e}")
                return None

def query_llama3_r1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the Llama3-R1 model with reasoning support"""
    client = create_qwq_client()
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            reasoning_content = response.choices[0].message.reasoning_content
            return content, reasoning_content
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying Llama3-R1 model: {e}")
                return None, None

def query_deepseek_r1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query DeepSeek-R1 model via direct HTTP request"""
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "DeepSeek-R1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        "temperature": 0.6,
        "stream": False,
        "max_tokens": 10000,
    }
    max_retry = 3
    curr_retry = 0
    
    while curr_retry < max_retry:
        try:
            response = requests.post(DEEPSEEK_R1_URL, json=data, headers=headers)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            reasoning = content.split('</think>')[0].replace('<think>', '').strip()
            answer = content.split('</think>')[1].strip()
            return reasoning, answer
        except Exception as e:
            curr_retry += 1
            print(f"Error: {e}, retrying... {curr_retry}/{max_retry}")
            time.sleep(2)
    
    return None, None

def query_o1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the O1/O3-mini model with round-robin API key selection"""
    max_try_times = 3
    curr_try = 0
    
    while curr_try < max_try_times:
        try:
            client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key=random.choice(O1_API_KEY_LIST)
            )
            completion = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            curr_try += 1
            if curr_try >= max_try_times:
                print(f"Error querying O1 model: {e}")
                return "Error."
            time.sleep(5)

def query_my_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """通用模型调用接口，支持思维链解析"""
    client = OpenAI(
        base_url=MY_MODEL_URL,
        api_key=MY_MODEL_API_KEY
    )
    
    try:
        response = client.chat.completions.create(
            model=MY_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        content = response.choices[0].message.content
        
        # 针对思维链模型的解析逻辑
        reasoning = ""
        if "</think>" in content:
            parts = content.split("</think>")
            reasoning = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip()
        else:
            answer = content
            
        return answer, reasoning
    except Exception as e:
        print(f"API Error: {e}")
        if "rate limit" in str(e).lower():
            time.sleep(30)
        return None, None

def query_gemini_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the Gemini model and handle rate limits"""
    client = OpenAI(
        base_url=GEMINI_URL,
        api_key=GEMINI_API_KEY
    )
    while True:
        try:
            response = client.chat.completions.create(
                model="gemini-2.0-flash-thinking-exp-01-21",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying Gemini model: {e}")
                return None

def query_baichuan_model(input_text, model, tokenizer, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the Baichuan model using transformers library"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate text
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Error querying Baichuan model: {e}")
        return None

# =====================
# DATA PROCESSING
# =====================

def load_instruction(template_path):
    """Load instruction template from file"""
    try:
        with open(template_path, encoding='utf-8') as fp:
            return fp.read()
    except Exception as e:
        print(f"Error loading instruction template: {e}")
        return None

def load_data(data_path):
    """Load and parse case data from JSON file"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

def save_results(data, filename):
    """Save results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

# =====================
# INFERENCE PROCESSORS
# =====================

def process_qwq_data(data_id, data, prompt_template):
    """Process a single data item with QwQ model"""   
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        response = query_qwq_model(prompt)
        result['content'] = response
        
        return data_id, result
    except Exception as e:
        print(f"Error processing QwQ data {data_id}: {e}")
        return data_id, None

def process_deepseek_r1_data(data_id, data, prompt_template):
    """Process a single data item with DeepSeek-R1 model"""
    
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        reasoning, answer = query_deepseek_r1_model(prompt)
        if reasoning and answer:
            result['out_reasoning'] = reasoning
            result['out_answer'] = answer
                        
            return data_id, result
        else:
            return data_id, None
    except Exception as e:
        print(f"Error processing DeepSeek-R1 data {data_id}: {e}")
        return data_id, None

def process_o1_data(data_id, data, prompt_template):
    """Process a single data item with O1/O3-mini model"""
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        response = query_o1_model(prompt)
        result['content'] = response
                
        return data_id, result
    except Exception as e:
        print(f"Error processing O1 data {data_id}: {e}")
        return data_id, None

def process_gemini_data(data_id, data, prompt_template):
    """Process a single data item with Gemini model"""
       
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        response = query_gemini_model(prompt)
        result['content'] = response
                
        return data_id, result
    except Exception as e:
        print(f"Error processing Gemini data {data_id}: {e}")
        return data_id, None
    
def process_my_model_data(data_id, data_item, prompt_template):
    """处理单个案例的推理"""
    try:
        result = {}
        # 对应你提供的 JSON 数据层级: data['generate_case']['case_summary']
        patient_case = data_item.get('generate_case', {}).get('case_summary', "")
        
        if not patient_case:
            return data_id, None
            
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        answer, reasoning = query_my_model(prompt)
        
        if answer is not None:
            result['out_answer'] = answer
            result['out_reasoning'] = reasoning
            return data_id, result
        else:
            return data_id, None
            
    except Exception as e:
        print(f"Error processing data {data_id}: {e}")
        return data_id, None

# =====================
# MAIN INFERENCE RUNNERS
# =====================

def run_inference_with_model(
    process_func, 
    model_name, 
    output_filename, 
    max_workers=8
):
    """Generic function to run inference with any model"""
    print(f"Running inference with {model_name} model")
    
    prompt_template = load_instruction(PROMPT_TEMPLATE_PATH)
    data = load_data(DATA_PATH)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data_id, data_item in data.items():
            futures.append(executor.submit(process_func, data_id, data_item, prompt_template))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing with {model_name}"):
            data_id, result = future.result()
            if result is not None:
                data[data_id][model_name.lower()] = result
    save_results(data, output_filename)


def run_oracle_inference(max_workers=4):
    """主运行函数"""
    print(f"Running Oracle Diagnosis with {MY_MODEL_NAME}")
    
    # 加载模板和数据
    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 这里可以限制测试数量，例如只测前5个：list(data.items())[:5]
    items = list(data.items())[:2]
    
    results_map = {}
    
    # 使用线程池进行并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(process_my_model_data, d_id, d_item, prompt_template): d_id 
            for d_id, d_item in items
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(items), desc="Inference"):
            data_id, result = future.result()
            if result:
                # 将结果存入原始数据的对应 ID 下
                data[data_id][f'result'] = result

    # 保存结果
    output_file = f"oracle_diagnosis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully saved results to {output_file}")

if __name__ == "__main__":
    # # Run inference with all models
    # inference_qwq()
    # inference_deepseek_r1()
    # inference_o1()
    # inference_gemini()
    # # Only run this if you have the Baichuan model and CUDA support
    # inference_baichuan()
    run_oracle_inference(max_workers=4)