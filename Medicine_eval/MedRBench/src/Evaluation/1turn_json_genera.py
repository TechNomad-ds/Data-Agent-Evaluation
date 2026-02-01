import json
import os
import re

def generate_model_outputs(source_json_path, log_folder_path, output_json_path, model_name='deepseek-r1'):
    """
    遍历log文件夹，结合源JSON，生成评估脚本所需的 model_outputs JSON文件。
    """
    # 1. 加载源病例数据 (包含 generate_case 等字段)
    if not os.path.exists(source_json_path):
        print(f"Error: Source file {source_json_path} not found.")
        return
        
    with open(source_json_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # 存储最终生成的 model_outputs 结构
    # 结构示例: { "ID": { "model_name": { "messages": [...] } } }
    model_outputs_final = {}

    # 2. 获取 log 文件夹下所有的 json 文件
    log_files = [f for f in os.listdir(log_folder_path) if f.endswith('.json')]
    print(f"Found {len(log_files)} log files in {log_folder_path}")

    # 3. 遍历每个 log 文件进行处理
    for log_file in log_files:
        # 使用正则表达式从文件名提取 PMC 编号 (例如从 log_PMC11625232.json 提取 PMC11625232)
        match = re.search(r'PMC\d+', log_file)
        if not match:
            continue
        
        case_id = match.group()
        log_file_path = os.path.join(log_folder_path, log_file)

        # 检查该 ID 是否在源数据中
        if case_id not in source_data:
            print(f"Skip: Case {case_id} found in logs but not in source JSON.")
            continue

        try:
            with open(log_file_path, 'r', encoding='utf-8') as lf:
                log_content = json.load(lf)
                
                # 提取对话历史 (对应 log 文件中的 output_messages 字段)
                messages = log_content.get("output_messages", [])
                
                if not messages:
                    print(f"Warning: {log_file} has no output_messages.")
                    continue

                # 构造符合评估脚本读取逻辑的字典层级:
                # 脚本通过 data['results']['messages'][-1]['content']['answer'] 访问
                # 而 log 文件中的 assistant 回复已经是 {'content': {'answer': '...'}} 格式
                model_outputs_final[case_id] = {
                    model_name: {
                        "messages": messages
                    }
                }
        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")

    # 4. 将结果保存为 JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as out_f:
        json.dump(model_outputs_final, out_f, ensure_ascii=False, indent=4)
    
    print(f"Success! Generated evaluation-ready JSON at: {output_json_path}")
    print(f"Total cases matched and processed: {len(model_outputs_final)}")

# --- 执行设置 ---
# 源文件路径
SOURCE_JSON = '../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json'
# 存放 log_PMCxxxx.json 的文件夹路径
LOG_FOLDER = '../../data/InferenceResults/1_turn_my_test_model' 
# 生成的输出文件路径 (对应你评估脚本 --model-outputs 参数)
OUTPUT_FILE = '../../data/InferenceResults/1_turn_assessment_recommendation+final_diagnosis.json'

if __name__ == "__main__":
    generate_model_outputs(SOURCE_JSON, LOG_FOLDER, OUTPUT_FILE)