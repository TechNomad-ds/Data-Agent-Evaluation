import json
import os
import re

def generate_reasoning_outputs(source_json_path, log_folder_path, output_json_path, model_name='deepseek-r1'):
    """
    结合源文件，遍历log文件夹生成推理评估所需的 model-outputs JSON。
    """
    # 1. 加载源文件，获取所有合法的 case_id
    if not os.path.exists(source_json_path):
        print(f"错误：找不到源文件 {source_json_path}")
        return
    
    with open(source_json_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    valid_ids = set(source_data.keys())
    model_outputs_final = {}

    # 2. 遍历 log 文件夹
    if not os.path.exists(log_folder_path):
        print(f"错误：找不到 log 文件夹 {log_folder_path}")
        return

    log_files = [f for f in os.listdir(log_folder_path) if f.endswith('.json')]
    print(f"开始处理，在文件夹中发现 {len(log_files)} 个文件...")

    for log_file in log_files:
        # 从文件名提取 PMC 编号
        match = re.search(r'PMC\d+', log_file)
        if not match:
            continue
        
        case_id = match.group()
        
        # 关键步骤：只处理源文件中存在的病例
        if case_id not in valid_ids:
            # print(f"跳过：{case_id} 不在源文件中")
            continue

        log_file_path = os.path.join(log_folder_path, log_file)
        try:
            with open(log_file_path, 'r', encoding='utf-8') as lf:
                log_data = json.load(lf)
                
                # 根据你提供的 log 样例，优先读取 deepseek_messages
                messages = log_data.get("deepseek_messages") or log_data.get("output_messages")
                
                if messages:
                    # 按照评估脚本要求的结构填充
                    model_outputs_final[case_id] = {
                        model_name: {
                            "messages": messages
                        }
                    }
        except Exception as e:
            print(f"解析文件 {log_file} 时出错: {e}")

    # 3. 保存最终生成的 model-outputs 文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as out_f:
        json.dump(model_outputs_final, out_f, ensure_ascii=False, indent=4)
    
    print(f"转换成功！共处理 {len(model_outputs_final)} 个匹配的病例。")
    print(f"结果已保存至: {output_json_path}")

# --- 配置路径 ---
SOURCE_FILE = '../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json' # 源文件
LOG_DIR = '../../data/InferenceResults/free_turn_my_test_model' # 存放 log_PMCxxx.json 的文件夹
OUTPUT_PATH = '../../data/InferenceResults/free_turn_assessment_recommendation+final_diagnosis.json'

if __name__ == "__main__":
    generate_reasoning_outputs(SOURCE_FILE, LOG_DIR, OUTPUT_PATH)