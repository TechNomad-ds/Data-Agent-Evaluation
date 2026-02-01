import re
import os

# ==================== 全局变量设置区 ====================
NEW_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
NEW_BASE_URL = "http://123.129.219.111:3000/v1"
NEW_MODEL_NAME = "deepseek-r1"
IS_FULL_EVAL = False  # True 为全量测试, False 为少量测试
# ========================================================

def update_file(file_path, patterns):
    if not os.path.exists(file_path):
        print(f" 找不到文件: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f" 已更新: {file_path}")

def run_configuration():
    print(f" 开始同步配置... (全量模式: {IS_FULL_EVAL})")

    # 1. DiagnosisArena
    update_file("DiagnosisArena_eval.py", [
        (r'"api_key":\s*".*?"', f'"api_key": "{NEW_API_KEY}"'),
        (r'"base_url":\s*".*?"', f'"base_url": "{NEW_BASE_URL}"'),
        (r'"model":\s*".*?"', f'"model": "{NEW_MODEL_NAME}"'),
        (r'"test_size":\s*\d+', f'"test_size": {0 if IS_FULL_EVAL else 3}')
    ])

    # 2. MedXpertQA
    # 修改主入口
    update_file("MedXpertQA_eval.py", [
        (r'--model",\s*default=".*?"', f'--model", default="{NEW_MODEL_NAME}"'),
        (r'--dataset",\s*default=".*?"', f'--dataset", default="{"medxpertqa" if IS_FULL_EVAL else "medxpertqa_sampled"}"')
    ])
    # 修改 API 代理
    update_file("MedXpertQA/eval/model/api_agent.py", [
        (r'self\.api_key\s*=\s*".*?"', f'self.api_key = "{NEW_API_KEY}"'),
        (r'self\.base_url\s*=\s*".*?"', f'self.base_url = "{NEW_BASE_URL}"')
    ])

    # 3. MedRBench (one_turn, free_turn, oracle_*)
    medrbench_files = [
        "MedRBench/src/Inference/one_turn.py",
        "MedRBench/src/Inference/free_turn.py",
        "MedRBench/src/Inference/oracle_diagnose.py",
        "MedRBench/src/Inference/oracle_treatment_planning.py"
    ]
    for rb_file in medrbench_files:
        rb_patterns = [
            (r"MY_MODEL_URL\s*=\s*'.*?'", f"MY_MODEL_URL = '{NEW_BASE_URL}'"),
            (r"MY_MODEL_API_KEY\s*=\s*'.*?'", f"MY_MODEL_API_KEY = '{NEW_API_KEY}'"),
            (r"MY_MODEL_NAME\s*=\s*'.*?'", f"MY_MODEL_NAME = '{NEW_MODEL_NAME}'")
        ]
        # 处理全量逻辑 (处理 keys = ...[:2] 或 items = ...[:2])
        if IS_FULL_EVAL:
            rb_patterns.append((r'(\.keys\(\)| \.items\(\))\[:\d+\]', r'\1'))
        else:
            rb_patterns.append((r'(\.keys\(\)| \.items\(\))(?![\[])', r'\1[:2]'))
        update_file(rb_file, rb_patterns)

    # 4. MedCaseReasoning
    update_file("MedCaseReasoning/inferce.py", [
        (r'--test_size",\s*type=int,\s*default=\d+', f'--test_size", type=int, default={897 if IS_FULL_EVAL else 3}'),
        (r'--model_name",\s*type=str,\s*default=".*?"', f'--model_name", type=str, default="{NEW_MODEL_NAME}"'),
        (r'--api_key",\s*type=str,\s*default=".*?"', f'--api_key", type=str, default="{NEW_API_KEY}"'),
        (r'--base_url",\s*type=str,\s*default=".*?"', f'--base_url", type=str, default="{NEW_BASE_URL}"')
    ])

    # 5. HELM
    # YAML 修改建议使用正则适配 name 字段
    helm_configs = [
        ("/opt/conda/envs/mineru/lib/python3.10/site-packages/helm/config/model_deployments.yaml", [
            (r'name: my_org/.*', f'name: my_org/{NEW_MODEL_NAME}'),
            (r'model_name: my_org/.*', f'model_name: my_org/{NEW_MODEL_NAME}'),
            (r'tokenizer_name: my_org/.*', f'tokenizer_name: my_org/{NEW_MODEL_NAME}'),
            (r'api_key:\s*".*?"', f'api_key: "{NEW_API_KEY}"'),
            (r'base_url:\s*".*?"', f'base_url: "{NEW_BASE_URL}"')
        ]),
        ("/opt/conda/envs/mineru/lib/python3.10/site-packages/helm/config/model_metadata.yaml", [
            (r'- name: my_org/.*', f'- name: my_org/{NEW_MODEL_NAME}')
        ])
    ]
    for h_file, h_patterns in helm_configs:
        update_file(h_file, h_patterns)

    # 6. 修改 Shell 运行脚本 (HELM 的 max-eval-instances)
    shell_run_script = "run_all_evals.sh" 
    if IS_FULL_EVAL:
        update_file(shell_run_script, [(r'--max-eval-instances\s+\d+\s*\\', '')])
    else:
        # 确保存在该参数
        update_file(shell_run_script, [(r'--suite medical_test_v1', '--suite medical_test_v1 --max-eval-instances 5 \\')])

    print("\n 所有配置同步完成！现在可以运行测试脚本了。")

if __name__ == "__main__":
    run_configuration()