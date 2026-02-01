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
    print(f"已更新: {file_path}")

def run_configuration():
    print(f"开始同步 Business_eval 配置... (全量模式: {IS_FULL_EVAL})")

    # 1. FinCDM
    update_file("FinCDM/src/api-eval.py", [
        (r'api_key\s*=\s*".*?"', f'api_key="{NEW_API_KEY}"'),
        (r'base_url\s*=\s*".*?"', f'base_url="{NEW_BASE_URL}"'),
        (r"models\s*=\s*\[\s*'.*?'\s*\]", f"models = ['{NEW_MODEL_NAME}']"),
        (r'num_trials\s*=\s*\d+', f'num_trials = {101 if IS_FULL_EVAL else 5}')
    ])

    # 2. FinEval-KR
    update_file("FinEval-KR/genera_a.py", [
        (r'api_key\s*=\s*".*?"', f'api_key="{NEW_API_KEY}"'),
        (r'base_url\s*=\s*".*?"', f'base_url="{NEW_BASE_URL}"'),
        (r'model\s*=\s*".*?"', f'model="{NEW_MODEL_NAME}"')
    ])

    # 3. XFinBench
    # 修改 API 密钥
    update_file("XFinBench/evaluate/load_models.py", [
        (r'MY_API_KEY\s*=\s*".*?"', f'MY_API_KEY = "{NEW_API_KEY}"'),
        (r'MY_BASE_URL\s*=\s*".*?"', f'MY_BASE_URL = "{NEW_BASE_URL}"')
    ])
    # 修改模型名称和数据量限制
    xfb_main_patterns = [
        (r'--model",\s*default=\'.*?\'', f'--model", default=\'{NEW_MODEL_NAME}\'')
    ]
    if IS_FULL_EVAL:
        xfb_main_patterns.append((r'\.index\.tolist\(\)\[:\d+\]', r'.index.tolist()'))
    else:
        xfb_main_patterns.append((r'\.index\.tolist\(\)(?!\[)', r'.index.tolist()[:15]'))
    update_file("XFinBench/evaluate/main.py", xfb_main_patterns)
    
    # 修改 run.sh
    update_file("XFinBench/run.sh", [
        (r'MODEL_NAME\s*=\s*".*?"', f'MODEL_NAME="{NEW_MODEL_NAME}"')
    ])

    # 4. PIXIU
    # 注意 PIXIU 的 URL 包含 /chat/completions
    full_url = NEW_BASE_URL.rstrip('/') + "/chat/completions"
    update_file("PIXIU/src/chatlm.py", [
        (r'url\s*=\s*".*?"', f'url="{full_url}"'),
        (r'api_key\s*=\s*".*?"', f'api_key = "{NEW_API_KEY}"')
    ])
    
    # 修改外部运行脚本 (run_all_evals.sh) 中的 PIXIU 部分
    shell_run_script = "run_all_evals.sh"
    pixiu_patterns = [(r'--model\s+[^\s\\]+', f'--model {NEW_MODEL_NAME}')]
    if IS_FULL_EVAL:
        pixiu_patterns.append((r'--limit\s+\d+\s*\\', ''))
    else:
        # 如果没有 --limit 则尝试加上
        pixiu_patterns.append((r'--batch_size 1', '--batch_size 1 --limit 5 \\'))
    update_file(shell_run_script, pixiu_patterns)

    # 5. CFA_ESSAY_REPRODUCER
    update_file("CFA_ESSAY_REPRODUCER/src/llm_clients/openrouter_client.py", [
        (r'base_url\s*=\s*".*?"', f'base_url="{NEW_BASE_URL}"'),
        (r'api_key\s*=\s*".*?"', f'api_key= "{NEW_API_KEY}"')
    ])
    update_file("CFA_ESSAY_REPRODUCER/src/main.py", [
        (r'selected_model_ids_for_run\s*=\s*\[\s*".*?"\s*\]', f'selected_model_ids_for_run = ["{NEW_MODEL_NAME}"]')
    ])

    print("\nBusiness_eval 所有配置同步完成！")

if __name__ == "__main__":
    run_configuration()