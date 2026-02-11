import re
import os

# ==================== 全局变量设置区 ====================
NEW_API_KEY = "sk-ptwvi0XgJUL6JyjyMS9faFUSDbw5vxC1hX0EqQvxeUUG0Kv7"
NEW_BASE_URL = "http://123.129.219.111:3000/v1"
NEW_MODEL_NAME = "deepseek-r1"
IS_FULL_EVAL = False  # True 为全量测试, False 为少量测试
# ========================================================

def update_file(file_path, patterns):
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已更新: {file_path}")

def run_configuration():
    print(f"开始同步 Law_eval 配置... (全量模式: {IS_FULL_EVAL})")

    # 1. legalbench (修改 eval.py)
    legalbench_patterns = [
        (r'API_KEY\s*=\s*".*?"', f'API_KEY = "{NEW_API_KEY}"'),
        (r'BASE_URL\s*=\s*".*?"', f'BASE_URL = "{NEW_BASE_URL}"'),
        (r'MODEL_NAME\s*=\s*".*?"', f'MODEL_NAME = "{NEW_MODEL_NAME}"')
    ]
    if IS_FULL_EVAL:
        # 全量：删除 .head(SAMPLE_LIMIT)
        legalbench_patterns.append((r'\.to_pandas\(\)\.head\(SAMPLE_LIMIT\)', r'.to_pandas()'))
    else:
        # 少量：添加 .head(SAMPLE_LIMIT) (如果已经有了则不重复添加)
        legalbench_patterns.append((r'\.to_pandas\(\)(?!\.head)', r'.to_pandas().head(SAMPLE_LIMIT)'))
    update_file("legalbench/eval.py", legalbench_patterns)

    # 2. LegalEval-Q (修改 eval.py)
    legaleval_q_patterns = [
        (r'API_KEY\s*=\s*".*?"', f'API_KEY = "{NEW_API_KEY}"'),
        (r'BASE_URL\s*=\s*".*?"', f'BASE_URL = "{NEW_BASE_URL}"'),
        (r'MODEL_NAME\s*=\s*".*?"', f'MODEL_NAME = "{NEW_MODEL_NAME}"'),
        (r'SAMPLE_SIZE\s*=\s*.*', f'SAMPLE_SIZE = {"None" if IS_FULL_EVAL else "3"}')
    ]
    update_file("LegalEval-Q/eval.py", legaleval_q_patterns)

    # 3. LEXam (修改 eval.py)
    lexam_patterns = [
        (r'API_BASE_URL\s*=\s*".*?"', f'API_BASE_URL = "{NEW_BASE_URL}"'),
        (r'API_KEY\s*=\s*".*?"', f'API_KEY = "{NEW_API_KEY}"'),
        (r'MODEL_NAME\s*=\s*".*?"', f'MODEL_NAME = "{NEW_MODEL_NAME}"'),
        (r'NUM_SAMPLES\s*=\s*.*', f'NUM_SAMPLES = {"-1" if IS_FULL_EVAL else "3"}')
    ]
    update_file("LEXam/eval.py", lexam_patterns)

    print("\n Law_eval 所有配置同步完成！")

if __name__ == "__main__":
    run_configuration()