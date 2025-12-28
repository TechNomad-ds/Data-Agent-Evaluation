import os
import subprocess
import shutil
from pathlib import Path
from openai import OpenAI
import json

# --- 配置区域 ---
PROJECT_NAME = "MedCaseReasoning"  # 你的项目文件夹名称
RESULT_DIR = Path("MedCaseReasoning_result")
INTERMEDIATE_DIR = RESULT_DIR / "intermediate_data"
EVALUATION_DIR = RESULT_DIR / "final_evaluation"

# API 配置
API_KEY = "sk-OqIPE7A0rEMX8Rwt5NFrxB5TKAruSRGQVw7dUPRh78QpwGUi"
BASE_URL = "http://123.129.219.111:3000/v1"
MODEL_NAME = "deepseek-r1"
USER_EMAIL = "your_email@example.com" # NCBI 接口需要

def setup_directories():
    """创建结果输出目录结构"""
    for d in [RESULT_DIR, INTERMEDIATE_DIR, EVALUATION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ 目录结构已准备就绪: {RESULT_DIR}")

def run_command(command, description):
    """模拟终端执行命令"""
    print(f"\n🚀 正在执行: {description}...")
    try:
        # 在项目文件夹内执行脚本
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            cwd=PROJECT_NAME,
            capture_output=True, 
            text=True
        )
        print(f"✅ 完成: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {description}\n错误信息: {e.stderr}")
        return None

def move_intermediate_files():
    """将项目产生的中间文件移动到结果文件夹"""
    print("\n📦 整理中间产物...")
    # 定义需要移动的文件列表（基于项目 README 提到的产物）
    files_to_move = [
        "case_report_pmcids.csv",
        "metadata.csv",
        "case_reports_text.parquet",
        "extracted_case_reports"
    ]
    
    project_path = Path(PROJECT_NAME)
    for item in files_to_move:
        src = project_path / item
        dst = INTERMEDIATE_DIR / item
        if src.exists():
            if dst.exists():
                if dst.is_dir(): shutil.rmtree(dst)
                else: os.remove(dst)
            shutil.move(str(src), str(dst))
            print(f"  - 已移动: {item}")

def run_llm_inference_api():
    """使用提供的 API 接口进行最终评测模拟"""
    print(f"\n🧠 启动 LLM API 推理测试 (模型: {MODEL_NAME})...")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个医疗诊断助手，请根据提供的病历输出诊断结果。"},
                {"role": "user", "content": "模拟测试：患者表现为高热、咳嗽、咳痰，胸片显示肺部浸润影。最可能的诊断是？"}
            ],
            max_tokens=200
        )
        
        result_content = response.choices[0].message.content
        
        # 保存最终评测结果
        output_file = EVALUATION_DIR / "api_inference_result.json"
        result_data = {
            "model": response.model,
            "diagnosis_response": result_content,
            "usage": response.usage.dict() if response.usage else "N/A"
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 最终评测结果已保存至: {output_file}")
        print(f"📢 API 回复摘要: {result_content[:50]}...")
        
    except Exception as e:
        print(f"❌ API 调用失败: {e}")

def main():
    setup_directories()
    
    # 1. 下载数据
    run_command("python download_pmc.py", "下载 PMC 原始数据")
    
    # 2. 获取 ID 列表
    run_command(f"python get_case_report_pmcids.py --start-date 2015/01/01 --email {USER_EMAIL}", "获取病例 ID")
    
    # 3. 提取 XML
    run_command("python process_pmc.py", "并行提取匹配 XML")
    
    # 4. 生成元数据与文本
    run_command("python extract_metadata.py", "提取文章元数据")
    run_command("python extract_text.py", "提取并清洗正文文本")
    
    # 5. 整理文件
    move_intermediate_files()
    
    # 6. API 评测
    run_llm_inference_api()

    print("\n" + "="*30)
    print("🎉 全流程模拟运行完成！")
    print(f"中间产物见: {INTERMEDIATE_DIR}")
    print(f"评测结果见: {EVALUATION_DIR}")
    print("="*30)

if __name__ == "__main__":
    main()