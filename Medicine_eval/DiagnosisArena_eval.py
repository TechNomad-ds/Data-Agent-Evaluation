# diagnosis_arena_auto_test.py
#!/usr/bin/env python3
"""
DiagnosisArena 自动化测试脚本
自动调用项目内的脚本来测试两种模式
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# 项目配置
PROJECT_CONFIG = {
    "api_key": "sk-OqIPE7A0rEMX8Rwt5NFrxB5TKAruSRGQVw7dUPRh78QpwGUi",
    "base_url": "http://123.129.219.111:3000/v1",
    "model": "deepseek-r1",  # 使用你API支持的模型
    "project_dir": "DiagnosisArena/code",  # 项目脚本目录
    "hf_data_path": "SII-SPIRAL-MED/DiagnosisArena",
    "test_size": 10,  # 测试样本数（可选）
    "folk_nums": 4,  # 并发数，根据API限制调整
    "test_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

class DiagnosisArenaAutoTester:
    """自动化测试管理器"""
    
    def __init__(self, config):
        self.config = config
        self.project_dir = Path(self.config["project_dir"])
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 测试结果文件路径
        self.test_id = f"test_{self.config['test_timestamp']}"
        self.open_ended_files = {
            "answers": self.results_dir / f"{self.test_id}_open_answers.jsonl",
            "evaluated": self.results_dir / f"{self.test_id}_open_evaluated.jsonl",
            "metrics": self.results_dir / f"{self.test_id}_open_metrics.txt"
        }
        self.mcq_files = {
            "answers": self.results_dir / f"{self.test_id}_mcq_answers.jsonl",
            "metrics": self.results_dir / f"{self.test_id}_mcq_metrics.txt"
        }
        
    def print_header(self, title):
        """打印标题"""
        print("\n" + "=" * 70)
        print(f"🔬 {title}")
        print("=" * 70)
    
    def print_step(self, step_num, description):
        """打印步骤信息"""
        print(f"\n📋 步骤 {step_num}: {description}")
        print("-" * 50)
    
    def run_command(self, command, description, output_file=None):
        """运行命令并捕获输出"""
        print(f"  执行: {description}")
        print(f"  命令: {' '.join(command)}")
        
        try:
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    # 写入输出
                    f.write("STDOUT:\n" + result.stdout)
                    f.write("\n\nSTDERR:\n" + result.stderr)
                    f.write(f"\n\n返回码: {result.returncode}")
            else:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False
                )
            
            if result.returncode != 0:
                print(f"  ⚠️  警告: 返回码 {result.returncode}")
                if result.stderr:
                    print(f"  错误输出:\n{result.stderr[:500]}...")
            else:
                print("  ✅ 完成")
                
            return result
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            return None
    
    def modify_script_for_testing(self, script_path, test_size=None):
        """修改脚本以限制测试样本数"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找并取消注释测试行
            test_line = '# input_datas = input_datas.select(range(10))'
            if test_line in content:
                if test_size:
                    new_line = f'input_datas = input_datas.select(range({test_size}))'
                else:
                    new_line = 'input_datas = input_datas.select(range(10))'
                content = content.replace(test_line, new_line)
                print(f"  🔧 修改脚本以测试 {test_size if test_size else 10} 个样本")
            
            # 写入临时文件
            temp_path = script_path.with_suffix('.temp.py')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return temp_path
            
        except Exception as e:
            print(f"  ⚠️  无法修改脚本: {e}")
            return script_path
    
    def test_open_ended_mode(self):
        """测试模式一：开放式诊断"""
        self.print_header("模式一测试：开放式诊断 (Top-5诊断)")
        
        # 步骤1: 生成诊断结果
        self.print_step(1, "生成诊断结果")
        
        # 修改inference.py以限制测试样本
        inference_script = self.project_dir / "inference.py"
        temp_inference = self.modify_script_for_testing(inference_script, self.config["test_size"])
        
        cmd = [
            "python", str(temp_inference),
            "--hf_data_path", self.config["hf_data_path"],
            "--model_name", self.config["model"],
            "--output_path", str(self.open_ended_files["answers"]),
            "--api_key", self.config["api_key"],
            "--base_url", self.config["base_url"],
            "--folk_nums", str(self.config["folk_nums"])
        ]
        
        result = self.run_command(cmd, "运行推理脚本", 
                                 self.open_ended_files["metrics"].with_suffix(".inference.log"))
        
        # 清理临时文件
        if temp_inference != inference_script and temp_inference.exists():
            temp_inference.unlink()
        
        if not result or result.returncode != 0:
            print("  ❌ 推理步骤失败，跳过后续步骤")
            return False
        
        # 检查生成了多少条结果
        if self.open_ended_files["answers"].exists():
            with open(self.open_ended_files["answers"], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"  📊 生成了 {len(lines)} 条诊断结果")
        
        # 步骤2: 使用LLM裁判评分
        self.print_step(2, "使用LLM裁判评分")
        
        cmd = [
            "python", str(self.project_dir / "evaluation.py"),
            "--input_path", str(self.open_ended_files["answers"]),
            "--output_path", str(self.open_ended_files["evaluated"]),
            "--model_name", self.config["model"],
            "--api_key", self.config["api_key"],
            "--base_url", self.config["base_url"],
            "--folk_nums", str(self.config["folk_nums"])
        ]
        
        result = self.run_command(cmd, "运行评估脚本",
                                 self.open_ended_files["metrics"].with_suffix(".evaluation.log"))
        
        if not result or result.returncode != 0:
            print("  ⚠️  评估步骤可能未完全完成")
        
        # 步骤3: 计算指标
        self.print_step(3, "计算Top-k指标")
        
        cmd = [
            "python", str(self.project_dir / "metric.py"),
            "--model_name", self.config["model"],
            "--metric_path", str(self.open_ended_files["evaluated"])
        ]
        
        result = self.run_command(cmd, "运行指标计算脚本")
        
        if result and result.stdout:
            # 保存指标结果
            with open(self.open_ended_files["metrics"], 'w', encoding='utf-8') as f:
                f.write(f"开放式诊断测试结果 - {self.test_id}\n")
                f.write(f"测试时间: {datetime.now()}\n")
                f.write(f"模型: {self.config['model']}\n")
                f.write(f"测试样本数: {self.config['test_size']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result.stdout)
            
            print(f"\n📈 指标结果已保存到: {self.open_ended_files['metrics']}")
            print("\n指标摘要:")
            print(result.stdout)
        
        return True
    
    def test_mcq_mode(self):
        """测试模式二：多选题"""
        self.print_header("模式二测试：多选题 (四选一)")
        
        # 步骤1: 生成答案
        self.print_step(1, "生成多选题答案")
        
        # 修改inference_mcq.py以限制测试样本
        inference_mcq_script = self.project_dir / "inference_mcq.py"
        temp_inference_mcq = self.modify_script_for_testing(inference_mcq_script, self.config["test_size"])
        
        cmd = [
            "python", str(temp_inference_mcq),
            "--hf_data_path", self.config["hf_data_path"],
            "--model_name", self.config["model"],
            "--output_path", str(self.mcq_files["answers"]),
            "--api_key", self.config["api_key"],
            "--base_url", self.config["base_url"],
            "--folk_nums", str(self.config["folk_nums"])
        ]
        
        result = self.run_command(cmd, "运行多选题推理脚本",
                                 self.mcq_files["metrics"].with_suffix(".inference.log"))
        
        # 清理临时文件
        if temp_inference_mcq != inference_mcq_script and temp_inference_mcq.exists():
            temp_inference_mcq.unlink()
        
        if not result or result.returncode != 0:
            print("  ❌ 多选题推理步骤失败，跳过后续步骤")
            return False
        
        # 检查生成了多少条结果
        if self.mcq_files["answers"].exists():
            with open(self.mcq_files["answers"], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"  📊 生成了 {len(lines)} 条多选题答案")
        
        # 步骤2: 计算准确率
        self.print_step(2, "计算准确率")
        
        cmd = [
            "python", str(self.project_dir / "metric_mcq.py"),
            "--model_name", self.config["model"],
            "--metric_path", str(self.mcq_files["answers"])
        ]
        
        result = self.run_command(cmd, "运行多选题指标计算脚本")
        
        if result and result.stdout:
            # 保存指标结果
            with open(self.mcq_files["metrics"], 'w', encoding='utf-8') as f:
                f.write(f"多选题测试结果 - {self.test_id}\n")
                f.write(f"测试时间: {datetime.now()}\n")
                f.write(f"模型: {self.config['model']}\n")
                f.write(f"测试样本数: {self.config['test_size']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result.stdout)
            
            print(f"\n📈 指标结果已保存到: {self.mcq_files['metrics']}")
            print("\n指标摘要:")
            print(result.stdout)
        
        return True
    
    def generate_summary_report(self):
        """生成测试摘要报告"""
        summary_file = self.results_dir / f"{self.test_id}_summary.md"
        
        # 确保结果目录存在
        self.results_dir.mkdir(exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# DiagnosisArena 测试摘要\n\n")
            f.write(f"**测试ID**: {self.test_id}\n")
            f.write(f"**测试时间**: {datetime.now()}\n")
            f.write(f"**模型**: {self.config['model']}\n")
            f.write(f"**API端点**: {self.config['base_url']}\n")
            f.write(f"**测试样本数**: {self.config['test_size']}\n")
            f.write(f"**并发数**: {self.config['folk_nums']}\n\n")
            
            f.write("## 文件输出\n\n")
            
            # 开放式诊断结果
            f.write("### 模式一：开放式诊断\n")
            f.write(f"- 诊断结果: `{self.open_ended_files['answers'].name}`\n")
            f.write(f"- 评估结果: `{self.open_ended_files['evaluated'].name}`\n")
            f.write(f"- 指标文件: `{self.open_ended_files['metrics'].name}`\n\n")
            
            # 多选题结果
            f.write("### 模式二：多选题\n")
            f.write(f"- 答案文件: `{self.mcq_files['answers'].name}`\n")
            f.write(f"- 指标文件: `{self.mcq_files['metrics'].name}`\n\n")
            
            f.write("## 快速查看结果\n\n")
            f.write("```bash\n")
            f.write("# 查看开放式诊断指标\n")
            # 修复：使用相对于当前目录的路径
            rel_open_path = self.open_ended_files['metrics'].relative_to(Path.cwd())
            f.write(f"cat {rel_open_path}\n\n")
            
            f.write("# 查看多选题指标\n")
            rel_mcq_path = self.mcq_files['metrics'].relative_to(Path.cwd())
            f.write(f"cat {rel_mcq_path}\n\n")
            
            f.write("# 查看所有测试文件\n")
            f.write(f"ls -la {self.results_dir.name}/\n")
            f.write("```\n")
        
        print(f"\n📋 测试摘要已保存到: {summary_file}")
        return summary_file
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 DiagnosisArena 自动化测试开始")
        print(f"📁 项目目录: {self.project_dir.absolute()}")
        print(f"💾 结果目录: {self.results_dir.absolute()}")
        print(f"🤖 测试模型: {self.config['model']}")
        print(f"🔗 API端点: {self.config['base_url']}")
        print(f"📊 测试样本: {self.config['test_size']} 个\n")
        
        start_time = time.time()
        
        # 检查项目目录是否存在
        if not self.project_dir.exists():
            print(f"❌ 错误: 项目目录不存在: {self.project_dir}")
            print("请确保项目结构为:")
            print("  /当前目录")
            print("  ├── diagnosis_arena_auto_test.py  (本脚本)")
            print("  └── DiagnosisArena/")
            print("      └── code/  (项目脚本目录)")
            return False
        
        # 检查必要脚本
        required_scripts = ["inference.py", "evaluation.py", "metric.py", "inference_mcq.py", "metric_mcq.py"]
        missing = []
        for script in required_scripts:
            if not (self.project_dir / script).exists():
                missing.append(script)
        
        if missing:
            print(f"❌ 错误: 缺少必要的脚本文件: {missing}")
            return False
        
        # 运行测试
        test_results = {}
        
        try:
            # 测试模式一
            test_results["open_ended"] = self.test_open_ended_mode()
            
            # 等待一下避免API过载
            time.sleep(2)
            
            # 测试模式二
            test_results["mcq"] = self.test_mcq_mode()
            
            # 生成摘要报告
            summary_file = self.generate_summary_report()
            
            total_time = time.time() - start_time
            
            print("\n" + "=" * 70)
            print("🎉 测试完成！")
            print("=" * 70)
            print(f"⏱️  总用时: {total_time:.1f} 秒")
            print(f"📁 结果目录: {self.results_dir.absolute()}")
            
            if summary_file.exists():
                print(f"📋 测试摘要: {summary_file.relative_to(Path.cwd())}")
            
            print("\n📊 测试结果文件:")
            for file in self.results_dir.glob(f"{self.test_id}_*"):
                print(f"  - {file.name}")
            
            print("\n🔍 查看结果:")
            print(f"  cat {self.results_dir.name}/{self.test_id}_summary.md")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️  测试被用户中断")
            return False
        except Exception as e:
            print(f"\n❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DiagnosisArena 自动化测试脚本')
    parser.add_argument('--project_dir', type=str, default="DiagnosisArena/code",
                       help='项目脚本目录路径')
    parser.add_argument('--model', type=str, default="deepseek-r1",
                       help='要测试的模型名称')
    parser.add_argument('--test_size', type=int, default=10,
                       help='测试样本数量')
    parser.add_argument('--folk_nums', type=int, default=4,
                       help='并发请求数')
    parser.add_argument('--skip_open', action='store_true',
                       help='跳过开放式诊断测试')
    parser.add_argument('--skip_mcq', action='store_true',
                       help='跳过多选题测试')
    
    args = parser.parse_args()
    
    # 更新配置
    config = PROJECT_CONFIG.copy()
    config.update({
        "project_dir": args.project_dir,
        "model": args.model,
        "test_size": args.test_size,
        "folk_nums": args.folk_nums
    })
    
    # 创建测试器
    tester = DiagnosisArenaAutoTester(config)
    
    # 运行测试
    tester.run_all_tests()

if __name__ == "__main__":
    main()