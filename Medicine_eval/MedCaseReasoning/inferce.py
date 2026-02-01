import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="MedCaseReasoning LLM 推理脚本 - 优化推理内容提取")
    parser.add_argument("--input_path", type=str, default="./data/test.jsonl", help="本地测试集路径")
    parser.add_argument("--output_path", type=str, default="./results/medcase_results.jsonl", help="输出结果路径")
    parser.add_argument("--mode", type=str, choices=["few_shot", "all"], default="few_shot", help="模式：few_shot/all")
    parser.add_argument("--test_size", type=int, default=3, help="few_shot数量")
    parser.add_argument("--model_name", type=str, default="deepseek-r1", help="模型名称")
    parser.add_argument("--api_key", type=str, default="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", help="你的 API Key")
    parser.add_argument("--base_url", type=str, default="http://123.129.219.111:3000/v1", help="API URL")
    parser.add_argument("--threads", type=int, default=5, help="并发线程数")
    return parser.parse_args()

PROMPT_TEMPLATE = """Read the following case presentation and give the most likely diagnosis.
First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.
Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.

----------------------------------------
CASE PRESENTATION
----------------------------------------
%s
----------------------------------------
OUTPUT TEMPLATE
----------------------------------------
<think>
...your internal reasoning for the diagnosis...
</think>
<answer>
...the name of the disease/entity...
</answer>"""

def process_item(item, client, args):
    case_input = item.get("case_prompt", "")
    full_prompt = PROMPT_TEMPLATE % case_input

    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
        )
        
        # --- 核心修改部分：提取推理内容 ---
        message = response.choices[0].message
        content = message.content or ""
        
        # 很多提供商（如 DeepSeek, OpenRouter）将 R1 的思维链放在 reasoning_content 中
        reasoning = getattr(message, 'reasoning_content', None)
        
        # 如果有独立的推理字段，将其包裹在 <think> 中并拼接到正文前
        if reasoning:
            full_response = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            # 如果没有独立字段，看看 content 里是否已经自带了 <think> 标签
            if "<think>" not in content and content.strip():
                # 如果既没标签也没独立字段，为了评分脚本能跑，我们将整个 content 视为推理
                # 这通常发生在非 R1 模型但被要求推理时
                full_response = f"<think>\n{content}\n</think>\n<answer>\n{content}\n</answer>"
            else:
                full_response = content

        result = {
            "pmcid": item.get("pmcid", "N/A"),
            "final_diagnosis": item.get("final_diagnosis", ""),
            "diagnostic_reasoning": item.get("diagnostic_reasoning", ""),
            "llm_response": full_response
        }
        
        with open(args.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"\n出错 [PMCID: {item.get('pmcid')}]: {e}")

def main():
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    if not os.path.exists(args.input_path):
        print(f"找不到输入文件: {args.input_path}")
        return

    with open(args.input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    if args.mode == "few_shot":
        print(f"[小样本模式] 抽取前 {args.test_size} 条")
        all_data = all_data[:args.test_size]
    
    # 过滤掉已经处理过的 PMCID
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            done_pmcids = {json.loads(line).get("pmcid") for line in f if line.strip()}
        all_data = [d for d in all_data if d.get("pmcid") not in done_pmcids]
        if done_pmcids:
            print(f"已跳过 {len(done_pmcids)} 条已完成的数据")

    if not all_data:
        print("所有任务已完成！")
        return

    print(f"开始测试... 并发数: {args.threads}")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        list(tqdm(executor.map(lambda x: process_item(x, client, args), all_data), total=len(all_data)))

if __name__ == "__main__":
    main()