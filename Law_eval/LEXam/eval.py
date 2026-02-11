import os
import pandas as pd
import ast
from tqdm import tqdm
from openai import OpenAI

API_BASE_URL = "http://123.129.219.111:3000/v1"
API_KEY = "sk-ptwvi0XgJUL6JyjyMS9faFUSDbw5vxC1hX0EqQvxeUUG0Kv7"
MODEL_NAME = "gpt-4o-mini"
NUM_SAMPLES = 3  # 测试数量

GENE_ARGS = {
    "temperature": 0.1,
    "max_tokens": 4096
}

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


QA_PROMPT = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
Assume Swiss law applies unless specifically mentioned; if the course context justifies, address legal issues beyond Swiss law as well.
Use precise legal language and formal "Sie" when answering.
Do NOT state any disclaimer or refer to the need for external legal advice.
Do NOT request the user to consult laws or to research on their own.
Offer focused legal analyses and individualized advice.
Speak directly and authoritatively without mentioning that your response is merely for general information.
Incorporate Swiss-specific legal terminology.
If you have discovered relevant legal considerations (Erwägungen), respond with a concise, clear legal analysis.
Cite only from your identified considerations.
Always cite the specific legal provision, explicitly indicating paragraphs (Abs.), numbers (Ziff.), or letters (lit.) where available (e.g., “'Art. 74 Abs. 2 Ziff. 2 OR”, “Art. 336 lit. a StGB”). Avoid general references (such as 'Art. 3 ZGB') without mentioning the specific paragraph, number, or letter, if applicable.
If no relevant considerations are found, explicitly state that no pertinent information is available.
If you do have reliable sources, share practical guidance or insights from them.
Respond in the same language as the question.
If the question specifically requests a short answer, provide a concise response.
If the prompt asks you to analyze a specific case provided in the exam, but the text or details of that case have not been provided in the prompt, explicitly flag that the required case material is missing.

Question:
{question}

Answer:"""


MCQ_PROMPT = {
    "letters": """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., A, B, C, etc.) with a brief justification for why it best fits the legal analysis.

Format your final answer as follows:
 Correct Answer: ###C### 

Question:
 {question}

Answer:""",
    "numbers": """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., 1, 2, 3, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., 1, 2, 3, etc.) with a brief justification for why it best fits the legal analysis.

Format your final answer as follows:
 Correct Answer: ###3### 

Question:
 {question}

Answer:""",
}



def get_model_response(messages):
    """发送同步请求给模型"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            **GENE_ARGS
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling API: {e}")
        return f"Error: {str(e)}"

def run_evaluation_task(input_file, task_type, output_file):
    """
    执行单个评测任务（MCQ 或 QA）
    task_type 选项: 'mcq_letters', 'mcq_numbers', 'open_questions'
    """
    print(f"\n>>> 开始执行任务: {task_type}")
    print(f"Reading file: {input_file}")

    # 1. 读取数据 (支持 Excel 和 CSV)
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    # 2. 小样本截断
    if NUM_SAMPLES > 0:
        print(f"Running on first {NUM_SAMPLES} samples only.")
        df = df.head(NUM_SAMPLES)

    # 3. 准备数据字段
    questions = df['question'].tolist()
    # 如果没有 course 列，默认填充 'Law'
    course_names = df['course'].tolist() if 'course' in df.columns else ["Law"] * len(df)
    
    prompts = []

    # 4. 格式化题目 (核心逻辑：处理选项拼接)
    if task_type.startswith('mcq'):
        # 确保存在 choices 列
        if 'choices' not in df.columns:
            raise ValueError(f"Task type {task_type} requires a 'choices' column in input file.")

        # 将字符串类型的列表 "['A', 'B']" 转回 Python 列表
        choices = [ast.literal_eval(c) if isinstance(c, str) else c for c in df['choices'].tolist()]
        
        formatted_questions = []
        # 定义标签生成器 (A,B,C... 或 1,2,3...)
        is_letter = (task_type == 'mcq_letters')
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if is_letter else [str(i+1) for i in range(26)]
        
        # 循环拼接：题干 + 换行 + 选项
        for q, opts in zip(questions, choices):
            formatted_q = q
            for i, opt in enumerate(opts):
                # 防止选项超过标签范围
                label = labels[i] if i < len(labels) else str(i)
                formatted_q += f'\n{label}. {opt}'
            formatted_questions.append(formatted_q)
            
        # 填入 Prompt
        template = MCQ_PROMPT['letters'] if is_letter else MCQ_PROMPT['numbers']
        for q, c_name in zip(formatted_questions, course_names):
            prompts.append([{'role': 'user', 'content': template.format(course_name=c_name, question=q)}])
            
    else: # Open Questions (简答题)
        for q, c_name in zip(questions, course_names):
            prompts.append([{'role': 'user', 'content': QA_PROMPT.format(course_name=c_name, question=q)}])

    # 5. 同步循环调用模型
    print(f"Total prompts to process: {len(prompts)}")
    responses = []
    
    # 使用 tqdm 显示进度条
    for p in tqdm(prompts, desc=f"Processing {task_type}"):
        ans = get_model_response(p)
        responses.append(ans)

    # 6. 保存结果 (增强兼容性版)
    # 自动识别原始数据中的答案列
    if 'gold' in df.columns:
        gold_vals = df['gold'].tolist()
    elif 'answer' in df.columns:
        gold_vals = df['answer'].tolist()
    else:
        gold_vals = df.get('gold_answer', ['N/A'] * len(df)).tolist()

    output_df = pd.DataFrame({
        'prompt': [p[0]['content'] for p in prompts],
        'question': questions,
        'course': course_names, # 建议保留，方便后续分析
        'gold_answer': gold_vals,  # 统一列名，对接 evaluation.py
        f'{MODEL_NAME}_answer': responses 
    })

    # 针对 MCQ 的特殊处理：如果 gold_answer 是字符串形式的 A,B,C，需确保它与 evaluation.py 的预期一致
    # 如果你的原始数据 'gold' 列已经是数字索引 (0, 1, 2)，则直接保持即可。

    os.makedirs("./results", exist_ok=True)
    output_df.to_csv(output_file, encoding='utf-8', index=False)


# ==========================================
# 4. 主程序入口 (在此指定文件名)
# ==========================================
if __name__ == '__main__':
    # 请根据你实际的文件名修改这里
    MCQ_FILE = "./data/MCQs_test_8.xlsx"   # 你的多选题文件路径
    OQ_FILE = "./data/open_questions_test.xlsx"     # 你的简答题文件路径
    
    # 任务 1: 运行多选题评测
    # task_type 可选: 'mcq_letters' 或 'mcq_numbers'
    if os.path.exists(MCQ_FILE):
        run_evaluation_task(
            input_file=MCQ_FILE, 
            task_type="mcq_letters", 
            output_file=f"./results/result_mcq_{MODEL_NAME}.csv"
        )
    else:
        print(f"文件 {MCQ_FILE} 不存在，跳过 MCQ 任务。")

    # 任务 2: 运行简答题评测
    if os.path.exists(OQ_FILE):
        run_evaluation_task(
            input_file=OQ_FILE, 
            task_type="open_questions", 
            output_file=f"./results/result_oq_{MODEL_NAME}.csv"
        )
    else:
        print(f"文件 {OQ_FILE} 不存在，跳过 QA 任务。")

