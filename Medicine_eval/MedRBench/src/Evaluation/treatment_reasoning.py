import os
import json
import logging
import argparse
import multiprocessing
from multiprocessing import Manager

from utils import split_reasoning
from metrics.reasoning_eval import (
    eval_reasoning_efficiency_factuality,
    eval_reasoning_completeness
)

# Configuration constants
NUM_WORKERS = 8  # Number of worker processes for parallel execution
MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts for API calls
EVALUATION_MODEL = "gpt-4o-2024-11-20"  # Model to be used for evaluation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_case(data, save_root, model_name):
    """Evaluate reasoning quality for a specific model's output on a single case."""
    logger.info(f'Evaluating case {data["id"]} for model {model_name}')
    error_log_file = f'{model_name}_error.log'
    

    # 1. 提取基础信息 (基于你提供的 generate_case 结构)
    gen_case = data.get('generate_case', {})
    case_info = gen_case.get('case_summary', "")
    
    # 标准答案 (Ground Truth)
    gt_answer = gen_case.get('treatment_plan_results', "")
    # 标准推理路径
    gt_analysis = gen_case.get('treatment_planning_analysis', "")
    gt_reasoning = f"{gt_analysis}\n Treatment plan results:\n{gt_answer}"
            
    # 2. 提取模型输出 (基于你提供的 result 结构)
    res_obj = data.get('result', {})
    
    # 逻辑：优先取 reasoning 字段，如果没有则取 content
    # DeepSeek-R1 这种模型通常会将 <think> 里的内容放在 reasoning
    raw_model_text = res_obj.get('reasoning', "").strip()
    if not raw_model_text:
        raw_model_text = res_obj.get('content', "").strip()
    
    if not raw_model_text:
        logger.warning(f"ID: {data['id']} has no model output text to evaluate.")
        return

    # 3. 拆分推理步骤
    try:
        reasoning_steps = split_reasoning(raw_model_text)
        if not reasoning_steps:
            # 如果 split_reasoning 返回空列表，则将全文作为一个 step
            reasoning_steps = [raw_model_text]
    except Exception as e:
        logger.error(f"Error calling split_reasoning for {data['id']}: {e}")
        reasoning_steps = [raw_model_text]

    combined_reasoning = '\n'.join(reasoning_steps)
    print("DEBUG: Extracted reasoning steps:")
    # 4. 评估：效率与事实性 (Efficiency and Factuality)
    efficiency_factuality_results = None
    for attempt in range(MAX_RETRY_ATTEMPTS):
        print("DEBUG: Starting efficiency and factuality evaluation, attempt", attempt + 1)
        efficiency_factuality_results = eval_reasoning_efficiency_factuality(
            case_info=case_info,
            pred_reasoning_steps_list=reasoning_steps,
            gt_answer=gt_answer,
            is_treatment=True,
            evaluation_model=EVALUATION_MODEL
        )
        print(f"Attempt {attempt + 1} results: {efficiency_factuality_results}")
        if efficiency_factuality_results:
            break
        
    

    # 5. 评估：完整性 (Recall/Completeness)
    print("Starting completeness evaluation")
    completeness_results = None
    for attempt in range(MAX_RETRY_ATTEMPTS):
        completeness_results = eval_reasoning_completeness(
            gt_reasoning=gt_reasoning,
            pred_reasoning_steps_string=combined_reasoning,
            evaluation_model=EVALUATION_MODEL
        )
        print(f"Attempt {attempt + 1} results: {completeness_results}")
        if completeness_results:
            break
        
    
    # 6. 整合并保存结果
    if efficiency_factuality_results and completeness_results:
        # 这里的 Key 名与主脚本保持一致
        data['reasoning_eval'] = efficiency_factuality_results.get('evaluated_steps', [])
        data['gt_reasoning_eval'] = completeness_results.get('ground_truth_steps', [])
        data['efficiency'] = efficiency_factuality_results.get('efficiency_score', 0)
        data['factulity'] = efficiency_factuality_results.get('factuality_score', 0)
        data['recall'] = completeness_results.get('recall_score', 0)

        # 确保保存路径存在
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
            
        output_path = os.path.join(save_root, f'{data["id"]}.json')
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Successfully evaluated case {data['id']}")
    # try:
    #     # 1. 提取基础信息 (基于你提供的 generate_case 结构)
    #     gen_case = data.get('generate_case', {})
    #     case_info = gen_case.get('case_summary', "")
        
    #     # 标准答案 (Ground Truth)
    #     gt_answer = gen_case.get('treatment_plan_results', "")
    #     # 标准推理路径
    #     gt_analysis = gen_case.get('treatment_planning_analysis', "")
    #     gt_reasoning = f"{gt_analysis}\n Treatment plan results:\n{gt_answer}"
                
    #     # 2. 提取模型输出 (基于你提供的 result 结构)
    #     res_obj = data.get('result', {})
        
    #     # 逻辑：优先取 reasoning 字段，如果没有则取 content
    #     # DeepSeek-R1 这种模型通常会将 <think> 里的内容放在 reasoning
    #     raw_model_text = res_obj.get('reasoning', "").strip()
    #     if not raw_model_text:
    #         raw_model_text = res_obj.get('content', "").strip()
        
    #     if not raw_model_text:
    #         logger.warning(f"ID: {data['id']} has no model output text to evaluate.")
    #         return

    #     # 3. 拆分推理步骤
    #     try:
    #         reasoning_steps = split_reasoning(raw_model_text)
    #         if not reasoning_steps:
    #             # 如果 split_reasoning 返回空列表，则将全文作为一个 step
    #             reasoning_steps = [raw_model_text]
    #     except Exception as e:
    #         logger.error(f"Error calling split_reasoning for {data['id']}: {e}")
    #         reasoning_steps = [raw_model_text]

    #     combined_reasoning = '\n'.join(reasoning_steps)
        
    #     # 4. 评估：效率与事实性 (Efficiency and Factuality)
    #     efficiency_factuality_results = None
    #     for attempt in range(MAX_RETRY_ATTEMPTS):
    #         try:
    #             efficiency_factuality_results = eval_reasoning_efficiency_factuality(
    #                 case_info=case_info,
    #                 pred_reasoning_steps_list=reasoning_steps,
    #                 gt_answer=gt_answer,
    #                 is_treatment=True,
    #                 evaluation_model=EVALUATION_MODEL
    #             )
    #             if efficiency_factuality_results:
    #                 break
    #         except Exception as e:
    #             if attempt == MAX_RETRY_ATTEMPTS - 1:
    #                 raise e
        

    #     # 5. 评估：完整性 (Recall/Completeness)
    #     completeness_results = None
    #     for attempt in range(MAX_RETRY_ATTEMPTS):
    #         try:
    #             completeness_results = eval_reasoning_completeness(
    #                 gt_reasoning=gt_reasoning,
    #                 pred_reasoning_steps_string=combined_reasoning,
    #                 evaluation_model=EVALUATION_MODEL
    #             )
    #             if completeness_results:
    #                 break
    #         except Exception as e:
    #             if attempt == MAX_RETRY_ATTEMPTS - 1:
    #                 raise e
        
    #     # 6. 整合并保存结果
    #     if efficiency_factuality_results and completeness_results:
    #         # 这里的 Key 名与主脚本保持一致
    #         data['reasoning_eval'] = efficiency_factuality_results.get('evaluated_steps', [])
    #         data['gt_reasoning_eval'] = completeness_results.get('ground_truth_steps', [])
    #         data['efficiency'] = efficiency_factuality_results.get('efficiency_score', 0)
    #         data['factulity'] = efficiency_factuality_results.get('factuality_score', 0)
    #         data['recall'] = completeness_results.get('recall_score', 0)

    #         # 确保保存路径存在
    #         if not os.path.exists(save_root):
    #             os.makedirs(save_root, exist_ok=True)
                
    #         output_path = os.path.join(save_root, f'{data["id"]}.json')
    #         with open(output_path, 'w', encoding="utf-8") as f:
    #             json.dump(data, f, ensure_ascii=False, indent=4)
            
    #         logger.info(f"Successfully evaluated case {data['id']}")
            
    # except Exception as e:
    #     logger.error(f"Critical error evaluating case {data['id']}: {str(e)}")
    #     with open(error_log_file, 'a', encoding='utf-8') as f:
    #         f.write(f"ID: {data['id']}, error: {str(e)}\n")


# def evaluate_case(data, save_root, model_name):
#     """Evaluate reasoning quality for a specific model's output on a single case."""
#     logger.info(f'Evaluating case {data["id"]} for model {model_name}')
#     error_log_file = f'{model_name}_error.log'
    
#     try:
#         case_info = data['generate_case']['case_summary']
#         gt_answer = data['parsed']['treatment_plan_results']
#         gt_reasoning = data['generate_case']["treatment_planning_analysis"] + "\n Treatment plan results:\n" + data['generate_case']["treatment_plan_results"]
                
#         # Extract reasoning steps based on model type
#         if model_name == 'deepseek-r1-thinkingprocess':
#             reasoning_steps = split_reasoning(data['results']['thinking_process'])
#         else:
#             reasoning_steps = split_reasoning(data['results']['content'])
            
#         # Combine all steps for recall evaluation
#         combined_reasoning = '\n'.join(reasoning_steps)
        
#         # Evaluate efficiency and factuality
#         for attempt in range(MAX_RETRY_ATTEMPTS):
#             try:
#                 efficiency_factuality_results = eval_reasoning_efficiency_factuality(
#                     case_info=case_info,
#                     pred_reasoning_steps_list=reasoning_steps,
#                     gt_answer=gt_answer,
#                     is_treatment=True,
#                     evaluation_model=EVALUATION_MODEL
#                 )
#                 break
#             except Exception as e:
#                 with open(error_log_file, 'a', encoding='utf-8') as f:
#                     f.write(f"ID: {data['id']}, efficiency_factuality_evaluation, Attempt: {attempt + 1}, Error: {str(e)}\n")
#                 if attempt == MAX_RETRY_ATTEMPTS - 1:
#                     logger.error(f"Failed to evaluate efficiency/factuality after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
#                     logger.error(str(e))
#                     return
        
#         # Evaluate recall/completeness
#         for attempt in range(MAX_RETRY_ATTEMPTS):
#             try:
#                 completeness_results = eval_reasoning_completeness(
#                     gt_reasoning=gt_reasoning,
#                     pred_reasoning_steps_string=combined_reasoning,
#                     evaluation_model=EVALUATION_MODEL
#                 )
#                 break
#             except Exception as e:
#                 with open(error_log_file, 'a', encoding='utf-8') as f:
#                     f.write(f"ID: {data['id']}, completeness_evaluation, Attempt: {attempt + 1}, Error: {str(e)}\n")
#                 if attempt == MAX_RETRY_ATTEMPTS - 1:
#                     logger.error(f"Failed to evaluate completeness after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
#                     logger.error(str(e))
#                     return
        
#         # Store evaluation results
#         data['reasoning_eval'] = efficiency_factuality_results['evaluated_steps']
#         data['gt_reasoning_eval'] = completeness_results['ground_truth_steps']
#         data['efficiency'] = efficiency_factuality_results['efficiency_score']
#         data['factulity'] = efficiency_factuality_results['factuality_score']
#         data['recall'] = completeness_results['recall_score']

#         # Save evaluation results
#         with open(os.path.join(save_root, f'{data["id"]}.json'), 'w', encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=4)
            
#     except Exception as e:
#         logger.error(f"Error evaluating case {data['id']}: {str(e)}")
#         with open(error_log_file, 'a', encoding='utf-8') as f:
#             f.write(f"ID: {data['id']}, general_error: {str(e)}\n")


def worker(task_queue):
    """Worker process function to process evaluation tasks from queue."""
    while not task_queue.empty():
        try:
            data, save_root, model_name = task_queue.get()
            evaluate_case(data, save_root, model_name)
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")


def main(model_name, patient_case_filepath, model_output_filepath, output_directory, use_parallel=True):
    """Main function to orchestrate the evaluation process."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load patient cases and model outputs
    with open(patient_case_filepath, 'r', encoding='utf-8') as f:
        patient_cases = json.load(f)
    
    with open(model_output_filepath, 'r', encoding='utf-8') as f:
        model_outputs = json.load(f)
        
    # Filter already processed data
    cases_to_evaluate = []
    
    completed_cases = os.listdir(output_directory)
    completed_case_ids = [name.split('.')[0] for name in completed_cases]
    # 修改这里的筛选逻辑
    for case_id in patient_cases.keys():
        # 跳过已完成的
        if case_id in completed_case_ids:
            continue

        if case_id in model_outputs and 'result' in model_outputs[case_id]:
            case_data = patient_cases[case_id].copy()  # Create a copy to avoid modifying the original
            case_data['id'] = case_id
            # 统一提取推理结果到 case_data 中
            case_data['result'] = model_outputs[case_id]['result']
            cases_to_evaluate.append(case_data)

    
    logger.info(f'Total cases to evaluate: {len(cases_to_evaluate)}')

    if use_parallel and len(cases_to_evaluate) > 0:
        # Create multiprocessing task queue
        manager = Manager()
        task_queue = manager.Queue()
        
        for case_data in cases_to_evaluate:
            task_queue.put((case_data, output_directory, model_name))

        # Start worker processes
        processes = []
        worker_count = min(NUM_WORKERS, len(cases_to_evaluate))
        logger.info(f"Starting {worker_count} worker processes")
        
        for _ in range(worker_count):
            p = multiprocessing.Process(target=worker, args=(task_queue,))
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join()
    else:
        logger.info("Processing cases sequentially")
        for case_data in cases_to_evaluate:
            evaluate_case(case_data, output_directory, model_name)
            
    logger.info(f"Evaluation completed for model {model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model reasoning on treatment planning tasks')
    parser.add_argument('--model', type=str, default = "deepseek-r1", 
                      help='Model to evaluate')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run sequentially instead of using parallel processing')
    parser.add_argument('--output-dir', type=str, default='../../data/EvalResults/reasoning_results_treatment',
                      help='Base directory for evaluation results')
    parser.add_argument('--patient-cases', type=str,
                      default='../../data/MedRBench/treatment_496_cases_with_rare_disease_165.json',
                      help='Path to patient cases file')
    parser.add_argument('--model-outputs', type=str,
                      default='../../data/InferenceResults/oracle_treatment.json',
                      help='Path to model outputs file')
    
    args = parser.parse_args()
    
    # Define input and output file paths
    model_output_filepath = args.model_outputs
    patient_case_filepath = args.patient_cases
    output_directory = f'{args.output_dir}/{args.model}'
    
    # Run main evaluation process
    main(
        args.model, 
        patient_case_filepath, 
        model_output_filepath, 
        output_directory, 
        not args.sequential
    )