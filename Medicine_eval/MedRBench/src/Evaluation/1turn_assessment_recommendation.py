import os
import json
import logging
import argparse
import multiprocessing
from multiprocessing import Manager

from utils import split_reasoning, extract_ancillary_tests
from metrics.assessment_recommendation_eval import eval_dynamic_asking_info_precision_recall

# Configuration constants
NUM_WORKERS = 8  # Number of worker processes for parallel execution
MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts for API calls
EVALUATION_MODEL = "gpt-4o"  # Model to be used for evaluation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_case(data, save_root, model_name):
    """Evaluate reasoning quality for a specific model's output on a single case."""
    case_id = data["id"]
    logger.info(f'Evaluating case {case_id} for model {model_name}')
    
    try:
        # Extract case information
        case_info = data['generate_case']['case_summary']
        case_info_without_ancillary_test, ancillary_test = extract_ancillary_tests(case_info)
        gt_reasoning = data['generate_case']["differential_diagnosis"] + "\nFinal diagnosis:\n" + data['generate_case']["final_diagnosis"]
        
        # Get model outputs
        model_output = data['results']['messages'][2]['content']['answer']
        pred_info_required = model_output.split('### Additional Information Required:')[-1] if '### Additional Information Required:' in model_output else ""
        gt_info_required = ancillary_test if ancillary_test else ""
        
        # Evaluate with retries
        eval_results = None
        last_error = None
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                eval_results = eval_dynamic_asking_info_precision_recall(
                    pred_info_required,
                    gt_info_required
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for case {case_id}: {str(e)}")
                continue
        
        if not eval_results:
            logger.error(f"Failed to evaluate case {case_id} after {MAX_RETRY_ATTEMPTS} attempts: {str(last_error)}")
            return
        
        if 'error' in eval_results:
            logger.error(f"Evaluation error for case {case_id}: {eval_results['error']}")
            return
            
        # Store results
        data['evaluation'] = {
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'infer_info_split': eval_results['infer_info_split'],
            'gt_info_split': eval_results['gt_info_split']
        }
        
        # Save results
        output_path = os.path.join(save_root, f'{case_id}.json')
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logger.error(f"Unexpected error evaluating case {case_id}: {str(e)}", exc_info=True)


def worker(task_queue):
    """Worker process function to process evaluation tasks from queue."""
    while not task_queue.empty():
        try:
            data, save_root, model_name = task_queue.get()
            evaluate_case(data, save_root, model_name)
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")


# def main(model_name, patient_case_filepath, model_output_filepath, output_directory, use_parallel=True):
#     """Main function to orchestrate the evaluation process."""
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
    
#     # Load patient cases and model outputs
#     with open(patient_case_filepath, 'r', encoding='utf-8') as f:
#         patient_cases = json.load(f)
    
#     with open(model_output_filepath, 'r', encoding='utf-8') as f:
#         model_outputs = json.load(f)
        
#     # Filter already processed data
#     cases_to_evaluate = []
    
#     completed_cases = os.listdir(output_directory)
#     completed_case_ids = [name.split('.')[0] for name in completed_cases]
    
#     for case_id in patient_cases.keys():
#         if case_id not in completed_case_ids and case_id in model_outputs and model_name in model_outputs[case_id]:
#             case_data = patient_cases[case_id].copy()  # Create a copy to avoid modifying the original
#             case_data['id'] = case_id
#             case_data['results'] = model_outputs[case_id][model_name]
#             cases_to_evaluate.append(case_data)    
    
#     logger.info(f'Total cases to evaluate: {len(cases_to_evaluate)}')

#     if use_parallel and len(cases_to_evaluate) > 0:
#         # Create multiprocessing task queue
#         manager = Manager()
#         task_queue = manager.Queue()
        
#         for case_data in cases_to_evaluate:
#             task_queue.put((case_data, output_directory, model_name))

#         # Start worker processes
#         processes = []
#         worker_count = min(NUM_WORKERS, len(cases_to_evaluate))
#         logger.info(f"Starting {worker_count} worker processes")
        
#         for _ in range(worker_count):
#             p = multiprocessing.Process(target=worker, args=(task_queue,))
#             p.start()
#             processes.append(p)

#         # Wait for completion
#         for p in processes:
#             p.join()
#     else:
#         logger.info("Processing cases sequentially")
#         for case_data in cases_to_evaluate:
#             evaluate_case(case_data, output_directory, model_name)
            
#     logger.info(f"Evaluation completed for model {model_name}")

def main(model_name, patient_case_filepath, model_output_folder, output_directory, use_parallel=True):
    """
    修改后的 Main 函数：
    1. 支持直接读取 one_turn.py 生成的文件夹离散结果。
    2. 自动匹配病例 ID 和模型输出。
    """
    # 1. 创建评估结果保存目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 2. 加载原始病例数据（用于获取 Gold Standard 检查结果）
    logger.info(f"正在从以下路径加载原始病例: {patient_case_filepath}")
    if not os.path.exists(patient_case_filepath):
        logger.error(f"错误: 找不到病例文件 {patient_case_filepath}")
        return
        
    with open(patient_case_filepath, 'r', encoding='utf-8') as f:
        patient_cases = json.load(f)
    
    # 3. 遍历推理结果文件夹，读取每个病例的输出
    logger.info(f"正在扫描推理结果文件夹: {model_output_folder}")
    model_outputs = {}
    
    if not os.path.exists(model_output_folder):
        logger.error(f"错误: 推理结果文件夹 {model_output_folder} 不存在！")
        return

    # 遍历文件夹下所有的 .json 文件并加载到内存
    files = [f for f in os.listdir(model_output_folder) if f.endswith('.json')]
    for file_name in files:
        case_id = file_name.replace('.json', '')
        file_path = os.path.join(model_output_folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                # 兼容 one_turn.py 的存储格式
                if 'results' in case_data:
                    model_outputs[case_id] = case_data['results']
                else:
                    # 如果结构不同，尝试直接读取
                    model_outputs[case_id] = case_data
        except Exception as e:
            logger.error(f"读取文件 {file_name} 失败: {str(e)}")

    # 4. 筛选待评估的案例
    cases_to_evaluate = []
    
    # 获取已经评估过的案例（断点续传逻辑）
    completed_cases = os.listdir(output_directory)
    completed_case_ids = [name.split('.')[0] for name in completed_cases]
    
    for case_id in patient_cases.keys():
        # 只有在“有推理结果”且“还没评估过”的情况下才加入任务队列
        if case_id in model_outputs and case_id not in completed_case_ids:
            case_data = patient_cases[case_id].copy()
            case_data['id'] = case_id
            # 将模型结果注入，以便 evaluate_case 函数使用
            case_data['results'] = model_outputs[case_id]
            cases_to_evaluate.append(case_data) 
    
    logger.info(f'文件夹中找到的有效结果数: {len(model_outputs)}')
    logger.info(f'剩余需要评估的案例数: {len(cases_to_evaluate)}')

    # 5. 执行评估
    if use_parallel and len(cases_to_evaluate) > 0:
        manager = Manager()
        task_queue = manager.Queue()
        
        for case_data in cases_to_evaluate:
            task_queue.put((case_data, output_directory, model_name))

        processes = []
        # 根据任务数调整进程数
        worker_count = min(NUM_WORKERS, len(cases_to_evaluate))
        logger.info(f"启动 {worker_count} 个并行进程进行评测...")
        
        for _ in range(worker_count):
            p = multiprocessing.Process(target=worker, args=(task_queue,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        if len(cases_to_evaluate) > 0:
            logger.info("正在顺序执行评估...")
            for case_data in cases_to_evaluate:
                evaluate_case(case_data, output_directory, model_name)
        else:
            logger.info("没有需要评估的新案例。")
            
    logger.info(f"评估完成！结果保存在: {output_directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model reasoning on treatment planning tasks')
    parser.add_argument('--model', type=str, required=True, 
                      help='Model to evaluate')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run sequentially instead of using parallel processing')
    parser.add_argument('--output-dir', type=str, default='./reasoning_results',
                      help='Base directory for evaluation results')
    parser.add_argument('--patient-cases', type=str,
                      default='../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json',
                      help='Path to patient cases file')
    parser.add_argument('--model-outputs', type=str,
                      default='../../data/InferenceResults/1turn_assessment_recommendation+final_diagnosis.json',
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