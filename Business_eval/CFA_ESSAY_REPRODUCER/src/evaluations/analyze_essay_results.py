import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PALETTE = sns.color_palette("viridis", 30)
STRATEGY_PALETTE = sns.color_palette("plasma", 4)

REASONING_MODEL_CONFIG_IDS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash", 
    "o3-mini",
    "o4-mini",
    "grok-3",
    "grok-3-mini-beta-high-effort", 
    "deepseek-r1",  
    "gemini-2.5-pro-cot",
    "gemini-2.5-flash-cot",
    "claude-3.7-sonnet-cot",
    "grok-3-cot",
    "deepseek-r1-cot",  
    "gpt-4o-self-discover",
    "gpt-o3-self-discover",
    "grok-3-self-discover",
    "gemini-2.5-pro-self-discover",
    "gemini-2.5-flash-self-discover",
    "deepseek-r1-self-discover", 
    "grok-3-mini-beta-self-discover-high-effort" 
]

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "all_results_reproduced.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "essay_analysis_charts"

NUMERIC_COLUMNS = [
    'cosine_similarity', 'self_grade_score', 'rouge_l_precision', 
    'rouge_l_recall', 'rouge_l_f1measure', 'latency_ms', 
    'input_tokens', 'output_tokens', 'api_cost', 'answer_length'
]

MAX_POSSIBLE_SCORE = 149

QUALITY_METRICS = ['cosine_similarity', 'rouge_l_f1measure', 'self_grade_score']

def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Loads data from CSV and prepares it for analysis."""
    logging.info(f"Loading data from {csv_path}")
    if not csv_path.exists():
        logging.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} rows.")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logging.warning(f"Numeric column {col} not found in CSV. It will be ignored.")

    df['successful_run'] = df['error'].isna() | (df['error'] == '') | (df['error'] == 'nan')
    
    logging.info(f"Data preparation complete. {len(df[df['successful_run']])} successful runs identified.")
    return df

def create_output_directory(output_dir: Path):
    """Creates the output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

def save_plot(fig, filename: str, output_dir: Path):
    """Saves the given matplotlib figure."""
    filepath = output_dir / filename
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    logging.info(f"Saved plot: {filepath}")
    plt.close(fig)

def categorize_model(model_name: str, reasoning_models_list: list) -> str:
    """Categorizes models as 'Reasoning' or 'Non-Reasoning' based on a provided list."""
    if model_name in reasoning_models_list:
        return "Reasoning"
    else:
        return "Non-Reasoning"


def perform_overall_performance_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes overall model performance by identifying the best strategy for each model
       based on self_grade_score sum, and then reports metrics for that combination."""
    logging.info("Performing Overall Performance Analysis (Best Strategy per Model)...")
    
    successful_df = df[df['successful_run']].copy()
    
    if successful_df.empty:
        logging.warning("No successful runs found. Skipping Overall Performance Analysis.")
        return

    if 'model' not in successful_df.columns or 'strategy' not in successful_df.columns or 'self_grade_score' not in successful_df.columns:
        logging.error("Essential columns ('model', 'strategy', 'self_grade_score') not found. Skipping Overall Performance Analysis.")
        return

    model_strategy_scores = successful_df.groupby(['model', 'strategy'])['self_grade_score'].sum().reset_index()
    model_strategy_scores = model_strategy_scores.rename(columns={'self_grade_score': 'self_grade_score_sum_for_strategy'})

    best_strategy_indices = model_strategy_scores.loc[model_strategy_scores.groupby('model')['self_grade_score_sum_for_strategy'].idxmax()]
    best_strategies_df = best_strategy_indices[['model', 'strategy', 'self_grade_score_sum_for_strategy']]
    best_strategies_df = best_strategies_df.rename(columns={'self_grade_score_sum_for_strategy': 'best_strategy_self_grade_score_sum'})

    logging.info(f"Identified best performing strategies for each model based on self_grade_score sum:\n{best_strategies_df.to_string()}")

    
    overall_performance_list = []
    for _, row in best_strategies_df.iterrows():
        model_name = row['model']
        best_strategy_name = row['strategy']
        best_score_sum = row['best_strategy_self_grade_score_sum']
        
        model_strategy_data = successful_df[
            (successful_df['model'] == model_name) & 
            (successful_df['strategy'] == best_strategy_name)
        ]
        
        if model_strategy_data.empty:
            logging.warning(f"No data found for model {model_name} with its identified best strategy {best_strategy_name}. Skipping.")
            continue
            
        cosine_mean = model_strategy_data['cosine_similarity'].mean() if 'cosine_similarity' in model_strategy_data else np.nan
        cosine_std = model_strategy_data['cosine_similarity'].std() if 'cosine_similarity' in model_strategy_data else np.nan
        cosine_count = model_strategy_data['cosine_similarity'].count() if 'cosine_similarity' in model_strategy_data else 0
        
        rouge_mean = model_strategy_data['rouge_l_f1measure'].mean() if 'rouge_l_f1measure' in model_strategy_data else np.nan
        rouge_std = model_strategy_data['rouge_l_f1measure'].std() if 'rouge_l_f1measure' in model_strategy_data else np.nan
        rouge_count = model_strategy_data['rouge_l_f1measure'].count() if 'rouge_l_f1measure' in model_strategy_data else 0
        
        self_grade_std = model_strategy_data['self_grade_score'].std() if 'self_grade_score' in model_strategy_data else np.nan
        self_grade_count = model_strategy_data['self_grade_score'].count() if 'self_grade_score' in model_strategy_data else 0

        overall_performance_list.append({
            'model': model_name,
            'best_strategy': best_strategy_name,
            'cosine_similarity_mean': cosine_mean,
            'cosine_similarity_std': cosine_std,
            'cosine_similarity_count': cosine_count,
            'rouge_l_f1measure_mean': rouge_mean,
            'rouge_l_f1measure_std': rouge_std,
            'rouge_l_f1measure_count': rouge_count,
            'self_grade_score_sum': best_score_sum,
            'self_grade_score_normalized': (best_score_sum / MAX_POSSIBLE_SCORE) * 100 if best_score_sum is not None else None,
            'self_grade_score_std': self_grade_std,
            'self_grade_score_count': self_grade_count
        })

    if not overall_performance_list:
        logging.warning("No data to compile for overall model performance summary after filtering for best strategies. Skipping.")
        return

    model_performance = pd.DataFrame(overall_performance_list)
    
    if 'self_grade_score_sum' in model_performance.columns:
        model_performance = model_performance.sort_values(by='self_grade_score_sum', ascending=False)
    elif 'cosine_similarity_mean' in model_performance.columns:
        model_performance = model_performance.sort_values(by='cosine_similarity_mean', ascending=False)
        
    table_path = output_dir / "overall_model_performance_summary.csv"
    model_performance.to_csv(table_path, index=False)
    logging.info(f"Saved overall model performance summary (best strategy per model) to {table_path}")
    print("\nOverall Model Performance Summary (Best Strategy per Model):")
    print(model_performance)

    for metric in QUALITY_METRICS:
        if metric in successful_df.columns:
            plt.figure(figsize=(14, 8))
            plot_data = successful_df['self_grade_score'] if metric == 'self_grade_score' else successful_df[metric]
            plot_model = successful_df['model']
            sns.boxplot(x=plot_model, y=plot_data, hue=plot_model, palette=MODEL_PALETTE, legend=False)
            title_metric_name = metric.replace("_", " ").title()
            if metric == 'self_grade_score':
                 title_metric_name += " (Individual Distribution - All Strategies)" 

            plt.title(f'Distribution of {title_metric_name} by Model', fontsize=16)
            plt.xlabel("Model", fontsize=14)
            plt.ylabel(metric.replace("_", " ").title(), fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            save_plot(plt.gcf(), f"boxplot_{metric}_by_model.png", output_dir)
        else:
            logging.warning(f"Metric {metric} not found for boxplot generation.")

def perform_model_type_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes performance by model type (Reasoning vs. Non-Reasoning)."""
    logging.info("Performing Model Type Analysis (Reasoning vs. Non-Reasoning)...")
    
    successful_df = df[df['successful_run']].copy()
    if successful_df.empty:
        logging.warning("No successful runs found. Skipping Model Type Analysis.")
        return

    successful_df['model_category'] = successful_df['model'].apply(lambda x: categorize_model(x, REASONING_MODEL_CONFIG_IDS))
    
    model_scores = successful_df.groupby('model')['self_grade_score'].sum().reset_index()
    model_scores = model_scores.rename(columns={'self_grade_score': 'self_grade_score_sum'})
    
    model_scores['model_category'] = model_scores['model'].apply(lambda x: categorize_model(x, REASONING_MODEL_CONFIG_IDS))
    
    category_score_stats = model_scores.groupby('model_category')['self_grade_score_sum'].agg([
        'mean',
        'max',
        'min',
        'std',
        'count'
    ]).reset_index()
    
    category_score_stats.columns = [
        'model_category',
        'self_grade_score_sum_avg_across_models',
        'self_grade_score_sum_max_across_models', 
        'self_grade_score_sum_min_across_models',
        'self_grade_score_sum_std_across_models',
        'models_in_category_count'
    ]
    
    category_score_stats['self_grade_score_sum_avg_normalized'] = (category_score_stats['self_grade_score_sum_avg_across_models'] / MAX_POSSIBLE_SCORE) * 100
    category_score_stats['self_grade_score_sum_max_normalized'] = (category_score_stats['self_grade_score_sum_max_across_models'] / MAX_POSSIBLE_SCORE) * 100
    
    model_type_quality_metrics = ['cosine_similarity', 'rouge_l_f1measure']
    category_other_metrics = successful_df.groupby('model_category')[model_type_quality_metrics].agg(['mean', 'std', 'count']).reset_index()
    category_other_metrics.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in category_other_metrics.columns.values]
    
    category_performance = category_score_stats.merge(category_other_metrics, on='model_category', how='left')

    primary_sort_metric_model_type = 'self_grade_score_sum_avg_across_models'
    if primary_sort_metric_model_type in category_performance.columns:
         category_performance = category_performance.sort_values(by=primary_sort_metric_model_type, ascending=False)
    else:
        logging.warning("No primary sort metric found for model category performance. Table will be unsorted.")

    table_path = output_dir / "model_category_performance_summary.csv"
    category_performance.to_csv(table_path, index=False)
    logging.info(f"Saved model category performance summary to {table_path}")
    print("\nModel Category Performance Summary (Focus: Self Grade Score Sums):")
    print(category_performance)

    
    plt.figure(figsize=(10, 7))
    sorted_categories = category_performance.sort_values(by='self_grade_score_sum_avg_across_models', ascending=False)
    sns.barplot(data=sorted_categories, x='model_category', y='self_grade_score_sum_avg_across_models', hue='model_category', palette="Paired", legend=False)
    plt.title('Average Self Grade Score Sum Across Models by Category', fontsize=16)
    plt.xlabel("Model Category", fontsize=14)
    plt.ylabel('Average Self Grade Score Sum', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(plt.gcf(), f"barchart_avg_self_grade_score_sum_by_model_category.png", output_dir)
    
    plt.figure(figsize=(10, 7))
    sorted_categories_max = category_performance.sort_values(by='self_grade_score_sum_max_across_models', ascending=False)
    sns.barplot(data=sorted_categories_max, x='model_category', y='self_grade_score_sum_max_across_models', hue='model_category', palette="Paired", legend=False)
    plt.title('Maximum Self Grade Score Sum Achieved by Model Category', fontsize=16)
    plt.xlabel("Model Category", fontsize=14)
    plt.ylabel('Maximum Self Grade Score Sum', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(plt.gcf(), f"barchart_max_self_grade_score_sum_by_model_category.png", output_dir)

    for metric in ['cosine_similarity', 'rouge_l_f1measure']:
        if metric + '_mean' in category_performance.columns:
            plt.figure(figsize=(10, 7))
            sorted_categories_other = category_performance.sort_values(by=metric + '_mean', ascending=False)
            sns.barplot(data=sorted_categories_other, x='model_category', y=metric + '_mean', hue='model_category', palette="Paired", legend=False)
            plt.title(f'Average {metric.replace("_", " ").title()} by Model Category', fontsize=16)
            plt.xlabel("Model Category", fontsize=14)
            plt.ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            save_plot(plt.gcf(), f"barchart_avg_{metric}_by_model_category.png", output_dir)

def perform_strategy_effectiveness_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes the effectiveness of different prompting strategies."""
    logging.info("Performing Prompt Strategy Effectiveness Analysis...")
    successful_df = df[df['successful_run']].copy()
    if successful_df.empty:
        logging.warning("No successful runs found. Skipping Prompt Strategy Effectiveness Analysis.")
        return

    
    model_strategy_scores = successful_df.groupby(['model', 'strategy'])['self_grade_score'].sum().reset_index()
    model_strategy_scores = model_strategy_scores.rename(columns={'self_grade_score': 'self_grade_score_sum'})
    
    strategy_score_stats = model_strategy_scores.groupby('strategy')['self_grade_score_sum'].agg([
        'mean',
        'max',
        'min',
        'std',
        'count'
    ]).reset_index()
    
    strategy_score_stats.columns = [
        'strategy', 
        'self_grade_score_sum_avg_across_models',
        'self_grade_score_sum_max_across_models', 
        'self_grade_score_sum_min_across_models',
        'self_grade_score_sum_std_across_models',
        'models_tested_count'
    ]
    
    strategy_score_stats['self_grade_score_sum_avg_normalized'] = (strategy_score_stats['self_grade_score_sum_avg_across_models'] / MAX_POSSIBLE_SCORE) * 100
    strategy_score_stats['self_grade_score_sum_max_normalized'] = (strategy_score_stats['self_grade_score_sum_max_across_models'] / MAX_POSSIBLE_SCORE) * 100
    
    strategy_score_stats = strategy_score_stats.sort_values(by='self_grade_score_sum_avg_across_models', ascending=False)
    
    strategy_other_metrics = successful_df.groupby('strategy').agg({
        'cosine_similarity': ['mean', 'std', 'count'],
        'rouge_l_f1measure': ['mean', 'std', 'count']
    }).reset_index()
    strategy_other_metrics.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in strategy_other_metrics.columns.values]
    
    strategy_quality_performance = strategy_score_stats.merge(strategy_other_metrics, on='strategy', how='left')

    strategy_metrics_efficiency = ['latency_ms', 'api_cost', 'input_tokens', 'output_tokens']
    strategy_efficiency_metrics_agg = {metric: ['mean', 'std', 'count'] for metric in strategy_metrics_efficiency if metric in df.columns}
    strategy_efficiency_performance = df.groupby('strategy').agg(strategy_efficiency_metrics_agg).reset_index()
    strategy_efficiency_performance.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in strategy_efficiency_performance.columns.values]

    strategy_quality_table_path = output_dir / "strategy_quality_performance.csv"
    strategy_quality_performance.to_csv(strategy_quality_table_path, index=False)
    logging.info(f"Saved strategy quality performance to {strategy_quality_table_path}")
    print("\nStrategy Quality Performance (Focus: Self Grade Score Sums):")
    print(strategy_quality_performance)

    strategy_efficiency_table_path = output_dir / "strategy_efficiency_performance.csv"
    strategy_efficiency_performance.to_csv(strategy_efficiency_table_path, index=False)
    logging.info(f"Saved strategy efficiency performance to {strategy_efficiency_table_path}")
    print("\nStrategy Efficiency Performance:")
    print(strategy_efficiency_performance)
    
    
    plt.figure(figsize=(12, 7))
    sorted_strategies = strategy_quality_performance.sort_values(by='self_grade_score_sum_avg_across_models', ascending=False)
    sns.barplot(data=sorted_strategies, x='strategy', y='self_grade_score_sum_avg_across_models', hue='strategy', palette=STRATEGY_PALETTE, legend=False)
    plt.title('Average Self Grade Score Sum Across Models by Strategy', fontsize=16)
    plt.xlabel("Strategy", fontsize=14)
    plt.ylabel('Average Self Grade Score Sum', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(plt.gcf(), f"barchart_avg_self_grade_score_sum_by_strategy.png", output_dir)
    
    plt.figure(figsize=(12, 7))
    sorted_strategies_max = strategy_quality_performance.sort_values(by='self_grade_score_sum_max_across_models', ascending=False)
    sns.barplot(data=sorted_strategies_max, x='strategy', y='self_grade_score_sum_max_across_models', hue='strategy', palette=STRATEGY_PALETTE, legend=False)
    plt.title('Maximum Self Grade Score Sum Achieved by Strategy', fontsize=16)
    plt.xlabel("Strategy", fontsize=14)
    plt.ylabel('Maximum Self Grade Score Sum', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(plt.gcf(), f"barchart_max_self_grade_score_sum_by_strategy.png", output_dir)

    for metric in ['cosine_similarity', 'rouge_l_f1measure']:
        if metric + '_mean' in strategy_quality_performance.columns:
            plt.figure(figsize=(10, 6))
            sorted_strategies_other = strategy_quality_performance.sort_values(by=metric + '_mean', ascending=False)
            sns.barplot(data=sorted_strategies_other, x='strategy', y=metric + '_mean', hue='strategy', palette=STRATEGY_PALETTE, legend=False)
            plt.title(f'Average {metric.replace("_", " ").title()} by Strategy', fontsize=16)
            plt.xlabel("Strategy", fontsize=14)
            plt.ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            save_plot(plt.gcf(), f"barchart_avg_{metric}_by_strategy.png", output_dir)

    sc_n3 = df[(df['strategy'].str.contains("N=3", case=False)) & df['successful_run']]
    sc_n5 = df[(df['strategy'].str.contains("N=5", case=False)) & df['successful_run']]

    if not sc_n3.empty and not sc_n5.empty:
        logging.info("Comparing Self-Consistency N=3 vs N=5...")
        
        sc_n3_sums = sc_n3.groupby('model')['self_grade_score'].sum()
        sc_n5_sums = sc_n5.groupby('model')['self_grade_score'].sum()
        
        comparison_data = []
        
        n3_sum_avg = sc_n3_sums.mean()
        n5_sum_avg = sc_n5_sums.mean()
        n3_sum_max = sc_n3_sums.max()
        n5_sum_max = sc_n5_sums.max()
        
        comparison_data.append({
            'metric': 'self_grade_score_sum_avg',
            'N=3_value': n3_sum_avg,
            'N=5_value': n5_sum_avg,
            'improvement_N5_over_N3': n5_sum_avg - n3_sum_avg,
            'N=3_normalized': (n3_sum_avg / MAX_POSSIBLE_SCORE) * 100,
            'N=5_normalized': (n5_sum_avg / MAX_POSSIBLE_SCORE) * 100,
            'normalized_improvement_N5_over_N3': ((n5_sum_avg - n3_sum_avg) / MAX_POSSIBLE_SCORE) * 100
        })
        
        comparison_data.append({
            'metric': 'self_grade_score_sum_max',
            'N=3_value': n3_sum_max,
            'N=5_value': n5_sum_max,
            'improvement_N5_over_N3': n5_sum_max - n3_sum_max,
            'N=3_normalized': (n3_sum_max / MAX_POSSIBLE_SCORE) * 100,
            'N=5_normalized': (n5_sum_max / MAX_POSSIBLE_SCORE) * 100,
            'normalized_improvement_N5_over_N3': ((n5_sum_max - n3_sum_max) / MAX_POSSIBLE_SCORE) * 100
        })
        
        for metric in ['cosine_similarity', 'rouge_l_f1measure', 'api_cost', 'latency_ms']:
            if metric in sc_n3.columns and metric in sc_n5.columns:
                mean_n3 = sc_n3[metric].mean()
                mean_n5 = sc_n5[metric].mean()
                
                comparison_data.append({
                    'metric': metric,
                    'N=3_value': mean_n3,
                    'N=5_value': mean_n5,
                    'improvement_N5_over_N3': mean_n5 - mean_n3 if pd.notna(mean_n5) and pd.notna(mean_n3) else None
                })
        
        if comparison_data:
            sc_comparison_df = pd.DataFrame(comparison_data)
            sc_comp_table_path = output_dir / "self_consistency_n3_vs_n5_comparison.csv"
            sc_comparison_df.to_csv(sc_comp_table_path, index=False)
            logging.info(f"Saved Self-Consistency N=3 vs N=5 comparison to {sc_comp_table_path}")
            print("\nSelf-Consistency N=3 vs N=5 Comparison (Focus: Score Sums):")
            print(sc_comparison_df)
    else:
        logging.warning("Not enough data for Self-Consistency N=3 vs N=5 comparison.")


def perform_efficiency_tradeoffs_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes efficiency tradeoffs (performance vs. cost/latency)."""
    logging.info("Performing Efficiency Tradeoffs Analysis...")
    
    successful_df = df[df['successful_run']].copy()
    if successful_df.empty:
        logging.warning("No successful runs found. Skipping part of Efficiency Tradeoffs Analysis.")
    
    if not successful_df.empty:
        agg_df = successful_df.groupby(['model', 'strategy']).agg(
            avg_cosine_similarity=('cosine_similarity', 'mean'),
            avg_rouge_l_f1measure=('rouge_l_f1measure', 'mean'),
            total_self_grade_score=('self_grade_score', 'sum'),
            avg_api_cost=('api_cost', 'mean'),
            avg_latency_ms=('latency_ms', 'mean'),
            count=('model', 'count') 
        ).reset_index()

        agg_df = agg_df[agg_df['count'] >= 3] 

        if agg_df.empty:
            logging.warning("Not enough aggregated data (after filtering by count) for scatter plots.")
            return

        scatter_metrics = [
            ('avg_cosine_similarity', 'avg_api_cost'),
            ('avg_rouge_l_f1measure', 'avg_api_cost'),
            ('total_self_grade_score', 'avg_api_cost'),
            ('avg_cosine_similarity', 'avg_latency_ms'),
            ('avg_rouge_l_f1measure', 'avg_latency_ms'),
            ('total_self_grade_score', 'avg_latency_ms'),
        ]

        for perf_metric, cost_metric in scatter_metrics:
            if perf_metric in agg_df.columns and cost_metric in agg_df.columns:
                plt.figure(figsize=(15, 10))
                sns.scatterplot(data=agg_df, x=cost_metric, y=perf_metric, hue='model', style='strategy', s=150, palette=MODEL_PALETTE)
                
                for i, point in agg_df.iterrows():
                    plt.text(point[cost_metric] + 0.01 * agg_df[cost_metric].max(),
                             point[perf_metric], 
                             f"{point['model'][:10]}-{point['strategy'][:3]}", 
                             fontsize=7)

                plt.title(f'{perf_metric.replace("_", " ").title()} vs. {cost_metric.replace("_", " ").title()}', fontsize=16)
                plt.xlabel(cost_metric.replace("_", " ").title(), fontsize=14)
                plt.ylabel(perf_metric.replace("_", " ").title(), fontsize=14)
                plt.legend(title='Model / Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                save_plot(plt.gcf(), f"scatter_{perf_metric}_vs_{cost_metric}.png", output_dir)
            else:
                logging.warning(f"Skipping scatter plot: Columns {perf_metric} or {cost_metric} not found in aggregated data.")
    else:
        logging.warning("No successful runs, so cannot generate quality vs cost/latency scatter plots.")


def main(csv_path: Path, output_dir: Path):
    """Main function to run all analyses."""
    create_output_directory(output_dir)
    df = load_and_prepare_data(csv_path)

    perform_overall_performance_analysis(df, output_dir)
    perform_model_type_analysis(df, output_dir)
    perform_strategy_effectiveness_analysis(df, output_dir)
    perform_efficiency_tradeoffs_analysis(df, output_dir)
    
    logging.info("All analyses complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze essay evaluation results from a CSV file.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the input CSV file (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save generated charts and tables (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    try:
        main(args.csv_path, args.output_dir)
    except FileNotFoundError:
        logging.error("Exiting due to missing input file.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) 