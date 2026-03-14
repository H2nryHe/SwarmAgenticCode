import os
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from tqdm import tqdm

from func import *
from role import Team
from logger import setup_logger, log, log_all
from eval import evaluate, get_scores
from llm_utils import build_chat_openai


def execute(team_with_task, data, ref_info, i, func, llm_extract):
    """Execute a single task and evaluate the result."""
    task_description = f'''Given reference information: {ref_info}\n\nQuery: {data['query']}\n'''
    team_with_task.reset_task(task_description)
    
    res = func(team_with_task)
    final_answer, plan_log = extract_plan(llm_extract, res)
    
    # Evaluation 
    try:
        false_item, problem = evaluate(data, final_answer)
        
        if problem != '': 
            score = 0.0
        else:
            score = 1.0
    except:
        score = 0.0
    
    result = {
        "idx": i,
        "query": data['query'],
        "plan": final_answer,
        "score": score
    }
    return result


async def evaluate_particle(team, func, testset, infoset, llm_extract, logger, save_dir, 
                           start_index=0, end_index=None, max_workers=16):
    """Evaluate a particle on the test set."""
    if end_index is None:
        end_index = len(testset)
    
    dataset = testset[start_index:end_index]
    ref_info_set = infoset[start_index:end_index]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_running_loop()
        tasks = []
        
        for i, (data, info) in enumerate(zip(dataset, ref_info_set)):
            team_with_task = team.deepcopy() 
            tasks.append(loop.run_in_executor(
                executor, execute, team_with_task, data, info, i+start_index, func, llm_extract
            ))
        
        results = await asyncio.gather(*tasks)
    
    # Sort results by index
    results.sort(key=lambda x: x['idx'])
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, 'results.jsonl')
    write_jsonl(result_path, results, 'w')
    
    # Calculate fitness
    scores = [result['score'] for result in results]
    fitness = sum(scores) / len(scores) if scores else 0.0
    
    log(logger, 'Evaluation Complete', f'Average Score: {fitness:.4f}')
    
    return results, fitness


def aggregate_results(aggregate_folder):
    """Aggregate results from multiple result files in a folder."""
    import glob
    
    result_files = glob.glob(os.path.join(aggregate_folder, 'results-*.jsonl'))
    all_results = []
    
    for file_path in result_files:
        results = load_jsonl(file_path)
        all_results.extend(results)
    
    # Sort by index
    all_results.sort(key=lambda x: x['idx'])
    
    # Save aggregated results
    aggregated_path = os.path.join(aggregate_folder, 'results.jsonl')
    write_jsonl(aggregated_path, all_results, 'w')
    
    print(f"Aggregated {len(all_results)} results to {aggregated_path}")
    return aggregated_path


async def main(particle_idx=-1, model='gpt-4o-mini', save_dir='evaluation/test', 
               start_index=0, end_index=None, max_workers=16,
               dataset_path=r'data/validation.jsonl',
               ref_info_path=r'data/validation_ref_info.jsonl',
               aggregate_folder=None):
    """
    Main function to evaluate a particle.
    
    Args:
        particle_idx: Index of the particle to evaluate (-1 for the last one)
        model: LLM model to use for role initialization
        save_dir: Directory to save results
        start_index: Start index of the test set
        end_index: End index of the test set (None for full set)
        max_workers: Number of worker threads
        dataset_path: Path to the dataset
        ref_info_path: Path to the reference info dataset
        aggregate_folder: If provided, aggregate results from this folder instead of running evaluation
    """
    
    # If aggregation mode, just aggregate and exit
    if aggregate_folder:
        aggregated_path = aggregate_results(aggregate_folder)
        
        # Load dataset for score calculation
        testset = load_dataset(path=dataset_path)
        
        # Calculate detailed scores
        print("\nCalculating detailed scores...")
        s1, s2 = get_scores(testset, aggregated_path)
        print(f"\nCommonsense Constraint Pass Rate: {s1}")
        print(f"Hard Constraint Pass Rate: {s2}")
        return
    
    # Setup logger
    logger = setup_logger(particle_idx)
    
    # Initialize LLMs
    llm_role = build_chat_openai(model=model, temperature=0.001)
    llm_extract = build_chat_openai(model="gpt-4o-mini", temperature=0.001)
    
    # Load particle
    try:
        particles = load_particles(particle_idx)
        if not particles:
            raise ValueError("No particles found in save.jsonl")
        
        particle = particles[particle_idx]
        team_dict = particle['team']
        code = particle['code']
    except FileNotFoundError:
        log(logger, 'Error', 'save.jsonl not found. Please run pso.py first.')
        raise FileNotFoundError("save.jsonl not found. Please run pso.py first to generate particles.")
    except (IndexError, KeyError) as e:
        log(logger, 'Error', f'Failed to load particle at index {particle_idx}: {e}')
        raise
    except Exception as e:
        log(logger, 'Error', f'Unexpected error loading particle: {e}')
        raise
    
    # Setup team
    team = Team(llm_role, logger)
    team.update(team_dict)
    func = set_forward(code)
    
    log(logger, 'Particle', f'{team}\n\n{code}')
    
    # Load datasets
    testset = load_dataset(path=dataset_path)
    infoset = load_ref_info(path=ref_info_path)
    
    log(logger, 'Dataset', f'Loaded {len(testset)} test examples from {dataset_path}')
    log(logger, 'Dataset', f'Loaded {len(infoset)} reference info from {ref_info_path}')
    
    if end_index is None:
        end_index = len(testset)
    
    log(logger, 'Evaluation', f'Evaluating from index {start_index} to {end_index}')
    
    # Evaluate
    results, fitness = await evaluate_particle(
        team, func, testset, infoset, llm_extract, logger, save_dir, 
        start_index, end_index, max_workers
    )
    
    log(logger, 'Results', f'Evaluated {len(results)} examples with average score: {fitness:.4f}')
    
    # Calculate detailed scores
    result_path = os.path.join(save_dir, 'results.jsonl')
    print("\nCalculating detailed scores...")
    score_dataset = testset[start_index:end_index]
    s1, s2 = get_scores(score_dataset, result_path)
    print(f"\nCommonsense Constraint Pass Rate: {s1}")
    print(f"Hard Constraint Pass Rate: {s2}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a PSO particle on the travelplanner task')
    
    parser.add_argument('--particle_idx', type=int, default=-1,
                       help='Index of the particle to evaluate (-1 for the last one)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125'],
                       help='LLM model to use')
    parser.add_argument('--save_dir', type=str, default='evaluation/test',
                       help='Directory to save results')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index of the test set')
    parser.add_argument('--end_index', type=int, default=None,
                       help='End index of the test set (None for full set)')
    parser.add_argument('--max_workers', type=int, default=16,
                       help='Number of worker threads')
    parser.add_argument('--dataset', type=str, default=r'data/validation.jsonl',
                       help='Path to the dataset')
    parser.add_argument('--ref_info', type=str, default=r'data/validation_ref_info.jsonl',
                       help='Path to the reference info dataset')
    parser.add_argument('--aggregate_folder', type=str, default=None,
                       help='If provided, aggregate results from this folder instead of running evaluation')
    
    args = parser.parse_args()
    
    asyncio.run(main(
        particle_idx=args.particle_idx,
        model=args.model,
        save_dir=args.save_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        max_workers=args.max_workers,
        dataset_path=args.dataset,
        ref_info_path=args.ref_info,
        aggregate_folder=args.aggregate_folder
    ))

