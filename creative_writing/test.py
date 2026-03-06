"""
Testing and Evaluation Script for Creative Writing Task
Evaluate a saved particle on test dataset
"""

from langchain_openai import ChatOpenAI
from func_timeout import func_timeout, FunctionTimedOut

import asyncio

import argparse
import json
import os
from tqdm import tqdm
from func import *
from role import Team
from logger import setup_logger, log
from eval import evaluate, get_fitness

LLM_TIMEOUT_SECONDS = 5
LLM_MAX_RETRIES = 0


def execute(team_with_task, data, i, func, llm_eval):
    """Execute the team workflow on a single data point.
    
    Args:
        team_with_task: Team object with task
        data: Input data (text line)
        i: Index of the data
        func: Forward function
        llm_eval: LLM for evaluation
        
    Returns:
        dict: Result with response and score
    """
    task_description = f'''Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {data}'''
    team_with_task.reset_task(task_description)
    
    try:
        res = func_timeout(10, func, args=(team_with_task,))
    except FunctionTimedOut:
        return {
            "idx": i,
            "response": "",
            "score": 0.0,
            "error": "function_timeout",
        }
    except Exception as e:
        return {
            "idx": i,
            "response": "",
            "score": 0.0,
            "error": f"function_error: {e}",
        }

    # Evaluation
    try:
        score, _ = evaluate(llm_eval, task_description, res)
    except Exception as e:
        score = 0.0
        return {
            "idx": i,
            "response": res,
            "score": score,
            "error": f"eval_error: {e}",
        }

    result = {
        "idx": i,
        "response": res,
        "score": score,
    }
    return result


async def evaluate_particle(team, func, testset, llm_eval, save_dir, 
                            start_index=5, end_index=None, max_workers=16):
    """Evaluate a particle on the test dataset.
    
    Args:
        team: Team object
        func: Forward function
        testset: Test dataset
        llm_eval: LLM for evaluation
        save_dir: Directory to save results
        start_index: Starting index in dataset
        end_index: Ending index in dataset (None = to end)
        max_workers: Maximum number of worker threads
        
    Returns:
        float: Average fitness score
    """
    if end_index is None:
        end_index = len(testset)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Evaluating on indices {start_index} to {end_index} ({end_index - start_index} examples)")
    
    results = []
    for i, data in enumerate(testset[start_index:end_index]):
        team_with_task = team.deepcopy()
        result = execute(team_with_task, data, i, func, llm_eval)
        results.append(result)
        if (i + 1) % 5 == 0:
            print(f"Progress: {i + 1}/{end_index - start_index}")

    # Save results
    result_file = f'{save_dir}/results.jsonl'
    write_jsonl(result_file, results, 'w')
    print(f"Results saved to: {result_file}")
    
    # Calculate and print fitness
    fitness = get_fitness(results)
    print(f"\nFitness Score: {fitness:.4f}")
    
    return fitness


async def main(particle_idx=-1, model='gpt-4o-mini', eval_model='gpt-4o-mini',
               dataset_path='data/data_100_random_text.jsonl', 
               save_dir='results/test', start_index=5, end_index=None, 
               max_workers=16):
    """Main function to evaluate a particle on test dataset.
    
    Args:
        particle_idx: Index of particle to load from save.jsonl (default: -1, last particle)
        model: Model to use for team execution
        eval_model: Model to use for evaluation
        dataset_path: Path to test dataset file
        save_dir: Directory to save results
        start_index: Starting index in dataset
        end_index: Ending index in dataset (None = to end)
        max_workers: Maximum number of worker threads
    """
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as file:
        testset = file.readlines()
    
    print(f"Loaded {len(testset)} test examples")
    
    # Setup models
    llm_role = ChatOpenAI(
        model=model,
        temperature=0.001,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    llm_eval = ChatOpenAI(
        model=eval_model,
        temperature=0.001,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    
    print(f"Using execution model: {model}")
    print(f"Using evaluation model: {eval_model}")
    
    # Load particle
    logger = setup_logger(9)
    team = Team(llm_role, logger)
    particles = load_particles(particle_idx)
    
    if not particles:
        print(f"Error: No particles found at index {particle_idx}")
        return
    
    team_dict = particles[0]['team']
    code = particles[0]['code']
    team.update(team_dict)
    func = set_forward(code)
    
    log(logger, f'Particle {particle_idx}', f'''{team}\n\n{code}''')
    print(f"Loaded particle {particle_idx}")
    
    # Run evaluation
    await evaluate_particle(
        team, func, testset, llm_eval, save_dir,
        start_index, end_index, max_workers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a saved particle on test dataset')
    parser.add_argument('--particle_idx', type=int, default=-1,
                        help='Index of particle to load from save.jsonl (default: -1, last particle)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model to use for team execution (default: gpt-4o-mini)')
    parser.add_argument('--eval_model', type=str, default='gpt-4o-mini',
                        help='Model to use for evaluation (default: gpt-4o-mini)')
    parser.add_argument('--dataset', type=str, 
                        default='data/data_100_random_text.txt',
                        help='Path to test dataset file')
    parser.add_argument('--save_dir', type=str, default='results/test',
                        help='Directory to save results (default: results/test)')
    parser.add_argument('--start_index', type=int, default=5,
                        help='Starting index in dataset (default: 5)')
    parser.add_argument('--end_index', type=int, default=6,
                        help='Ending index in dataset (default: None, to end)')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Maximum number of worker threads (default: 16)')
    
    args = parser.parse_args()
    
    asyncio.run(main(
        particle_idx=args.particle_idx,
        model=args.model,
        eval_model=args.eval_model,
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        max_workers=args.max_workers
    ))

