"""
Testing and Evaluation Script for Creative Writing Task
Evaluate a saved particle on test dataset
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import argparse
import json
import os
from tqdm import tqdm
from func import *
from role import Team
from logger import setup_logger, log
from eval import evaluate, get_fitness
from llm_utils import build_chat_openai


def load_archive_entry(archive_idx):
    """Load one archive entry from save.jsonl and resolve negative indices."""
    archives = read_jsonl('save.jsonl')
    if not archives:
        return None, None

    resolved_idx = archive_idx if archive_idx >= 0 else len(archives) + archive_idx
    if resolved_idx < 0 or resolved_idx >= len(archives):
        return None, None

    return resolved_idx, archives[resolved_idx]['archive']


def select_particle(particles, particle_mode):
    """Select a particle deterministically from one archive entry."""
    if not particles:
        return None, None

    if particle_mode == 'best':
        best_idx = max(range(len(particles)), key=lambda idx: particles[idx].get('score', float('-inf')))
        return best_idx, particles[best_idx]

    return 0, particles[0]


def get_particle_metadata(particle):
    team_dict = particle['team']
    roles = [role['Name'] for role in team_dict['roles']]
    workflow_order = [step['Role'] for step in team_dict['workflow']]
    return {
        "archive_score": particle.get('score'),
        "roles": roles,
        "workflow_order": workflow_order,
    }


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
        res = func(team_with_task)
        score, _ = evaluate(llm_eval, None, res)
        return {
            "idx": i,
            "response": res,
            "score": score,
        }
    except Exception as exc:
        return {
            "idx": i,
            "response": "",
            "score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


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
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_running_loop()
        tasks = []
        for i, data in enumerate(testset[start_index:end_index]):
            team_with_task = team.deepcopy() 
            tasks.append(loop.run_in_executor(
                executor, execute, team_with_task, data, i, func, llm_eval
            ))

        results = await asyncio.gather(*tasks)

    # Save results
    result_file = f'{save_dir}/results.jsonl'
    write_jsonl(result_file, results, 'w')
    print(f"Results saved to: {result_file}")
    
    # Calculate and print fitness
    fitness = get_fitness(results)
    print(f"\nFitness Score: {fitness:.4f}")
    
    return fitness


async def main(particle_idx=-1, particle_mode='first', model='gpt-4o-mini', eval_model='gpt-4o-mini',
               dataset_path='data/data_100_random_text.jsonl', 
               save_dir='results/test', start_index=5, end_index=None, 
               max_workers=16):
    """Main function to evaluate a particle on test dataset.
    
    Args:
        particle_idx: Archive entry index to load from save.jsonl (default: -1, last archive entry)
        particle_mode: Which particle to evaluate from the selected archive entry
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
    llm_role = build_chat_openai(model=model, temperature=0.001)
    llm_eval = build_chat_openai(model=eval_model, temperature=0.001)
    
    print(f"Using execution model: {model}")
    print(f"Using evaluation model: {eval_model}")
    
    # Load particle
    logger = setup_logger(9)
    team = Team(llm_role, logger)
    resolved_archive_idx, particles = load_archive_entry(particle_idx)
    
    if not particles:
        print(f"Error: No archive entry found at index {particle_idx}")
        return

    selected_particle_idx, selected_particle = select_particle(particles, particle_mode)
    metadata = get_particle_metadata(selected_particle)

    team_dict = selected_particle['team']
    code = selected_particle['code']
    team.update(team_dict)
    func = set_forward(code)

    selection_summary = {
        "selected_archive_entry_index": resolved_archive_idx,
        "selected_particle_mode": particle_mode,
        "selected_particle_index": selected_particle_idx,
        "selected_particle_archive_score": metadata["archive_score"],
        "selected_particle_roles": metadata["roles"],
        "selected_workflow_order": metadata["workflow_order"],
    }

    print(json.dumps(selection_summary, indent=2))
    log(logger, 'Particle Selection', json.dumps(selection_summary, indent=2))
    log(logger, f'Particle archive {resolved_archive_idx} member {selected_particle_idx}', f'''{team}\n\n{code}''')
    print(f"Loaded archive entry {resolved_archive_idx}, particle {selected_particle_idx} ({particle_mode})")
    
    # Run evaluation
    await evaluate_particle(
        team, func, testset, llm_eval, save_dir,
        start_index, end_index, max_workers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a saved particle on test dataset')
    parser.add_argument('--particle_idx', type=int, default=-1,
                        help='Archive entry index to load from save.jsonl (default: -1, last archive entry)')
    parser.add_argument('--particle_mode', type=str, default='first', choices=['first', 'best'],
                        help='Which particle to evaluate from the selected archive entry (default: first)')
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
        particle_mode=args.particle_mode,
        model=args.model,
        eval_model=args.eval_model,
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        max_workers=args.max_workers
    ))

