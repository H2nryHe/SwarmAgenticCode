# RUN LOG

## Stage 0
- Workspace: /tmp/work/swarmagentic/SwarmAgenticCode
- Repo: yaoz720/SwarmAgenticCode

## Stage 1 - Environment inventory
### sw_vers
ProductName:		macOS
ProductVersion:		26.3
BuildVersion:		25D125
### uname -m
arm64
### sysctl -n machdep.cpu.brand_string
### which git && git --version
/opt/homebrew/bin/git
git version 2.39.2
### which python3 && python3 --version
/Users/linruihe/anaconda3/bin/python3
Python 3.10.9
### which conda && conda --version
/Users/linruihe/anaconda3/bin/conda
conda 23.9.0

## Stage 2 - Conda env
- Created env: swarm (python 3.11, osx-arm64)
### python/arch in env
/Users/linruihe/anaconda3/envs/swarm/bin/python
arm64


## Stage 3 - Dependencies
- Installed requirements with minimal fixes
- Change 1: requirements.txt pydantic==2.12.3 -> pydantic==2.11.9
- Change 2: pip override pandas to 2.3.3 (compatible with numpy 2.3.4 and gradio<3)
### sanity import check
imports_ok


## Stage 4 - API key load
### API_keys.txt permissions
-rw------- API_keys.txt
- detected_format=OpenAI API:VALUE
OPENAI_API_KEY_set= True len= 164 prefix= sk-pro...

## Stage 5 - Smoke test (travelplanner/swarm)
### PSO command
python pso.py \
  --max_iteration 1 \
  --settings 0.2 0.8 \
  --sample_step 45 \
  --max_workers 1 \
  --save_dir evaluation_smoke \
  --dataset data/train_45.jsonl \
  --ref_info data/train_ref_info.jsonl

### PSO result
- Loaded 1 examples (sampled every 45 items)
- Optimization completed
- Global best fitness: 0.0
- Global best trend: [0.0]

### Artifact check
- travelplanner/swarm/save.jsonl exists
- travelplanner/swarm/evaluation_smoke/results-0-0.jsonl
- travelplanner/swarm/evaluation_smoke/results-0-1.jsonl

### Eval command
python test.py \
  --particle_idx -1 \
  --start_index 0 \
  --end_index 2 \
  --max_workers 1 \
  --save_dir evaluation_smoke/test \
  --dataset data/validation.jsonl \
  --ref_info data/validation_ref_info.jsonl

### Eval printed metrics
- Commonsense Constraint Pass Rate: {'Commonsense Constraint Micro Pass Rate': 0.0, 'Hard Constraint Micro Pass Rate': 0.0}
- Hard Constraint Pass Rate: {'fallback_reason': "[Errno 2] No such file or directory: '../database/flights/clean_Flights_2022.csv'"}

### Eval artifacts
- travelplanner/swarm/evaluation_smoke/test/results.jsonl

## Stage 6 - Normal run commands (heavier)
From repo root:
- cd travelplanner/swarm
- python pso.py --max_iteration 5 --dataset data/train_45.jsonl --ref_info data/train_ref_info.jsonl
- python test.py --particle_idx -1
- Default model: gpt-4o-mini (unless --model is provided)

## Stage 7 - Deliverables
- RUN_LOG.md (this file)
- pip_freeze.txt
- conda_env.txt

## Code/Dependency changes applied
1) requirements.txt
- pydantic==2.12.3 -> pydantic==2.11.9 (gradio pin compatibility)

2) Runtime package override (kept in environment)
- pandas upgraded to 2.3.3 (to resolve numpy/pandas ABI issue with numpy 2.3.4)

3) travelplanner/swarm/eval.py
- Added fallback evaluator path when travelplanner/database assets are missing
- Restored cwd after evaluator import attempt to avoid side effects

4) travelplanner/swarm/pso.py
- Added save_particles(particles) before save_state(...) so save.jsonl is created for test.py
