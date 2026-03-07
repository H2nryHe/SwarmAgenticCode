# CW Reproduction Report

## 1) Environment
- OS: macOS 26.3 (Build 25D125)
- Arch: arm64
- Python (conda env `swarm`): 3.11.14
- Platform string: `macOS-26.3-arm64-i386-64bit`
- Key package versions:
  - `langchain==1.0.1`
  - `langchain_openai==1.0.0`
  - `openai==2.6.0`
  - `pydantic==2.11.9`

## 2) Commands Run

### Main training (existing archive used for evaluation)
From `creative_writing/`:
```bash
python pso.py --max_iteration 10 --dataset data/train_5.txt --max_workers 1 --save_dir results_cw_main --model gpt-4o-mini
```

### 1-example timeout debug
From `creative_writing/`:
```bash
CW_OPENAI_TIMEOUT_SECONDS=300 CW_OPENAI_MAX_RETRIES=6 CW_LLM_INVOKE_ATTEMPTS=3 CW_EVAL_SCORE_RUNS=1 \
python test.py --particle_idx -1 --dataset data/data_100_random_text.txt \
  --save_dir results_cw_debug --start_index 5 --end_index 6 \
  --max_workers 1 --model gpt-4o-mini --eval_model gpt-4o-mini
```

### Main evaluation rerun (95 tasks: indices 5..99)
From `creative_writing/`:
```bash
CW_OPENAI_TIMEOUT_SECONDS=300 CW_OPENAI_MAX_RETRIES=6 CW_LLM_INVOKE_ATTEMPTS=3 \
python test.py --particle_idx -1 --dataset data/data_100_random_text.txt \
  --save_dir results_cw_main/test_95 --start_index 5 --end_index 100 \
  --max_workers 1 --model gpt-4o-mini --eval_model gpt-4o-mini
```

## 3) Timeout Diagnosis
- The strings `"function_timeout"` and `"function_error: Request timed out."` are not produced anywhere in the checked-in Creative Writing code.
- They appear only in the previously saved invalid file `creative_writing/results_cw_main/test_95/results.jsonl`, which means that run was wrapped by an external timeout/error harness outside the current CW source tree.
- In the repo itself, the vulnerable call sites were:
  - `creative_writing/test.py`: `ChatOpenAI(...)` without explicit timeout/retry settings.
  - `creative_writing/pso.py`: same issue for role/eval/init models.
  - `creative_writing/eval.py`, `creative_writing/role.py`, and CW prompt modules: direct `chain.invoke(...)` calls without local retry handling.

## 4) Timeout Fix Applied
- Added `creative_writing/llm_utils.py` to centralize:
  - `ChatOpenAI(..., request_timeout=300, max_retries=6)`
  - local `invoke_with_retries(...)` wrapper for transient invoke failures
  - env-controlled `CW_EVAL_SCORE_RUNS` with default `5`
- Switched CW `ChatOpenAI` construction in `test.py` and `pso.py` to the shared helper.
- Wrapped CW `chain.invoke(...)` calls with the retry helper.
- In `test.py`, changed evaluation-only runs to call `evaluate(llm_eval, None, res)` so the unused problem-explanation request is skipped during final scoring.
- Added ignore rules for `API_keys.txt`, `results*/`, `evaluation*/`, and `*.log` remained covered.

## 5) 1-Example Debug Validation
File: `creative_writing/results_cw_debug/results.jsonl`

- Rows: **1**
- Response non-empty: **1 / 1**
- Error rows: **0 / 1**
- Score: **6.0**
- Response length: **1168**

This confirms the patched path returns a real model response and a valid score in `[1,10]`.

## 6) Final Evaluation Metrics (95-task rerun)
File: `creative_writing/results_cw_main/test_95/results.jsonl`
Timestamp: **2026-03-06 20:05:46 PST**

- Rows: **95**
- Fitness Score (mean coherence): **6.0589**
- Mean from saved per-example scores: **6.0589**
- Std from saved per-example scores (population): **1.1099**
- Median score: **6.0**
- Score range: **3.0 .. 8.0**
- Non-empty responses: **95 / 95**
- Non-zero scores: **95 / 95**
- Error rows: **0 / 95**
- Timeout rows: **0 / 95**
- Successful rows (`response` non-empty, `score>0`, no timeout error): **95 / 95**

The timeout-invalid all-zero evaluation has been replaced by a valid 95-example run.

## 7) Metric Alignment Check
- Confirmed from `creative_writing/eval.py`: coherence is scored on **1–10** and the default final evaluation still averages **5 runs** per example.
- The 1-example debug run temporarily set `CW_EVAL_SCORE_RUNS=1`; the final 95-example rerun used the default **5-run** average.
- Paper comparison reference: arXiv paper appendix table excerpt reports Creative Writing setting `SwarmAgentic(5,10)` at **8.15**.
- Current rerun score (**6.0589**) is valid, but still below the paper-reported level.

## 8) Structure Alignment Check vs Appendix F.2
Reference role list (Appendix F.2):
- Sentence Analyzer
- Narrative Architect
- Narrative Coherence Reviewer
- Feedback Integrator
- Thematic Integration Specialist
- Integration Clarity Review
- Integrated Feedback Review
- Feedback Review Discussion
- Paragraph Developer
- Final Integrator

### Best particle from archive (max score in `save.jsonl`)
- Best archive score: **7.4**
- Roles:
  - Sentence Collector
  - Narrative Developer
  - Editor
- Workflow order:
  1. Sentence Collector
  2. Narrative Developer
  3. Editor

#### Diff vs Appendix F.2 role list
- Overlap count: **0 / 10**
- Missing (paper roles absent):
  - Sentence Analyzer
  - Narrative Architect
  - Narrative Coherence Reviewer
  - Feedback Integrator
  - Thematic Integration Specialist
  - Integration Clarity Review
  - Integrated Feedback Review
  - Feedback Review Discussion
  - Paragraph Developer
  - Final Integrator
- Extra (in run, not in Appendix F.2 list):
  - Sentence Collector
  - Narrative Developer
  - Editor

### Note about what `test.py` actually loads
Current `test.py` uses `particles = load_particles(-1)` and then `particles[0]` (first particle), not the max-score particle.
- Evaluated particle score in archive: **6.04**
- Evaluated particle roles:
  - Task Coordinator
  - Sentence Analyzer
  - Narrative Developer
  - Thematic Reviewer
  - Editor
  - Final Integrator

## 9) Constraint Check Status
- The report from the earlier invalid run referenced `tools/check_cw_constraints.py`, but that checker script is not present in the current checkout.
- No fresh constraint-check rerun was performed as part of the timeout fix.

## 10) Notes
- The prior `0.0000` evaluation was invalid because the saved file contained only external timeout-wrapper errors and empty responses.
- The patched rerun shows the CW pipeline can complete normally in this environment when explicit OpenAI request timeouts and retries are configured.
- Remaining gap versus the paper is now about model/team-quality alignment, not an evaluation-time timeout collapse.
