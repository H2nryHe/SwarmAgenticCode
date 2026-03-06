# CW Reproduction Report

## 1) Environment
- OS: macOS 26.3 (Build 25D125)
- Arch: arm64
- Python (conda env `swarm`): 3.11.14
- Platform string: `macOS-26.3-arm64-i386-64bit`
- Pip freeze: `pip_freeze_cw.txt`

## 2) Commands Run

### Main training (paper-aligned intent)
From `creative_writing/`:
```bash
python pso.py --max_iteration 10 --dataset data/train_5.txt --max_workers 1 --save_dir results_cw_main --model gpt-4o-mini
```

### Main evaluation (95 tasks: indices 5..99)
From `creative_writing/`:
```bash
python test.py --particle_idx -1 --dataset data/data_100_random_text.txt --save_dir results_cw_main/test_95 \
  --start_index 5 --end_index 100 --max_workers 1 --model gpt-4o-mini --eval_model gpt-4o-mini
```

## 3) Key Artifacts
- Training archive: `creative_writing/save.jsonl`
- Training iteration outputs: `creative_writing/results_cw_main/results-*-*.jsonl`
- Evaluation outputs (95 tasks): `creative_writing/results_cw_main/test_95/results.jsonl`
- Constraint checker output (3 random eval samples): `creative_writing/results_cw_main/test_95/constraint_check_3.json`
- Checker script: `tools/check_cw_constraints.py`

## 4) Final Evaluation Metrics (95-task run)
File: `creative_writing/results_cw_main/test_95/results.jsonl` (95 rows)

- Fitness Score (mean coherence): **0.0000**
- Mean from saved per-example scores: **0.0000**
- Std from saved per-example scores (population): **0.0000**

## 5) Metric Alignment Check (A)
- Confirmed from `creative_writing/eval.py`: coherence is scored on **1–10** and averaged over **5 runs** per example.
- Paper comparison reference: arXiv paper appendix table excerpt reports Creative Writing setting `SwarmAgentic(5,10)` at **8.15** (Table 9 style ablation row).
  - Source: https://arxiv.org/pdf/2506.15672
- This run’s 95-task evaluation score (**0.0000**) is far below the paper-reported level.

## 6) Structure Alignment Check vs Appendix F.2 (B)
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

## 7) Output Format Constraint Check (C)
Checker script: `tools/check_cw_constraints.py`

Random sample settings:
- Source eval set: `results_cw_main/test_95/results.jsonl`
- Sample size: 3
- Seed: 42

Checks:
- 4 paragraphs present
- Paragraph i ends with sentence i from task line

Result:
- Pass count: **0 / 3**
- Pass rate: **0.0**

## 8) Notes on Mismatches / Drift
- Model family nominally aligned (`gpt-4o-mini`) but exact hosted model backend/version can differ from paper-time execution.
- Repository code required runtime hardening to finish in this environment (timeouts, stuck-task handling, workflow patching).
- `test.py` particle selection behavior (always first archived particle) is not equivalent to selecting the best archived particle.
- The final 95-task evaluation used strict timeout guards to guarantee completion; this materially affects score quality.

