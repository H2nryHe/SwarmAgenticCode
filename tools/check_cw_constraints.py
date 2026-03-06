#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path


def extract_sentences(task_text: str):
    text = task_text.strip()
    sents = [s.strip() for s in re.findall(r"[^.!?]+[.!?]", text)]
    if len(sents) < 4:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        sents = parts
    return sents[:4]


def clean_block(block: str):
    lines = []
    for line in block.splitlines():
        if line.strip().startswith("#"):
            continue
        lines.append(line)
    cleaned = " ".join(" ".join(lines).split()).strip()
    return cleaned


def extract_paragraphs(response: str):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", response.strip()) if b.strip()]
    paras = []
    for block in blocks:
        c = clean_block(block)
        lc = c.lower()
        if not c:
            continue
        if lc.startswith("feedback integration report"):
            continue
        if lc.startswith("checklist"):
            continue
        if c == "---":
            continue
        paras.append(c)
    return paras


def norm(s: str):
    return " ".join(s.strip().strip('"\'').split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="creative_writing/data/data_100_random_text.txt")
    ap.add_argument("--results", default="creative_writing/results_cw_main/test_95/results.jsonl")
    ap.add_argument("--start_index", type=int, default=5)
    ap.add_argument("--sample_n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="creative_writing/results_cw_main/test_95/constraint_check_3.json")
    args = ap.parse_args()

    data_lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in Path(args.results).read_text(encoding="utf-8").splitlines() if x.strip()]

    random.seed(args.seed)
    sample_rows = random.sample(rows, min(args.sample_n, len(rows)))

    checks = []
    pass_count = 0
    for row in sample_rows:
        local_idx = int(row.get("idx", -1))
        global_idx = args.start_index + local_idx
        task = data_lines[global_idx] if 0 <= global_idx < len(data_lines) else ""
        target = extract_sentences(task)
        paras = extract_paragraphs(row.get("response", ""))
        paras4 = paras[:4]
        para_count_ok = len(paras4) == 4
        ending_checks = []
        for i in range(4):
            if i < len(paras4) and i < len(target):
                ending_checks.append(norm(paras4[i]).endswith(norm(target[i])))
            else:
                ending_checks.append(False)
        all_ok = para_count_ok and all(ending_checks)
        if all_ok:
            pass_count += 1
        checks.append({
            "local_idx": local_idx,
            "global_idx": global_idx,
            "para_count_ok": para_count_ok,
            "ending_checks": ending_checks,
            "all_ok": all_ok,
        })

    result = {
        "sample_n": len(checks),
        "pass_count": pass_count,
        "pass_rate": (pass_count / len(checks)) if checks else 0.0,
        "checks": checks,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
