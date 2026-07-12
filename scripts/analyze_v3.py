#!/usr/bin/env python3
"""Phase C analysis over the v3 grid (see docs/benchmark_v3.md §3).

Reads experiments/v3/{reader}/{parser}.jsonl and reports, for each of the four
reader x judge configs and each question subset:
  - per-parser RCRR with cluster-bootstrap 95% CIs (pages resampled, not questions)
  - paired per-page diffs vs anchors (azure_di, nebula) with CIs
  - ranking-stability table across configs
  - fine-tuning delta: qwen32b_base vs nebula (same pages/questions/readers/judges)

Question subsets:
  core    = v3_status "kept"            (primary)
  +review = kept + review               (sensitivity)
  mapping = the 285 human-reviewed mapping questions (separate column, never blended)

Config notes: (gemini35flash reader x gemini35flash judge) is SELF-GRADING — reported but
flagged; primary configs are the cross-family pairs.
Offline; no API keys. Usage: python analyze_v3.py [B]   (default 10000)
Output: printed tables + experiments/v3/analysis_summary.md"""
import os, sys, json, random
from pathlib import Path
from collections import defaultdict

# Windows consoles may default to a non-UTF-8 codepage; never let printing kill the run
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
V3 = ROOT / "experiments" / "v3"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
READERS = ["gemini35flash", "gpt54mini"]
JUDGES = ["gpt54mini", "gemini35flash"]
ANCHORS = ["azure_di", "nebula"]
SELF_GRADING = ("gemini35flash", "gemini35flash")   # (reader, judge)
random.seed(20260705)

# data[reader][parser] = list of rows
data = defaultdict(dict)
for reader in READERS:
    for f in sorted((V3 / reader).glob("*.jsonl")):
        rows = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
        data[reader][f.stem] = rows
parsers = sorted(set.intersection(*(set(data[r]) for r in READERS)))
print(f"parsers={len(parsers)} B={B}")

def bydoc(rows, judge, subset):
    d = defaultdict(lambda: [0.0, 0])
    for r in rows:
        st = r.get("v3_status")
        if subset == "core" and st != "kept": continue
        if subset == "core+review" and st not in ("kept", "review"): continue
        if subset == "mapping" and st != "mapping": continue
        s = r.get(f"score_{judge}", -1)
        if s < 0: continue
        d[r["doc_id"]][0] += s / 2.0; d[r["doc_id"]][1] += 1
    return {k: tuple(v) for k, v in d.items()}

def ci(v):
    v = sorted(v); return v[int(0.025 * len(v))], v[int(0.975 * len(v))]

lines = ["# v3 analysis summary", ""]
rank_table = defaultdict(dict)   # parser -> config label -> rank

for subset in ("core", "mapping", "core+review"):
    for reader in READERS:
        for judge in JUDGES:
            label = f"{reader}-reader/{judge}-judge"
            flag = "  [SELF-GRADING — excluded from primary]" if (reader, judge) == SELF_GRADING else ""
            agg = {p: bydoc(data[reader][p], judge, subset) for p in parsers}
            docs = sorted(set.intersection(*(set(v) for v in agg.values())))
            if not docs: continue
            def score(p, sample):
                num = sum(agg[p][d][0] for d in sample); den = sum(agg[p][d][1] for d in sample)
                return num / den * 100 if den else float("nan")
            reps = {p: [] for p in parsers}
            for _ in range(B):
                sample = [random.choice(docs) for _ in docs]
                for p in parsers: reps[p].append(score(p, sample))
            order = sorted(parsers, key=lambda p: -score(p, docs))
            if subset == "core":
                for i, p in enumerate(order, 1): rank_table[p][label] = i
            lines.append(f"\n## {subset} | {label}{flag}  (pages={len(docs)})")
            lines.append(f"{'parser':20} {'RCRR':>6}  95% CI")
            for p in order:
                lo, hi = ci(reps[p])
                lines.append(f"{p:20} {score(p, docs):6.1f}  [{lo:5.1f},{hi:5.1f}]")
            for anchor in ANCHORS:
                if anchor not in parsers: continue
                lines.append(f"paired diffs vs {anchor}:")
                for p in order:
                    if p == anchor: continue
                    diffs = sorted(a - b for a, b in zip(reps[p], reps[anchor]))
                    lo, hi = ci(diffs)
                    sig = "SIG" if (lo > 0 or hi < 0) else "tie"
                    lines.append(f"  {p:20} {score(p, docs) - score(anchor, docs):+6.1f} [{lo:+5.1f},{hi:+5.1f}] {sig}")

lines.append("\n\n# Ranking stability across configs (core subset)")
labels = sorted({l for v in rank_table.values() for l in v})
lines.append(f"{'parser':20} " + "  ".join(f"{l[:22]:>24}" for l in labels))
for p in sorted(rank_table, key=lambda x: rank_table[x].get(labels[0], 99)):
    lines.append(f"{p:20} " + "  ".join(f"{rank_table[p].get(l, '-'):>24}" for l in labels))

out = V3 / "analysis_summary.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:60]))
print(f"\nfull summary -> {out}")
