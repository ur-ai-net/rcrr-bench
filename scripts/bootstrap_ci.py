#!/usr/bin/env python3
"""Cluster-bootstrap 95% CIs for the final RCRR scores (reader = Gemini 3.5 Flash).
Resamples PAGES (not questions) with replacement: questions on the same page share one
parse and are correlated, so the page is the independent sampling unit. Also reports
paired per-page differences vs anchor parsers (page difficulty cancels), which is the
correct test for "A beats B" / "statistical tie" claims.
Offline — reads experiments/rcrr_v2_reader35/*.jsonl, no API keys needed.
Usage: python bootstrap_ci.py [B]   (default B=10000 replicates)"""
import os, sys, json, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "experiments" / "rcrr_v2"
NEW = ROOT / "experiments" / "rcrr_v2_reader35"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
ANCHORS = ["azure_di", "nebula"]          # paired diffs reported against these
random.seed(20260621)                      # fixed seed -> reproducible tables

ver = json.load(open(SRC / "unfair_audit_b8.json", encoding="utf-8"))
IMP = {"chart_geometry", "not_on_page", "gold_wrong", "ambiguous"}
def is_imp(v):
    a = (v.get("verdict") or "").lower(); c = (v.get("category") or "").lower()
    if a == "fair" or c == "answerable": return False
    return a == "impossible" or a in IMP or c in IMP
removed = {t for t, v in ver.items() if is_imp(v)}

# per parser: doc_id -> [(cell, score 0..1), ...]
parsers = sorted(f.stem for f in NEW.glob("*.jsonl"))
bydoc = {p: defaultdict(list) for p in parsers}
for p in parsers:
    for line in open(NEW / f"{p}.jsonl", encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        if r["task_id"] in removed or r.get("score_gpt54mini", -1) < 0: continue
        bydoc[p][r["doc_id"]].append((r["cell"], r["score_gpt54mini"] / 2.0))

docs = sorted(set.intersection(*(set(bydoc[p]) for p in parsers)))
print(f"parsers={len(parsers)} pages={len(docs)} B={B}", flush=True)

# precompute per-page (sum, n) so each replicate is O(pages)
agg = {p: {sc: {d: (sum(s for c, s in bydoc[p][d] if sc is None or c == sc),
                    sum(1 for c, s in bydoc[p][d] if sc is None or c == sc))
                for d in docs}
           for sc in (None, "jp_table", "jp_chart")} for p in parsers}

def score(p, sample, cell=None):
    a = agg[p][cell]
    num = sum(a[d][0] for d in sample); den = sum(a[d][1] for d in sample)
    return num / den * 100 if den else float("nan")

reps = {p: {sc: [] for sc in (None, "jp_table", "jp_chart")} for p in parsers}
for _ in range(B):
    sample = [random.choice(docs) for _ in docs]
    for p in parsers:
        for sc in (None, "jp_table", "jp_chart"):
            reps[p][sc].append(score(p, sample, sc))

def ci(v):
    v = sorted(v); return v[int(0.025 * len(v))], v[int(0.975 * len(v))]

print(f"\n{'parser':20} " + "".join(f"{lbl:>24}" for lbl in ("ALL", "tables", "charts")))
for p in sorted(parsers, key=lambda x: -score(x, docs)):
    cells = []
    for sc in (None, "jp_table", "jp_chart"):
        lo, hi = ci(reps[p][sc])
        cells.append(f"{score(p, docs, sc):5.1f} [{lo:5.1f},{hi:5.1f}]")
    print(f"{p:20} " + "".join(f"{c:>24}" for c in cells))

for anchor in ANCHORS:
    if anchor not in parsers: continue
    print(f"\npaired diffs vs {anchor} (overall; CI straddling 0 = statistical tie):")
    for p in sorted(parsers, key=lambda x: -score(x, docs)):
        if p == anchor: continue
        diffs = sorted(a - b for a, b in zip(reps[p][None], reps[anchor][None]))
        lo, hi = ci(diffs)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "tie"
        print(f"  {p:20} {score(p, docs) - score(anchor, docs):+5.1f}  [{lo:+5.1f},{hi:+5.1f}]  {sig}")
