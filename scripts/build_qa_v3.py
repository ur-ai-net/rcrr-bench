#!/usr/bin/env python3
"""Build the v3 question manifest by combining every Phase-A filter. Offline.

Filters, in order of application (a question is excluded by the FIRST filter that
catches it; all applicable reasons are still recorded):
  1. legacy_audit   — the original B8 audit (unfair_audit_b8.json): 46 impossible.
  2. self_contained — calculation questions that embed >=2 multi-digit numbers plus a
                      calc keyword IN THE QUESTION. These test the reader's arithmetic
                      and grounding discipline, not OCR quality (see docs/benchmark_v3.md).
  3. nodoc          — questions the reader answered correctly (full credit) from an
                      EMPTY document (experiments/controls/nodoc_compromised.json):
                      self-contained or contaminated from pretraining.
  4. audit_v3       — full dual-family image audit (audit_v3.py):
                      both auditors impossible -> excluded;
                      exactly one auditor impossible -> status "review" (kept in scoring
                      until a human rules; listed for the research team).

Output: public_data/qa_v3_manifest.json
  {"summary": {...}, "questions": {task_id: {"status": kept|excluded|review,
   "reasons": [...], "cell":..., "level":...}}}
Usage: python build_qa_v3.py   (rerun any time; picks up whichever inputs exist)"""
import os, json, re
from pathlib import Path
from collections import Counter

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
QA_DIR = ROOT / "public_data" / "qa"
SRC = ROOT / "experiments" / "rcrr_v2"
CTRL = ROOT / "experiments" / "controls"
OUT = ROOT / "public_data" / "qa_v3_manifest.json"
CELLS = ["jp_table", "jp_chart"]

meta = {}
for cell in CELLS:
    for jf in sorted((QA_DIR / cell).glob("*.json")):
        for r in json.loads(jf.read_text(encoding="utf-8")):
            r.setdefault("cell", cell); meta[r["task_id"]] = r

# 1. legacy B8 audit
ver = json.load(open(SRC / "unfair_audit_b8.json", encoding="utf-8"))
IMP = {"chart_geometry", "not_on_page", "gold_wrong", "ambiguous"}
def is_imp(v):
    a = (v.get("verdict") or "").lower(); c = (v.get("category") or "").lower()
    if a == "fair" or c == "answerable": return False
    return a == "impossible" or a in IMP or c in IMP
legacy = {t for t, v in ver.items() if is_imp(v)}

# 2. self-contained calc questions
NUM = re.compile(r"\d[\d,\.]{2,}")
CALCKW = re.compile(r"差額|差は|何倍|合計|平均|引く|増加幅|何ポイント|変化は")
selfc = {t for t, r in meta.items()
         if len(NUM.findall(r["question"])) >= 2 and CALCKW.search(r["question"])}

# 3. no-doc control
nodoc = set()
f = CTRL / "nodoc_compromised.json"
if f.exists(): nodoc = set(json.loads(f.read_text(encoding="utf-8")))
else: print("NOTE: no-doc control results not found — run nodoc_control.py")

# 4. dual-family audit
def load_audit(name):
    d = {}
    f = CTRL / f"audit_v3_{name}.jsonl"
    if not f.exists():
        print(f"NOTE: audit_v3_{name}.jsonl not found — run audit_v3.py"); return d
    for l in f.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l); d[r["task_id"]] = is_imp(r)
    return d
# canonical second auditor = gemini 3.5 flash (3.1-pro proved flaky at scale; its partial
# verdicts fill any gaps but flash is authoritative where both exist)
aud_g = load_audit("gemini31pro"); aud_g.update(load_audit("gemini35flash"))
aud_o = load_audit("gpt54mini")

manifest = {}
for t, r in meta.items():
    reasons = []
    if t in legacy: reasons.append("legacy_audit:" + (ver[t].get("category") or "impossible"))
    if t in selfc: reasons.append("self_contained")
    if t in nodoc: reasons.append("nodoc_compromised")
    both = aud_g.get(t) and aud_o.get(t)
    one = (aud_g.get(t) or aud_o.get(t)) and not both
    if both: reasons.append("audit_v3:both_impossible")
    status = "excluded" if (t in legacy or t in selfc or t in nodoc or both) else \
             ("review" if one else "kept")
    if status == "review": reasons.append("audit_v3:split_verdict")
    manifest[t] = {"status": status, "reasons": reasons,
                   "cell": r["cell"], "level": r.get("level")}

summary = {
    "total": len(manifest),
    "by_status": dict(Counter(v["status"] for v in manifest.values())),
    "excluded_by_reason": dict(Counter(x for v in manifest.values()
                                       if v["status"] == "excluded" for x in v["reasons"])),
    "kept_by_cell": dict(Counter(v["cell"] for v in manifest.values() if v["status"] == "kept")),
    "review_task_ids": sorted(t for t, v in manifest.items() if v["status"] == "review"),
}
OUT.write_text(json.dumps({"summary": summary, "questions": manifest},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nmanifest -> {OUT}")
