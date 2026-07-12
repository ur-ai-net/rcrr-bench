#!/usr/bin/env python3
"""Validate the generated mapping questions (public_data/qa_mapping/) and build the
human-review sheet. Offline. Checks:
  - JSON validity, required keys, task_id uniqueness/format
  - self-containment (values leaked into the question text — the v3 exclusion regex)
  - duplicate/near-duplicate questions within a doc
  - gold length, empty evidence
Output: docs/mapping_qs_review.md (grouped per page, with evidence quote and PDF path,
verdict column for the reviewer) + console summary.
Usage: python validate_mapping_qs.py [qa_dir_name] [review_md_name]
       (defaults: qa_mapping mapping_qs_review.md; round 2: qa_mapping2 mapping_qs_round2_review.md)
Round-2 extra check: forbidden positional/visual vocabulary in question text."""
import os, sys, json, re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
QM = ROOT / "public_data" / (sys.argv[1] if len(sys.argv) > 1 else "qa_mapping")
SP = ROOT / "public_data" / "single_pages"
OUT = ROOT / "docs" / (sys.argv[2] if len(sys.argv) > 2 else "mapping_qs_review.md")
FORBIDDEN = re.compile(r"上部|下部|右側|左側|真ん中|中央|一番上|一番下|左から|右から|の右|の左|"
                       r"グラフ|チャート|ドーナツ|バブル|ウォーターフォール|凡例|折れ線|吹き出し|"
                       r"矢印|点線|左軸|右軸|縦軸|横軸|パネル")

REQ = {"question", "answer", "evidence", "level", "qtype", "doc_id", "cell",
       "content_type", "lang", "task_id"}
NUM = re.compile(r"\d[\d,\.]{2,}")
CALCKW = re.compile(r"差額|差は|何倍|合計|平均|引く|増加幅|何ポイント|変化は")

docs, problems, all_rows = {}, [], []
for f in sorted(QM.glob("tdnet_*.json")):
    try:
        rows = json.loads(f.read_text(encoding="utf-8-sig"))
    except Exception as e:
        problems.append(f"{f.name}: JSON PARSE FAIL {e}"); continue
    docs[f.stem] = rows
    for i, r in enumerate(rows):
        miss = REQ - set(r)
        if miss: problems.append(f"{f.stem}[{i}]: missing keys {miss}")
        if not r.get("evidence", "").strip(): problems.append(f"{r.get('task_id')}: empty evidence")
        if r.get("level") not in ("M1", "M2"): problems.append(f"{r.get('task_id')}: bad level {r.get('level')}")
        all_rows.append(r)

ids = [r["task_id"] for r in all_rows]
for t, c in Counter(ids).items():
    if c > 1: problems.append(f"duplicate task_id: {t} x{c}")

flags = defaultdict(list)
for r in all_rows:
    q = r["question"]
    if len(NUM.findall(q)) >= 2 and CALCKW.search(q):
        flags[r["task_id"]].append("SELF_CONTAINED?")
    fb = FORBIDDEN.findall(q)
    if fb: flags[r["task_id"]].append(f"FORBIDDEN_VOCAB:{','.join(sorted(set(fb)))}")
    nums_in_q = [n for n in NUM.findall(q)]
    # numeric tokens that are not years/periods (rough heuristic)
    susp = [n for n in nums_in_q if not re.fullmatch(r"(19|20)\d\d(?:[/年].*)?", n)]
    if susp: flags[r["task_id"]].append(f"NUM_IN_Q:{','.join(susp[:3])}")
    if len(r["answer"]) > 100: flags[r["task_id"]].append("LONG_GOLD")
for d, rows in docs.items():
    qs = [r["question"] for r in rows]
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            a, b = set(qs[i]), set(qs[j])
            if len(a & b) / max(len(a | b), 1) > 0.8:
                flags[rows[j]["task_id"]].append("NEAR_DUP_IN_DOC")

by_level = Counter(r["level"] for r in all_rows)
by_cell = Counter(r["cell"] for r in all_rows)
print(f"docs={len(docs)} questions={len(all_rows)} levels={dict(by_level)} cells={dict(by_cell)}")
print(f"structural problems: {len(problems)}")
for p in problems: print("  !", p)
print(f"flagged for attention: {len(flags)}")
for t, fl in sorted(flags.items()): print("  ?", t, fl)

lines = ["# Mapping questions — human review sheet",
         "",
         f"{len(all_rows)} questions over {len(docs)} pages, authored by Fable 5 agents from the page images.",
         "For each item: verify (1) the gold is correct per the page, (2) the question has exactly one",
         "defensible answer, (3) the answer is printed on the page (evidence quote should match).",
         "Mark verdict: OK / FIX (edit inline) / DROP. Auto-flags below are hints, not verdicts.",
         ""]
for d in sorted(docs):
    cell = docs[d][0]["cell"] if docs[d] else "?"
    lines.append(f"\n## {d}  ({cell})")
    lines.append(f"page: `public_data/single_pages/{cell}/{d}.pdf`")
    for r in docs[d]:
        fl = " ".join(f"**[{x}]**" for x in flags.get(r["task_id"], []))
        se = f" [{r['source_element']}]" if r.get("source_element") else ""
        lines.append(f"\n- **{r['task_id']}** ({r['level']}){se} {fl}")
        lines.append(f"  - Q: {r['question']}")
        lines.append(f"  - GT: {r['answer']}")
        lines.append(f"  - evidence: {r['evidence']}")
        lines.append(f"  - verdict: [ ]")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nreview sheet -> {OUT}")
