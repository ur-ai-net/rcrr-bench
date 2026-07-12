#!/usr/bin/env python3
"""Full-coverage, dual-family fairness audit (v3). Addresses two gaps in the original
audit (audit_unfair.py): (1) it only reviewed the 97 questions where every OCR product
failed, leaving 92% of questions unaudited; (2) the auditor was a Gemini model checking
Gemini-generated golds (family circularity).

Here EVERY validated question is audited independently by two model families, each
looking at the original page image:
  - Gemini 3.1 Pro   (PDF upload, as in the original audit)
  - GPT-5.4-mini     (page rendered to PNG at 150 dpi)
Same rubric as the original audit. Verdicts -> experiments/controls/audit_v3_{name}.jsonl
(resumable per question). Exclusion policy is applied later in build_qa_v3.py:
both-impossible = excluded, split verdict = flagged for human review.
Keys: GEMINI_API_KEY, OPENAI_API_KEY (.env). ~2 x 1,194 image calls."""
import os, sys, json, re, glob, base64, threading
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
QA_DIR = ROOT / "public_data" / "qa"
SP = ROOT / "public_data" / "single_pages"
SRC = ROOT / "experiments" / "rcrr_v2"
OUT_DIR = ROOT / "experiments" / "controls"; OUT_DIR.mkdir(parents=True, exist_ok=True)
CELLS = ["jp_table", "jp_chart"]
AUDITORS = {"gemini31pro": "gemini-3.1-pro-preview", "gpt54mini": "gpt-5.4-mini",
            "gemini35flash": "gemini-3.5-flash"}
# NOTE: gemini-3.1-pro-preview proved flaky at scale here (stalled uploads, ~20% empty
# responses). gemini35flash is the canonical second auditor (same model as the v2 audit);
# any partial 3.1-pro verdicts are kept as supplementary evidence only.
WORKERS = 6
_lock = threading.Lock()

# same rubric as audit_unfair.py (kept verbatim so verdicts are comparable)
PROMPT = """あなたはOCRベンチマークの公正性を監査する専門家です。添付した1ページの資料画像と、その内容に関する「質問」「正解」を見て、この問いが OCR（画像→テキスト・表への変換）で公正に答えられるかを判定してください。
判定:
- "fair": 正解に必要な情報が、ページ上に文字・数値・表として明示されており、忠実にテキスト化すれば答えられる。
- "impossible"（理不尽）: chart_geometry（正解値がグラフの棒の高さ等、図形からしか読めずデータ数字が書かれていない）/ not_on_page（正解がページに無い）/ gold_wrong（提示された正解がページ内容と矛盾）/ ambiguous（質問が曖昧で一意に定まらない）。
必ずJSONのみ: {"verdict":"fair"|"impossible","category":"answerable|chart_geometry|not_on_page|gold_wrong|ambiguous","reason":"日本語簡潔"}
質問: %s
正解: %s"""

# validated set = everything the legacy B8 audit did not already remove
ver = json.load(open(SRC / "unfair_audit_b8.json", encoding="utf-8"))
IMP = {"chart_geometry", "not_on_page", "gold_wrong", "ambiguous"}
def is_imp(v):
    a = (v.get("verdict") or "").lower(); c = (v.get("category") or "").lower()
    if a == "fair" or c == "answerable": return False
    return a == "impossible" or a in IMP or c in IMP
removed = {t for t, v in ver.items() if is_imp(v)}

def retry(fn, n=4):
    import time
    for i in range(n):
        try: return fn()
        except Exception:
            if i == n - 1: raise
            time.sleep(2 * (i + 1))

def parse(t):
    m = re.search(r'\{.*\}', t or "", re.DOTALL)
    try:
        d = json.loads(m.group(0))
        return {"verdict": d.get("verdict", "?"), "category": d.get("category", "?"),
                "reason": d.get("reason", "")}
    except Exception:
        return {"verdict": "?", "category": "parse_error", "reason": ""}

def pdf_for(doc_id):
    g = glob.glob(str(SP / "*" / f"{doc_id}.pdf")); return g[0] if g else None

def audit_gemini(pdf, qas, out_file, done, name="gemini31pro"):
    from google import genai
    from google.genai import types
    import httpx
    cli = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                       http_options={"client_args": {"timeout": httpx.Timeout(120.0)}})
    up = cli.files.upload(file=pdf)
    # flash: thinking off (README troubleshooting), modest cap — same recipe as the v2 audit.
    # 3.1 Pro: thinking can't be disabled and spends from max_output_tokens — keep it high.
    if "flash" in name:
        cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=4000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json")
    else:
        cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=16000,
            response_mime_type="application/json")
    for q in qas:
        if q["task_id"] in done: continue
        def _call():
            return cli.models.generate_content(model=AUDITORS[name],
                contents=[PROMPT % (q["question"], q["answer"]), up],
                config=cfg)
        v = None
        for attempt in range(3):   # also retry PARSE failures (empty/odd text), not just API errors
            try: resp = retry(_call)
            except Exception as e:
                v = {"verdict": "?", "category": "err", "reason": str(e)[:80]}; break
            v = parse(resp.text)
            if v.get("category") != "parse_error": break
            fr = ""
            try: fr = str(resp.candidates[0].finish_reason)
            except Exception: pass
            v["reason"] = f"finish={fr} text={str(resp.text)[:40]}"
        v.update(task_id=q["task_id"], doc_id=q["doc_id"], cell=q["cell"],
                 question=q["question"], reference=q["answer"])
        with _lock, out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

def audit_gpt(pdf, qas, out_file, done):
    import fitz
    from openai import OpenAI
    png = fitz.open(pdf)[0].get_pixmap(dpi=150).tobytes("png")
    b64 = base64.b64encode(png).decode()
    cli = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120)
    for q in qas:
        if q["task_id"] in done: continue
        def _call():
            return cli.responses.create(model=AUDITORS["gpt54mini"], input=[{"role": "user", "content": [
                {"type": "input_text", "text": PROMPT % (q["question"], q["answer"])},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}]}])
        try: v = parse(retry(_call).output_text)
        except Exception as e: v = {"verdict": "?", "category": "err", "reason": str(e)[:80]}
        v.update(task_id=q["task_id"], doc_id=q["doc_id"], cell=q["cell"],
                 question=q["question"], reference=q["answer"])
        with _lock, out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(AUDITORS)
    by_doc = defaultdict(list)
    for cell in CELLS:
        for jf in sorted((QA_DIR / cell).glob("*.json")):
            for r in json.loads(jf.read_text(encoding="utf-8")):
                if r["task_id"] in removed: continue
                r.setdefault("cell", cell); by_doc[r["doc_id"]].append(r)
    total = sum(len(v) for v in by_doc.values())
    for name in only:
        out_file = OUT_DIR / f"audit_v3_{name}.jsonl"
        done = set()
        if out_file.exists():
            for l in out_file.open(encoding="utf-8"):
                if l.strip(): done.add(json.loads(l)["task_id"])
        todo = [(d, qs) for d, qs in sorted(by_doc.items()) if any(q["task_id"] not in done for q in qs)]
        print(f"[{name}] questions={total} done={len(done)} docs_todo={len(todo)}", flush=True)
        def work(arg):
            d, qs = arg
            pdf = pdf_for(d)
            if not pdf: return
            if name.startswith('gemini'): audit_gemini(pdf, qs, out_file, done, name)
            else: audit_gpt(pdf, qs, out_file, done)
        cnt = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for _ in as_completed([ex.submit(work, a) for a in todo]):
                cnt += 1
                if cnt % 10 == 0: print(f"  [{name}] {cnt}/{len(todo)} docs", flush=True)
        rows = [json.loads(l) for l in out_file.open(encoding="utf-8") if l.strip()]
        print(f"[{name}] verdicts: {Counter(r['verdict'] for r in rows)}")
        print(f"[{name}] categories: {Counter(r['category'] for r in rows)}", flush=True)

if __name__ == "__main__":
    main()
