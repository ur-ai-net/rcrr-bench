#!/usr/bin/env python3
"""No-document control run: the reader answers every validated question with an EMPTY
document. Any question it still answers correctly is compromised — either self-contained
(needed values embedded in the question) or contaminated (reader knows these public TDnet
filings from pretraining). Same reader/judge/prompts as reader35_rescore.py.
Output -> experiments/controls/nodoc_reader35.jsonl (resumable per doc) + summary.
Keys: GEMINI_API_KEY, OPENAI_API_KEY (.env).  ~99 reader + ~99 judge calls."""
import os, json, re, threading
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
QA_DIR = ROOT / "public_data" / "qa"
SRC = ROOT / "experiments" / "rcrr_v2"
OUT_DIR = ROOT / "experiments" / "controls"; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "nodoc_reader35.jsonl"
READER = "gemini-3.5-flash"; JUDGE = "gpt-5.4-mini"
EMPTY_DOC = "（空のドキュメント — 内容はありません）"
CELLS = ["jp_table", "jp_chart"]; WORKERS = 8
_lock = threading.Lock()

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
            time.sleep(1.5 * (i + 1))

def pj(t, key):
    m = re.search(r'\{.*\}', t or "", re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if key in d: return d[key]
        except Exception: pass
    return None

def read_batch(qs):
    from google import genai
    from google.genai import types
    import httpx
    ql = "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
    prompt = ("以下のドキュメントに基づいて各質問に簡潔に日本語で回答。情報がなければ「情報なし」。"
              '出力は必ずJSON {"answers":["回答1",...]} で質問数と同数。\n\nドキュメント:\n'
              + EMPTY_DOC + "\n\n質問:\n" + ql)
    def c():
        cl = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options={"client_args": {"timeout": httpx.Timeout(90.0)}})
        return cl.models.generate_content(model=READER, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=8000,
                thinking_config=types.ThinkingConfig(thinking_budget=0), response_mime_type="application/json"))
    a = pj(retry(c).text, "answers") or []
    a = [str(x) for x in a]
    if len(a) < len(qs): a += ["[NO ANSWER]"] * (len(qs) - len(a))
    return a[:len(qs)]

RUB = ('厳格な評価者。各項目の候補回答を採点。2=正解と実質同じ/1=部分/0=不正解・情報なし・無関係。'
       '出力は必ずJSON {"scores":[n,...]} 項目数と同数。\n\n項目:\n')
def judge(items):
    from openai import OpenAI
    body = "\n".join(f"{i+1}. 質問:{q}/正解:{r}/候補:{c}" for i, (q, r, c) in enumerate(items))
    def c():
        cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120)
        try: return cl.responses.create(model=JUDGE, input=[{"role": "user", "content": RUB + body}], temperature=0.0)
        except Exception: return cl.responses.create(model=JUDGE, input=[{"role": "user", "content": RUB + body}])
    sc = pj(retry(c).output_text, "scores") or []
    out = []
    for s in sc:
        try: out.append(min(max(int(s), 0), 2))
        except Exception: out.append(-1)
    if len(out) < len(items): out += [-1] * (len(items) - len(out))
    return out[:len(items)]

def load_qa_by_doc():
    by_doc = defaultdict(list)
    for cell in CELLS:
        for jf in sorted((QA_DIR / cell).glob("*.json")):
            for r in json.loads(jf.read_text(encoding="utf-8")):
                if r["task_id"] in removed: continue
                r.setdefault("cell", cell); by_doc[r["doc_id"]].append(r)
    return by_doc

def main():
    by_doc = load_qa_by_doc()
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            if line.strip(): done.add(json.loads(line)["doc_id"])
    todo = [d for d in sorted(by_doc) if d not in done]
    print(f"docs={len(by_doc)} validated_qa={sum(len(v) for v in by_doc.values())} todo={len(todo)}", flush=True)

    def work(doc_id):
        qas = by_doc[doc_id]
        try:
            ans = read_batch([q["question"] for q in qas])
            sc = judge([(q["question"], q["answer"], a) for q, a in zip(qas, ans)])
        except Exception as e:
            ans = [f"[ERR:{str(e)[:60]}]"] * len(qas); sc = [-1] * len(qas)
        with _lock, OUT.open("a", encoding="utf-8") as f:
            for q, a, s in zip(qas, ans, sc):
                f.write(json.dumps({"task_id": q["task_id"], "doc_id": doc_id, "cell": q["cell"],
                                    "level": q.get("level"), "question": q["question"],
                                    "reference_answer": q["answer"], "answer": a,
                                    "score_gpt54mini": s}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(as_completed([ex.submit(work, d) for d in todo]))

    rows = [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r["score_gpt54mini"] >= 0]
    hit2 = [r for r in ok if r["score_gpt54mini"] == 2]
    hit1 = [r for r in ok if r["score_gpt54mini"] == 1]
    print(f"\n=== no-document control summary ===")
    print(f"scored: {len(ok)}   full credit (2): {len(hit2)} ({len(hit2)/len(ok)*100:.1f}%)   "
          f"partial (1): {len(hit1)} ({len(hit1)/len(ok)*100:.1f}%)")
    print("full-credit by cell:", Counter(r["cell"] for r in hit2))
    print("full-credit by level:", Counter(r.get("level") for r in hit2))
    comp = OUT_DIR / "nodoc_compromised.json"
    comp.write_text(json.dumps(sorted(r["task_id"] for r in hit2), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"compromised task_ids -> {comp}")

if __name__ == "__main__":
    main()
