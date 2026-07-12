#!/usr/bin/env python3
"""RCRR v3 scoring grid — the revised protocol (see docs/benchmark_v3.md §3 Phase B).

Changes vs reader35_rescore.py:
  * PER-QUESTION reader calls (v2 pooled all of a page's questions into one call, allowing
    cross-question leakage; one question's numbers could help answer another).
  * PER-ITEM judging (v2 judged a page's answers as a positional JSON array, which showed
    occasional index-misalignment noise — ~51/1,194 anomalies).
  * 2 readers x 2 judges: readers Gemini 3.5 Flash / GPT-5.4-mini; judges GPT-5.4-mini /
    Gemini 3.1 Pro; every answer scored by BOTH judges. Primary reported configs are the
    cross-family pairs (Gemini reads + GPT judges; GPT reads + Gemini judges).
  * Question set = qa_v3_manifest.json, statuses "kept" and "review" ("review" rows carry
    that flag through so analysis can include/exclude them).

Output: experiments/v3/{reader}/{parser}.jsonl   (resumable per parser x doc)
Usage:  python rcrr_v3_eval.py [reader] [parser,parser,...]
        (no args = both readers, all parsers found in public_data/parser_outputs)
Keys: GEMINI_API_KEY, OPENAI_API_KEY (.env).
Scale: ~15 parsers x ~1,100 Q x 2 readers reader calls + 2 judge calls each — plan a few
hours wall-clock and tens of USD. Temperature 0, thinking off."""
import os, sys, json, re, threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
QA_DIR = ROOT / "public_data" / "qa"
PD = ROOT / "public_data" / "parser_outputs"
MANIFEST = ROOT / "public_data" / "qa_v3_manifest.json"
OUT_ROOT = ROOT / "experiments" / "v3"
READERS = {"gemini35flash": "gemini-3.5-flash", "gpt54mini": "gpt-5.4-mini"}
# gemini judge = 3.5 flash: 3.1-pro-preview proved unstable at scale (stalls, empty output)
# in both the audit and the calibration pilot. The (gemini35flash reader x gemini35flash
# judge) cell is SELF-GRADING — exclude it from primary reporting; primary configs are the
# cross-family pairs.
JUDGES = {"gpt54mini": "gpt-5.4-mini", "gemini35flash": "gemini-3.5-flash"}
CELLS = ["jp_table", "jp_chart"]
WORKERS = int(os.environ.get("RCRR_WORKERS", "56"))
               # OpenAI Tier 5 (no practical limit here); Gemini Tier 2 is the binding
               # constraint (~1000 RPM) — 56 workers ≈ 800-900 Gemini RPM, just under cap.
               # Override with RCRR_WORKERS when Gemini is returning 503 "high demand"
               # (throttle down to avoid contaminating results with hard-fails).
_lock = threading.Lock()

def retry(fn, n=4):
    import time
    for i in range(n):
        try: return fn()
        except Exception:
            if i == n - 1: raise
            time.sleep(1.5 * (i + 1))

def pj(t, key):
    # Robust JSON-value extraction. The old greedy `\{.*\}` regex silently
    # dropped correct answers when a reader (esp. Gemini 3.5 Flash) emitted
    # slightly-malformed JSON — a doubled closing brace (`{...}\n}\n`) made the
    # greedy match unparseable, and a truncated close (`{"answer":"3470"` with no
    # `"}`) matched nothing. Those became [NO ANSWER] -> scored 0, penalizing any
    # parser whose output made the reader produce such JSON (measured: 27% of
    # nebula_bbox vs 0.2% of the baseline on the Gemini reader — see PR).
    # Strategy: (1) raw_decode the first object (tolerates trailing junk / extra
    # braces); (2) fall back to the original greedy parse; (3) last resort,
    # recover a quoted string value even if its closing `"}` is missing.
    t = t or ""
    i = t.find("{")
    if i >= 0:
        try:
            d, _ = json.JSONDecoder().raw_decode(t, i)
            if isinstance(d, dict) and key in d: return d[key]
        except Exception: pass
    m = re.search(r'\{.*\}', t, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if key in d: return d[key]
        except Exception: pass
    m2 = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)', t)
    if m2:
        return m2.group(1)
    # tolerant fallback for a BARE (unquoted) value — same truncation, numeric
    # field. The judge emits {"reasoning":"...","score": 2 and stops before the
    # closing brace (finish_reason=STOP): score is an int, so the quoted-string
    # regex above can't see it. Recover the number/bool/null directly.
    m3 = re.search(r'"' + re.escape(key) + r'"\s*:\s*(-?\d+(?:\.\d+)?|true|false|null)', t)
    if m3:
        v = m3.group(1)
        return {"true": True, "false": False, "null": None}.get(v, v)
    return None

def gcli():
    from google import genai
    import httpx
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                        http_options={"client_args": {"timeout": httpx.Timeout(90.0)}})

def ocli():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=90)

READ_PROMPT = ("以下のドキュメントに基づいて質問に簡潔に日本語で回答してください。"
               "ドキュメントに情報がなければ「情報なし」と回答。"
               '出力は必ずJSON {"answer":"..."} のみ。\n\nドキュメント:\n%s\n\n質問: %s')

def read_one(reader, md, q):
    if len(md) > 80000: md = md[:80000]
    prompt = READ_PROMPT % (md, q)
    if reader.startswith("gemini"):
        from google.genai import types
        def c():
            cl = gcli()   # keep client in a local until the call returns (README: "client has been closed")
            return cl.models.generate_content(model=READERS[reader], contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=2000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json"))
        a = pj(retry(c).text, "answer")
    else:
        def c():
            cl = ocli()
            try: return cl.responses.create(model=READERS[reader],
                     input=[{"role": "user", "content": prompt}], temperature=0.0)
            except Exception: return cl.responses.create(model=READERS[reader],
                     input=[{"role": "user", "content": prompt}])
        a = pj(retry(c).output_text, "answer")
    return str(a) if a is not None else "[NO ANSWER]"

# v3.1 rubric — validated by judge_calibration.py (see docs/benchmark_v3.md §2.7):
# explicit unit conversion, not-stated gold handling, language neutrality, contradiction cap,
# mandatory brief reasoning emitted before the score.
JUDGE_PROMPT = ('あなたは金融文書QAの厳格な採点者です。正解と候補回答を比較し、以下の規則で採点してください。\n\n'
    '規則:\n'
    '- 数値は単位換算して比較（1億円=100百万円=100,000千円）。換算後に一致すれば正解。'
    '表示桁までの四捨五入差は許容。単位・桁の誤りは不正解。\n'
    '- 正解が「存在しない・該当なし・記載なし」という内容の場合、候補の「情報なし」「ありません」等は正解(2)。'
    'それ以外の場合、「情報なし」は不正解(0)。\n'
    '- 言語の違い（日本語/英語）は無視し、内容で判定。\n'
    '- 2=内容が実質的に一致 / 1=核心は正しいが一部欠落・一部不正確 / 0=不正解・無関係。\n'
    '- 候補に正解と矛盾する内容が含まれる場合は、正しい部分があっても最大1。\n\n'
    '出力は必ずJSON: {"id":"%s","reasoning":"換算・比較を1-2文で","score":n}\n\n'
    'id: %s\n質問: %s\n正解: %s\n候補: %s')

def judge_one(judge, task_id, q, ref, cand):
    prompt = JUDGE_PROMPT % (task_id, task_id, q, ref, cand)
    if judge.startswith("gemini"):
        from google.genai import types
        def c():
            cl = gcli()   # keep client in a local until the call returns
            return cl.models.generate_content(model=JUDGES[judge], contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json"))
        s = pj(retry(c).text, "score")
    else:
        def c():
            cl = ocli()
            try: return cl.responses.create(model=JUDGES[judge],
                     input=[{"role": "user", "content": prompt}], temperature=0.0)
            except Exception: return cl.responses.create(model=JUDGES[judge],
                     input=[{"role": "user", "content": prompt}])
        s = pj(retry(c).output_text, "score")
    try: return min(max(int(s), 0), 2)
    except Exception: return -1

def load_questions():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))["questions"]
    by_doc = defaultdict(list)
    for cell in CELLS:
        for jf in sorted((QA_DIR / cell).glob("*.json")):
            for r in json.loads(jf.read_text(encoding="utf-8")):
                st = man.get(r["task_id"], {}).get("status")
                if st not in ("kept", "review"): continue
                r.setdefault("cell", cell); r["v3_status"] = st
                by_doc[r["doc_id"]].append(r)
    # mapping questions (human-reviewed; see docs/benchmark_v3.md) live outside the manifest
    # round 1 = qa_mapping (m00-m02); round 2 chart-ink-only = qa_mapping2 (m03+)
    for d in ("qa_mapping", "qa_mapping2"):
        for jf in sorted((ROOT / "public_data" / d).glob("tdnet_*.json")):
            for r in json.loads(jf.read_text(encoding="utf-8-sig")):
                r["v3_status"] = "mapping"
                by_doc[r["doc_id"]].append(r)
    return by_doc

def main():
    readers = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] in READERS else list(READERS)
    parsers = (sys.argv[2].split(",") if len(sys.argv) > 2
               else sorted(d.name for d in PD.iterdir() if d.is_dir()))
    by_doc = load_questions()
    nq = sum(len(v) for v in by_doc.values())
    chunks = []
    for reader in readers:
        out_dir = OUT_ROOT / reader; out_dir.mkdir(parents=True, exist_ok=True)
        for parser in parsers:
            done = set()
            f = out_dir / f"{parser}.jsonl"
            if f.exists():
                for l in f.open(encoding="utf-8"):
                    if l.strip(): done.add(json.loads(l)["task_id"])
            for doc, qs in by_doc.items():
                todo = [q for q in qs if q["task_id"] not in done]
                if todo and (PD / parser / f"{doc}.md").exists():
                    chunks.append((reader, parser, doc, todo))
    print(f"readers={readers} parsers={len(parsers)} questions={nq} chunks_todo={len(chunks)}", flush=True)

    done_cnt = [0]
    def work(chunk):
        reader, parser, doc, qs = chunk
        md = (PD / parser / f"{doc}.md").read_text(encoding="utf-8")
        rows = []
        for q in qs:
            try: ans = read_one(reader, md, q["question"])
            except Exception as e: ans = f"[READER ERR:{str(e)[:60]}]"
            row = {"task_id": q["task_id"], "doc_id": doc, "cell": q["cell"],
                   "level": q.get("level"), "v3_status": q["v3_status"], "parser": parser,
                   "answer": ans}
            for jname in JUDGES:
                try: row[f"score_{jname}"] = judge_one(jname, q["task_id"], q["question"], q["answer"], ans)
                except Exception: row[f"score_{jname}"] = -1
            rows.append(row)
        with _lock:
            with (OUT_ROOT / reader / f"{parser}.jsonl").open("a", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done_cnt[0] += 1
            if done_cnt[0] % 50 == 0: print(f"  {done_cnt[0]}/{len(chunks)} chunks", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(as_completed([ex.submit(work, c) for c in chunks]))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
