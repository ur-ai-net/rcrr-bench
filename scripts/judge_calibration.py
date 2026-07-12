#!/usr/bin/env python3
"""Judge-prompt calibration pilot (pre-Phase-B gate; see docs/benchmark_v3.md).

Compares the v2 judge rubric against the proposed v3.1 rubric on a stratified sample of
existing (question, gold, answer) triples from the rcrr_v2 dual-judge run, scored by BOTH
judge families under BOTH prompts (4 scores per item, per-item calls).

Strata:
  A. all 0-vs-2 inter-judge disagreements from rcrr_v2 (the known hard cases)
  B. questions whose gold says the info does not exist (rubric-contradiction cases)
  C. unit-mismatch cases (gold in 億円, answer in 百万円 or vice versa)
  D. random control sample of exact-agreement items

v3.1 rubric changes: explicit unit-conversion rule, "not stated" gold handling, language
neutrality, contradiction cap, and a mandatory 1-2 sentence reasoning field before the score
(targets the observed unit-arithmetic judge errors).

Output: experiments/controls/judge_calibration.jsonl + printed flip-rate summary.
Keys: GEMINI_API_KEY, OPENAI_API_KEY (.env). ~4 x ~190 calls."""
import os, json, re, random, threading
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
SRC = ROOT / "experiments" / "rcrr_v2"
OUT = ROOT / "experiments" / "controls" / "judge_calibration.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
JUDGE_MODELS = {"gpt": "gpt-5.4-mini", "gemini": "gemini-3.1-pro-preview"}
WORKERS = 8
random.seed(31)
_lock = threading.Lock()

OLD_RUBRIC = ('厳格な評価者。候補回答を採点。2=正解と実質同じ/1=部分/0=不正解・情報なし・無関係。'
              '出力は必ずJSON {"score":n} のみ。\n\n質問:%s\n正解:%s\n候補:%s')

NEW_RUBRIC = ('あなたは金融文書QAの厳格な採点者です。正解と候補回答を比較し、以下の規則で採点してください。\n\n'
    '規則:\n'
    '- 数値は単位換算して比較（1億円=100百万円=100,000千円）。換算後に一致すれば正解。'
    '表示桁までの四捨五入差は許容。単位・桁の誤りは不正解。\n'
    '- 正解が「存在しない・該当なし・記載なし」という内容の場合、候補の「情報なし」「ありません」等は正解(2)。'
    'それ以外の場合、「情報なし」は不正解(0)。\n'
    '- 言語の違い（日本語/英語）は無視し、内容で判定。\n'
    '- 2=内容が実質的に一致 / 1=核心は正しいが一部欠落・一部不正確 / 0=不正解・無関係。\n'
    '- 候補に正解と矛盾する内容が含まれる場合は、正しい部分があっても最大1。\n\n'
    '出力は必ずJSON: {"reasoning":"換算・比較を1-2文で","score":n}\n\n'
    '質問:%s\n正解:%s\n候補:%s')

NOTSTATED = re.compile(r"記載(が)?(あり)?(され)?(てい)?ませ|記載なし|情報なし|記載はあり|ありません|示されていません|存在しません|見当たりません")
OKU = re.compile(r"億円"); HYAKUMAN = re.compile(r"百万円")

def build_sample():
    rows = []
    for f in sorted(SRC.glob("*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            g, o = r.get("score_gemini35flash", -1), r.get("score_gpt54mini", -1)
            if g < 0 or o < 0: continue
            rows.append(r)
    sample, seen = [], set()
    def add(r, stratum):
        k = (r["parser"], r["task_id"])
        if k in seen: return
        seen.add(k)
        sample.append({"key": f'{r["parser"]}|{r["task_id"]}', "stratum": stratum,
                       "question": r["question"], "reference": r["reference_answer"],
                       "answer": r["answer"],
                       "v2_gemini": r["score_gemini35flash"], "v2_gpt": r["score_gpt54mini"]})
    for r in rows:                                   # A: 0-vs-2 disagreements
        if abs(r["score_gemini35flash"] - r["score_gpt54mini"]) == 2: add(r, "A_disagree")
    ns_qs = {r["task_id"] for r in rows if NOTSTATED.search(r["reference_answer"])}
    ns_rows = [r for r in rows if r["task_id"] in ns_qs]
    for r in random.sample(ns_rows, min(25, len(ns_rows))): add(r, "B_notstated")
    um = [r for r in rows if (OKU.search(r["reference_answer"]) and HYAKUMAN.search(r["answer"]))
          or (HYAKUMAN.search(r["reference_answer"]) and OKU.search(r["answer"]))]
    for r in random.sample(um, min(30, len(um))): add(r, "C_units")
    agree = [r for r in rows if r["score_gemini35flash"] == r["score_gpt54mini"]]
    for r in random.sample(agree, 100): add(r, "D_control")
    return sample

def pjs(t):
    m = re.search(r'\{.*\}', t or "", re.DOTALL)
    try: return json.loads(m.group(0)).get("score")
    except Exception: return None

def retry(fn, n=4):
    import time
    for i in range(n):
        try: return fn()
        except Exception:
            if i == n - 1: raise
            time.sleep(1.5 * (i + 1))

def call(family, prompt):
    if family == "gemini":
        from google import genai
        from google.genai import types
        import httpx
        def c():
            cl = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options={"client_args": {"timeout": httpx.Timeout(90.0)}})
            # 3.1 Pro spends thinking tokens from max_output_tokens — keep high
            return cl.models.generate_content(model=JUDGE_MODELS["gemini"], contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=16000,
                    response_mime_type="application/json"))
        return pjs(retry(c).text)
    else:
        from openai import OpenAI
        def c():
            cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=90)
            try: return cl.responses.create(model=JUDGE_MODELS["gpt"],
                     input=[{"role": "user", "content": prompt}], temperature=0.0)
            except Exception: return cl.responses.create(model=JUDGE_MODELS["gpt"],
                     input=[{"role": "user", "content": prompt}])
        return pjs(retry(c).output_text)

def main():
    sample = build_sample()
    done = {}
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            if l.strip(): r = json.loads(l); done[r["key"]] = r
    todo = [s for s in sample if s["key"] not in done]
    print(f"sample={len(sample)} by stratum {Counter(s['stratum'] for s in sample)}; todo={len(todo)}", flush=True)

    def work(s):
        for fam in ("gpt", "gemini"):
            for tag, rub in (("old", OLD_RUBRIC), ("new", NEW_RUBRIC)):
                k = f"{tag}_{fam}"
                if k in s and s[k] is not None: continue
                try: s[k] = call(fam, rub % (s["question"], s["reference"], s["answer"]))
                except Exception: s[k] = None
        with _lock, OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(as_completed([ex.submit(work, s) for s in todo]))

    rows = list(done.values())
    for l in OUT.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["key"] not in done: rows.append(r); done[r["key"]] = r
    ok = [r for r in rows if all(r.get(k) is not None for k in
          ("old_gpt", "old_gemini", "new_gpt", "new_gemini"))]
    print(f"\n=== calibration summary (n={len(ok)}) ===")
    for tag in ("old", "new"):
        agree = sum(1 for r in ok if r[f"{tag}_gpt"] == r[f"{tag}_gemini"])
        print(f"{tag} rubric inter-judge exact agreement: {agree/len(ok)*100:.1f}%")
    for fam in ("gpt", "gemini"):
        flips = [r for r in ok if r[f"old_{fam}"] != r[f"new_{fam}"]]
        print(f"{fam}: old->new flips {len(flips)}/{len(ok)} "
              f"(by stratum {Counter(r['stratum'] for r in flips)})")

if __name__ == "__main__":
    main()
