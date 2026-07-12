#!/usr/bin/env python3
"""Base (un-fine-tuned) Qwen3-VL-32B OCR -> Markdown for the 99 benchmark pages.
This is the "before" leg of the fine-tuning before/after comparison (docs/benchmark_v3.md
§4 beat 4): same pages, same eval harness as every other parser; only the weights differ
from the fine-tuned Nebula model.

Serving: team-provided vLLM endpoint (OpenAI-compatible), base Qwen/Qwen3-VL-32B-Instruct
with the structured-OCR system prompt. Reference config per their README: temperature 0,
max_tokens 2048, PDF rendered at 150 dpi. finish_reason is recorded so truncation-at-cap
can be quantified (a fairness check — dense pages may exceed 2048 tokens).

Output -> public_data/parser_outputs/qwen32b_base/{doc_id}.md  (resumable)
        + experiments/controls/qwen32b_base_meta.jsonl (finish_reason per doc)
Keys from .env: QWEN_BASE_URL, QWEN_BASE_API_KEY.
Usage: python qwen_base_ocr.py"""
import os, json, glob, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
QA_DIR = ROOT / "public_data" / "qa"
SP = ROOT / "public_data" / "single_pages"
OUT = ROOT / "public_data" / "parser_outputs" / "qwen32b_base"; OUT.mkdir(parents=True, exist_ok=True)
META = ROOT / "experiments" / "controls" / "qwen32b_base_meta.jsonl"
MODEL = "Qwen/Qwen3-VL-32B-Instruct"
CELLS = ["jp_table", "jp_chart"]
WORKERS = 4          # single-GPU vLLM server — be polite
# default 2048 per team README ("matches their reference eval") — but 9/99 dense pages hit
# the cap (finish_reason=length), and on one page the fine-tuned model needed ~25k chars,
# so capped output would mismeasure the base model. Those 9 were regenerated with
# QWEN_MAX_TOKENS=8192 (see experiments/controls/qwen32b_base_meta.jsonl for provenance).
MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "2048"))
DPI = 150
_lock = threading.Lock()

SYSTEM_PROMPT = """あなたは日本のIR資料・財務諸表の構造化OCRパーサーです。与えられた画像を読み取り、以下のスキーマに従ってMarkdownで出力してください。出力はMarkdownブロックのみ。説明・前置き・コードフェンス（```）は禁止。日本語で出力。

## テーブルの場合

[table] {表の名称。なければ空欄}
- unit: {単位（百万円、億円、%等）。判別不能なら「単位不明」}
- sign_convention: {記号ルール（▲ = 負値、△ = 負値等）。該当なしなら省略}
- header_hierarchy: {多段ヘッダの階層構造。例: "列1 > 列2" の形式。平坦なら省略}
| 列1 | 列2 | ... |
|---|---|---|
| 値1 | 値2 | ... |

## チャートの場合

[chart_{subtype}] {名称。判別不能なら「名称不明」}
- x_axis: {軸ラベル} ({値一覧または範囲})
- y_axis: {軸ラベル} ({単位。判別不能なら「単位不明」})
- series:
  - {系列名}: {x値: y値, x値: y値, ...}
- annotations:
  - {at: 位置, value: 数値またはnull, note: "注記"}
- caption: {キャプション。なければ空欄}
- estimation_note: {推定値を含む場合の注記。なければ空欄}"""

def retry(fn, n=4):
    import time
    for i in range(n):
        try: return fn()
        except Exception:
            if i == n - 1: raise
            time.sleep(3 * (i + 1))

def docs_with_qa():
    ids = []
    for cell in CELLS:
        for jf in sorted((QA_DIR / cell).glob("*.json")):
            ids.append((cell, jf.stem))
    return ids

def ocr_one(cell, doc_id):
    out = OUT / f"{doc_id}.md"
    if out.exists() and out.stat().st_size > 0: return doc_id, "skip", None
    pdf = SP / cell / f"{doc_id}.pdf"
    if not pdf.exists(): return doc_id, "no_pdf", None
    import base64, fitz
    from openai import OpenAI
    png = fitz.open(str(pdf))[0].get_pixmap(dpi=DPI).tobytes("png")
    b64 = base64.b64encode(png).decode()
    def _call():
        cli = OpenAI(base_url=os.environ["QWEN_BASE_URL"],
                     api_key=os.environ["QWEN_BASE_API_KEY"], timeout=900)
        return cli.chat.completions.create(model=MODEL, temperature=0, max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                          {"type": "text", "text": "この画像を構造化Markdownに変換してください。"}]}])
    r = retry(_call)
    md = r.choices[0].message.content or ""
    fr = r.choices[0].finish_reason
    if md.strip(): out.write_text(md, encoding="utf-8")
    with _lock, META.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"doc_id": doc_id, "finish_reason": fr, "chars": len(md)},
                           ensure_ascii=False) + "\n")
    return doc_id, "ok" if md.strip() else "EMPTY", fr

def main():
    ids = docs_with_qa()
    print(f"docs={len(ids)} -> {OUT}", flush=True)
    trunc = 0; done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(ocr_one, c, d) for c, d in ids]):
            try: doc_id, status, fr = fut.result()
            except Exception as e:
                print(f"  ERR {str(e)[:100]}", flush=True); continue
            done += 1
            if fr == "length": trunc += 1
            if status != "skip" and (done % 10 == 0 or status != "ok"):
                print(f"  [{done}/{len(ids)}] {doc_id} {status} finish={fr}", flush=True)
    print(f"DONE. truncated-at-cap (finish_reason=length): {trunc}", flush=True)

if __name__ == "__main__":
    main()
