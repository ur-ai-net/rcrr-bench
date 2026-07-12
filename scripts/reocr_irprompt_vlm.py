#!/usr/bin/env python3
"""Re-OCR the JP single-page PDFs with the detailed Japanese-IR prompt, using 4 VLMs.
Output -> public_data/parser_outputs/{name}/{doc_id}.md. Resumable.
Keys from env (.env): OPENAI_API_KEY, GEMINI_API_KEY."""
import os, sys, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
SRC = ROOT / "public_data" / "single_pages"
OUT = ROOT / "public_data" / "parser_outputs"
CELLS = ["jp_table", "jp_chart"]

IR_PROMPT = """あなたは日本のIR資料・財務諸表の専門OCRパーサーです。与えられた画像を解析し、以下のルールに従って全ての情報を抽出してください。

最重要ルール
- 画像内の全てのテキスト・数値を漏れなく転記すること
- ページを上部・下部に分けて、各セクションの内容を記述すること
- 会社名・資料タイトルは必ず最初に記載すること
- 表の全ての行・列を省略せず完全にリストアップすること
- 説明文・対話文は一字一句正確に転記すること（話者名も含める）
- 数値はカンマ・単位を含めて正確に記録すること
- グラフ・チャートの系列データを全て列挙すること
- 矢印の色・向き、グラフのデザインなど視覚的要素も説明すること
- 注釈（*1, *2等）の内容を必ず含めること

出力はMarkdown形式。説明・前置き・コードフェンスは禁止。全て日本語で出力。"""

MODELS = {
    "gpt55_ir":         ("openai", "gpt-5.5"),
    "gpt54mini_ir":     ("openai", "gpt-5.4-mini"),
    "gemini35flash_ir": ("gemini", "gemini-3.5-flash"),
    "gemini31pro_ir":   ("gemini", "gemini-3.1-pro-preview"),
    # v4 additions (2026-07-12)
    "gpt56sol_ir":      ("openai", "gpt-5.6-sol"),
    "fable5_ir":        ("anthropic", "claude-fable-5"),   # ANTHROPIC_API_KEY in .env
}

def call_openai(pdf_path, model):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=240)
    up = client.files.create(file=open(pdf_path, "rb"), purpose="user_data")
    msg = [{"role": "user", "content": [{"type": "input_file", "file_id": up.id},
                                        {"type": "input_text", "text": IR_PROMPT}]}]
    try:
        r = client.responses.create(model=model, input=msg, temperature=0.1)
    except Exception:
        r = client.responses.create(model=model, input=msg)
    return r.output_text

def call_gemini(pdf_path, model):
    from google import genai
    from google.genai import types
    import httpx
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options={"client_args": {"timeout": httpx.Timeout(240.0)}})
    up = client.files.upload(file=str(pdf_path))
    try:
        r = client.models.generate_content(model=model, contents=[IR_PROMPT, up],
              config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=30000))
    except Exception:
        r = client.models.generate_content(model=model, contents=[IR_PROMPT, up])
    return r.text

def call_anthropic(pdf_path, model):
    import base64
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300)
    b64 = base64.standard_b64encode(open(pdf_path, "rb").read()).decode()
    # NOTE: Fable 5 rejects the temperature param ("deprecated for this model") —
    # other parsers here use temperature=0.1; Fable 5 runs at its API default.
    r = client.messages.create(model=model, max_tokens=32000,
        messages=[{"role": "user", "content": [
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
            {"type": "text", "text": IR_PROMPT}]}])
    return "".join(b.text for b in r.content if b.type == "text")

def do(name, backend, model, pdf):
    doc_id = os.path.splitext(os.path.basename(pdf))[0]
    out = OUT / name / (doc_id + ".md")
    if out.exists() and out.stat().st_size > 0:
        return (name, doc_id, "skip")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        md = (call_openai(pdf, model) if backend == "openai"
              else call_anthropic(pdf, model) if backend == "anthropic"
              else call_gemini(pdf, model))
        if md and md.strip():
            out.write_text(md, encoding="utf-8"); return (name, doc_id, "ok")
        return (name, doc_id, "EMPTY")
    except Exception as e:
        return (name, doc_id, f"ERR:{str(e)[:120]}")

def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    pdfs = []
    for c in CELLS: pdfs += sorted(glob.glob(str(SRC / c / "*.pdf")))
    tasks = [(n, b, m, p) for n, (b, m) in MODELS.items() if n in only for p in pdfs]
    print(f"{len(only)} models x {len(pdfs)} pdfs = {len(tasks)} tasks", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, f in enumerate(as_completed([ex.submit(do, *t) for t in tasks]), 1):
            name, doc_id, status = f.result()
            if status.startswith("ERR") or status == "EMPTY" or i % 20 == 0:
                print(f"[{i}/{len(tasks)}] {name:18} {doc_id:26} {status}", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
