#!/usr/bin/env python3
"""Image-input condition: re-OCR the JP single-page PDFs with the IR prompt, but feed
each model a rendered PAGE IMAGE (PyMuPDF, DPI 300, max edge 3200 px, PNG, detail=high)
instead of the PDF file. This mirrors practical deployment (long/multiple PDFs are
parsed page-by-page as images) and removes vendor PDF ingestion (server-side rendering
+ embedded-text extraction) from the comparison. Output ->
public_data/parser_outputs/{name}_img/{doc_id}.md. Resumable.
Keys from .env: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY."""
import os, sys, glob, base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import fitz

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
load_dotenv(ROOT / ".env")
SRC = ROOT / "public_data" / "single_pages"
OUT = ROOT / "public_data" / "parser_outputs"
CELLS = ["jp_table", "jp_chart"]
DPI = 300; MAX_EDGE = 3200  # matches the Nebula worker's raster settings

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
    "gpt56sol_img":      ("openai", "gpt-5.6-sol"),
    "gpt55_img":         ("openai", "gpt-5.5"),
    "gpt54mini_img":     ("openai", "gpt-5.4-mini"),
    "gpt52_img":         ("openai", "gpt-5.2"),
    "gemini35flash_img": ("gemini", "gemini-3.5-flash"),
    "gemini31pro_img":   ("gemini", "gemini-3.1-pro-preview"),
    "fable5_img":        ("anthropic", "claude-fable-5"),
}

def render(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = DPI / 72.0
    w, h = page.rect.width * zoom, page.rect.height * zoom
    if max(w, h) > MAX_EDGE:
        zoom *= MAX_EDGE / max(w, h)
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")

def call_openai(png, model):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=240)
    b64 = base64.standard_b64encode(png).decode()
    msg = [{"role": "user", "content": [
        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": "high"},
        {"type": "input_text", "text": IR_PROMPT}]}]
    try:
        r = client.responses.create(model=model, input=msg, temperature=0.1)
    except Exception:
        r = client.responses.create(model=model, input=msg)
    return r.output_text

def call_gemini(png, model):
    from google import genai
    from google.genai import types
    import httpx
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options={"client_args": {"timeout": httpx.Timeout(240.0)}})
    part = types.Part.from_bytes(data=png, mime_type="image/png")
    try:
        r = client.models.generate_content(model=model, contents=[IR_PROMPT, part],
              config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=30000))
    except Exception:
        r = client.models.generate_content(model=model, contents=[IR_PROMPT, part])
    return r.text

def call_anthropic(png, model):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300)
    b64 = base64.standard_b64encode(png).decode()
    # Fable 5 rejects the temperature param (deprecated for this model)
    r = client.messages.create(model=model, max_tokens=32000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": IR_PROMPT}]}])
    return "".join(b.text for b in r.content if b.type == "text")

def do(name, backend, model, pdf):
    doc_id = os.path.splitext(os.path.basename(pdf))[0]
    out = OUT / name / (doc_id + ".md")
    if out.exists() and out.stat().st_size > 0:
        return (name, doc_id, "skip")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        png = render(pdf)
        md = (call_openai(png, model) if backend == "openai"
              else call_anthropic(png, model) if backend == "anthropic"
              else call_gemini(png, model))
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
    print(f"{len(only)} models x {len(pdfs)} pages = {len(tasks)} tasks (raster {DPI}dpi/{MAX_EDGE}px)", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, f in enumerate(as_completed([ex.submit(do, *t) for t in tasks]), 1):
            name, doc_id, status = f.result()
            if status.startswith("ERR") or status == "EMPTY" or i % 25 == 0:
                print(f"[{i}/{len(tasks)}] {name:18} {doc_id:26} {status}", flush=True)
    print("done", flush=True)
    for n in only:
        miss = [Path(p).stem for p in pdfs if not (OUT / n / (Path(p).stem + ".md")).exists()]
        if miss: print(f"MISSING {n}: {len(miss)}", flush=True)

if __name__ == "__main__":
    main()
