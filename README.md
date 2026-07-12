# RCRR Benchmark — how much *meaning* survives document conversion?

Companion repository for the Ur AI technical report
**“Which document conversion systems preserve the meaning of Japanese IR materials?”**
(https://ur-ai.net/blog/rcrr-technical-report / Japanese: https://ur-ai.net/ja/blog/rcrr-technical-report).

RCRR (reading-comprehension recovery) measures document-conversion quality by outcome:
after a page is converted to Markdown, can an AI reader still answer validated questions
that a human answers from the original page? 14 systems in the primary comparison
(results for every scored configuration are included here), 99 real Japanese IR pages,
1,410 validated questions.

Two of the fourteen systems are Ur AI's own product, Nebula, in its two deployment
configurations: **Nebula Frontier** (frontier orchestration: a frontier VLM API,
currently the GPT-5.6 generation, inside Ur AI's conversion pipeline with whole-page
conversion and PDF text-layer fusion; live at
https://nebula.ur-ai.net) and **Nebula Sovereign** (fully self-hosted, fine-tuned: a
fine-tuned Qwen3-VL-32B served on customer-controlled GPUs). API documentation:
https://ocr.ur-ai.net/docs.

## Input conditions

Frontier VLM APIs are evaluated on **page-image inputs** (300-dpi renders): real-world
workloads involve PDFs over 100 pages and multiple files at once, so per-page image
conversion is the practical deployment mode. These systems carry the `_img` suffix in
the data files. For reference, the same VLMs measured through **vendor PDF-file
ingestion** carry the `_ir` suffix (e.g. GPT-5.6 Sol scores 94.6 via PDF ingestion vs
94.0 on images); both conditions are fully included so the effect of the input path is
itself verifiable. OCR products (Azure DI, Mistral OCR, ...) use their native PDF
ingestion — that is their deployment mode.

## Headline results: the primary comparison (details and confidence intervals in the report, §4)

| # | System | Overall | Text & tables | Charts |
|---|--------|--------:|--------------:|-------:|
| 1 | Fable 5 (VLM API, image input) §| 94.6 | 94.0 | 98.1 |
| 2 | **Nebula Frontier (Ur AI orchestration)** ¹ | **94.4** | **94.3** | 94.7 |
| 3 | GPT-5.6 Sol (VLM API, image input) | 94.0 | 93.4 | 97.1 |
| 4 | Gemini 3.1 Pro (VLM API, image input) § | 93.7 | 93.1 | 97.1 |
| 5 | Azure Document Intelligence | 88.2 | 91.5 | 69.1 |
| 6 | **Nebula Sovereign (Ur AI, self-hosted, fine-tuned)** ¹ | 87.3 | 89.0 | 77.3 |
| 7 | Reducto | 85.9 | 88.5 | 70.8 |
| 8 | Qwen3-VL-32B (base) | 81.1 | 80.9 | 82.4 |
| 9 | olmOCR | 78.8 | 84.3 | 46.4 |
| 10 | Mistral OCR | 73.6 | 82.4 | 22.2 |
| 11 | Marker | 69.7 | 78.6 | 17.6 |
| 12 | Docling | 65.9 | 74.2 | 17.1 |
| 13 | LlamaParse | 65.8 | 71.3 | 34.1 |
| 14 | AWS Textract | 20.2 | 20.7 | 17.4 |

¹ Data files (see mapping below). § Gold-provenance family: the benchmark's gold answers
were authored with Gemini 3.1 Pro (text & tables) and Fable 5 (charts) assistance, all
human-reviewed. Scores of those families are reported with this provenance disclosed
(report §3); Fable 5's chart score is the chart-gold author's own and should be read as
a reference ceiling.

Statistical framing (report §4, paired per-page tests): Nebula Frontier vs Fable 5 is
-0.2 [-1.2, +0.8], vs GPT-5.6 Sol +0.4 [-0.8, +1.6], vs Gemini 3.1 Pro +0.7 [-0.6, +2.1]
— all statistical ties; vs Azure DI +6.2 [+4.0, +8.5], significant. Nebula Sovereign vs
Azure DI overall is -1.0 [-3.8, +1.8], a statistical tie. Fine-tuning effect (Sovereign
vs base Qwen3-VL-32B): +6.2 [+2.1, +10.3] overall, +8.1 on text & tables.

## Data-file mapping (`experiments/v3/{reader}/{system}.jsonl`)

| File key | System |
|---|---|
| `nebula_staging_tuned_v6` | **Nebula Frontier** (published configuration) |
| `nebula` | **Nebula Sovereign** |
| `qwen32b_base` | Qwen3-VL-32B base (fine-tuning baseline) |
| `fable5_img`, `gpt56sol_img`, `gemini31pro_img`, `gemini35flash_img`, `gpt55_img`, `gpt54mini_img`, `gpt52_img` | VLM APIs, image-input condition |
| `fable5_ir`, `gpt56sol_ir`, `gemini31pro_ir`, `gemini35flash_ir`, `gpt55_ir`, `gpt54mini_ir`, `gpt52_ir` | the same VLMs, vendor PDF-ingestion condition (reference) |
| `azure_di`, `mistral_ocr`, `reducto`, `llamaparse`, `textract`, `docling`, `marker`, `olmocr` | OCR products / open-source converters |
| `nebula_cloud` | previous-generation Nebula Frontier (GPT-5.2 backend, whole-page) |
| `nebula_staging_sol_wp` | Frontier pipeline ablation: untuned prompt, whole-page |
| `nebula_staging_tuned` | Frontier pipeline ablation: intermediate prompt (chart rules only) |
| `nebula_staging_sol` | Frontier pipeline ablation: region-crop (bbox) pipeline. Internal ablation only: on 12/100 pages a handwriting-detection path re-OCRed one region via a Gemini model, so this run does not satisfy the backend-independence constraint the published configurations meet. Included for completeness. |

Published configuration: GPT-5.4-mini reader × Gemini 3.5 Flash judge (chosen so no model
family both authors gold answers and reads documents; see report §3). Per-question outputs
for both readers and both judges are included in full for transparency, so the complete
reader × judge grid is derivable by third parties.

## What's here

```
public_data/qa*/            questions + gold answers (validated; human-reviewed mapping sets)
public_data/qa_v3_manifest.json   which questions are kept/excluded and why
public_data/tdnet_doc_manifest.json  the 99 source pages (securities code + page)
experiments/v3/{reader}/{system}.jsonl   per-question answers + both judges' scores
experiments/controls/       fairness audits (2 model families), contamination control,
                            judge-calibration pilot
scripts/                    the full evaluation harness
```

## Verify the published numbers (offline, no API keys)

```bash
pip install -r requirements.txt
python scripts/analyze_v3.py          # recompute all tables + bootstrap CIs from the
                                      # included per-question results
```

## Re-run the evaluation (your own API keys)

`scripts/rcrr_v3_eval.py` re-scores any system from its Markdown conversions
(`OPENAI_API_KEY`, `GEMINI_API_KEY` in `.env`). To evaluate a NEW conversion system,
produce `parser_outputs/<name>/<doc_id>.md` for the 99 pages and run the harness; it is
auto-discovered. `scripts/reocr_irprompt_vlm_images.py` reproduces the image-input VLM
conversions (`ANTHROPIC_API_KEY` additionally required for Fable 5);
`scripts/reocr_irprompt_vlm.py` reproduces the PDF-ingestion condition.

## Source documents

The 99 pages are single pages from publicly available corporate disclosure filings
published via TDnet (Tokyo Stock Exchange). We do not redistribute the source PDFs;
`public_data/tdnet_doc_manifest.json` identifies each page (securities code + page
number), and the filings are retrievable from public disclosure archives. Questions,
gold answers, and model answers contain short factual excerpts from these public filings,
included for evaluation and research purposes. All documents remain the property of their
issuers. Removal requests: contact@ur-ai.net.

## Acknowledgements

Nebula Sovereign is fine-tuned from Qwen3-VL-32B, an open-weights model by the Qwen
team (Alibaba Cloud); the base model is also evaluated here in unmodified form. We
thank the Qwen team and the maintainers of every system evaluated. All model and
product names are trademarks of their respective owners and appear solely for factual
benchmarking.

## License & citation

Code and data are released under the Apache License 2.0 (see `LICENSE`). Source
documents remain the property of their issuers (see Source documents above).

> Ur AI (2026). *RCRR: a meaning-survival benchmark for Japanese document AI.*
> https://ur-ai.net/blog/rcrr-technical-report
