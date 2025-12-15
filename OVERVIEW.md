# EduGuide AI – Solution Overview and Technical Details

## Solution Overview
EduGuide AI is a multilingual learning and guidance assistant purpose‑built for Nigeria and Africa. It helps learners understand complex topics with clear, culturally relevant explanations in English, Hausa, Yoruba, and Igbo. The app is fully self‑hosted: it downloads the N‑ATLaS checkpoint from Hugging Face and serves inference locally on GPU (Kaggle/Colab/local). The UX focuses on speed and clarity (Topic Presets, Response Level control, Fast Mode) so users get helpful answers quickly on modest hardware.

Key outcomes:
- Local, privacy‑friendly inference with zero external API dependency.
- Language‑aware, culturally grounded explanations.
- Fast responses on a free T4 GPU using 4‑bit quantization, prompt trimming, and caching.
- Clean UI that feels professional and approachable for real learners.

Primary users:
- Students, job seekers, and young professionals across Nigeria.
- Schools, NGOs, and training centres looking for inclusive learning tools.

Value proposition:
- Removes language and cultural barriers to understanding.
- Makes learning and career guidance more accessible and confidence‑building.

---

## Technical Details

### Inference Stack
- PyTorch + Transformers with `AutoModelForCausalLM` and `AutoTokenizer` (`trust_remote_code=True`).
- Model: `NCAIR1/N-ATLaS` (Llama‑3 8B fine‑tuned multilingual instruction model).

### Quantization & Memory
- bitsandbytes 4‑bit quantization (`nf4`, optional double‑quant) with bfloat16 compute.
- Falls back to 8‑bit or full precision if GPU bnb is not available.
- `device_map="auto"` for balanced placement.

### Latency Optimizations
- Fast Mode sets conservative generation params (low temperature, short `max_new_tokens`).
- Prompt trimming to a configurable `MAX_INPUT_TOKENS` to avoid long context.
- In‑memory response caching keyed by tail of messages + generation settings.
- Runtime accelerations: CUDA TF32 and SDPA attention kernels where supported.

### Tokenizer & Output Robustness
- Fast tokenizer preferred; safe fallback to slow (sentencepiece) with forced re‑download if needed.
- Output cleaner removes chat‑template markers (e.g., `<|end_header_id|>`), yielding clean text.

### Serving & API
- Gradio UI and FastAPI backend run in one process.
- REST endpoint: `POST /api/generate` returns `{ reply, history }`.
- Port conflict retry; Kaggle forces public link and bind to `0.0.0.0`.

### Platform Guidance
- Kaggle/Colab setup cells provided; Torch CUDA 12.1 + bitsandbytes 0.43.x recommended.
- Vision/audio packages uninstalled to avoid custom ops conflicts.

---

## Technical Architecture (Text)
- UI: Gradio Blocks (chat, controls for response level, temperature/top‑p, Fast Mode; export history).
- Controller: Small Python callbacks manage preset loading, input validation, and calling inference.
- Inference Layer: Builds system/user/assistant messages, applies chat template, tokenizes, and generates via N‑ATLaS in 4‑bit.
- Cache: Optional in‑memory cache avoids recomputation for identical recent prompts/settings.
- API: FastAPI mounted in the same app under `/api/generate`.

Mermaid diagram: see README for a renderable flowchart and ASCII fallback.

---

## Compliance & Attribution
- Self‑hosted inference (no managed API).
- Attribution in UI and docs:
  - “N‑ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.”
- Respect model license and ≤1000 active user cap (non‑enterprise) unless a commercial license is obtained.

---

## Performance Notes (Summary)
- After load, logs show: `Attn q_proj module: Linear4bit` (confirms 4‑bit).
- T4 GPU: Fast Mode with short prompts typically responds within a few seconds.
- Keep prompts brief; raise `max_new_tokens` only when needed.
