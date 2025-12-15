# EduGuide AI – One‑Pager (Awarri Developer Challenge 2025)

## Problem
Millions of Nigerian learners struggle with comprehension because most tools aren’t built for our languages or cultural context. English‑only content, abstract examples, and complex explanations create barriers for students, job seekers, and young professionals who need practical guidance they can relate to.

## Solution
EduGuide AI is a multilingual learning and guidance assistant powered by N‑ATLaS (Llama‑3 8B fine‑tuned for Nigerian languages). It explains topics clearly in English, Hausa, Yoruba, and Igbo, with local context and step‑by‑step guidance. The product runs entirely on user infrastructure (Kaggle/Colab/local GPU) with 4‑bit quantized inference for fast, affordable performance.

## Core Use Cases
- Learning support: math, science, programming, exam prep
- Career guidance: CV improvement, interview prep, skill planning
- Language‑aware tutoring: explain in a user’s language with cultural relevance
- NGO/School deployments: localized learning assistance at low cost

## Impact
- Accessibility: removes language and context barriers; more inclusive learning
- Speed: fast responses on a T4 GPU via 4‑bit quantization and Fast Mode
- Confidence: step‑by‑step, relatable explanations improve comprehension and retention

## Market & Users
- Primary: secondary/tertiary students, job seekers, and self‑learners
- Buyers/Partners: schools, NGOs, training centers, youth organizations
- Expansion: pan‑African multilingual deployments and skill‑building programs

## Technical Approach
- Self‑hosted inference with Transformers/PyTorch, `AutoModelForCausalLM`
- bitsandbytes 4‑bit quantization (nf4 + optional double‑quant) for speed and memory
- Prompt trimming, response caching, CUDA TF32/SDPA optimizations
- Robust tokenizer fallbacks; cleaned chat‑template tokens for polished UI output
- Gradio front‑end + FastAPI in one process; `/api/generate` endpoint

## Roadmap
- Voice I/O (ASR/TTS) and mobile‑first UX
- Structured learning paths and progress analytics
- Admin CMS and content library
- Improved Yoruba performance and additional languages
- Docker images and CI/CD for managed deployments

## Compliance & Attribution
- Self‑hosted inference; no managed API usage
- License: ≤1000 active users without a commercial license (see model terms)
- Attribution: “N‑ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.”
