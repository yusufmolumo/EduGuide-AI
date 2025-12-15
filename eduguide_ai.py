#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EduGuide AI - N-ATLaS Powered Multilingual Learning Assistant

Features
- Self-installs required packages if missing
- Uses Hugging Face token from env var HF_TOKEN or interactive login
- Loads N-ATLaS (NCAIR1/N-ATLaS) with 4-bit quantization for speed (bitsandbytes)
- Gradio UI: stunning theme with presets, language-aware chat, response level control
- Chat history export (JSON), session reset
- REST API endpoint /api/generate via the same Gradio FastAPI app
- Prompt templating via tokenizer.apply_chat_template

How to run in Colab
1) Open this file in a Colab notebook cell using:
   %pip install -q -U pip
   !python eduguide_ai.py --colab 1
   Or copy its content to a Colab cell and run.

2) Provide your HF token safely by either:
   - Setting env variable before running: 
       import os
       os.environ['HF_TOKEN'] = 'your_hf_token_here'
   - Or interactively login when prompted.

For local CUDA GPU: python eduguide_ai.py
"""
import os
import sys
import json
import time
import argparse
import threading
from datetime import datetime
import re

# Guard against torchvision/torchao slow/breaking imports in text-only usage
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")

# --- Bootstrap installs ---
REQUIRED = [
    ("transformers", "transformers==4.46.2"),
    ("accelerate", "accelerate==0.34.2"),
    ("bitsandbytes", "bitsandbytes==0.43.1"),
    ("gradio", "gradio>=4.44.0"),
    ("fastapi", "fastapi>=0.110.0"),
    ("uvicorn", "uvicorn>=0.30.0"),
    ("sentencepiece", "sentencepiece>=0.1.99"),
]


def ensure_installed():
    # Always upgrade core deps to ensure compatible bitsandbytes/transformers in Colab
    specs = [spec for _mod, spec in REQUIRED]
    print("Ensuring packages (upgrade):", specs)
    os.system(f"{sys.executable} -m pip install -q -U " + " ".join(specs))
    time.sleep(1)


ensure_installed()

# Now safe to import heavy deps
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
from fastapi import Request

# --- Config ---
MODEL_NAME = os.environ.get("NATLAS_MODEL", "NCAIR1/N-ATLaS")
APP_TITLE = "EduGuide AI – Multilingual Learning & Guidance (Powered by Awarri)"
APP_TAGLINE = "Learn faster with Nigerian languages. Hausa · Igbo · Yoruba · English"

# Generation defaults (tuned for quality + speed on T4)
GEN_CFG = {
    "max_new_tokens": 192,
    "temperature": 0.25,
    "top_p": 0.9,
    "repetition_penalty": 1.07,
}
MAX_INPUT_TOKENS = 2048  # cap prompt length to keep latency low on T4

# --- Hugging Face token handling ---
def get_hf_token(interactive: bool = True) -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token.strip()
    if not interactive:
        return None
    try:
        print("No HF token found in env. You can paste one now (input hidden). Press Enter to skip.")
        import getpass
        token = getpass.getpass("HF token: ")
        if token:
            os.environ["HF_TOKEN"] = token.strip()
            return token.strip()
    except Exception:
        pass
    return None


# --- Model Loader ---
class NATLaSModel:
    def __init__(self, model_name: str = MODEL_NAME, hf_token: Optional[str] = None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self._load()

    def _load(self):
        print("Loading tokenizer...")
        # Try fast tokenizer first; if the tokenizers JSON is incompatible, fall back to slow
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token if self.hf_token else None,
                trust_remote_code=True,
            )
        except Exception as e_fast:
            print("Fast tokenizer load failed, falling back to slow tokenizer:", e_fast)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=self.hf_token if self.hf_token else None,
                    trust_remote_code=True,
                    use_fast=False,
                    force_download=True,
                )
            except Exception as e_slow:
                print("Slow tokenizer load failed; retrying without trust_remote_code:", e_slow)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=self.hf_token if self.hf_token else None,
                    use_fast=False,
                )
        self.tokenizer = tokenizer
        # Detect if bitsandbytes is available and CUDA is on; do not require triton.ops explicitly
        can_bnb = False
        try:
            import importlib
            importlib.import_module("bitsandbytes")
            can_bnb = torch.cuda.is_available()
        except Exception:
            can_bnb = False

        print("Loading model (this may take a few minutes)...")
        if can_bnb:
            print("Attempting 4-bit quantization (bitsandbytes)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            )
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    token=self.hf_token if self.hf_token else None,
                    trust_remote_code=True,
                )
            except Exception as e:
                print("4-bit load failed, falling back to 8-bit:", e)
                try:
                    bnb_config8 = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config8,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                        token=self.hf_token if self.hf_token else None,
                        trust_remote_code=True,
                    )
                except Exception as e2:
                    print("8-bit load failed, loading without quantization:", e2)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                        token=self.hf_token if self.hf_token else None,
                        trust_remote_code=True,
                    )
        else:
            print("bitsandbytes GPU/triton not available – loading without quantization.")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    token=self.hf_token if self.hf_token else None,
                    trust_remote_code=True,
                )
            except RuntimeError as oom:
                if "out of memory" in str(oom).lower():
                    print("OOM during load; retrying with CPU offload and lower dtype (fp16)")
                    max_mem = {"cuda": "12GiB", "cpu": "32GiB"}
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        max_memory=max_mem,
                        torch_dtype=torch.float16,
                        token=self.hf_token if self.hf_token else None,
                        trust_remote_code=True,
                    )
                else:
                    raise
        try:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass
        print("Model loaded.")
        try:
            # Extra runtime speed opts (safe no-ops if unsupported)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
                torch.backends.cudnn.allow_tf32 = True  # type: ignore
                try:
                    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)  # type: ignore
                except Exception:
                    pass
            self.model.eval()
            # Brief quantization status check
            try:
                q_proj = self.model.model.layers[0].self_attn.q_proj  # type: ignore[attr-defined]
                print("Attn q_proj module:", q_proj.__class__.__name__)
            except Exception:
                pass
        except Exception:
            pass

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        current_date = datetime.now().strftime('%d %b %Y')
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                date_string=current_date,
            )
            return text
        except TypeError:
            # Older/newer Transformers may not accept date_string
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                return text
            except Exception:
                pass
        # Fallback: simple role-prefixed conversation if template is unavailable
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role}] {content}")
        parts.append("[assistant]")
        return "\n".join(parts)

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        text = self._format_messages(messages)
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        kwargs = GEN_CFG.copy()
        kwargs.update({k: v for k, v in gen_kwargs.items() if v is not None})
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.3),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            do_sample=kwargs.get("do_sample", True),
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        # Extract only assistant's latest segment after last <|start_header_id|>assistant
        # Fallback to full decoded
        try:
            marker = "<|start_header_id|>assistant"
            last = full_text.rsplit(marker, 1)[-1]
            # remove special chat template tokens
            cleaned = last
            for tok in ("<|end_header_id|>", "<|eot_id|>", "<|start_header_id|>"):
                cleaned = cleaned.replace(tok, "")
            # collapse excessive newlines/spaces introduced by token removal
            cleaned = re.sub(r"^[\s:]+", "", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            return cleaned
        except Exception:
            return full_text


# --- App State & Utilities ---
LEVEL_PROMPTS = {
    "Beginner": "Explain simply with step-by-step, everyday Nigerian examples. Use short sentences.",
    "Intermediate": "Explain with moderate depth and practical examples relevant to Nigeria.",
    "Advanced": "Provide deep, technical, and rigorous explanation with references and caveats.",
}

SYSTEM_PROMPT_BASE = (
    "You are EduGuide AI, a helpful multilingual learning assistant powered by N-ATLaS (Awarri). "
    "Communicate clearly in the user's language (English, Hausa, Yoruba, Igbo) and keep cultural context Nigerian. "
    "Detect the user's language automatically and respond in it unless they request otherwise. "
    "Be supportive, unbiased, and avoid unsafe or harmful content. If in doubt, ask a clarifying question. "
    "When explaining, give local examples and analogies."
)

TOPIC_PRESETS = [
    {"label": "Math – Algebra Basics", "prompt": "Explain linear equations with examples relevant to Nigeria."},
    {"label": "Science – Human Biology", "prompt": "Explain how the human digestive system works in simple steps."},
    {"label": "Career – CV Writing", "prompt": "Help me write a strong CV for an entry-level role in Nigeria."},
    {"label": "Tech – Python Basics", "prompt": "Teach me Python variables and data types with examples."},
    {"label": "Business – Small Retail", "prompt": "How can a small shop in Lagos improve profits?"},
]


def build_messages(history: List[Dict[str, Any]], system_text: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_text}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        if "assistant" in turn and turn["assistant"]:
            messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def trim_messages_to_tokens(tokenizer, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    # Keep the end of the conversation; drop oldest turns until tokenized length <= max_tokens
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True, return_tensors=None)
    length = len(encoded) if isinstance(encoded, list) else len(tokenizer(encoded, add_special_tokens=False)["input_ids"])
    if length <= max_tokens:
        return messages
    kept = [messages[0]]  # system message
    ua = messages[1:]
    rev = ua[::-1]
    acc: List[Dict[str, str]] = []
    for m in rev:
        acc.append(m)
        cand = kept + acc[::-1]
        ids = tokenizer.apply_chat_template(cand, add_generation_prompt=False, tokenize=True)
        if len(ids) > max_tokens:
            acc.pop()
            break
    return kept + acc[::-1]


# --- Global model holder ---
MODEL: Optional[NATLaSModel] = None
# Simple in-memory response cache
CACHE: Dict[str, str] = {}


# --- API Logic ---
def generate_reply(user_text: str, level: str, history_json: str, temperature: float, top_p: float, max_new_tokens: int, do_sample: bool = True) -> (str, str):
    global MODEL
    if MODEL is None:
        raise RuntimeError("Model not loaded yet.")

    try:
        history = json.loads(history_json) if history_json else []
    except Exception:
        history = []
    system_text = SYSTEM_PROMPT_BASE + " " + LEVEL_PROMPTS.get(level, LEVEL_PROMPTS["Beginner"]) + "\n" \
                  + "If the user mixes languages, keep the main response in their most recent language."

    messages = build_messages(history + [{"user": user_text}], system_text)
    # Trim input context to avoid slow, long prompts
    try:
        messages = trim_messages_to_tokens(MODEL.tokenizer, messages, MAX_INPUT_TOKENS)
    except Exception:
        pass

    # Cache on tail of messages + gen settings
    cache_key = json.dumps({
        "lvl": level,
        "temp": float(temperature),
        "top_p": float(top_p),
        "max": int(max_new_tokens),
        "sample": bool(do_sample),
        "tail": messages[-6:],
    }, ensure_ascii=False)

    if cache_key in CACHE:
        reply = CACHE[cache_key]
    else:
        reply = MODEL.chat(
            messages,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
        )
        CACHE[cache_key] = reply

    # append reply to history
    history.append({"user": user_text, "assistant": reply})
    return reply, json.dumps(history, ensure_ascii=False)


def reset_history() -> str:
    return json.dumps([], ensure_ascii=False)


def export_history(history_json: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"eduguide_chat_{ts}.json"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(history_json or "[]")
        return f"Saved {fname}"
    except Exception as e:
        return f"Failed to save: {e}"


# --- Gradio UI ---
CUSTOM_CSS = """
.gradio-container {max-width: 1024px !important}
#title h1 {font-size: 2.0rem; margin-bottom: 0.2rem}
#title p {opacity: 0.85}
.footer {opacity: 0.65; font-size: 0.9rem}
:root { --radius-xl: 16px }
"""


def build_ui():
    # Theme and CSS will be passed to launch() for Gradio 6+
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"""
        <div id='title'>
          <h1>{APP_TITLE}</h1>
          <p>{APP_TAGLINE}</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=520, avatar_images=(None, None))
                user_in = gr.Textbox(placeholder="Type in English, Hausa, Yoruba, or Igbo...", label="Your message")
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    reset_btn = gr.Button("Reset Chat")
                    export_btn = gr.Button("Export History")
            with gr.Column(scale=3):
                level = gr.Radio(["Beginner", "Intermediate", "Advanced"], value="Beginner", label="Response Level")
                temperature = gr.Slider(0.0, 1.5, value=GEN_CFG["temperature"], step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=GEN_CFG["top_p"], step=0.05, label="Top-p")
                max_new_tokens = gr.Slider(64, 1024, value=GEN_CFG["max_new_tokens"], step=16, label="Max New Tokens")
                fast_mode = gr.Checkbox(value=True, label="Fast Mode (128 tokens, temp 0.2)")
                gr.Markdown("**Topic Presets**")
                preset_dd = gr.Dropdown(choices=[p["label"] for p in TOPIC_PRESETS], label="Choose a topic", value=None)
                preset_btn = gr.Button("Load Preset")
                hidden_history = gr.Textbox(visible=False, value=json.dumps([], ensure_ascii=False))
                status = gr.Markdown(visible=True)

        def on_send(user_text, level, history_json, temperature, top_p, max_new_tokens, chat_messages, fast_mode):
            if not user_text or not user_text.strip():
                return gr.update(), history_json, gr.update(value="Type something to begin.")
            # Apply fast mode overrides
            if fast_mode:
                temperature = 0.2
                max_new_tokens = 128
            # Greedy decoding in Fast Mode
            do_sample = False if fast_mode else True
            reply, new_hist = generate_reply(
                user_text.strip(),
                level,
                history_json,
                float(temperature),
                float(top_p),
                int(max_new_tokens),
                do_sample=do_sample,
            )
            # Update UI chat as role/content dicts (compatible with newer Gradio)
            chat_messages = (chat_messages or []) + [
                {"role": "user", "content": user_text.strip()},
                {"role": "assistant", "content": reply},
            ]
            return chat_messages, new_hist, gr.update(value="")

        def on_reset():
            return [], reset_history(), gr.update(value="Chat reset.")

        def on_export(history_json):
            msg = export_history(history_json)
            return gr.update(value=msg)

        def on_preset(choice):
            if not choice:
                return gr.update()
            for p in TOPIC_PRESETS:
                if p["label"] == choice:
                    return gr.update(value=p["prompt"])
            return gr.update()

        send_btn.click(
            on_send,
            inputs=[user_in, level, hidden_history, temperature, top_p, max_new_tokens, chatbot, fast_mode],
            outputs=[chatbot, hidden_history, user_in],
        )
        user_in.submit(
            on_send,
            inputs=[user_in, level, hidden_history, temperature, top_p, max_new_tokens, chatbot, fast_mode],
            outputs=[chatbot, hidden_history, user_in],
        )
        reset_btn.click(on_reset, inputs=None, outputs=[chatbot, hidden_history, status])
        export_btn.click(on_export, inputs=[hidden_history], outputs=[status])
        preset_btn.click(on_preset, inputs=[preset_dd], outputs=[user_in])

        # Add footer
        gr.Markdown(
            """
            <div class="footer">
            N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.\
            EduGuide AI – Built for the Awarri Developer Challenge.\
            </div>
            """
        )

        # Expose REST API on same app
        app = demo.app  # FastAPI instance

        @app.post("/api/generate")
        async def api_generate(request: Request):
            body = await request.json()
            prompt = body.get("prompt", "")
            level_v = body.get("level", "Beginner")
            history = body.get("history", [])
            temperature_v = float(body.get("temperature", GEN_CFG["temperature"]))
            top_p_v = float(body.get("top_p", GEN_CFG["top_p"]))
            max_tokens_v = int(body.get("max_new_tokens", GEN_CFG["max_new_tokens"]))

            reply, new_hist_json = generate_reply(prompt, level_v, json.dumps(history, ensure_ascii=False), temperature_v, top_p_v, max_tokens_v)
            return {"reply": reply, "history": json.loads(new_hist_json)}

    return demo


# --- Main ---

def main(colab: bool = False, share: bool = True, server_port: int = 7860):
    global MODEL
    token = get_hf_token(interactive=colab or sys.stdin.isatty())

    print("Loading N-ATLaS model... This can take 2-5 minutes on first run.")
    MODEL = NATLaSModel(MODEL_NAME, hf_token=token)

    demo = build_ui()
    # Launch Gradio. In Colab, enable share for a public link.
    try:
        demo.launch(
            share=share,
            server_port=server_port,
            theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue),
            css=CUSTOM_CSS,
        )
    except OSError as e:
        if "Cannot find empty port" in str(e):
            print(f"Port {server_port} busy; retrying on a random free port...")
            demo.launch(
                share=share,
                server_port=0,
                theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue),
                css=CUSTOM_CSS,
            )
        else:
            raise


if __name__ == "__main__":
    # Auto-detect Colab to set sensible defaults and ignore Jupyter's extra argv like '-f ...json'
    in_colab = False
    try:
        import google.colab  # type: ignore
        in_colab = True
    except Exception:
        in_colab = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--colab", type=int, default=1 if in_colab else 0)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-share", action="store_true")
    # Use parse_known_args to ignore unknown args injected by IPython/Colab
    args, _unknown = parser.parse_known_args()

    try:
        main(colab=bool(args.colab), share=not args.no_share, server_port=args.port)
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
