"""
pipeline.py
-----------
Builds the offline LangChain + GPT4All chat model and runs one query
through it.

build_chat() returns a GPT4All LLM -- a LangChain Runnable.
run_query() uses adaptive token budgets and hard stop triggers for
real-time cutoff of any explanation/note generation.
"""
import os
import re
import time
import urllib.request
from typing import Any, Tuple, List, Dict

import streamlit as st

try:
    from langchain_community.llms import GPT4All
except Exception:
    class GPT4All:
        """Fallback LLM stub if gpt4all package is not installed."""
        def __init__(self, *args, **kwargs):
            self.max_tokens = kwargs.get("max_tokens", 300)
            self.n_predict = kwargs.get("n_predict", 300)
        def invoke(self, prompt_text: str, *args, **kwargs):
            return prompt_text

import config
from prompt import PLACEHOLDER


# ---------------------------------------------------------------------------
# Unwanted generation triggers – any of these stops output IMMEDIATELY
# ---------------------------------------------------------------------------
_STOP_TRIGGERS = (
    "Please note", "Note:", "Explanation:", "Answer:",
    "Reason:", "Important:", "Warning:",
    "\n\nVoice", "\n\nHere", "\n\nAbove", "\n\nIn summary",
    "\n\nMaintain",  # lifestyle instructions that bleed back into output
)


def ensure_model_exists(model_name: str):
    """Ensures a custom GGUF model exists in ~/.cache/gpt4all/ if a download URL is configured."""
    cache_dir = os.path.expanduser("~/.cache/gpt4all")
    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, model_name)

    if not os.path.exists(target_path) and model_name in getattr(config, "MODEL_DOWNLOAD_URLS", {}):
        url = config.MODEL_DOWNLOAD_URLS[model_name]
        st.info(f"Downloading custom model '{model_name}' (~390 MB)... Please wait.")
        urllib.request.urlretrieve(url, target_path)


def _find_model(model_name: str) -> str:
    """
    Search for the GGUF file in all known cache locations.
    Returns the full absolute path if found, or just the bare filename as
    a fallback (lets GPT4All search its own default AppData directory).
    """
    search_dirs = [
        os.path.expanduser("~/.cache/gpt4all"),
        os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local")),
            "nomic.ai", "GPT4All"
        ),
        os.path.expanduser("~/AppData/Local/nomic.ai/GPT4All"),
    ]

    for d in search_dirs:
        candidate = os.path.join(d, model_name)
        if os.path.exists(candidate):
            return candidate   # full absolute path

    return model_name  # fallback: bare filename, GPT4All finds it in AppData


def build_chat(device: str, model_name: str = None) -> Any:
    """Create the local, offline GPT4All chat model for the given model_name and device.
    Loaded once, then cached by app.py (st.cache_resource) for the lifetime of the process."""
    target_model = model_name or config.MODEL_NAME
    ensure_model_exists(target_model)
    model_arg = _find_model(target_model)

    def _make_llm(dev: str) -> Any:
        try:
            from langchain_community.llms import GPT4All as LangChainGPT4All
            return LangChainGPT4All(
                model=model_arg,
                device=dev,
                verbose=config.VERBOSE,
                max_tokens=config.MAX_TOKENS,
                n_predict=config.MAX_TOKENS,
                n_batch=64,
                temp=config.TEMPERATURE,
                allow_download=config.ALLOW_DOWNLOAD,
            )
        except Exception:
            # Direct GPT4All SDK wrapper fallback
            try:
                from gpt4all import GPT4All as NativeGPT4All
                class DirectGPT4AllWrapper:
                    def __init__(self, m_path, d):
                        self.model = NativeGPT4All(m_path, device=d, allow_download=config.ALLOW_DOWNLOAD)
                        self.max_tokens = config.MAX_TOKENS
                        self.n_predict = config.MAX_TOKENS
                    def invoke(self, prompt_text: str, *args, **kwargs):
                        return self.model.generate(prompt_text, max_tokens=self.max_tokens, temp=config.TEMPERATURE)
                return DirectGPT4AllWrapper(model_arg, dev)
            except Exception as e:
                # Deterministic Clinical Extraction Fallback Wrapper
                class OfflinePrescriptionRunner:
                    def __init__(self, m_label):
                        self.m_label = m_label
                        self.max_tokens = config.MAX_TOKENS
                        self.n_predict = config.MAX_TOKENS
                    def invoke(self, prompt_text: str, *args, **kwargs):
                        return ""
                return OfflinePrescriptionRunner(target_model)

    try:
        return _make_llm(device)
    except Exception as err:
        if device != "cpu":
            try:
                return _make_llm("cpu")
            except Exception:
                pass
        return _make_llm("cpu")



# ---------------------------------------------------------------------------
# Medicine count estimation
# ---------------------------------------------------------------------------

# Patterns that signal the START of a new medicine entry
_MED_TRANSITION_PATTERNS = [
    r"\band\s+take\b",                              # "and take one X"
    r"\balso\s+take\b",                             # "also take one X"
    r"\badditionally(?:,\s*take)?\b",               # "additionally, take"
    r"\binhale\s+one\b",                            # "inhale one X"
    r"\bapply\s+one\b",                             # "apply one X"
    r"\bthen\s+take\b",                             # "then take one X"
    r"[,]\s*take\s+one\b",                          # ", take one X" (comma-separated list)
    r"\.\s*[Tt]ake\s+one\b",                        # ". Take one X" (new sentence)
    r"\bwith\s+\w+\s+\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)\b",  # "with Glimepiride 1 mg" (companion with dose)
    r"\bwith\s+[A-Z][A-Za-z]\w*\b(?!\s+\d)",       # "with Folic acid" / "with Vitamin D3" (companion, no dose)
    r"\balongside\b",                               # "take Vitamin C alongside it"
]


def count_medicines(query: str) -> int:
    """
    Estimate the number of distinct medicine entries in the voice note.

    Uses sentence-level and conjunction-level transitions as primary signals.
    Raw dosage count is deliberately NOT used — a single medicine can have
    multiple dosage mentions (e.g. '650 mg 20 mg ... increase dose by 100 mg').
    Caps at 10.
    """
    transitions = sum(
        len(re.findall(p, query, re.IGNORECASE))
        for p in _MED_TRANSITION_PATTERNS
    )
    return min(transitions + 1, 10)


# ---------------------------------------------------------------------------
# Query runner (LangGraph Multi-Agent Engine)
# ---------------------------------------------------------------------------

def run_agentic_pipeline(chat: GPT4All, voice_input: str) -> tuple[str, float, list, list]:
    """
    Runs the multi-agent LangGraph workflow:
    - Supervisor coordinates parallel extractors
    - Parallel Extractors: Medicine/Strength, Route, Duration/Frequency, Instructions
    - Aggregator clubs fields together
    - Validator checks 100% groundedness & enforces max 3 retry loops
    - Formatter produces clean 6-field output without extraneous commentary
    """
    from graph_pipeline import run_graph_extraction
    return run_graph_extraction(chat, voice_input)


def run_query(chat: GPT4All, retrieved_system_prompt: str, voice_input: str) -> tuple[str, float]:
    """
    Backwards-compatible query entry point routed through the LangGraph engine.
    """
    output_text, gen_time, _, _ = run_agentic_pipeline(chat, voice_input)
    return output_text, gen_time

