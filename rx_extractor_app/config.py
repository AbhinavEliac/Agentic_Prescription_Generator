"""
config.py
---------
Central configuration. Edit this file to change the model, storage
locations, or defaults -- nothing else in the project should need to change.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# LLM settings (GPT4All, fully offline)
# ---------------------------------------------------------------------------
# Must already be downloaded into the GPT4All cache
# (~/.cache/gpt4all on Linux/macOS, %LOCALAPPDATA%\nomic.ai\GPT4All on Windows)
# Models offered in the Streamlit UI -> actual GGUF file name
MODEL_OPTIONS = {
    "GPT4All (Llama 3 8B)": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "Qwen3 0.6B (Qwen2.5 0.5B)": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Qwen2 1.5B Instruct": "qwen2-1_5b-instruct-q4_0.gguf",
    "DeepSeek-R1 Distill Qwen 1.5B": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
}
MODEL_DOWNLOAD_URLS = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen2-1_5b-instruct-q4_0.gguf": "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_0.gguf",
    "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf": "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
}
DEFAULT_MODEL_LABEL = "GPT4All (Llama 3 8B)"
MODEL_NAME = MODEL_OPTIONS[DEFAULT_MODEL_LABEL]

MAX_TOKENS = 300
TEMPERATURE = 0.0  # deterministic extraction -- important for medical data
VERBOSE = False
ALLOW_DOWNLOAD = True

# ---------------------------------------------------------------------------
# Speech-to-Text (STT) Multi-Model Options & Settings
# ---------------------------------------------------------------------------
STT_MODEL_OPTIONS = {
    "Whisper Ayush (Fine-Tuned Turbo Rx v1)": "whisper_ayush",
    "NVIDIA Canary 1B (High-Accuracy Multilingual)": "canary_1b",
    "NVIDIA Parakeet TDT 1.1B (Ultra-Low Latency)": "parakeet_tdt",
    "Useful Sensors Moonshine Base (Edge Device Optimized)": "moonshine_base",
    "Useful Sensors Moonshine Tiny (Fastest Edge ASR)": "moonshine_tiny",
    "OpenAI Whisper Large v3 Turbo": "whisper_large_turbo",
    "OpenAI Whisper Base (Local Offline)": "whisper_base",
    "OpenAI Whisper Tiny (Lightweight)": "whisper_tiny",
}
DEFAULT_STT_MODEL_LABEL = "Whisper Ayush (Fine-Tuned Turbo Rx v1)"
WHISPER_MODEL = STT_MODEL_OPTIONS[DEFAULT_STT_MODEL_LABEL]

AYUSH_WHISPER_PATH = os.path.join(os.path.dirname(BASE_DIR), "Whisper_Ayush", "Whisper large", "saved_models_whisper_large", "merged_turbo_rx_v1")
if not os.path.exists(AYUSH_WHISPER_PATH):
    AYUSH_WHISPER_PATH = os.path.join(BASE_DIR, "Whisper_Ayush", "Whisper large", "saved_models_whisper_large", "merged_turbo_rx_v1")

# Choices offered in the Streamlit sidebar -> actual GPT4All device string
DEVICE_OPTIONS = {"CPU": "cpu", "GPU (CUDA)": "cuda"}
DEFAULT_DEVICE_LABEL = "GPU (CUDA)"

# ---------------------------------------------------------------------------
# Embedding model for FAISS (also GPT4All-backed -> stays fully offline)
# ---------------------------------------------------------------------------
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2.gguf2.f16.gguf"

# ---------------------------------------------------------------------------
# Storage locations (all local disk -- this is what survives a page refresh
# or an app restart)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
AUDIO_DIR = os.path.join(DATA_DIR, "audio_files")
SQLITE_PATH = os.path.join(DATA_DIR, "app_state.db")

for _d in (DATA_DIR, FAISS_DIR, OUTPUT_DIR, AUDIO_DIR):
    os.makedirs(_d, exist_ok=True)
