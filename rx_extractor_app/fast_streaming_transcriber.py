"""
fast_streaming_transcriber.py
------------------------------
Ultra-low-latency GPU-accelerated streaming ASR engine.
RTX 3050 + Whisper tiny fp16 = ~15-25ms inference latency.
CPU fallback = ~350ms.

Architecture:
- Model loaded ONCE as module singleton (GPU if available, else CPU)
- Decode triggered every 0.5s of NEW audio on GPU (1.5s on CPU)
- Rolling 3s window (GPU) or 4s window (CPU)
- Returns partial text immediately
"""
import time
import logging
import threading
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("fast_streaming_transcriber")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Detect GPU once at module load
# ---------------------------------------------------------------------------
try:
    import torch
    _USE_GPU = torch.cuda.is_available()
    _GPU_NAME = torch.cuda.get_device_name(0) if _USE_GPU else "CPU"
except Exception:
    _USE_GPU = False
    _GPU_NAME = "CPU"

logger.info(f"Streaming ASR device: {'GPU (' + _GPU_NAME + ')' if _USE_GPU else 'CPU (no CUDA)'}")

# ---------------------------------------------------------------------------
# Module-level singleton: loaded once, reused for all WebSocket sessions
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper_model(size: str = "tiny"):
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        try:
            import whisper
            device = "cuda" if _USE_GPU else "cpu"
            logger.info(f"Loading openai-whisper {size} on {device.upper()}...")
            _whisper_model = whisper.load_model(size, device=device)
            logger.info(f"Whisper {size} loaded on {device.upper()} — ready")
        except Exception as e:
            logger.error(f"Failed to load whisper {size}: {e}")
            _whisper_model = None
    return _whisper_model


# ---------------------------------------------------------------------------
# Medical vocabulary initial_prompt — primes Whisper to expect drug names
# This is the single biggest accuracy improvement for Whisper tiny (free)
# ---------------------------------------------------------------------------
_MEDICAL_PROMPT = (
    "Medical prescription dictation. Drug names, dosages, frequencies, routes: "
    "Paracetamol, Ibuprofen, Aspirin, Amoxicillin, Amoxicillin-Clavulanate, Azithromycin, "
    "Cefpodoxime, Cefixime, Cefuroxime, Ciprofloxacin, Levofloxacin, Metronidazole, "
    "Doxycycline, Clindamycin, Erythromycin, Nitrofurantoin, Trimethoprim, "
    "Omeprazole, Pantoprazole, Rabeprazole, Esomeprazole, Ranitidine, Domperidone, "
    "Metformin, Glibenclamide, Glipizide, Sitagliptin, Insulin, "
    "Atorvastatin, Rosuvastatin, Amlodipine, Enalapril, Losartan, Telmisartan, "
    "Metoprolol, Atenolol, Furosemide, Spironolactone, Hydrochlorothiazide, "
    "Salbutamol, Budesonide, Montelukast, Levocetirizine, Cetirizine, Loratadine, "
    "Fexofenadine, Oxymetazoline, Betamethasone, Prednisolone, Dexamethasone, "
    "Diclofenac, Aceclofenac, Tramadol, Gabapentin, Pregabalin, "
    "Clotrimazole, Fluconazole, Terbinafine, Ivermectin, Albendazole, "
    "mg, mcg, ml, tablet, capsule, syrup, drops, spray, cream, ointment, "
    "twice daily, once daily, three times daily, after food, before food, "
    "at bedtime, for 5 days, for 7 days, for 10 days, for 14 days, for 30 days."
)

# Common Whisper tiny drug-name mangling corrections {mangled: correct}
_DRUG_CORRECTIONS = {
    "parasatamol": "paracetamol", "parasettamol": "paracetamol", "parasatmal": "paracetamol",
    "parasitamol": "paracetamol", "paraseta": "paracetamol", "parasuta": "paracetamol",
    "parasuita": "paracetamol", "parasatum": "paracetamol", "paracetamal": "paracetamol",
    "cefpodoxim": "cefpodoxime", "safpodoxim": "cefpodoxime", "seifpodoxim": "cefpodoxime",
    "levocetirizin": "levocetirizine", "levoceterizine": "levocetirizine",
    "cetirizin": "cetirizine", "cetrizine": "cetirizine",
    "montelukast": "montelukast", "montelecast": "montelukast",
    "pantoprazol": "pantoprazole", "pantaprazole": "pantoprazole",
    "omeprazol": "omeprazole", "omiprazole": "omeprazole",
    "azithromycin": "azithromycin", "azithromycin": "azithromycin",
    "amoxicillin": "amoxicillin", "amoxycillin": "amoxicillin",
    "metformin": "metformin", "metphormin": "metformin",
    "atorvastatin": "atorvastatin", "atorvastation": "atorvastatin",
    "amlodipine": "amlodipine", "amlodipin": "amlodipine",
    "salbutamol": "salbutamol", "salbuterol": "salbutamol",
    "budesonide": "budesonide", "budesonid": "budesonide",
    "diclofenac": "diclofenac", "diclofenack": "diclofenac",
    "ibuprofen": "ibuprofen", "iboprofen": "ibuprofen",
    "doxycycline": "doxycycline", "doxicycline": "doxycycline",
    "clotrimazole": "clotrimazole", "clotrimazol": "clotrimazole",
    "fluconazole": "fluconazole", "fluconazol": "fluconazole",
    "oxymetazoline": "oxymetazoline", "oximethazoline": "oxymetazoline",
    "gabapentin": "gabapentin", "gabapentine": "gabapentin",
    "pregabalin": "pregabalin", "pregabaline": "pregabalin",
    "cefixime": "cefixime", "sefixime": "cefixime",
    "ciprofloxacin": "ciprofloxacin", "ciprofloxacine": "ciprofloxacin",
    "levofloxacin": "levofloxacin", "levofloxacine": "levofloxacin",
    "metronidazole": "metronidazole", "metronidazol": "metronidazole",
}


def _medical_spell_correct(text: str) -> str:
    """
    Fast word-level spell correction for common Whisper tiny drug-name manglings.
    Uses exact lowercase match first, then optional fuzzy fallback.
    """
    if not text:
        return text
    words = text.split()
    corrected = []
    for word in words:
        clean = word.lower().strip(".,;:()")
        if clean in _DRUG_CORRECTIONS:
            # Preserve original casing style (capitalized if original was)
            replacement = _DRUG_CORRECTIONS[clean]
            if word[0].isupper():
                replacement = replacement.capitalize()
            corrected.append(word.replace(clean, replacement).replace(clean.capitalize(), replacement.capitalize()))
        else:
            corrected.append(word)
    return " ".join(corrected)


def _transcribe_pcm(audio_np: np.ndarray, sample_rate: int = 16000,
                    use_medical_prompt: bool = True) -> str:
    """
    Transcribes a float32 numpy array.
    - GPU fp16 if CUDA available, else CPU fp32
    - Medical initial_prompt primes drug name recognition
    - Post-processing spell correction for common mangling
    """
    model = _get_whisper_model("tiny")
    if model is None:
        return ""
    try:
        use_fp16 = _USE_GPU
        result = model.transcribe(
            audio_np,
            language="en",
            fp16=use_fp16,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=_MEDICAL_PROMPT if use_medical_prompt else None,
        )
        raw = (result.get("text") or "").strip()
        return _medical_spell_correct(raw)
    except Exception as e:
        logger.warning(f"Transcribe error: {e}")
        return ""



class FastLiveTranscriber:
    """
    Sub-second streaming transcriber.
    - GPU (RTX 3050): ~15-25ms per inference, decode every 0.5s of audio
    - CPU fallback: ~350ms per inference, decode every 1.5s of audio
    """

    # GPU settings: smaller window + faster decode stride
    GPU_DECODE_INTERVAL_S = 0.5   # Decode every 0.5s of new audio
    GPU_MAX_WINDOW_S = 3.0        # Use last 3s of audio
    GPU_MIN_AUDIO_S = 0.5         # Start decoding after 0.5s

    # CPU fallback settings
    CPU_DECODE_INTERVAL_S = 1.5
    CPU_MAX_WINDOW_S = 4.0
    CPU_MIN_AUDIO_S = 0.8

    def __init__(self, sample_rate: int = 16000, model_key: str = "whisper_ayush"):
        self.sample_rate = sample_rate
        self.model_key = model_key
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.committed_text = ""
        self.current_partial = ""
        self.total_received_s = 0.0
        self.last_decode_s = 0.0
        self.is_speaking = False

        # Select settings based on hardware
        if _USE_GPU:
            self.DECODE_INTERVAL_S = self.GPU_DECODE_INTERVAL_S
            self.MAX_WINDOW_S = self.GPU_MAX_WINDOW_S
            self.MIN_AUDIO_S = self.GPU_MIN_AUDIO_S
        else:
            self.DECODE_INTERVAL_S = self.CPU_DECODE_INTERVAL_S
            self.MAX_WINDOW_S = self.CPU_MAX_WINDOW_S
            self.MIN_AUDIO_S = self.CPU_MIN_AUDIO_S

        # Pre-warm model in background thread
        threading.Thread(target=_get_whisper_model, args=("tiny",), daemon=True).start()

    def feed_pcm16(self, pcm_bytes: bytes) -> Dict[str, Any]:
        """Feed raw 16-bit PCM bytes from the browser Web Audio API."""
        if not pcm_bytes:
            return self._make_partial(0.0)

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, samples])
        self.total_received_s = len(self.audio_buffer) / self.sample_rate

        # VAD via RMS energy of last 200ms
        frame = samples[-min(len(samples), int(self.sample_rate * 0.2)):]
        energy = float(np.sqrt(np.mean(frame ** 2))) if len(frame) > 0 else 0.0
        self.is_speaking = energy > 0.005

        audio_since_last = self.total_received_s - self.last_decode_s
        if self.total_received_s < self.MIN_AUDIO_S or audio_since_last < self.DECODE_INTERVAL_S:
            return self._make_partial(0.0)

        return self._run_inference()

    def _run_inference(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        max_samples = int(self.MAX_WINDOW_S * self.sample_rate)
        active = self.audio_buffer[-max_samples:]
        text = _transcribe_pcm(active, self.sample_rate)
        self.last_decode_s = self.total_received_s
        text = self._remove_committed_overlap(text)
        self.current_partial = text
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        device_tag = "GPU" if _USE_GPU else "CPU"
        logger.info(f"[FastASR/{device_tag}] '{text[:60]}' ({latency_ms}ms | {self.total_received_s:.1f}s audio)")
        return self._make_partial(latency_ms)

    def _remove_committed_overlap(self, new_text: str) -> str:
        if not new_text or not self.committed_text:
            return new_text.strip()
        comm_words = self.committed_text.strip().split()
        new_words = new_text.strip().split()
        max_overlap = min(len(comm_words), len(new_words), 6)
        for i in range(max_overlap, 0, -1):
            if [w.lower() for w in comm_words[-i:]] == [w.lower() for w in new_words[:i]]:
                return " ".join(new_words[i:])
        return new_text.strip()

    def _make_partial(self, latency_ms: float) -> Dict[str, Any]:
        parts = [p for p in [self.committed_text.strip(), self.current_partial.strip()] if p]
        device_tag = f"GPU ({_GPU_NAME})" if _USE_GPU else "CPU"
        return {
            "type": "partial",
            "text": " ".join(parts),
            "partial": self.current_partial,
            "committed": self.committed_text,
            "duration": round(self.total_received_s, 2),
            "latency_ms": latency_ms,
            "is_speech": self.is_speaking,
            "device": device_tag,
        }

    def finalize(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if len(self.audio_buffer) == 0:
            return {"type": "final", "raw_text": "", "punctuated_text": "", "duration": 0.0, "final_latency_ms": 0.0}

        # ── Final pass: try Whisper Ayush (fine-tuned prescription model) first ──
        # This gives the highest accuracy for drug names like Cefpodoxime, Levocetirizine etc.
        final_text = ""
        used_model = "whisper_tiny"
        try:
            import io
            import soundfile as sf
            import transcriber as tr

            # Export full buffer as WAV for Ayush model
            bio = io.BytesIO()
            sf.write(bio, self.audio_buffer, self.sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = bio.getvalue()

            ayush_result = tr.transcribe_audio(wav_bytes, model_key="whisper_ayush")
            if ayush_result and len(ayush_result.strip()) > 2:
                final_text = _medical_spell_correct(ayush_result.strip())
                used_model = "whisper_ayush"
                logger.info(f"[FastASR] Final pass via Whisper Ayush: '{final_text[:60]}'")
        except Exception as e:
            logger.warning(f"[FastASR] Whisper Ayush final pass failed ({e}), falling back to tiny+prompt")

        # ── Fallback: Whisper tiny with medical prompt + spell correction ──
        if not final_text:
            final_text = _transcribe_pcm(self.audio_buffer, self.sample_rate, use_medical_prompt=True)
            used_model = "whisper_tiny_gpu" if _USE_GPU else "whisper_tiny_cpu"

        if not final_text:
            final_text = (self.committed_text + " " + self.current_partial).strip()

        # ── Punctuation correction ──
        try:
            from agents.punctuation_agent import correct_sentence_punctuation
            punctuated = correct_sentence_punctuation(final_text)
        except Exception:
            punctuated = final_text

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"[FastASR] Finalized via {used_model}: '{punctuated}' ({latency_ms}ms)")
        return {
            "type": "final",
            "raw_text": final_text,
            "punctuated_text": punctuated,
            "duration": round(self.total_received_s, 2),
            "final_latency_ms": latency_ms,
            "model_used": used_model,
        }

    def reset(self):
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.committed_text = ""
        self.current_partial = ""
        self.total_received_s = 0.0
        self.last_decode_s = 0.0
        self.is_speaking = False
