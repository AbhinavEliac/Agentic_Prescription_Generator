"""
app/stt/benchmark.py
--------------------
Empirical benchmarking suite for STT subsystem performance.
Measures:
- Cold start latency (ms)
- Warm start latency (ms)
- Full audio transcription latency (ms) and Real-Time Factor (RTF)
- CPU utilization (%)
- Process RAM usage (MB)
- NVIDIA GPU VRAM allocation (MB)
- Streaming frame latency (ms)
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure project root in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.stt.manager import STTManager
from app.stt.schemas import STTConfig
from app.stt.vad import VADDetector
from app.stt.streaming import StreamingTranscriber

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_memory_metrics() -> Dict[str, float]:
    """Measures current process RAM and GPU VRAM."""
    ram_mb = 0.0
    vram_mb = 0.0
    if _HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
    if _HAS_TORCH and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    return {"ram_mb": round(ram_mb, 1), "vram_mb": round(vram_mb, 1)}


def run_benchmark() -> Dict[str, Any]:
    """Runs complete performance benchmarks on available models."""
    print("=" * 65)
    print("[BENCHMARK] STT SUBSYSTEM EMPIRICAL BENCHMARK SUITE")
    print("=" * 65)

    audio_path = _ROOT / "rx_extractor_app" / "data" / "audio_files" / "proc_70_20260828_134911.wav"
    if not audio_path.exists():
        print(f"Benchmark audio not found at: {audio_path}")
        return {}

    import soundfile as sf
    data, sr = sf.read(str(audio_path))
    audio_duration_s = len(data) / sr
    print(f"Loaded benchmark audio: {audio_duration_s:.2f}s at {sr}Hz ({os.path.getsize(audio_path)/1024:.1f} KB)")

    results: List[Dict[str, Any]] = []

    # 1. Benchmark Mock Engine
    print("\n[1/3] Benchmarking MockSTTEngine...")
    mgr = STTManager(config=STTConfig(default_model="mock"))
    t0 = time.perf_counter()
    mock_eng = mgr.get_engine("mock", auto_load=True)
    mock_cold_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res = mock_eng.transcribe(str(audio_path))
    mock_warm_ms = (time.perf_counter() - t0) * 1000
    mock_rtf = (mock_warm_ms / 1000.0) / audio_duration_s

    results.append({
        "engine": "Mock Engine",
        "device": "CPU",
        "cold_start_ms": round(mock_cold_ms, 2),
        "warm_latency_ms": round(mock_warm_ms, 2),
        "rtf": round(mock_rtf, 4),
        "transcript": res.text[:40] + "...",
        **get_memory_metrics(),
    })

    # 2. Benchmark Whisper Ayush (CTranslate2)
    ct2_dir = _ROOT / "Whisper_Ayush_ct2"
    if ct2_dir.exists() and (ct2_dir / "model.bin").exists():
        print("\n[2/3] Benchmarking CTranslate2 Whisper Ayush...")
        mgr_ct2 = STTManager(config=STTConfig(default_model="whisper_ayush"))

        # Cold start
        t0 = time.perf_counter()
        ct2_eng = mgr_ct2.get_engine("whisper_ayush", auto_load=True)
        ct2_cold_ms = (time.perf_counter() - t0) * 1000
        print(f"   -> Cold load time: {ct2_cold_ms:.1f}ms on {ct2_eng.device.upper()} ({ct2_eng.compute_type})")

        # First pass (transcription)
        t0 = time.perf_counter()
        res_pass1 = ct2_eng.transcribe(str(audio_path))
        ct2_pass1_ms = (time.perf_counter() - t0) * 1000
        print(f"   -> Pass 1 inference: {ct2_pass1_ms:.1f}ms")

        # Warm pass
        t0 = time.perf_counter()
        res_warm = ct2_eng.transcribe(str(audio_path))
        ct2_warm_ms = (time.perf_counter() - t0) * 1000
        ct2_rtf = (ct2_warm_ms / 1000.0) / audio_duration_s
        print(f"   -> Warm inference: {ct2_warm_ms:.1f}ms (RTF: {ct2_rtf:.3f})")
        print(f"   -> Transcript: '{res_warm.text}'")

        results.append({
            "engine": f"Whisper Ayush (CT2 {ct2_eng.device.upper()})",
            "device": ct2_eng.device.upper(),
            "cold_start_ms": round(ct2_cold_ms, 1),
            "warm_latency_ms": round(ct2_warm_ms, 1),
            "rtf": round(ct2_rtf, 3),
            "transcript": res_warm.text[:40] + "...",
            **get_memory_metrics(),
        })

        # Also benchmark CPU mode
        print("\n   [2b] Benchmarking CTranslate2 Whisper Ayush on CPU (int8)...")
        from app.stt.adapters.ctranslate2_engine import CTranslate2Engine
        t0 = time.perf_counter()
        ct2_cpu = CTranslate2Engine(name="whisper_ayush_cpu", model_dir=str(ct2_dir), device="cpu", compute_type="int8")
        ct2_cpu.load()
        cpu_cold_ms = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        res_cpu = ct2_cpu.transcribe(str(audio_path))
        cpu_warm_ms = (time.perf_counter() - t0) * 1000
        cpu_rtf = (cpu_warm_ms / 1000.0) / audio_duration_s
        print(f"   -> CPU load time: {cpu_cold_ms:.1f}ms, inference: {cpu_warm_ms:.1f}ms (RTF: {cpu_rtf:.3f})")

        results.append({
            "engine": "Whisper Ayush (CT2 CPU)",
            "device": "CPU",
            "cold_start_ms": round(cpu_cold_ms, 1),
            "warm_latency_ms": round(cpu_warm_ms, 1),
            "rtf": round(cpu_rtf, 3),
            "transcript": res_cpu.text[:40] + "...",
            **get_memory_metrics(),
        })

    # 3. Benchmark VAD & Streaming Transcriber
    print("\n[3/3] Benchmarking VAD & Streaming Frame Latency...")
    vad = VADDetector(sample_rate=16000)
    frame = (0.2 * np.sin(np.linspace(0, 0.03 * 1000, 480))).astype(np.float32)

    # 10,000 VAD frame evaluations
    t0 = time.perf_counter()
    for _ in range(10000):
        vad.is_speech_frame(frame)
    vad_us = ((time.perf_counter() - t0) / 10000) * 1_000_000
    print(f"   -> Average VAD frame decision: {vad_us:.2f} µs (0.00{int(vad_us):02d}ms)")

    mem = get_memory_metrics()
    print(f"\nResource Footprint: RAM = {mem['ram_mb']} MB, GPU VRAM = {mem['vram_mb']} MB")

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Engine':<28} | {'Device':<6} | {'Cold (ms)':<9} | {'Warm (ms)':<9} | {'RTF':<6} | {'RAM (MB)':<8} | {'VRAM (MB)'}")
    print("-" * 88)
    for r in results:
        print(f"{r['engine']:<28} | {r['device']:<6} | {r['cold_start_ms']:<9} | {r['warm_latency_ms']:<9} | {r['rtf']:<6} | {r['ram_mb']:<8} | {r['vram_mb']}")

    return {"results": results, "vad_us": round(vad_us, 2), "duration_s": audio_duration_s}


if __name__ == "__main__":
    run_benchmark()
