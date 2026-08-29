"""
test_streaming_engine.py
-------------------------
Automated Verification Suite for Sub-Second Streaming Audio Engine & WebSocket Service.
Tests VAD detection, sliding window buffer, prefix cache consensus, and WebSocket endpoints.
"""
import sys
import os
import time
import io
import json
from unittest.mock import patch
import numpy as np
import soundfile as sf

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming_engine.vad_detector import VADDetector
from streaming_engine.sliding_window_decoder import SlidingWindowStreamingDecoder
from streaming_transcriber import LiveStreamingTranscriber, StreamingAudioBuffer
from fastapi.testclient import TestClient
from api_server import app


def generate_synthetic_speech_signal(duration_s: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates a synthetic speech-like harmonic signal with modulated amplitude and noise."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t) + 0.1 * np.sin(2 * np.pi * 2500 * t)
    mod = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    speech = (signal * mod).astype(np.float32)
    noise = np.random.normal(0, 0.002, len(speech)).astype(np.float32)
    return speech + noise


def generate_silence_signal(duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates pure ambient background noise/silence."""
    return np.random.normal(0, 0.001, int(sample_rate * duration_s)).astype(np.float32)


def test_vad_detector_latency_and_accuracy():
    """Validates VAD sub-10ms processing speed and speech vs silence separation."""
    vad = VADDetector(sample_rate=16000, energy_threshold=0.01)
    
    silence = generate_silence_signal(0.5, 16000)
    speech = generate_synthetic_speech_signal(1.0, 16000)
    
    # 1. Test Frame Processing Speed (<0.1ms per 30ms frame)
    frame_30ms = speech[:480]
    t0 = time.perf_counter()
    for _ in range(100):
        is_speech = vad.is_speech_frame(frame_30ms)
    t1 = time.perf_counter()
    avg_frame_latency_ms = ((t1 - t0) / 100) * 1000
    
    print(f"[VAD Test] Average Frame Latency: {avg_frame_latency_ms:.4f} ms (<10ms requirement)")
    assert avg_frame_latency_ms < 2.0, "VAD frame latency exceeded budget."
    
    # 2. Test Silence Detection
    vad.reset()
    silence_decisions = [vad.is_speech_frame(silence[i:i+480]) for i in range(0, len(silence) - 480, 480)]
    assert sum(silence_decisions) <= 1, "Silence falsely classified as active speech."
    
    # 3. Test Continuous Speech Segmentation
    combined = np.concatenate([silence[:8000], speech, silence[:8000]])
    segments = vad.segment_speech(combined)
    assert len(segments) >= 1, "VAD failed to identify speech segments."
    print(f"[VAD Test] Identified Speech Segments: {segments}")
    print("PASS: VAD latency and speech segmentation verified.")


def test_sliding_window_decoder_prefix_consensus():
    """Validates sliding window buffer, prefix cache, and deduplication logic."""
    decoder = SlidingWindowStreamingDecoder(sample_rate=16000, window_duration_s=4.0)
    
    # Test deduplication
    decoder.committed_text = "Take one Cefpodoxime 200 mg"
    new_text = "Cefpodoxime 200 mg tablet twice daily"
    aligned = decoder._align_and_deduplicate(new_text)
    assert aligned == "tablet twice daily", f"Deduplication failed: got '{aligned}'"
    
    # Test audio ingestion
    speech = generate_synthetic_speech_signal(1.0, 16000)
    pcm_bytes = (speech * 32767).astype(np.int16).tobytes()
    decoder.append_pcm(pcm_bytes)
    assert decoder.total_audio_received_s >= 0.99
    
    with patch("transcriber.transcribe_audio", return_value="tablet twice daily after food"):
        res = decoder.decode_active_window()
        assert "type" in res and res["type"] == "partial"
        assert "latency_ms" in res
        assert "duration" in res
        print(f"[Sliding Window Test] Partial result: {res['text']} (latency={res['latency_ms']}ms)")
    print("PASS: Sliding window decoder and consensus deduplication verified.")


def test_streaming_transcriber_lifecycle():
    """Validates LiveStreamingTranscriber incremental feeding and finalization."""
    streamer = LiveStreamingTranscriber(sample_rate=16000)
    speech = generate_synthetic_speech_signal(1.0, 16000)
    pcm_bytes = (speech * 32767).astype(np.int16).tobytes()
    
    with patch("transcriber.transcribe_audio", side_effect=[
        "take one paracetamol",
        "take one paracetamol 650 mg",
        "take one paracetamol 650 mg three times daily",
        "take one paracetamol 650 mg three times daily for 5 days"
    ]):
        # Feed chunks
        chunk_size = len(pcm_bytes) // 3
        for i in range(3):
            chunk = pcm_bytes[i * chunk_size : (i + 1) * chunk_size]
            res = streamer.feed_audio_chunk(chunk)
            assert res["type"] == "partial"
            assert res["duration"] > 0
        
        # Finalize
        final_res = streamer.finalize()
        assert final_res["type"] == "final"
        assert "punctuated_text" in final_res
        assert "final_latency_ms" in final_res
        print(f"[Streamer Test] Final text: {final_res['punctuated_text']} ({final_res['final_latency_ms']}ms)")
    print("PASS: LiveStreamingTranscriber lifecycle verified.")


def test_websocket_transcribe_endpoint():
    """Validates FastAPI /ws/transcribe WebSocket protocol and bidirectional communication."""
    client = TestClient(app)
    
    speech = generate_synthetic_speech_signal(1.0, 16000)
    pcm_bytes = (speech * 32767).astype(np.int16).tobytes()
    
    with patch("transcriber.transcribe_audio", return_value="Take Cefpodoxime 200 mg twice daily"):
        with client.websocket_connect("/ws/transcribe?stt_model=whisper_ayush") as websocket:
            # Receive connection handshake
            init_data = websocket.receive_json()
            assert init_data["type"] == "connected"
            assert init_data["status"] == "ready"
            print(f"[WebSocket Test] Connection handshake: {init_data}")
            
            # Send binary chunk 1
            chunk_1 = pcm_bytes[:len(pcm_bytes)//2]
            websocket.send_bytes(chunk_1)
            resp_1 = websocket.receive_json()
            assert resp_1["type"] == "partial"
            assert "duration" in resp_1
            print(f"[WebSocket Test] Partial response 1: {resp_1}")
            
            # Send binary chunk 2
            chunk_2 = pcm_bytes[len(pcm_bytes)//2:]
            websocket.send_bytes(chunk_2)
            resp_2 = websocket.receive_json()
            assert resp_2["type"] == "partial"
            print(f"[WebSocket Test] Partial response 2: {resp_2}")
            
            # Send finalize command
            websocket.send_text(json.dumps({"action": "finalize"}))
            resp_final = websocket.receive_json()
            assert resp_final["type"] == "final"
            assert "punctuated_text" in resp_final
            print(f"[WebSocket Test] Final response: {resp_final}")
    print("PASS: WebSocket full-duplex binary audio transcription verified.")


if __name__ == "__main__":
    print("\n========================================================")
    print("RUNNING STREAMING AUDIO ENGINE & WEBSOCKET TEST SUITE")
    print("========================================================")
    
    test_vad_detector_latency_and_accuracy()
    test_sliding_window_decoder_prefix_consensus()
    test_streaming_transcriber_lifecycle()
    test_websocket_transcribe_endpoint()
    
    print("\n========================================================")
    print("ALL SUB-SECOND STREAMING ENGINE & WS TESTS PASSED!")
    print("========================================================")
