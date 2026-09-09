"""
tests/unit/test_stt_engine.py
-----------------------------
Comprehensive unit tests for the Canonical STT subsystem.

Verifies:
1. Mock STT engine lifecycle (load, unload, inference)
2. Canonical TranscriptionResult schema validation
3. Polymorphic audio input handling (bytes, str path, Path, np.ndarray, file buffer)
4. Lazy loading of engines (zero models loaded on initialization)
5. Explicit fallback auditing (never silently switches models)
6. Disabling fallback strictly raises standardized STTError
7. Standardized exception hierarchy (ModelNotFoundError, AudioFormatError)
8. Reusable standalone VADDetector (energy, ZCR, hangover, speech filtering)
9. Decoupled StreamingTranscriber (chunk feeding, interim text, finalization)
10. Real CTranslate2 Whisper Ayush inference on benchmark wav (if local weights present)
"""

import io
import os
import pytest
import numpy as np
from pathlib import Path

from app.stt.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    STTFallbackRecord,
    STTConfig,
)
from app.stt.base import (
    STTEngine,
    STTError,
    ModelNotFoundError,
    AudioFormatError,
    prepare_audio_input,
)
from app.stt.vad import VADDetector
from app.stt.streaming import StreamingTranscriber
from app.stt.adapters.mock_engine import MockSTTEngine
from app.stt.adapters.ctranslate2_engine import CTranslate2Engine
from app.stt.manager import STTManager


@pytest.fixture
def mock_engine():
    return MockSTTEngine(name="test_mock", default_transcript="Take Paracetamol 650 mg.")


@pytest.fixture
def manager():
    # Fresh manager for isolation
    return STTManager(config=STTConfig(default_model="mock", fallback_chain=["mock"]))


class TestMockSTTEngine:
    """Tests for MockSTTEngine lifecycle and basic inference."""

    def test_mock_lifecycle(self, mock_engine: MockSTTEngine):
        assert not mock_engine.is_loaded
        mock_engine.load()
        assert mock_engine.is_loaded
        mock_engine.unload()
        assert not mock_engine.is_loaded

    def test_mock_transcription(self, mock_engine: MockSTTEngine):
        audio_bytes = b"\x00" * 3200  # 0.1s of silence
        res = mock_engine.transcribe(audio_bytes)
        assert isinstance(res, TranscriptionResult)
        assert res.text == "Take Paracetamol 650 mg."
        assert res.model == "test_mock"
        assert res.latency > 0
        assert len(res.segments) > 0
        assert res.confidence == 0.98


class TestInputPolymorphism:
    """Verifies that all input formats (bytes, str, Path, ndarray, buffer) work seamlessly."""

    def test_bytes_input(self, mock_engine: MockSTTEngine):
        res = mock_engine.transcribe(b"\x01\x02\x03\x04" * 100)
        assert res.text != ""

    def test_numpy_array_input(self, mock_engine: MockSTTEngine):
        pcm = np.zeros(16000, dtype=np.float32)
        res = mock_engine.transcribe(pcm)
        assert res.text != ""

    def test_file_stream_input(self, mock_engine: MockSTTEngine):
        buf = io.BytesIO(b"\x00" * 1600)
        buf.name = "sample.wav"
        res = mock_engine.transcribe(buf)
        assert res.text != ""

    def test_string_path_input(self, mock_engine: MockSTTEngine, tmp_path):
        dummy_file = tmp_path / "audio_test.wav"
        dummy_file.write_bytes(b"RIFF" + b"\x00" * 500)
        res = mock_engine.transcribe(str(dummy_file))
        assert res.text != ""
        # Original file must NOT be deleted
        assert dummy_file.exists()

    def test_path_object_input(self, mock_engine: MockSTTEngine, tmp_path):
        dummy_file = tmp_path / "audio_test2.wav"
        dummy_file.write_bytes(b"RIFF" + b"\x00" * 500)
        res = mock_engine.transcribe(dummy_file)
        assert res.text != ""
        assert dummy_file.exists()

    def test_empty_audio_raises_error(self, mock_engine: MockSTTEngine):
        with pytest.raises(AudioFormatError):
            mock_engine.transcribe(b"")

    def test_none_audio_raises_error(self, mock_engine: MockSTTEngine):
        with pytest.raises(AudioFormatError):
            mock_engine.transcribe(None)


class TestSTTManagerLazyLoading:
    """Verifies zero startup overhead and lazy loading."""

    def test_manager_init_loads_zero_engines(self, manager: STTManager):
        # On instantiation, no engines should be loaded in memory
        assert len(manager._active_engines) == 0

    def test_get_engine_lazy_loads(self, manager: STTManager):
        engine = manager.get_engine("mock", auto_load=False)
        assert engine.name == "mock"
        assert not engine.is_loaded
        # Auto-load on transcribe
        res = manager.transcribe(b"\x00" * 1600, model_key="mock")
        assert res.model == "mock"
        assert manager._active_engines["mock"].is_loaded

    def test_unknown_model_raises_error(self, manager: STTManager):
        with pytest.raises(ModelNotFoundError):
            manager.get_engine("non_existent_model_xyz")


class TestExplicitFallbackTracking:
    """Verifies that model fallbacks are explicitly recorded and never silent."""

    def test_explicit_fallback_logged(self, manager: STTManager):
        # Create a failing primary engine
        failing_engine = MockSTTEngine(name="failing_primary", should_fail=True)
        backup_engine = MockSTTEngine(name="backup_model", default_transcript="Backup Transcript.")

        manager.register_engine("failing_primary", lambda: failing_engine)
        manager.register_engine("backup_model", lambda: backup_engine)

        manager.config.fallback_chain = ["backup_model"]
        manager.config.enable_fallback = True

        res = manager.transcribe(b"\x00" * 1600, model_key="failing_primary")
        assert res.text == "Backup Transcript."
        assert res.model == "backup_model"

        # Explicit fallback record must be present
        assert res.fallback_info is not None
        assert res.fallback_info.primary_model == "failing_primary"
        assert res.fallback_info.fallback_model == "backup_model"
        assert "Simulated" in res.fallback_info.reason

    def test_disabling_fallback_raises_exception(self, manager: STTManager):
        failing_engine = MockSTTEngine(name="failing_primary_2", should_fail=True)
        manager.register_engine("failing_primary_2", lambda: failing_engine)

        with pytest.raises(STTError):
            manager.transcribe(b"\x00" * 1600, model_key="failing_primary_2", allow_fallback=False)


class TestVADDetector:
    """Tests standalone pure-NumPy VADDetector."""

    def test_silence_detection(self):
        vad = VADDetector(sample_rate=16000, energy_threshold=0.01)
        silence = np.zeros(480, dtype=np.float32)  # 30ms silence
        is_speech = vad.is_speech_frame(silence)
        assert not is_speech

    def test_speech_detection_on_active_sine(self):
        vad = VADDetector(sample_rate=16000, energy_threshold=0.01, min_speech_duration_ms=30)
        # 1kHz sine wave with strong amplitude
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        speech_wave = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        # Feed frames
        detected = False
        step = 480
        for i in range(len(speech_wave) // step):
            if vad.is_speech_frame(speech_wave[i * step : (i + 1) * step]):
                detected = True
                break
        assert detected

    def test_filter_speech_leading_trailing_silence(self):
        vad = VADDetector(sample_rate=16000)
        silence_lead = np.zeros(16000, dtype=np.float32)  # 1s silence
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)  # 1s tone
        silence_trail = np.zeros(16000, dtype=np.float32)  # 1s silence

        combined = np.concatenate([silence_lead, tone, silence_trail])
        filtered = vad.filter_speech(combined)
        # Filtered audio should be substantially shorter than original 3s
        assert len(filtered) < len(combined)


class TestStreamingTranscriber:
    """Tests StreamingTranscriber with decoupled mock engine."""

    def test_streaming_chunk_feed_and_finalize(self, mock_engine: MockSTTEngine):
        streamer = StreamingTranscriber(
            engine=mock_engine,
            sample_rate=16000,
            window_duration_s=1.0,
            step_duration_s=0.2,
        )

        chunk = (0.3 * np.ones(3200, dtype=np.float32))  # 0.2s of signal
        state = streamer.feed_chunk(chunk)
        assert "is_speech" in state
        assert "interim_text" in state
        assert "full_transcript" in state

        final_res = streamer.finalize()
        assert isinstance(final_res, TranscriptionResult)
        assert final_res.text != ""

    def test_cumulative_utterance_retention(self, mock_engine: MockSTTEngine):
        """Verifies that earlier words are not dropped when multiple chunks arrive."""
        streamer = StreamingTranscriber(
            engine=mock_engine,
            sample_rate=16000,
            step_duration_s=0.1,
        )
        # Feed successive chunks of audio
        state = None
        for _ in range(5):
            chunk = (0.4 * np.ones(4096, dtype=np.float32))
            state = streamer.feed_chunk(chunk)
        assert state is not None
        assert state["text"] != ""
        assert "Paracetamol" in state["text"]


    def test_continuous_silence_produces_no_transcription_calls(self):
        """Verifies that pure silence never triggers Whisper inference and finalize yields empty text."""
        class CallCountingMockEngine(MockSTTEngine):
            def __init__(self):
                super().__init__(name="counting_mock", default_transcript="Hallucinated text")
                self.call_count = 0

            def transcribe(self, audio, **kwargs):
                self.call_count += 1
                return super().transcribe(audio, **kwargs)

        counting_engine = CallCountingMockEngine()
        streamer = StreamingTranscriber(
            engine=counting_engine,
            sample_rate=16000,
            step_duration_s=0.1,
        )

        # Feed 10 consecutive chunks of pure silence (each 0.1s = 1600 samples)
        for _ in range(10):
            silence_chunk = np.zeros(1600, dtype=np.float32)
            state = streamer.feed_chunk(silence_chunk)
            assert not state["is_speech"]
            assert not state["decoded"]
            assert state["text"] == ""

        # Inference should NOT have been called even once during silence
        assert counting_engine.call_count == 0

        # Finalizing on pure silence should return empty text and avoid calling transcribe
        final_res = streamer.finalize()
        assert final_res.text == ""
        assert counting_engine.call_count == 0

    def test_speech_followed_by_pause_commits_and_stops_decoding(self):
        """Verifies that speech pauses commit boundary cleanly and silence does not keep decoding."""
        class CallCountingMockEngine(MockSTTEngine):
            def __init__(self):
                super().__init__(name="counting_mock", default_transcript="Tab Paracetamol 650mg")
                self.call_count = 0

            def transcribe(self, audio, **kwargs):
                self.call_count += 1
                return super().transcribe(audio, **kwargs)

        counting_engine = CallCountingMockEngine()
        streamer = StreamingTranscriber(
            engine=counting_engine,
            sample_rate=16000,
            step_duration_s=0.1,
        )

        # Feed 3 speech chunks (0.3s total)
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        speech_pcm = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        for _ in range(3):
            streamer.feed_chunk(speech_pcm)

        calls_after_speech = counting_engine.call_count

        # Now feed 6 chunks of pure silence (0.6s of pause to cross 350ms boundary)
        silence_pcm = np.zeros(1600, dtype=np.float32)
        boundary_seen = False
        for _ in range(6):
            state = streamer.feed_chunk(silence_pcm)
            if state.get("boundary"):
                boundary_seen = True

        assert boundary_seen, "Natural speech pause boundary should be committed"
        assert "Paracetamol" in streamer.confirmed_text

        # Feed another 5 chunks of silence after boundary
        calls_at_boundary = counting_engine.call_count
        for _ in range(5):
            state = streamer.feed_chunk(silence_pcm)
            assert not state["decoded"], "Subsequent silence must NOT trigger Whisper decode"

        assert counting_engine.call_count == calls_at_boundary, "Whisper must not be called during idle silence"

    def test_hallucination_cleaning(self):
        """Verifies clean_hallucinations removes typical Whisper silence and subtitle hallucinations."""
        from app.stt.adapters.ctranslate2_engine import clean_hallucinations

        assert clean_hallucinations("Thank you.") == ""
        assert clean_hallucinations("Thank you for watching!") == ""
        assert clean_hallucinations("Please subscribe to my channel.") == ""
        assert clean_hallucinations("Subtitles by Amara.org") == ""
        assert clean_hallucinations("...") == ""
        assert clean_hallucinations(" ,,, ") == ""
        assert clean_hallucinations("Tab Paracetamol 500mg Thank you.") == "Tab Paracetamol 500mg"
        assert clean_hallucinations("Capsule Amoxicillin 500mg") == "Capsule Amoxicillin 500mg"


    def test_vad_process_samples_slicing(self):
        """Verifies that 4096-sample chunks are broken into 30ms frames accurately."""
        vad = VADDetector(sample_rate=16000)
        # 4096 samples of silence
        silence_chunk = np.zeros(4096, dtype=np.float32)
        assert not vad.process_samples(silence_chunk)

        # 4096 samples of tone (enough to cross min_speech_frames of 4*30ms=120ms)
        t = np.linspace(0, 4096 / 16000, 4096, endpoint=False)
        tone_chunk = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        assert vad.process_samples(tone_chunk)



class TestRealCT2WhisperAyush:
    """Integration test on actual Whisper_Ayush_ct2 model if files present."""

    def test_ct2_real_audio_transcription(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        audio_file = repo_root / "rx_extractor_app" / "data" / "audio_files" / "proc_70_20260828_134911.wav"
        ct2_dir = repo_root / "Whisper_Ayush_ct2"

        if not audio_file.exists() or not (ct2_dir / "model.bin").exists():
            pytest.skip("Audio file or Whisper_Ayush_ct2 model.bin not present; skipping local inference.")

        engine = CTranslate2Engine(name="whisper_ayush", model_dir=ct2_dir)
        engine.load()
        assert engine.is_loaded

        res = engine.transcribe(str(audio_file))
        assert "paracetamol" in res.text.lower()
        assert res.model == "whisper_ayush"
        assert res.latency > 0
        assert len(res.segments) > 0
