import os
p = r"C:\Users\ADMIN\AppData\Local\Programs\Ollama\lib\ollama\cuda_v12"
os.environ["PATH"] = p + ";" + os.environ["PATH"]
if hasattr(os, "add_dll_directory") and os.path.exists(p):
    os.add_dll_directory(p)

import time
import soundfile as sf
from faster_whisper import WhisperModel

audio_path = 'rx_extractor_app/data/audio_files/proc_70_20260828_134911.wav'
audio, sr = sf.read(audio_path)
print(f"Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz")

t0 = time.perf_counter()
model = WhisperModel('Whisper_Ayush_ct2', device='cuda', compute_type='float16')
print(f"Whisper Ayush GPU Model loaded in {(time.perf_counter()-t0)*1000:.1f}ms")

prompt = "Medical prescription dictation. Drug names, dosages, frequencies: Paracetamol, Pantoprazole, Amoxicillin, mg, tablets."

# Benchmark 1: Full 7.32s speech
t0 = time.perf_counter()
segs, info = model.transcribe(
    audio.astype('float32'),
    beam_size=1,
    best_of=1,
    temperature=0.0,
    language="en",
    initial_prompt=prompt,
    condition_on_previous_text=False
)
text = " ".join([s.text for s in segs]).strip()
dt = (time.perf_counter() - t0) * 1000
print(f"\n[BENCHMARK 1] Full 7.32s speech transcribed in: {dt:.1f} ms ({dt/1000:.3f}s)")
print(f"Transcript: '{text}'")

# Benchmark 2: Full 7.32s speech (Warm)
t0 = time.perf_counter()
segs, info = model.transcribe(
    audio.astype('float32'),
    beam_size=1,
    best_of=1,
    temperature=0.0,
    language="en",
    initial_prompt=prompt,
    condition_on_previous_text=False
)
text = " ".join([s.text for s in segs]).strip()
dt = (time.perf_counter() - t0) * 1000
print(f"\n[BENCHMARK 2 - WARM] Full 7.32s speech transcribed in: {dt:.1f} ms ({dt/1000:.3f}s)")
print(f"Transcript: '{text}'")

# Benchmark 3: Short 1.5s chunk
t0 = time.perf_counter()
segs, info = model.transcribe(
    audio[:int(sr*1.5)].astype('float32'),
    beam_size=1,
    best_of=1,
    temperature=0.0,
    language="en",
    initial_prompt=prompt,
    condition_on_previous_text=False
)
chunk_text = " ".join([s.text for s in segs]).strip()
dt_chunk = (time.perf_counter() - t0) * 1000
print(f"\n[BENCHMARK 3] 1.5s streaming chunk transcribed in: {dt_chunk:.1f} ms ({dt_chunk/1000:.3f}s)")
print(f"Chunk text: '{chunk_text}'")
