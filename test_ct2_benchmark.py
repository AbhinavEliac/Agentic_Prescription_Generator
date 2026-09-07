import time
import soundfile as sf
from faster_whisper import WhisperModel

audio_path = 'rx_extractor_app/data/audio_files/proc_70_20260828_134911.wav'
audio, sr = sf.read(audio_path)
print(f"Loaded audio: {len(audio)/sr:.2f}s")

ct2_model_path = r"c:\Users\ADMIN\Downloads\rx_extractor_app_agentic\Whisper_Ayush_ct2"
print(f"Loading local CT2 INT8 model from {ct2_model_path} with 6 threads...")
t0 = time.perf_counter()
model = WhisperModel(ct2_model_path, device="cpu", compute_type="int8", cpu_threads=6)
load_dt = (time.perf_counter() - t0) * 1000
print(f"CT2 INT8 Model loaded in {load_dt:.1f}ms ({load_dt/1000:.2f}s)!")

prompt = "Medical prescription dictation. Paracetamol 500mg, Pantoprazole 40mg, Amoxicillin."

# Pass 1: Full 7.32s speech
print("\n--- Pass 1: Full 7.32s Speech ---")
t0 = time.perf_counter()
segments, info = model.transcribe(
    audio.astype('float32'),
    beam_size=1,
    best_of=1,
    temperature=0.0,
    language="en",
    initial_prompt=prompt,
    condition_on_previous_text=False
)
text = " ".join([s.text for s in segments]).strip()
dt = (time.perf_counter() - t0) * 1000
print(f"Transcribed in {dt:.1f}ms ({dt/1000:.2f}s)!")
print(f"Transcript: '{text}'")

# Pass 2: Full 7.32s speech (Warm)
print("\n--- Pass 2: Full 7.32s Speech (Warm) ---")
t0 = time.perf_counter()
segments, info = model.transcribe(
    audio.astype('float32'),
    beam_size=1,
    best_of=1,
    temperature=0.0,
    language="en",
    initial_prompt=prompt,
    condition_on_previous_text=False
)
text = " ".join([s.text for s in segments]).strip()
dt = (time.perf_counter() - t0) * 1000
print(f"Transcribed in {dt:.1f}ms ({dt/1000:.2f}s)!")
print(f"Transcript: '{text}'")

# Pass 3: 1.5s streaming chunk
print("\n--- Pass 3: 1.5s Streaming Chunk ---")
t0 = time.perf_counter()
segments, info = model.transcribe(
    audio[:int(sr*1.5)].astype('float32'),
    beam_size=1,
    best_of=1,
    temperature=0.0,
    language="en",
    initial_prompt=prompt,
    condition_on_previous_text=False
)
chunk_text = " ".join([s.text for s in segments]).strip()
dt_chunk = (time.perf_counter() - t0) * 1000
print(f"1.5s chunk transcribed in {dt_chunk:.1f}ms ({dt_chunk/1000:.2f}s)!")
print(f"Chunk text: '{chunk_text}'")
