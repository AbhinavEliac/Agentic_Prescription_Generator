"""
app.py
------
Streamlit UI for the offline prescription-extraction pipeline.
"""
import os
import time
import datetime
import streamlit as st
import pandas as pd

import config
import db
import vectorstore
import pipeline
import exporter

st.set_page_config(page_title="Offline Prescription Extractor", layout="wide")
db.init_db()


@st.cache_resource(show_spinner="Loading local LLM model (first time only)...")
def get_chat(device: str, model_name: str):
    return pipeline.build_chat(device, model_name)


@st.cache_resource(show_spinner="Embedding + storing system prompt in FAISS (first time only)...")
def get_vector_store():
    return vectorstore.load_or_create_index()


def load_process_to_session(p: dict):
    """Load a process/thread entry from DB into Streamlit session state."""
    st.session_state.process_id = p["process_id"]
    st.session_state.process_status = p.get("status", "active")
    st.session_state.device = p["device"]
    st.session_state.csv_path = p["csv_path"]
    st.session_state.xlsx_path = p["xlsx_path"]
    st.session_state.process_name = p["name"]
    st.session_state.model_name = p.get("model_name") or config.MODEL_NAME
    st.session_state.model_label = p.get("model_label") or config.DEFAULT_MODEL_LABEL


def reattach_active_process():
    """Runs on script execution. If no thread is selected in session_state,
    reattach to the most recent active process if available."""
    if "process_id" not in st.session_state:
        active = db.get_active_process()
        if active:
            load_process_to_session(active)


reattach_active_process()

st.title("🩺 LangGraph Multi-Agent Prescription Extractor")
st.caption("Parallel Multi-Agent Architecture (Supervisor, Medicine & Strength, Route, Duration & Frequency, Instructions, Aggregator, Validator) -- Fully Offline & Grounded.")

# --- Sidebar: process & thread controls ------------------------------------
with st.sidebar:
    st.header("⚙️ Process Control")
    is_running = "process_id" in st.session_state and st.session_state.get("process_status") == "active"

    model_labels = list(config.MODEL_OPTIONS.keys())
    model_label = st.selectbox(
        "LLM Model",
        model_labels,
        index=model_labels.index(
            st.session_state.get("model_label", config.DEFAULT_MODEL_LABEL)
        ),
        disabled=False,
        help="Select offline LLM model. Can be switched dynamically for any query entry!",
    )
    model_name = config.MODEL_OPTIONS[model_label]
    st.session_state.model_label = model_label
    st.session_state.model_name = model_name

    device_labels = list(config.DEVICE_OPTIONS.keys())
    device_label = st.selectbox(
        "Run on",
        device_labels,
        index=device_labels.index(
            st.session_state.get("device_label", config.DEFAULT_DEVICE_LABEL)
        ),
        disabled=is_running,
        help="Locked while a process is running -- stop it to change device.",
    )
    device = config.DEVICE_OPTIONS[device_label]

    process_name = st.text_input("Process name (optional)", value="run", disabled=is_running)

    col1, col2 = st.columns(2)
    start_clicked = col1.button("Start Process", disabled=is_running, use_container_width=True)
    stop_clicked = col2.button("Stop Process", disabled=not is_running, use_container_width=True)

    if is_running:
        st.success(
            f"**Running**: {st.session_state.process_name}\n\n"
            f"🤖 **Selected Model**: {model_label}\n\n"
            f"⚙️ **Device**: {st.session_state.device}"
        )
    else:
        st.info("No active process running.")

    if start_clicked:
        csv_path, xlsx_path = exporter.new_output_paths(process_name)
        process_id = db.create_process(
            process_name, device, csv_path, xlsx_path, model_name=model_name, model_label=model_label
        )
        load_process_to_session({
            "process_id": process_id,
            "status": "active",
            "device": device,
            "csv_path": csv_path,
            "xlsx_path": xlsx_path,
            "name": process_name,
            "model_name": model_name,
            "model_label": model_label,
        })
        st.session_state.device_label = device_label
        st.rerun()

    if stop_clicked:
        db.stop_process(st.session_state.process_id)
        st.session_state.process_status = "stopped"
        # Clear FAISS index on disk
        import shutil
        if os.path.exists(config.FAISS_DIR):
            shutil.rmtree(config.FAISS_DIR, ignore_errors=True)
        # Invalidate the st.cache_resource so the next Start force-rebuilds the index
        get_vector_store.clear()
        # Drop any lingering session references
        st.session_state.pop("faiss_store", None)
        st.session_state.pop("vector_store", None)
        st.rerun()

    st.divider()
    st.subheader("🎙️ Speech-to-Text Engine")
    stt_labels = list(config.STT_MODEL_OPTIONS.keys())
    selected_stt_label = st.selectbox(
        "STT Model",
        stt_labels,
        index=stt_labels.index(
            st.session_state.get("stt_model_label", config.DEFAULT_STT_MODEL_LABEL)
        ),
        help="Select offline Speech-to-Text model. Can be switched dynamically for voice transcription!",
        key="stt_model_selector_sb",
    )
    st.session_state.stt_model_label = selected_stt_label
    st.session_state.stt_model_key = config.STT_MODEL_OPTIONS[selected_stt_label]

    st.divider()
    st.subheader("🧵 Conversation Threads")
    all_threads = db.list_processes()

    if all_threads:
        # Scrollable container displaying ~5 threads at a time
        with st.container(height=320):
            for t in all_threads:
                t_id = t["process_id"]
                is_selected = st.session_state.get("process_id") == t_id
                tag = "🟢" if t["status"] == "active" else "⚪"
                t_model = t.get("model_label") or config.DEFAULT_MODEL_LABEL
                date_str = t["created_at"].split("T")[0] if "T" in t["created_at"] else t["created_at"][:10]

                expander_label = f"{tag} {t['name']} (#{t_id})"
                with st.expander(expander_label, expanded=is_selected):
                    st.caption(f"🤖 **Model:** {t_model}")
                    st.caption(f"⚙️ **Device:** {t['device']} | 📅 {date_str}")
                    st.caption(f"Status: **{t['status'].upper()}**")

                    btn_c1, btn_c2 = st.columns(2)
                    if btn_c1.button("View 👁️", key=f"select_{t_id}", use_container_width=True, disabled=is_selected):
                        load_process_to_session(t)
                        st.rerun()

                    if btn_c2.button("Delete 🗑️", key=f"del_{t_id}", use_container_width=True):
                        db.delete_process(t_id)
                        if st.session_state.get("process_id") == t_id:
                            for key in (
                                "process_id", "process_status", "device", "csv_path",
                                "xlsx_path", "process_name", "model_name", "model_label",
                                "last_gen_time", "last_ret_time"
                            ):
                                st.session_state.pop(key, None)
                        st.rerun()

        if st.button("🗑️ Delete All History", use_container_width=True, type="secondary"):
            db.delete_all_processes()
            for key in (
                "process_id", "process_status", "device", "csv_path",
                "xlsx_path", "process_name", "model_name", "model_label",
                "last_gen_time", "last_ret_time"
            ):
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.caption("No saved threads yet.")

# --- Main panel --------------------------------------------------------------
if "process_id" not in st.session_state:
    st.warning("Select or start a process thread from the sidebar to begin.")
    st.stop()

current_proc = db.get_process(st.session_state.process_id)
if not current_proc:
    st.warning("Selected thread no longer exists.")
    st.session_state.pop("process_id", None)
    st.stop()

active_model_name = model_name
active_model_label = model_label
is_current_active = st.session_state.get("process_status") == "active"

st.subheader(f"🧵 Thread: {st.session_state.process_name} (ID: #{st.session_state.process_id})")
st.caption(
    f"🤖 **Current LLM Model**: {active_model_label} | "
    f"⚙️ **Device**: {st.session_state.device} | "
    f"Status: **{st.session_state.get('process_status', 'stopped').upper()}**"
)

if is_current_active:
    # Resource-cached: cheap after initial load for any selected model
    chat = get_chat(st.session_state.device, active_model_name)
    store = get_vector_store()

    st.subheader("Enter an instruction")
    input_mode = st.radio("Input Mode", ["Text Input 📝", "Voice Input 🎙️"], horizontal=True, key="input_mode_selector")

    query = ""
    if input_mode == "Text Input 📝":
        query = st.text_area("Text instruction", height=100, key="query_input")
    else:
        current_stt_label = st.session_state.get("stt_model_label", config.DEFAULT_STT_MODEL_LABEL)
        current_stt_key = st.session_state.get("stt_model_key", config.STT_MODEL_OPTIONS[config.DEFAULT_STT_MODEL_LABEL])

        st.markdown(f"##### 🎙️ Voice Input ({current_stt_label})")
        st.caption(f"Active Speech-to-Text Engine: **{current_stt_label}** — Toggle anytime in sidebar or below.")

        stt_col1, stt_col2 = st.columns([2, 1])
        with stt_col1:
            audio_source = st.radio("Audio Source", ["Microphone 🎤", "Upload Audio File 📁"], horizontal=True, key="audio_source_select")
        with stt_col2:
            stt_labels = list(config.STT_MODEL_OPTIONS.keys())
            inline_stt = st.selectbox("Switch STT Model", stt_labels, index=stt_labels.index(current_stt_label), key="inline_stt_select")
            if inline_stt != current_stt_label:
                st.session_state.stt_model_label = inline_stt
                st.session_state.stt_model_key = config.STT_MODEL_OPTIONS[inline_stt]
                st.rerun()

        audio_buffer = None
        if audio_source == "Microphone 🎤":
            audio_buffer = st.audio_input("Record doctor-patient voice note", key="mic_input")
        else:
            audio_buffer = st.file_uploader("Upload audio file (.wav, .mp3, .m4a, .ogg)", type=["wav", "mp3", "m4a", "ogg"], key="file_audio_input")

        if audio_buffer is not None:
            if hasattr(audio_buffer, "getvalue"):
                audio_bytes = audio_buffer.getvalue()
            else:
                if hasattr(audio_buffer, "seek"):
                    audio_buffer.seek(0)
                audio_bytes = audio_buffer.read()
                if hasattr(audio_buffer, "seek"):
                    audio_buffer.seek(0)

            if audio_bytes and len(audio_bytes) > 0:
                import hashlib
                audio_hash = hashlib.md5(audio_bytes).hexdigest() + "_" + str(current_stt_key)

                # Only transcribe when this audio clip + STT model combo has not been transcribed yet
                if st.session_state.get("last_transcribed_hash") != audio_hash:
                    with st.spinner(f"Transcribing audio using {current_stt_label}..."):
                        try:
                            import transcriber
                            transcribed_text = transcriber.transcribe_audio(audio_bytes, model_key=current_stt_key)
                            st.session_state.voice_transcript = transcribed_text
                            st.session_state.last_transcribed_hash = audio_hash
                            st.session_state.active_audio_bytes = audio_bytes
                            st.session_state["voice_query_input"] = transcribed_text
                            if transcribed_text:
                                st.success(f"✅ Audio Transcribed Successfully with {current_stt_label}!")
                            else:
                                st.warning("⚠️ No speech recognized in audio sample.")
                        except Exception as ex:
                            st.error(f"Speech transcription error with {current_stt_label}: {ex}")
                            st.session_state.voice_transcript = ""
                            st.session_state["voice_query_input"] = ""

                # Allow user to manually force re-transcription with another STT engine if desired
                re_trans_col1, re_trans_col2 = st.columns([4, 1])
                with re_trans_col2:
                    if st.button("🔄 Re-Transcribe", help="Force re-transcribe this audio with the selected STT model", use_container_width=True):
                        st.session_state.pop("last_transcribed_hash", None)
                        st.rerun()

            query = st.text_area(f"Transcribed Text ({current_stt_label}) — Edit if needed", value=st.session_state.get("voice_transcript", ""), height=100, key="voice_query_input")
        else:
            st.info(f"Record audio or upload an audio file above to transcribe using {current_stt_label}.")

    run_clicked = st.button("Run")

    if run_clicked and query.strip():
        progress_bar = st.progress(0, text="[1/5] Supervisor Agent initializing prescription graph...")
        time.sleep(0.1)

        progress_bar.progress(20, text="[2/5] Retrieving system context & initializing agents...")
        t_ret_start = time.perf_counter()
        retrieved_prompt = vectorstore.retrieve_system_prompt(store)
        t_ret_end = time.perf_counter()
        retrieval_time = round(t_ret_end - t_ret_start, 4)

        progress_bar.progress(
            45, text=f"[3/5] Dispatching parallel extractors (Medicine/Strength, Route, Duration/Frequency, Instructions)..."
        )
        time.sleep(0.1)

        progress_bar.progress(
            75, text=f"[4/5] Aggregating extractions & validating groundedness (anti-hallucination check)..."
        )
        output, generation_time, agent_logs, aggregated_blocks = pipeline.run_agentic_pipeline(chat, query.strip())

        progress_bar.progress(90, text="[5/5] Saving validated output to SQLite & exporting to CSV/XLSX...")

        # Save voice audio file to data/audio_files/ if recorded/uploaded
        audio_file_path = None
        if input_mode != "Text Input 📝" and st.session_state.get("active_audio_bytes"):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file_name = f"proc_{st.session_state.process_id}_{stamp}.wav"
            audio_file_path = os.path.join(config.AUDIO_DIR, audio_file_name)
            try:
                os.makedirs(config.AUDIO_DIR, exist_ok=True)
                with open(audio_file_path, "wb") as f_aud:
                    f_aud.write(st.session_state.active_audio_bytes)
            except Exception:
                audio_file_path = None

        db.add_history(st.session_state.process_id, query.strip(), output, generation_time, audio_path=audio_file_path)
        exporter.append_generation(
            st.session_state.csv_path,
            st.session_state.xlsx_path,
            query.strip(),
            output,
            generation_time,
            llm_model_used=active_model_label,
        )

        progress_bar.progress(100, text="✅ LangGraph Multi-Agent extraction & export complete!")

        st.session_state.last_gen_time = generation_time
        st.session_state.last_ret_time = retrieval_time
        st.session_state.last_agent_logs = agent_logs

        # Clear voice session cache after successful generation run
        for k in ("voice_transcript", "last_audio_hash", "active_audio_bytes", "voice_query_input"):
            st.session_state.pop(k, None)
        st.rerun()
else:
    st.info("ℹ️ This thread is stopped. You can view its generation history and download CSV/XLSX below, or start a new process in the sidebar.")

if "last_gen_time" in st.session_state:
    st.success(
        f"⏱️ **Latest Multi-Agent Timing ({active_model_label})**: "
        f"Graph Execution & Validation: **{st.session_state.last_gen_time:.3f}s** | "
        f"FAISS Retrieval: **{st.session_state.last_ret_time:.4f}s**"
    )

if st.session_state.get("last_agent_logs"):
    with st.expander("🤖 **LangGraph Multi-Agent Audit Log & Validator Trace**", expanded=False):
        for log in st.session_state.last_agent_logs:
            agent_name = log.get("agent", "Agent")
            action = log.get("action") or log.get("status") or ""
            feedback = log.get("feedback")
            st.markdown(f"- **{agent_name}**: `{action}`")
            if feedback and feedback != "All checks passed (Grounded).":
                st.caption(f"  *Feedback:* {feedback}")

st.subheader(f"📊 Generation History for Thread #{st.session_state.process_id}")
history = db.get_history(st.session_state.process_id)

if history:
    if os.path.exists(st.session_state.csv_path):
        df = pd.read_csv(st.session_state.csv_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        rows = []
        for r in history:
            parsed_items = exporter.parse_output_fields(r["output"], query=r["query"])
            created_str = str(r["created_at"])
            if "T" in created_str:
                d_val, t_val = created_str.split("T")[0], created_str.split("T")[1].split(".")[0]
            elif " " in created_str:
                d_val, t_val = created_str.split(" ")[0], created_str.split(" ")[1].split(".")[0]
            else:
                d_val, t_val = created_str[:10], created_str[11:19]

            for parsed in parsed_items:
                rows.append({
                    "date": d_val,
                    "time": t_val,
                    "query": r["query"],
                    "Drug_name": parsed["Drug_name"],
                    "strength": parsed["strength"],
                    "frequency": parsed["frequency"],
                    "duration": parsed["duration"],
                    "route": parsed["route"],
                    "instruction": parsed["instruction"],
                    "additional_instruction": parsed.get("additional_instruction", "NONE"),
                    "llm_model_used": active_model_label,
                    "generation_time_sec": r.get("generation_time") or "N/A",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    dl1, dl2 = st.columns(2)
    if os.path.exists(st.session_state.csv_path):
        with open(st.session_state.csv_path, "rb") as f:
            dl1.download_button(
                "📥 Download Thread CSV",
                f,
                file_name=os.path.basename(st.session_state.csv_path),
                use_container_width=True,
            )
    if os.path.exists(st.session_state.xlsx_path):
        with open(st.session_state.xlsx_path, "rb") as f:
            dl2.download_button(
                "📥 Download Thread XLSX",
                f,
                file_name=os.path.basename(st.session_state.xlsx_path),
                use_container_width=True,
            )
else:
    st.caption("No generations yet for this thread.")
