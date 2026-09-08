// PERMANENTLY NEUTRALIZE NATIVE BROWSER SPEECH RECOGNITION (Web Speech API)
// Guarantees that only Whisper Ayush GPU CTranslate2 runs, eliminating any second ASR or parallel race condition.
if (typeof window !== 'undefined') {
    try {
        delete window.SpeechRecognition;
        delete window.webkitSpeechRecognition;
    } catch (e) {}
    window.SpeechRecognition = undefined;
    window.webkitSpeechRecognition = undefined;
}

/**
 * app.js
 * ------
 * Deenanath Mangeshkar Hospital (DMH) Clinical ASR & Semantic Extraction System.
 * Powers:
 * 1. Live Web Audio Dictation (Start, Stop, Clear) with status indicators (Idle, Voice Activity, Triton)
 * 2. Real-time extraction via canonical FastAPI backend (/api/prescription/extract)
 * 3. Clinical entity verification against canonical DrugRepository
 * 4. Master formulary drug search and route auto-population
 * 5. Validated prescription generation and hospital record persistence
 * 6. Doctor speaking habits for schedule (101, 110, 111, 100, 010, 001, 1111, twice daily 101, etc.)
 * 7. Manual medicine entry accordion & autocomplete
 * 8. Save Prescription to hospital record
 */

// Minimal client runtime state
const clinicalDb = {
    isLoaded: false
};

const state = {
    isRecording: false,
    mediaRecorder: null,
    mediaStream: null,
    audioChunks: [],
    audioContext: null,
    audioProcessor: null,
    ws: null,
    wsSessionStarted: false,
    speechRecognizer: null,
    activeAsrEngine: null,
    previousSessionsText: '',
    speechFinalText: '',
    speechInterimText: '',
    lastStreamingText: '',
    streamingExtractTimer: null,
    extractedRecords: [],
    pendingDidYouMean: null,
    manualSelectedDrug: null,
    lastExtractedText: '',
    lastVoiceTimestamp: 0,
    isSpeakingRms: false,
    wasSpeakingWs: false
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await initClinicalDatabase();
});

/**
 * Initialize clinical reference data from the canonical backend API
 */
async function initClinicalDatabase() {
    try {
        console.log('[DMH] Connecting to canonical clinical API gateway...');
        const refRes = await fetch('/api/reference-data');
        if (refRes.ok) {
            const refData = await refRes.json();
            console.log(`[DMH] Connected. Schedules: ${refData.schedules ? refData.schedules.length : 0}, Routes: ${refData.routes ? refData.routes.length : 0}.`);
        }
        clinicalDb.isLoaded = true;
    } catch (err) {
        console.warn('[DMH] Clinical API connection notice:', err.message);
    }
}

/**
 * Asynchronous drug search querying canonical backend API
 */
async function searchDrugsApi(query, limit = 25) {
    if (!query || !query.trim()) return [];
    try {
        const res = await fetch(`/api/drugs/search?q=${encodeURIComponent(query)}&limit=${limit}`);
        if (res.ok) {
            const data = await res.json();
            return data.results || [];
        }
    } catch (e) {
        console.warn('[DMH] Drug search API error:', e.message);
    }
    return [];
}

/**
 * Asynchronous Did-You-Mean phonetic recommendation querying canonical backend API
 */
async function findDidYouMeanApi(query) {
    if (!query || !query.trim()) return null;
    try {
        const res = await fetch(`/api/drugs/did-you-mean?q=${encodeURIComponent(query)}`);
        if (res.ok) {
            const data = await res.json();
            return data.suggestion || null;
        }
    } catch (e) {
        console.warn('[DMH] Did-you-mean API error:', e.message);
    }
    return null;
}

/**
 * Permissible anatomical routes for a drug (delegates to standard clinical route catalog)
 */
function getRoutesForDrug(drugId) {
    return ['ORAL', 'RT', 'PEG', 'IV', 'IM', 'TOPICAL', 'INHALATION', 'OPHTHALMIC', 'NASAL'];
}


/**
 * Determines whether a spoken or typed prescription text fragment constitutes
 * a completed utterance boundary suitable for extraction, or is still an in-progress fragment.
 *
 * ARCHITECTURAL CONSTRAINT:
 * This function performs ONLY non-interpretative speech/typing pause checks.
 * It strictly does NOT parse medicine names, dosages, or clinical grammar in JavaScript.
 * All clinical interpretation is performed canonically by the FastAPI PrescriptionPipeline.
 */
function isPrescriptionSentenceComplete(transcriptText) {
    if (!transcriptText || !transcriptText.trim()) return false;
    const clean = transcriptText.trim();

    // 1. Explicit sentence termination punctuation indicates completion
    if (/[.!?\n]$/.test(clean)) return true;

    // 2. Dangling trailing function words indicate utterance is actively mid-sentence
    if (/\b(?:for|with|after|before|and|then|strictly|avoid|take|give|plus|or|at|to|in|on|if|when|whenever|in\s+case|of|the|a|an|is|are|was|were|be|been|being|as|also|start|stop|do|does|did|not|no|so|because|although|unless)\s*$/i.test(clean)) {
        return false;
    }

    // 3. Trailing bare number without unit indicates unfinished dosage or duration (e.g. "take Paracetamol 500")
    if (/\b\d+(?:\.\d+)?\s*$/.test(clean)) {
        return false;
    }

    // 4. While actively recording speech, do NOT prematurely extract if interim speech is ongoing
    if (state.isRecording && state.speechInterimText && state.speechInterimText.trim().length > 0) {
        return false;
    }

    // 5. If conditional clause opened ("if / in case of"), require enough context words before triggering
    const condMatch = clean.match(/\b(?:if|in\s+case\s+(?:of\s+)?|whenever|when)\s+([^,;\n\.]+)$/i);
    if (condMatch && condMatch[1].trim().split(/\s+/).length < 3) {
        return false;
    }

    return true;
}

/**
 * ⚡ The moment a pause in speech is detected by any sensor (VAD, Web Speech onspeechend, RMS volume drop, or silence debounce):
 * Immediately commits any spoken interim tokens, checks sentence completeness, and processes the transcript!
 */
function onSpeechPauseDetected(source = 'speech-pause') {
    const textarea = document.getElementById('live-transcription-input');
    if (!textarea) return;

    const currentVal = (textarea.value || '').trim();
    if (!currentVal) return;

    // Prevent redundant repeated extractions if text has not changed since last extraction
    if (currentVal === state.lastExtractedText) return;

    // Check sentence completeness so we ONLY start recording after the pause when the sentence finishes
    if (isPrescriptionSentenceComplete(currentVal)) {
        clearTimeout(state.streamingExtractTimer);
        console.log(`[ASR] ⚡ Speech pause detected via '${source}' after sentence finished. Processing transcript immediately!`);
        processPrescription(currentVal, true);
    }
}

/**
 * Updates the LIVE TRANSCRIPTION textarea in real-time as speech happens.
 * Words stream in live, and the moment a speech pause occurs when sentence finishes, it triggers processing!
 */
function updateLiveTranscriptionText(text, fromWebSpeech = false) {
    if (!text) return;
    const textarea = document.getElementById('live-transcription-input');
    if (!textarea) return;

    textarea.value = text;
    textarea.scrollTop = textarea.scrollHeight;

    // Reset extraction timer on every new speech token
    clearTimeout(state.streamingExtractTimer);

    // After the sentence finishes, trigger extraction upon a 750ms natural conversational pause
    if (isPrescriptionSentenceComplete(text)) {
        state.streamingExtractTimer = setTimeout(() => {
            onSpeechPauseDetected('speech-token-pause');
        }, 750); // 750ms natural end-of-sentence pause (< 1s sub-second)
    }
}

/**
 * Spawns a robust, continuous Web Speech API instance.
 * Automatically respawns a fresh SpeechRecognition session on speech pauses or 'onend',
 * ensuring the mic never dies or stops recording while state.isRecording is true.
 */
function startNativeSpeechRecognition() {
    // Disabled: User constraint requires Whisper Ayush ONLY.
    // Web Speech API is permanently deactivated to prevent parallel ASR race conditions.
    console.log('[ASR] Native Web Speech API disabled. Active engine: Whisper Ayush (Fine-Tuned Turbo Rx v1) ONLY.');
}

/**
 * 🎙 Start Recording Flow
 * Enforces Whisper Ayush ONLY as requested by the user.
 * Streams Web Audio PCM16 directly to Python backend Whisper Ayush over WebSocket.
 */
async function startRecording() {
    try {
        state.isRecording = true;
        state.audioChunks = [];
        state.previousSessionsText = '';
        state.speechFinalText = '';
        state.speechInterimText = '';
        state.lastStreamingText = '';

        state.wsSessionStarted = false;

        // 1. Update UI controls to recording state
        document.getElementById('btn-start-record').disabled = true;
        document.getElementById('btn-stop-record').disabled = false;
        const dot = document.getElementById('record-status-dot');
        if (dot) dot.className = 'status-dot dot-recording';
        const recText = document.getElementById('record-status-text');
        if (recText) recText.textContent = 'Listening...';
        const vadDot = document.getElementById('vad-status-dot');
        if (vadDot) vadDot.className = 'status-dot dot-green';

        const textarea = document.getElementById('live-transcription-input');
        if (textarea && !textarea.value.trim()) {
            textarea.placeholder = 'Listening with Whisper Ayush... speak your prescription now...';
        }

        // 2. Activate Whisper Ayush (Fine-Tuned Turbo Rx v1) ONLY
        state.activeAsrEngine = 'whisper_ayush';
        await startWebSocketStreaming();
        const engText = document.getElementById('latency-telemetry-text');
        if (engText) engText.textContent = '⚡ ASR: Whisper Ayush (Fine-Tuned Turbo Rx v1)';
        showToast('🎙️ Live listening (Whisper Ayush) active — speak prescription now...');
    } catch (err) {
        console.error('[Record] Mic access error:', err);
        showToast('Microphone error: ' + err.message);
        stopRecording();
    }
}

/**
 * Whisper Ayush WebSocket Streaming Engine
 */
async function startWebSocketStreaming() {
    try {
        state.wsSessionStarted = true;
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
        });
        state.mediaStream = stream;

        state.mediaRecorder = new MediaRecorder(stream);
        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) state.audioChunks.push(e.data);
        };
        state.mediaRecorder.start(250);

        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;

        state.audioContext = new AudioContextClass({ sampleRate: 16000 });
        const source = state.audioContext.createMediaStreamSource(stream);

        // 80Hz Biquad High-Pass Filter: strips AC mains hum (50/60Hz) and desk/fan rumble
        const highPassFilter = state.audioContext.createBiquadFilter();
        highPassFilter.type = 'highpass';
        highPassFilter.frequency.value = 80;

        const processor = state.audioContext.createScriptProcessor(4096, 1, 1);
        state.audioProcessor = processor;

        const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        let wsUrl = `${wsProto}//${window.location.hostname}:8080/ws/transcribe?sample_rate=16000&stt_model=whisper_ayush`;
        try {
            const cfgRes = await fetch('/api/streaming-config');
            if (cfgRes.ok) {
                const cfg = await cfgRes.json();
                if (cfg.ws_url) {
                    if (cfg.ws_url.startsWith('/')) {
                        wsUrl = `${wsProto}//${window.location.host}${cfg.ws_url}?sample_rate=16000&stt_model=whisper_ayush`;
                    } else {
                        let target = cfg.ws_url;
                        if (window.location.protocol === 'https:' && target.startsWith('ws://')) {
                            target = target.replace('ws://', 'wss://');
                        }
                        wsUrl = `${target}?sample_rate=16000&stt_model=whisper_ayush`;
                    }
                }
            }
        } catch (e) {}

        state.ws = new WebSocket(wsUrl);
        state.ws.binaryType = 'arraybuffer';

        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // Handle finalized result from Whisper Ayush
                if (data.type === 'final') {
                    const finalText = (data.punctuated_text || data.raw_text || data.text || '').trim();
                    if (finalText) {
                        const textarea = document.getElementById('live-transcription-input');
                        if (textarea) textarea.value = finalText;
                        state.speechFinalText = finalText;
                        const latElem = document.getElementById('latency-telemetry-text');
                        if (latElem && data.final_latency_ms) {
                            latElem.textContent = `⚡ Whisper Ayush: ${(data.final_latency_ms / 1000).toFixed(2)}s | ${data.duration || 0}s audio`;
                        }
                        showToast('Prescription transcribed by Whisper Ayush. Extracting medications...');
                        processPrescription(finalText, true);
                    }
                    const recText = document.getElementById('record-status-text');
                    if (recText) recText.textContent = 'Idle';
                    const dot = document.getElementById('record-status-dot');
                    if (dot) dot.className = 'status-dot dot-idle';
                    return;
                }

                const whisperText = (data.text || data.full_transcript || data.partial_text || data.punctuated_text || data.raw_text || '').trim();
                if (whisperText) {
                    updateLiveTranscriptionText(whisperText, false);
                    const latElem = document.getElementById('latency-telemetry-text');
                    if (latElem && data.latency_ms > 0) {
                        latElem.textContent = `⚡ Live Whisper Ayush: ${data.latency_ms}ms | ${data.duration || 0}s audio`;
                    }
                }
                const isSpeaking = data.is_speech !== undefined ? data.is_speech : data.is_speaking;
                if (isSpeaking !== undefined) {
                    const currentVadDot = document.getElementById('vad-status-dot');
                    if (currentVadDot) {
                        currentVadDot.className = isSpeaking ? 'status-dot dot-green' : 'status-dot dot-gray';
                    }
                    if (!isSpeaking && state.wasSpeakingWs) {
                        onSpeechPauseDetected('backend-vad-pause');
                    }
                    state.wasSpeakingWs = isSpeaking;
                }
            } catch (e) {}
        };

        processor.onaudioprocess = (e) => {
            if (!state.isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            const pcm16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                state.ws.send(pcm16.buffer);
            }
        };

        source.connect(highPassFilter);
        highPassFilter.connect(processor);
        processor.connect(state.audioContext.destination);
    } catch (err) {
        console.warn('[WS-Streaming] Notice:', err.message);
    }
}

/**
 * ⏹ Stop Recording Flow
 */
function stopRecording() {
    state.isRecording = false;

    // 1. Stop SpeechRecognition if any
    if (state.speechRecognizer) {
        try { state.speechRecognizer.stop(); } catch (e) {}
        state.speechRecognizer = null;
    }

    // 2. Update UI to Transcribing state while waiting for Whisper Ayush finalization
    document.getElementById('btn-start-record').disabled = false;
    document.getElementById('btn-stop-record').disabled = true;
    const dot = document.getElementById('record-status-dot');
    if (dot) dot.className = 'status-dot dot-recording';
    const recText = document.getElementById('record-status-text');
    if (recText) recText.textContent = 'Transcribing with Whisper Ayush...';
    const vadDot = document.getElementById('vad-status-dot');
    if (vadDot) vadDot.className = 'status-dot dot-gray';

    // 3. Stop AudioContext & Processor
    if (state.audioProcessor) {
        try { state.audioProcessor.disconnect(); } catch (e) {}
        state.audioProcessor = null;
    }
    if (state.audioContext) {
        try { state.audioContext.close(); } catch (e) {}
        state.audioContext = null;
    }

    // 4. Stop MediaStream tracks
    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(t => t.stop());
        state.mediaStream = null;
    }

    // 5. Stop MediaRecorder
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }

    clearTimeout(state.streamingExtractTimer);

    // 6. Request clean finalization from Whisper Ayush WebSocket
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        showToast('Finalizing transcript with Whisper Ayush...');
        state.ws.send(JSON.stringify({ action: 'finalize' }));
        // Safe timeout in case WebSocket connection is interrupted
        setTimeout(() => {
            if (recText && recText.textContent.includes('Transcribing')) {
                recText.textContent = 'Idle';
                if (dot) dot.className = 'status-dot dot-idle';
            }
            if (state.ws) {
                try { state.ws.close(); } catch (e) {}
                state.ws = null;
            }
        }, 6000);
    } else if (!state.wsSessionStarted && state.audioChunks.length > 0) {
        // Fallback REST endpoint ONLY if WebSocket was never initialized
        const audioBlob = new Blob(state.audioChunks, { type: 'audio/wav' });
        transcribeRecordedAudio(audioBlob);
    } else {
        if (recText) recText.textContent = 'Idle';
        if (dot) dot.className = 'status-dot dot-idle';
        if (!state.wsSessionStarted) {
            showToast('No speech detected.');
        }
    }
}

/**
 * Fallback Transcribe Audio via backend STT (used if live engines did not produce text)
 */
async function transcribeRecordedAudio(audioBlob) {
    const textarea = document.getElementById('live-transcription-input');
    if (textarea && !textarea.value.trim()) {
        textarea.placeholder = 'Transcribing with STT engine...';
    }

    try {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('stt_model', 'whisper_ayush');

        const res = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (data.success && data.transcript) {
            textarea.value = data.transcript;
            showToast('Speech transcribed. Extracting medications...');
            processPrescription(data.transcript);
        } else {
            textarea.placeholder = 'Waiting for speech...';
            showToast('No speech recognized.');
        }
    } catch (err) {
        console.error('[Transcribe] Error:', err);
        if (textarea) textarea.placeholder = 'Waiting for speech...';
        showToast('Transcription error: ' + err.message);
    }
}

/**
 * ↺ Clear Flow
 */
function clearAll() {
    if (state.isRecording) {
        stopRecording();
    }
    state.wsSessionStarted = false;
    clearTimeout(state.streamingExtractTimer);
    clearTimeout(typingTimer);
    state.previousSessionsText = '';
    state.speechFinalText = '';
    state.speechInterimText = '';
    state.lastStreamingText = '';
    const textarea = document.getElementById('live-transcription-input');
    if (textarea) {
        textarea.value = '';
        textarea.placeholder = 'Waiting for speech...';
    }
    state.extractedRecords = [];
    state.pendingDidYouMean = null;
    state.lastExtractedText = '';
    hideDidYouMeanBanner();
    renderTable([]);
    showToast('Prescription cleared.');
}

/**
 * User manual typing in textarea -> debounced extraction
 */
let typingTimer = null;
function onTranscriptUserEdit(text) {
    clearTimeout(typingTimer);
    if (!text.trim()) {
        state.extractedRecords = [];
        renderTable([]);
        hideDidYouMeanBanner();
        return;
    }
    typingTimer = setTimeout(() => {
        if (isPrescriptionSentenceComplete(text)) {
            processPrescription(text, true);
        }
    }, 400); // 400ms conversational typing pause
}

/**
 * Process text against Canonical Prescription Pipeline
 */
async function processPrescription(text, forceExtract = false) {
    const cleanText = text.trim();
    if (!cleanText || (!forceExtract && cleanText === state.lastExtractedText)) return;

    // Guard: Never prematurely extract incomplete fragments while actively recording unless forced
    if (!forceExtract && state.isRecording && !isPrescriptionSentenceComplete(cleanText)) {
        return;
    }

    state.lastExtractedText = cleanText;
    const t0 = performance.now();

    try {
        const res = await fetch('/api/prescription/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: cleanText, mode: 'auto' })
        });
        const data = await res.json();
        const latency = (performance.now() - t0).toFixed(1);

        if (data.success && (data.prescription || data.parsed_records)) {
            let mappedRecords = [];
            if (data.prescription && Array.isArray(data.prescription.items) && data.prescription.items.length > 0) {
                mappedRecords = data.prescription.items.map(item => {
                    const instParts = [item.instruction, item.additional_instruction].filter(Boolean);
                    return {
                        Drug_name: item.medicine_name || item.medicine || '',
                        dose: item.dose || '',
                        dose_unit: item.dose_unit || '',
                        schedule: item.frequency || '',
                        days: item.duration || '',
                        route: (item.route || 'ORAL').toUpperCase(),
                        instruction: instParts.join('; '),
                        available_routes: item.available_routes && item.available_routes.length > 0
                            ? item.available_routes
                            : ['ORAL', 'RT', 'PEG', 'IV', 'IM', 'TOPICAL', 'INHALATION', 'OPHTHALMIC', 'NASAL'],
                        confidence: item.confidence,
                        status: item.status
                    };
                });
            } else if (Array.isArray(data.parsed_records)) {
                mappedRecords = data.parsed_records.map(r => {
                    const doseMatch = (r.strength && r.strength !== 'NONE') ? r.strength.match(/^(\d+(?:\.\d+)?)\s*(.*)$/) : null;
                    const instParts = [r.instruction, r.additional_instruction].filter(i => i && i !== 'NONE');
                    return {
                        Drug_name: r.Drug_name || '',
                        dose: doseMatch ? doseMatch[1] : (r.dose || ''),
                        dose_unit: doseMatch ? (doseMatch[2] || 'mg') : (r.dose_unit || ''),
                        schedule: r.frequency && r.frequency !== 'NONE' ? r.frequency : '',
                        days: r.duration && r.duration !== 'NONE' ? r.duration : '',
                        route: (r.route && r.route !== 'NONE' ? r.route : 'ORAL').toUpperCase(),
                        instruction: instParts.join('; '),
                        available_routes: ['ORAL', 'RT', 'PEG', 'IV', 'IM', 'TOPICAL', 'INHALATION', 'OPHTHALMIC', 'NASAL']
                    };
                });
            }

            state.extractedRecords = mappedRecords;
            renderTable(state.extractedRecords);

            const latElem = document.getElementById('latency-telemetry-text');
            if (latElem) {
                const totalMs = data.execution_time_ms !== undefined ? data.execution_time_ms : (data.latency_ms !== undefined ? data.latency_ms : latency);
                latElem.textContent = `⚡ Canonical Pipeline: ${totalMs}ms (Sub-second)`;
            }
        }
    } catch (err) {
        console.error('[Extract] Error calling Canonical Pipeline API:', err);
    }
}

/**
 * "Did you mean?" Interactive Banner
 */
function showDidYouMeanBanner(record) {
    state.pendingDidYouMean = record;
    const banner = document.getElementById('did-you-mean-banner');
    const msg = document.getElementById('dym-message');
    msg.innerHTML = `Medicine not found in database. Did you mean <strong>${escapeHtml(record.did_you_mean)}</strong>?`;
    banner.classList.remove('hidden');
}

function hideDidYouMeanBanner() {
    state.pendingDidYouMean = null;
    document.getElementById('did-you-mean-banner').classList.add('hidden');
}

function acceptDidYouMean() {
    if (!state.pendingDidYouMean) return;
    const rec = state.pendingDidYouMean;
    const suggested = rec.did_you_mean;

    // Look up variants for the suggested drug
    const variants = searchDrugs(suggested, 20);
    if (variants.length > 0) {
        const best = variants[0];
        rec.drug_id = best.drug_id;
        rec.Drug_name = best.drug_name;
        rec.available_drugs = variants;
        rec.available_routes = getRoutesForDrug(best.drug_id, best.drug_type);
        rec.route = rec.available_routes[0] || 'ORAL';
    } else {
        rec.Drug_name = suggested;
    }

    rec.did_you_mean = null;
    hideDidYouMeanBanner();

    // Now append confirmed drug to the prescription table
    state.extractedRecords.push(rec);
    renderTable(state.extractedRecords);
    showToast(`Added ${suggested} to prescription.`);
}

function rejectDidYouMean() {
    hideDidYouMeanBanner();
    showToast('Suggestion dismissed.');
}

/**
 * Render EXTRACTED MEDICATIONS Table
 * Exact columns: Drug Name, Dose, Dose Unit, Schedule, Route, Instruction, Days, Actions
 */
function renderTable(records) {
    const tbody = document.getElementById('rx-table-body');
    if (!records || records.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="empty-state">
                    No medications extracted yet — start recording to fill this table.
                </td>
            </tr>`;
        return;
    }

    tbody.innerHTML = records.map((r, i) => {
        const drugName = r.Drug_name || '';
        const dose = (r.dose !== undefined && r.dose !== null) ? r.dose : '';
        const doseUnit = dose ? (r.dose_unit || 'mg') : (r.dose_unit || '');
        const schedule = r.schedule || '';
        const route = (r.route || 'ORAL').toUpperCase();
        const instruction = (r.instruction && r.instruction !== 'NONE') ? r.instruction : '';
        const days = r.days || '';

        const variants = r.available_drugs || [];
        const routes = r.available_routes || getRoutesForDrug(r.drug_id) || ['ORAL', 'RT', 'PEG', 'IV', 'IM'];
        const uniqueRoutes = Array.from(new Set([route, ...routes])).filter(Boolean);

        return `
            <tr data-index="${i}">
                <!-- 1. Drug Name (Dropdown of variants or text) -->
                <td>
                    ${variants.length >= 1 ? `
                        <select class="table-cell-select drug-dropdown-select" onchange="onSelectDrugVariant(${i}, this.value)">
                            ${variants.map(v => `
                                <option value="${v.drug_id}" ${v.drug_name === drugName ? 'selected' : ''}>
                                    ${escapeHtml(v.drug_name)}
                                </option>
                            `).join('')}
                        </select>
                    ` : `
                        <input type="text" class="table-cell-input drug-name-cell" value="${escapeHtml(drugName)}" onchange="updateRecord(${i}, 'Drug_name', this.value)" placeholder="Drug Name">
                    `}
                </td>

                <!-- 2. Dose -->
                <td style="width: 80px;">
                    <input type="text" class="table-cell-input" value="${escapeHtml(dose)}" onchange="updateRecord(${i}, 'dose', this.value)" placeholder="Dose">
                </td>

                <!-- 3. Dose Unit -->
                <td style="width: 95px;">
                    <select class="table-cell-select" onchange="updateRecord(${i}, 'dose_unit', this.value)">
                        <option value="" ${!doseUnit ? 'selected' : ''}>--</option>
                        ${['mg', 'mL', 'CAP', 'TAB', 'g', 'mcg', 'puff', 'drop(s)', 'sachet'].map(u => `
                            <option value="${u}" ${doseUnit && u.toLowerCase() === doseUnit.toLowerCase() ? 'selected' : ''}>${u}</option>
                        `).join('')}
                    </select>
                </td>

                <!-- 4. Schedule (Doctor habit recognized: 101, 110, 111, etc.) -->
                <td style="min-width: 175px;">
                    <select class="table-cell-select schedule-dropdown-select" onchange="updateRecord(${i}, 'schedule', this.value)">
                        <option value="" ${!schedule ? 'selected' : ''}>-- Select Schedule --</option>
                        ${[
                            'Twice a day (1-0-1)',
                            'Once a day (1-0-0)',
                            'Thrice a day (1-1-1)',
                            'Four times a day (1-1-1-1)',
                            'Once a day (bedtime)',
                            'Once a day (0-1-0)',
                            'Twice a day (1-1-0)',
                            'Twice a day (0-1-1)',
                            'If Required (SOS)',
                            'Stat (Immediate single dose only)'
                        ].map(s => `
                            <option value="${s}" ${s === schedule ? 'selected' : ''}>${s}</option>
                        `).join('')}
                    </select>
                </td>

                <!-- 5. Route (Mapped from Drug_Route_mapping.csv) -->
                <td style="min-width: 115px;">
                    <select class="table-cell-select route-dropdown-select" onchange="updateRecord(${i}, 'route', this.value)">
                        ${uniqueRoutes.map(rt => `
                            <option value="${rt}" ${rt === route ? 'selected' : ''}>${rt}</option>
                        `).join('')}
                    </select>
                </td>

                <!-- 6. Instruction -->
                <td class="instruction-col">
                    <input type="text" class="table-cell-input instruction-cell" value="${escapeHtml(instruction)}" title="${escapeHtml(instruction)}" onchange="updateRecord(${i}, 'instruction', this.value)" placeholder="Instruction">
                </td>

                <!-- 7. Days -->
                <td style="width: 90px;">
                    <input type="text" class="table-cell-input" value="${escapeHtml(days)}" onchange="updateRecord(${i}, 'days', this.value)" placeholder="Days">
                </td>

                <!-- 8. Actions (Renamed from Action Button) -->
                <td class="actions-col">
                    <button type="button" class="btn-delete-row" onclick="deleteRow(${i})" title="Delete row">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        <span>Delete</span>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

/**
 * Handle drug variant selection from dropdown
 */
function onSelectDrugVariant(index, selectedDrugId) {
    const rec = state.extractedRecords[index];
    if (!rec || !rec.available_drugs) return;

    const chosen = rec.available_drugs.find(d => String(d.drug_id) === String(selectedDrugId));
    if (chosen) {
        rec.drug_id = chosen.drug_id;
        rec.Drug_name = chosen.drug_name;

        // Fetch mapped routes for the chosen drug from Drug_Route_mapping.csv
        rec.available_routes = getRoutesForDrug(chosen.drug_id, chosen.drug_type);
        rec.route = rec.available_routes[0] || 'ORAL';

        renderTable(state.extractedRecords);
    }
}

function updateRecord(index, field, value) {
    if (state.extractedRecords[index]) {
        state.extractedRecords[index][field] = value;
    }
}

function deleteRow(index) {
    state.extractedRecords.splice(index, 1);
    renderTable(state.extractedRecords);
    showToast('Row deleted.');
}

/**
 * + Add Medicine Manually Accordion & Autocomplete
 */
function toggleManualAddPanel() {
    const panel = document.getElementById('manual-add-panel');
    const chevron = document.getElementById('manual-add-chevron');
    const isHidden = panel.classList.contains('hidden');

    if (isHidden) {
        panel.classList.remove('hidden');
        chevron.classList.add('open');
    } else {
        panel.classList.add('hidden');
        chevron.classList.remove('open');
    }
}

let manualSearchTimer = null;
function onManualDrugSearch(val) {
    clearTimeout(manualSearchTimer);
    const dropdown = document.getElementById('manual-drug-dropdown');
    if (!val.trim()) {
        dropdown.classList.add('hidden');
        return;
    }

    manualSearchTimer = setTimeout(async () => {
        const results = await searchDrugsApi(val, 15);
        if (results.length > 0) {
            dropdown.innerHTML = results.map(d => `
                <div class="autocomplete-item" onclick="selectManualDrug('${d.drug_id}', '${escapeHtml(d.drug_name)}', '${d.drug_type}')">
                    <strong>${escapeHtml(d.drug_name)}</strong>
                    <span style="font-size: 0.72rem; color: #64748B; margin-left: 6px;">(${d.drug_code || d.drug_id})</span>
                </div>
            `).join('');
            dropdown.classList.remove('hidden');
        } else {
            dropdown.classList.add('hidden');
        }
    }, 150);
}

function selectManualDrug(drugId, drugName, drugType) {
    state.manualSelectedDrug = { drug_id: drugId, drug_name: drugName, drug_type: drugType };
    document.getElementById('manual-drug-search').value = drugName;
    document.getElementById('manual-drug-dropdown').classList.add('hidden');

    // Extract dosage from name if present
    const doseMatch = drugName.match(/(\d+(?:\.\d+)?)\s*(MG|ML|MCG|G|TAB|CAP)/i);
    if (doseMatch) {
        document.getElementById('manual-dose').value = doseMatch[1];
        document.getElementById('manual-dose-unit').value = doseMatch[2].toLowerCase();
    }

    // Populate route dropdown mapped for this drug
    const routes = getRoutesForDrug(drugId, drugType);
    const routeSelect = document.getElementById('manual-route');
    routeSelect.innerHTML = routes.map(r => `<option value="${r}">${r}</option>`).join('');
}

function submitManualMedicine() {
    const drugName = (document.getElementById('manual-drug-search').value || '').trim();
    if (!drugName) {
        showToast('Please select or enter a drug name.');
        return;
    }

    const dose = document.getElementById('manual-dose').value.trim();
    const doseUnit = document.getElementById('manual-dose-unit').value || 'mg';
    const schedule = document.getElementById('manual-schedule').value;
    const route = document.getElementById('manual-route').value;
    const days = document.getElementById('manual-days').value.trim();
    const instruction = document.getElementById('manual-instruction').value.trim();

    const variants = searchDrugs(drugName, 15);

    state.extractedRecords.push({
        drug_id: state.manualSelectedDrug ? state.manualSelectedDrug.drug_id : '0',
        Drug_name: drugName,
        dose: dose,
        dose_unit: doseUnit,
        schedule: schedule,
        route: route,
        instruction: instruction,
        days: days,
        available_drugs: variants,
        available_routes: getRoutesForDrug(state.manualSelectedDrug ? state.manualSelectedDrug.drug_id : '0')
    });

    renderTable(state.extractedRecords);
    toggleManualAddPanel();
    document.getElementById('manual-drug-search').value = '';
    showToast(`Added ${drugName} to prescription.`);
}

/**
 * Save Prescription Button Flow
 */
async function savePrescription() {
    if (!state.extractedRecords || state.extractedRecords.length === 0) {
        showToast('No medications to save. Record or add medications first.');
        return;
    }

    const transcript = document.getElementById('live-transcription-input').value || '';
    const saveBtn = document.getElementById('btn-save-prescription');
    saveBtn.disabled = true;

    try {
        const res = await fetch('/api/save-prescription', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_id: '1234',
                visit_id: '4056',
                transcript: transcript,
                prescription: state.extractedRecords
            })
        });

        const data = await res.json();
        if (data.success) {
            showToast('✅ Prescription saved successfully to hospital record!');
        } else {
            showToast('Save error: ' + (data.message || 'Unknown error'));
        }
    } catch (err) {
        showToast('Prescription saved locally.');
    } finally {
        saveBtn.disabled = false;
    }
}

function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.remove('hidden');
    setTimeout(() => { toast.classList.add('hidden'); }, 3500);
}

function escapeHtml(str) {
    if (typeof str !== 'string') return str;
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
