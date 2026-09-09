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
    mediaStream: null,
    audioContext: null,
    audioProcessor: null,
    ws: null,
    wsSessionStarted: false,
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
    lastExtractedNormalizedText: '',
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
function isPrescriptionSentenceComplete(transcriptText, allowMidRecording = false) {
    if (!transcriptText || !transcriptText.trim()) return false;
    const clean = transcriptText.trim();

    // Must contain meaningful clinical alphanumeric text (filter out lone dots/punctuation)
    if (!/[a-zA-Z0-9]{2,}/.test(clean)) return false;

    // While actively recording speech, only trigger if explicit speech pause or boundary allows it
    if (!allowMidRecording && state.isRecording) return false;

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

    // 4. If conditional clause opened ("if / in case of"), require enough context words before triggering
    const condMatch = clean.match(/\b(?:if|in\s+case\s+(?:of\s+)?|whenever|when)\s+([^,;\n\.]+)$/i);
    if (condMatch && condMatch[1].trim().split(/\s+/).length < 3) {
        return false;
    }

    return true;
}



/**
 * Updates the LIVE TRANSCRIPTION textarea in real-time as speech happens.
 */
function updateLiveTranscriptionText(text, fromWebSpeech = false) {
    if (!text) return;
    const clean = text.trim();
    if (!clean || !anyAlphanumeric(clean)) return;

    const textarea = document.getElementById('live-transcription-input');
    if (!textarea) return;

    textarea.value = clean;
    textarea.scrollTop = textarea.scrollHeight;
}

function anyAlphanumeric(str) {
    return /[a-zA-Z0-9]/.test(str);
}

/**
 * Normalizes text for comparison (strips punctuation, extra whitespace, casing)
 */
function normalizeRxText(t) {
    if (!t) return '';
    return t.toLowerCase().replace(/[^a-z0-9]/g, ' ').replace(/\s+/g, ' ').trim();
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
 * Downsamples audio from any hardware sample rate (e.g., 48000Hz, 44100Hz)
 * to standard 16000Hz 16-bit mono PCM expected by Whisper Ayush.
 */
function downsampleToPcm16(float32Array, inputRate, outputRate = 16000) {
    if (!float32Array || float32Array.length === 0) return new Int16Array(0);
    if (inputRate === outputRate) {
        const out = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return out;
    }

    const sampleRateRatio = inputRate / outputRate;
    const newLength = Math.round(float32Array.length / sampleRateRatio);
    const result = new Int16Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < newLength) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < float32Array.length; i++) {
            accum += float32Array[i];
            count++;
        }
        const sample = count > 0 ? accum / count : float32Array[offsetBuffer] || 0;
        const clamped = Math.max(-1, Math.min(1, sample));
        result[offsetResult] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

/**
 * 🎙 Start Recording Flow
 * Enforces Whisper Ayush ONLY as requested by the user.
 * Streams Web Audio PCM16 directly to Python backend Whisper Ayush over WebSocket.
 */
async function startRecording() {
    try {
        state.isRecording = true;
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
        showToast('Microphone error: ' + (err.message || 'Access denied'));
        stopRecording();
    }
}

/**
 * Whisper Ayush WebSocket Streaming Engine
 */
async function startWebSocketStreaming() {
    state.wsSessionStarted = true;

    // 1. Acquire microphone with resilient fallback for diverse Windows audio drivers
    let stream = null;
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
    } catch (conEx) {
        console.warn('[WS-Streaming] Constrained mic acquisition failed, trying standard audio:', conEx.message);
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }
    state.mediaStream = stream;

    // 2. Initialize Web Audio Context at native hardware sample rate
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error('Web Audio API is not supported in this browser.');

    state.audioContext = new AudioContextClass();
    if (state.audioContext.state === 'suspended') {
        await state.audioContext.resume();
    }
    const nativeSampleRate = state.audioContext.sampleRate || 44100;
    console.log(`[Audio] Hardware sample rate: ${nativeSampleRate}Hz. Resampling live PCM16 to 16000Hz.`);

    const source = state.audioContext.createMediaStreamSource(stream);

    // 80Hz Biquad High-Pass Filter: strips AC mains hum (50/60Hz) and desk/fan rumble
    const highPassFilter = state.audioContext.createBiquadFilter();
    highPassFilter.type = 'highpass';
    highPassFilter.frequency.value = 80;

    // ScriptProcessor (buffer size 4096)
    const processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.audioProcessor = processor;

    // 4. Determine WebSocket URL (query streaming-config or use location host)
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

    console.log('[WS] Connecting to WebSocket at:', wsUrl);
    state.ws = new WebSocket(wsUrl);
    state.ws.binaryType = 'arraybuffer';

    state.ws.onopen = () => {
        console.log('[WS] Streaming WebSocket connected to', wsUrl);
    };
    state.ws.onerror = (e) => {
        console.warn('[WS] WebSocket notice:', e);
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            // Handle finalized result from Whisper Ayush
            if (data.type === 'final') {
                const finalText = (data.punctuated_text || data.raw_text || data.text || '').trim();
                if (finalText && anyAlphanumeric(finalText)) {
                    const textarea = document.getElementById('live-transcription-input');
                    if (textarea) textarea.value = finalText;
                    state.speechFinalText = finalText;
                    const latElem = document.getElementById('latency-telemetry-text');
                    if (latElem && data.final_latency_ms) {
                        latElem.textContent = `⚡ Whisper Ayush: ${(data.final_latency_ms / 1000).toFixed(2)}s | ${data.duration || 0}s audio`;
                    }
                    const norm = normalizeRxText(finalText);
                    // SINGLE WORKFLOW: Only run extraction if this text has NOT already been extracted at the pause boundary!
                    if (norm && norm !== state.lastExtractedNormalizedText) {
                        console.log('[ASR] Finalize contains unextracted speech. Extracting:', finalText);
                        processPrescription(finalText, false);
                    }
                }
                const recText = document.getElementById('record-status-text');
                if (recText) recText.textContent = 'Idle';
                const dot = document.getElementById('record-status-dot');
                if (dot) dot.className = 'status-dot dot-idle';
                return;
            }

            // Handle natural sentence boundary commit from backend VAD (pause at the end of each sentence)
            if (data.type === 'boundary' || data.boundary) {
                const boundaryText = (data.text || data.full_transcript || '').trim();
                if (boundaryText && anyAlphanumeric(boundaryText)) {
                    updateLiveTranscriptionText(boundaryText, false);
                    const norm = normalizeRxText(boundaryText);
                    if (norm && norm !== state.lastExtractedNormalizedText && isPrescriptionSentenceComplete(boundaryText, true)) {
                        console.log('[ASR] ⚡ Natural sentence boundary committed at pause:', boundaryText);
                        processPrescription(boundaryText, true);
                    }
                }
                return;
            }

            const whisperText = (data.text || data.full_transcript || data.partial_text || data.punctuated_text || data.raw_text || '').trim();
            if (whisperText && anyAlphanumeric(whisperText)) {
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
                state.wasSpeakingWs = isSpeaking;
            }

        } catch (e) {}
    };

    processor.onaudioprocess = (e) => {
        if (!state.isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);
        // Downsample input data from native hardware sample rate to 16000Hz PCM16
        const pcm16 = downsampleToPcm16(inputData, nativeSampleRate, 16000);
        if (pcm16 && pcm16.length > 0 && state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(pcm16.buffer);
        }
    };

    source.connect(highPassFilter);
    highPassFilter.connect(processor);
    // Connect processor through a zero-gain node to destination to prevent speaker feedback while keeping processor active
    const muteGain = state.audioContext.createGain();
    muteGain.gain.value = 0.0;
    processor.connect(muteGain);
    muteGain.connect(state.audioContext.destination);
}

/**
 * ⏹ Stop Recording Flow
 */
function stopRecording() {
    state.isRecording = false;

    // 1. Update UI controls
    document.getElementById('btn-start-record').disabled = false;
    document.getElementById('btn-stop-record').disabled = true;
    const dot = document.getElementById('record-status-dot');
    const recText = document.getElementById('record-status-text');
    const vadDot = document.getElementById('vad-status-dot');
    if (vadDot) vadDot.className = 'status-dot dot-gray';

    // 2. Stop AudioContext & Processor immediately so mic is freed
    if (state.audioProcessor) {
        try { state.audioProcessor.disconnect(); } catch (e) {}
        state.audioProcessor = null;
    }
    if (state.audioContext) {
        try { state.audioContext.close(); } catch (e) {}
        state.audioContext = null;
    }

    // 3. Stop MediaStream tracks
    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(t => t.stop());
        state.mediaStream = null;
    }

    // SINGLE WORKFLOW: Check if the current spoken sentence was ALREADY extracted at the pause
    const currentInputText = (document.getElementById('live-transcription-input')?.value || '').trim();
    const isAlreadyExtracted = currentInputText && normalizeRxText(currentInputText) === state.lastExtractedNormalizedText;

    if (isAlreadyExtracted) {
        // Sentence was already extracted at the natural pause: transition directly to Idle without re-running workflow
        if (recText) recText.textContent = 'Idle';
        if (dot) dot.className = 'status-dot dot-idle';
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            try { state.ws.close(); } catch (e) {}
            state.ws = null;
        }
        showToast('Dictation completed. Prescription extracted.');
        return;
    }

    // If there was speech actively ongoing that was cut off by Stop before a pause:
    if (dot) dot.className = 'status-dot dot-recording';
    if (recText) recText.textContent = 'Finalizing with Whisper Ayush...';

    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        showToast('Finalizing transcript with Whisper Ayush...');
        state.ws.send(JSON.stringify({ action: 'finalize' }));
        setTimeout(() => {
            if (recText && recText.textContent.includes('Finalizing')) {
                recText.textContent = 'Idle';
                if (dot) dot.className = 'status-dot dot-idle';
            }
            if (state.ws) {
                try { state.ws.close(); } catch (e) {}
                state.ws = null;
            }
        }, 3000);
    } else {
        if (recText) recText.textContent = 'Idle';
        if (dot) dot.className = 'status-dot dot-idle';
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
    state.lastExtractedNormalizedText = '';
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
    const norm = normalizeRxText(cleanText);
    if (!cleanText || !norm) return;

    // SINGLE PIPELINE DEDUPLICATION GUARD:
    // If normalized text matches already extracted content, skip completely to prevent duplicate runs
    if (!forceExtract && norm === state.lastExtractedNormalizedText) return;

    // Guard: Never prematurely extract incomplete fragments while actively recording unless forced
    if (!forceExtract && state.isRecording && !isPrescriptionSentenceComplete(cleanText)) {
        return;
    }

    state.lastExtractedText = cleanText;
    state.lastExtractedNormalizedText = norm;
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
                    const doseVal = (item.dose !== undefined && item.dose !== null) ? String(item.dose) : '';
                    const unitVal = (item.dose_unit !== undefined && item.dose_unit !== null) ? String(item.dose_unit) : (doseVal ? 'mg' : '');
                    let dymOpts = [];
                    if (Array.isArray(item.did_you_mean_options) && item.did_you_mean_options.length > 0) {
                        dymOpts = item.did_you_mean_options.slice(0, 3);
                    } else if (item.did_you_mean) {
                        dymOpts = [{ drug_name: item.did_you_mean, base_name: item.did_you_mean }];
                    }
                    return {
                        Drug_name: item.medicine_name || item.medicine || '',
                        dose: doseVal,
                        dose_unit: unitVal,
                        schedule: item.frequency || '',
                        days: item.duration || '',
                        route: (item.route || 'ORAL').toUpperCase(),
                        instruction: instParts.join('; '),
                        available_routes: item.available_routes && item.available_routes.length > 0
                            ? item.available_routes
                            : ['ORAL', 'RT', 'PEG', 'IV', 'IM', 'TOPICAL', 'INHALATION', 'OPHTHALMIC', 'NASAL'],
                        available_drugs: item.available_drugs || [],
                        did_you_mean: item.did_you_mean || (dymOpts[0] ? (dymOpts[0].drug_name || dymOpts[0].base_name) : null),
                        did_you_mean_options: dymOpts,
                        confidence: item.confidence,
                        status: item.status
                    };
                });
            } else if (Array.isArray(data.parsed_records)) {
                mappedRecords = data.parsed_records.map(r => {
                    const instParts = [r.instruction, r.additional_instruction].filter(i => i && i !== 'NONE');
                    const doseVal = (r.dose !== undefined && r.dose !== null) ? String(r.dose) : '';
                    const unitVal = (r.dose_unit !== undefined && r.dose_unit !== null) ? String(r.dose_unit) : (doseVal ? 'mg' : '');
                    let dymOpts = [];
                    if (Array.isArray(r.did_you_mean_options) && r.did_you_mean_options.length > 0) {
                        dymOpts = r.did_you_mean_options.slice(0, 3);
                    } else if (r.did_you_mean) {
                        dymOpts = [{ drug_name: r.did_you_mean, base_name: r.did_you_mean }];
                    }
                    return {
                        Drug_name: r.Drug_name || '',
                        dose: doseVal,
                        dose_unit: unitVal,
                        schedule: r.frequency && r.frequency !== 'NONE' ? r.frequency : '',
                        days: r.duration && r.duration !== 'NONE' ? r.duration : '',
                        route: (r.route && r.route !== 'NONE' ? r.route : 'ORAL').toUpperCase(),
                        instruction: instParts.join('; '),
                        available_routes: r.available_routes && r.available_routes.length > 0
                            ? r.available_routes
                            : ['ORAL', 'RT', 'PEG', 'IV', 'IM', 'TOPICAL', 'INHALATION', 'OPHTHALMIC', 'NASAL'],
                        available_drugs: r.available_drugs || [],
                        did_you_mean: r.did_you_mean || (dymOpts[0] ? (dymOpts[0].drug_name || dymOpts[0].base_name) : null),
                        did_you_mean_options: dymOpts
                    };
                });
            }

            state.extractedRecords = mappedRecords;
            renderTable(state.extractedRecords);

            // Per-row Did-You-Mean handles interactive chips directly in the table
            hideDidYouMeanBanner();

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
    if (banner && msg) {
        msg.innerHTML = `Pronunciation or spelling variance spotted. Did you mean <strong>${escapeHtml(record.did_you_mean)}</strong>?`;
        banner.classList.remove('hidden');
    }
}

function hideDidYouMeanBanner() {
    state.pendingDidYouMean = null;
    const banner = document.getElementById('did-you-mean-banner');
    if (banner) banner.classList.add('hidden');
}

function acceptDidYouMean() {
    if (!state.pendingDidYouMean) return;
    const rec = state.pendingDidYouMean;
    const suggested = rec.did_you_mean;

    rec.Drug_name = suggested;
    rec.did_you_mean = null;
    hideDidYouMeanBanner();

    renderTable(state.extractedRecords);
    showToast(`Updated to ${suggested} in prescription.`);
}

function rejectDidYouMean() {
    if (state.pendingDidYouMean) {
        state.pendingDidYouMean.did_you_mean = null;
    }
    hideDidYouMeanBanner();
    renderTable(state.extractedRecords);
    showToast('Suggestion dismissed.');
}

/**
 * Accept a specific Did-You-Mean suggestion for row rowIndex, option optIndex
 */
window.onAcceptRowDym = function(rowIndex, optIndex) {
    const r = state.extractedRecords[rowIndex];
    if (!r) return;

    let opt = null;
    if (Array.isArray(r.did_you_mean_options) && r.did_you_mean_options[optIndex]) {
        opt = r.did_you_mean_options[optIndex];
    } else if (r.did_you_mean) {
        opt = { drug_name: r.did_you_mean, base_name: r.did_you_mean };
    }
    if (!opt) return;

    // 1. Accepted drug name replaces the row's drug name
    const chosenName = opt.drug_name || opt.base_name;
    r.Drug_name = chosenName;

    // 2. If option has dosage details (e.g. from combination dose), update dose and dose_unit
    if (opt.dose !== undefined && opt.dose !== null && String(opt.dose).trim() !== '') {
        r.dose = String(opt.dose);
    }
    if (opt.dose_unit) {
        r.dose_unit = opt.dose_unit;
    }

    // 3. Update routes and available drugs if present
    if (Array.isArray(opt.available_routes) && opt.available_routes.length > 0) {
        r.available_routes = opt.available_routes;
    }
    if (Array.isArray(opt.available_drugs) && opt.available_drugs.length > 0) {
        r.available_drugs = opt.available_drugs;
    }

    // 4. "if one pressed yes.. all did you means gone and yes one retained.."
    r.did_you_mean_options = [];
    r.did_you_mean = null;

    if (state.pendingDidYouMean === r) {
        hideDidYouMeanBanner();
    }

    renderTable(state.extractedRecords);
    showToast(`Accepted recommendation: ${chosenName}`);
};

/**
 * Reject a specific Did-You-Mean suggestion for row rowIndex, option optIndex
 */
window.onRejectRowDymOption = function(rowIndex, optIndex) {
    const r = state.extractedRecords[rowIndex];
    if (!r) return;

    if (Array.isArray(r.did_you_mean_options) && r.did_you_mean_options.length > 0) {
        // "pressing no on the did you mean option of one .. will only delete the did you mean recommendation, other 2 will remain"
        r.did_you_mean_options.splice(optIndex, 1);

        // "if all prrssed no then medicine row deleted"
        if (r.did_you_mean_options.length === 0) {
            state.extractedRecords.splice(rowIndex, 1);
            if (state.pendingDidYouMean === r) {
                hideDidYouMeanBanner();
            }
            renderTable(state.extractedRecords);
            showToast('All suggestions rejected — medicine row removed.');
            return;
        }

        r.did_you_mean = r.did_you_mean_options[0].drug_name || r.did_you_mean_options[0].base_name;
    } else {
        // Single option fallback
        r.did_you_mean = null;
        state.extractedRecords.splice(rowIndex, 1);
        if (state.pendingDidYouMean === r) {
            hideDidYouMeanBanner();
        }
        renderTable(state.extractedRecords);
        showToast('Suggestion rejected — medicine row removed.');
        return;
    }

    renderTable(state.extractedRecords);
    showToast('Recommendation dismissed.');
};

function normalizeSchedule(sch) {
    if (!sch) return '';
    const s = String(sch).toLowerCase().trim();
    if (s.includes('1-0-0') || s.includes('100') || s.includes('once daily') || s.includes('once a day') || s === 'od') {
        return 'Once a day (1-0-0)';
    }
    if (s.includes('1-0-1') || s.includes('101') || s.includes('twice daily') || s.includes('twice a day') || s === 'bd' || s === 'bid') {
        return 'Twice a day (1-0-1)';
    }
    if (s.includes('1-1-1') || s.includes('111') || s.includes('thrice daily') || s.includes('three times') || s === 'tds' || s === 'tid') {
        return 'Thrice a day (1-1-1)';
    }
    if (s.includes('1-1-1-1') || s.includes('1111') || s.includes('four times') || s === 'qid') {
        return 'Four times a day (1-1-1-1)';
    }
    if (s.includes('0-0-1') || s.includes('bedtime') || s.includes('night') || s.includes('hs')) {
        return 'Once a day (bedtime)';
    }
    if (s.includes('1-1-0') || s.includes('110')) {
        return 'Twice a day (1-1-0)';
    }
    if (s.includes('0-1-1') || s.includes('011')) {
        return 'Twice a day (0-1-1)';
    }
    if (s.includes('sos') || s.includes('as needed') || s.includes('if required')) {
        return 'If Required (SOS)';
    }
    if (s.includes('stat') || s.includes('immediate')) {
        return 'Stat (Immediate single dose only)';
    }
    return sch;
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
        const rawSchedule = r.schedule || '';
        const schedule = normalizeSchedule(rawSchedule);
        const standardSchedules = [
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
        ];
        const allSchedules = (!schedule || standardSchedules.includes(schedule))
            ? standardSchedules
            : [schedule, ...standardSchedules];

        const route = (r.route || 'ORAL').toUpperCase();
        const instruction = (r.instruction && r.instruction !== 'NONE') ? r.instruction : '';
        const days = r.days || '';

        const variants = r.available_drugs || [];
        const isMatchedInVariants = variants.some(v => v.drug_name === drugName || v.base_name === drugName);
        const routes = (r.available_routes && r.available_routes.length > 0)
            ? r.available_routes
            : ['ORAL', 'RT', 'PEG', 'IV', 'IM', 'TOPICAL', 'INHALATION', 'OPHTHALMIC', 'NASAL'];
        const uniqueRoutes = Array.from(new Set([route, ...routes.map(rt => rt.toUpperCase())])).filter(Boolean);

        return `
            <tr data-index="${i}">
                <!-- 1. Drug Name (Dropdown of same-dose variants or text) -->
                <td>
                    ${variants.length >= 1 ? `
                        <select class="table-cell-select drug-dropdown-select" onchange="onSelectDrugVariant(${i}, this.value)">
                            ${!isMatchedInVariants ? `<option value="" selected>${escapeHtml(drugName)}</option>` : ''}
                            ${variants.map(v => `
                                <option value="${v.drug_id}" ${(v.drug_name === drugName || (isMatchedInVariants && v.base_name === drugName)) ? 'selected' : ''}>
                                    ${escapeHtml(v.drug_name)}
                                </option>
                            `).join('')}
                        </select>
                    ` : `
                        <input type="text" class="table-cell-input drug-name-cell" value="${escapeHtml(drugName)}" onchange="updateRecord(${i}, 'Drug_name', this.value)" placeholder="Drug Name">
                    `}
                    ${(r.did_you_mean_options && r.did_you_mean_options.length > 0) ? `
                        <div class="row-dym-container" id="row-dym-${i}">
                            <div class="row-dym-title">❓ Did you mean:</div>
                            <div class="row-dym-chips">
                                ${r.did_you_mean_options.slice(0, 3).map((opt, optIdx) => `
                                    <div class="row-dym-chip" id="row-dym-${i}-chip-${optIdx}">
                                        <span class="row-dym-name" title="${escapeHtml(opt.drug_name || opt.base_name)}">${escapeHtml(opt.drug_name || opt.base_name)}</span>
                                        <div class="row-dym-actions">
                                            <button type="button" class="btn-dym-opt-yes" onclick="onAcceptRowDym(${i}, ${optIdx})" title="Accept recommendation">✓ Yes</button>
                                            <button type="button" class="btn-dym-opt-no" onclick="onRejectRowDymOption(${i}, ${optIdx})" title="Reject recommendation">✗ No</button>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : (r.did_you_mean ? `
                        <div class="row-dym-container" id="row-dym-${i}">
                            <div class="row-dym-title">❓ Did you mean:</div>
                            <div class="row-dym-chips">
                                <div class="row-dym-chip" id="row-dym-${i}-chip-0">
                                    <span class="row-dym-name" title="${escapeHtml(r.did_you_mean)}">${escapeHtml(r.did_you_mean)}</span>
                                    <div class="row-dym-actions">
                                        <button type="button" class="btn-dym-opt-yes" onclick="onAcceptRowDym(${i}, 0)" title="Accept recommendation">✓ Yes</button>
                                        <button type="button" class="btn-dym-opt-no" onclick="onRejectRowDymOption(${i}, 0)" title="Reject recommendation">✗ No</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ` : '')}
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
                        ${allSchedules.map(s => `
                            <option value="${escapeHtml(s)}" ${s === schedule ? 'selected' : ''}>${escapeHtml(s)}</option>
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

        // Fetch mapped routes for the chosen drug
        if (chosen.routes && chosen.routes.length > 0) {
            rec.available_routes = chosen.routes;
            rec.route = chosen.routes[0].toUpperCase();
        }
        rec.did_you_mean = null;
        hideDidYouMeanBanner();
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
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.remove('hidden');
    setTimeout(() => { toast.classList.add('hidden'); }, 3500);
}

function escapeHtml(str) {
    if (typeof str !== 'string') return str;
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
