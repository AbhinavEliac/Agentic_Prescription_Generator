/**
 * app.js
 * ------
 * Client Application Logic for RxAgent Studio Node.js Web App.
 * Handles live audio recording, waveform visualizer, STT engine switching,
 * LangGraph extraction, interactive table manipulation, and history synchronization.
 */

// Global State
const state = {
    currentView: 'studio',
    inputMode: 'text',
    audioSource: 'mic',
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    recordedBlob: null,
    uploadedFile: null,
    audioContext: null,
    analyser: null,
    animFrameId: null,
    timerInterval: null,
    recordSeconds: 0,
    extractedRecords: [],
    processId: null
};

// Preset Clinical Samples
const SAMPLES = {
    1: "Take one Cefpodoxime proxetil 200 mg tablet orally twice daily after meals for 7 days, and take one Levocetirizine 5 mg with Montelukast 10 mg tablet once daily at bedtime for 10 days. Take one Paracetamol 650 mg tablet up to three times daily after food for pain or fever, take one Pantoprazole 40 mg tablet once daily before breakfast for 7 days, and administer two sprays of Oxymetazoline 0.05% nasal spray into each nostril twice daily for a strict maximum of 3 days. Use saline nasal irrigations twice daily, perform steam inhalation, and seek reassessment if eye swelling or severe headaches develop.",
    2: "Inhale 2 puffs of Budesonide 200 mcg twice daily using your dry powder inhaler and rinse your mouth immediately. Take Levosalbutamol 100 mcg via an MDI spacer as needed for sudden breathlessness.",
    3: "Take parasita mode, tablets 500 mg, 3 times a day, till 7 days, if the fever does not go away, increase the dosage by 20 mgs.",
    4: "Apply Clotrimazole 1% cream twice daily for 14 days over the affected skin areas. Instill 2 drops of Moxifloxacin 0.5% into both eyes every 4 hours for 7 days."
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    checkApiStatus();
    loadModels();
    setupCanvas();
});

/**
 * Checks API Gateway connectivity
 */
async function checkApiStatus() {
    const badge = document.getElementById('api-status-badge');
    const badgeText = document.getElementById('api-status-text');
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.status === 'online') {
            badge.style.background = 'rgba(16, 185, 129, 0.1)';
            badge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            badge.style.color = 'var(--emerald)';
            badgeText.textContent = `Agent Gateway Active (Python :8000)`;
            if (data.active_process) {
                state.processId = data.active_process.process_id;
                document.getElementById('metric-process').innerHTML = `Process: <strong>${data.active_process.name}</strong>`;
            }
        }
    } catch (err) {
        badge.style.background = 'rgba(244, 63, 94, 0.1)';
        badge.style.borderColor = 'rgba(244, 63, 94, 0.3)';
        badge.style.color = 'var(--rose)';
        badgeText.textContent = `Connecting to Python backend...`;
    }
}

/**
 * Loads available LLM & STT models into dropdowns
 */
async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        if (data.llm_models && data.llm_models.length > 0) {
            const llmSelect = document.getElementById('llm-model-select');
            llmSelect.innerHTML = data.llm_models.map(m => `<option value="${m.label}" ${m.is_default ? 'selected' : ''}>${m.label}</option>`).join('');
        }
        if (data.stt_models && data.stt_models.length > 0) {
            const sttSelect = document.getElementById('stt-model-select');
            sttSelect.innerHTML = data.stt_models.map(m => `<option value="${m.key}" ${m.is_default ? 'selected' : ''}>${m.label}</option>`).join('');
        }
    } catch (e) {
        console.warn('Could not load models dynamically:', e);
    }
}

/**
 * View Switcher
 */
function switchView(viewName) {
    state.currentView = viewName;
    const studioTab = document.getElementById('tab-studio-btn');
    const historyTab = document.getElementById('tab-history-btn');
    const studioView = document.getElementById('view-studio');
    const historyView = document.getElementById('view-history');

    if (viewName === 'studio') {
        studioTab.classList.add('active');
        historyTab.classList.remove('active');
        studioView.classList.remove('hidden');
        historyView.classList.add('hidden');
    } else {
        historyTab.classList.add('active');
        studioTab.classList.remove('active');
        historyView.classList.remove('hidden');
        studioView.classList.add('hidden');
        loadHistory();
    }
}

/**
 * Input Mode Switcher
 */
function setInputMode(mode) {
    state.inputMode = mode;
    const textBtn = document.getElementById('mode-text-btn');
    const voiceBtn = document.getElementById('mode-voice-btn');
    const voiceSec = document.getElementById('voice-controls-section');

    if (mode === 'text') {
        textBtn.classList.add('active');
        voiceBtn.classList.remove('active');
        voiceSec.classList.add('hidden');
    } else {
        voiceBtn.classList.add('active');
        textBtn.classList.remove('active');
        voiceSec.classList.remove('hidden');
    }
}

/**
 * Audio Source Switcher
 */
function setAudioSource(source) {
    state.audioSource = source;
    const micBtn = document.getElementById('audio-mic-btn');
    const fileBtn = document.getElementById('audio-file-btn');
    const micBox = document.getElementById('mic-box');
    const fileBox = document.getElementById('file-box');

    if (source === 'mic') {
        micBtn.classList.add('active');
        fileBtn.classList.remove('active');
        micBox.classList.remove('hidden');
        fileBox.classList.add('hidden');
    } else {
        fileBtn.classList.add('active');
        micBtn.classList.remove('active');
        fileBox.classList.remove('hidden');
        micBox.classList.add('hidden');
    }
}

/**
 * Load Sample Prescription
 */
function loadSample(id) {
    const text = SAMPLES[id];
    if (text) {
        document.getElementById('rx-input-text').value = text;
        showToast('Sample prescription loaded!');
    }
}

/**
 * Waveform Canvas Setup
 */
function setupCanvas() {
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

/**
 * Live Microphone Audio Recording & Waveform Animation
 */
async function toggleRecording() {
    const btn = document.getElementById('record-toggle-btn');
    const btnText = document.getElementById('record-btn-text');
    const timerText = document.getElementById('record-timer');
    const preview = document.getElementById('audio-preview');

    if (!state.isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            state.audioChunks = [];
            state.mediaRecorder = new MediaRecorder(stream);

            // Web Audio Analyser for Waveform
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = state.audioContext.createMediaStreamSource(stream);
            state.analyser = state.audioContext.createAnalyser();
            state.analyser.fftSize = 256;
            source.connect(state.analyser);

            drawWaveform();

            state.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) state.audioChunks.push(e.data);
            };

            state.mediaRecorder.onstop = () => {
                state.recordedBlob = new Blob(state.audioChunks, { type: 'audio/wav' });
                preview.src = URL.createObjectURL(state.recordedBlob);
                preview.classList.remove('hidden');
                document.getElementById('transcribe-btn').disabled = false;
                stream.getTracks().forEach(track => track.stop());
                cancelAnimationFrame(state.animFrameId);
                setupCanvas();
            };

            state.mediaRecorder.start();
            state.isRecording = true;
            state.recordSeconds = 0;
            btn.classList.add('recording');
            btnText.textContent = 'Stop Recording';

            state.timerInterval = setInterval(() => {
                state.recordSeconds++;
                const mins = String(Math.floor(state.recordSeconds / 60)).padStart(2, '0');
                const secs = String(state.recordSeconds % 60).padStart(2, '0');
                timerText.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            showToast('Microphone access denied: ' + err.message);
        }
    } else {
        // Stop recording
        state.mediaRecorder.stop();
        state.isRecording = false;
        clearInterval(state.timerInterval);
        btn.classList.remove('recording');
        btnText.textContent = 'Record Again';
    }
}

function drawWaveform() {
    if (!state.isRecording) return;
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    const bufferLength = state.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    state.analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#06B6D4';
    ctx.beginPath();

    const sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    state.animFrameId = requestAnimationFrame(drawWaveform);
}

/**
 * Handle File Selection
 */
function handleFileSelected(e) {
    const file = e.target.files[0];
    if (file) {
        state.uploadedFile = file;
        document.getElementById('selected-file-name').textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        document.getElementById('transcribe-btn').disabled = false;
    }
}

/**
 * Transcribe Audio via Selected STT Engine
 */
async function transcribeAudio() {
    const sttModel = document.getElementById('stt-model-select').value;
    const statusMsg = document.getElementById('transcribe-status');
    const transcribeBtn = document.getElementById('transcribe-btn');

    let audioBlob = (state.audioSource === 'mic') ? state.recordedBlob : state.uploadedFile;
    if (!audioBlob) {
        showToast('Please record or select an audio file first.');
        return;
    }

    transcribeBtn.disabled = true;
    statusMsg.textContent = `Transcribing with ${sttModel}...`;

    try {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('stt_model', sttModel);

        const res = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (data.success && data.transcript) {
            document.getElementById('rx-input-text').value = data.transcript;
            statusMsg.textContent = `✅ Transcribed in ${data.transcription_time}s`;
            showToast('Audio transcribed successfully!');
        } else {
            statusMsg.textContent = '⚠️ No speech recognized';
        }
    } catch (err) {
        statusMsg.textContent = '❌ Transcription error';
        showToast('Error transcribing audio: ' + err.message);
    } finally {
        transcribeBtn.disabled = false;
    }
}

/**
 * Run LangGraph Multi-Agent Prescription Extraction
 */
async function runExtraction() {
    const text = document.getElementById('rx-input-text').value.trim();
    if (!text) {
        showToast('Please enter prescription text or transcribe audio first.');
        return;
    }

    const runBtn = document.getElementById('run-extract-btn');
    const pipeline = document.getElementById('progress-pipeline');
    const llmModel = document.getElementById('llm-model-select').value;
    const device = document.getElementById('device-select').value;

    runBtn.disabled = true;
    pipeline.classList.remove('hidden');
    animatePipeline();

    try {
        const res = await fetch('/api/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                llm_model: llmModel,
                device: device,
                process_id: state.processId
            })
        });

        const data = await res.json();
        if (data.success) {
            state.extractedRecords = data.parsed_records || [];
            renderTable(state.extractedRecords);
            document.getElementById('metric-meds').innerHTML = `<strong>${data.total_medicines}</strong> Medications`;
            document.getElementById('metric-latency').innerHTML = `<strong>${data.generation_time}s</strong> Latency`;
            showToast(`Extracted ${data.total_medicines} medicines successfully!`);
        } else {
            showToast('Extraction failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        showToast('API Gateway error: ' + err.message);
    } finally {
        runBtn.disabled = false;
        completePipeline();
    }
}

function animatePipeline() {
    const s1 = document.getElementById('step-supervisor');
    const s2 = document.getElementById('step-agents');
    const s3 = document.getElementById('step-validator');
    const s4 = document.getElementById('step-export');

    s1.className = 'pipeline-step active';
    s2.className = 'pipeline-step';
    s3.className = 'pipeline-step';
    s4.className = 'pipeline-step';

    setTimeout(() => { s1.className = 'pipeline-step completed'; s2.className = 'pipeline-step active'; }, 300);
    setTimeout(() => { s2.className = 'pipeline-step completed'; s3.className = 'pipeline-step active'; }, 700);
    setTimeout(() => { s3.className = 'pipeline-step completed'; s4.className = 'pipeline-step active'; }, 1100);
}

function completePipeline() {
    const s4 = document.getElementById('step-export');
    s4.className = 'pipeline-step completed';
    setTimeout(() => {
        document.getElementById('progress-pipeline').classList.add('hidden');
    }, 1500);
}

/**
 * Render Interactive 7-Column Clinical Table
 */
function renderTable(records) {
    const tbody = document.getElementById('rx-table-body');
    if (!records || records.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="empty-state">
                    <div class="empty-content">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        <p>No medication records found in text.</p>
                    </div>
                </td>
            </tr>`;
        return;
    }

    tbody.innerHTML = records.map((r, i) => {
        const route = (r.route || 'oral').toLowerCase();
        const routeClass = `route-${route}`;
        return `
            <tr data-index="${i}">
                <td>${i + 1}</td>
                <td><input type="text" class="table-input drug-name-cell" value="${escapeHtml(r.Drug_name || '')}" onchange="updateRecord(${i}, 'Drug_name', this.value)"></td>
                <td><input type="text" class="table-input" value="${escapeHtml(r.strength || 'NONE')}" onchange="updateRecord(${i}, 'strength', this.value)"></td>
                <td><input type="text" class="table-input" value="${escapeHtml(r.frequency || 'NONE')}" onchange="updateRecord(${i}, 'frequency', this.value)"></td>
                <td><input type="text" class="table-input" value="${escapeHtml(r.duration || 'NONE')}" onchange="updateRecord(${i}, 'duration', this.value)"></td>
                <td>
                    <select class="table-input route-badge ${routeClass}" onchange="updateRecord(${i}, 'route', this.value); this.className='table-input route-badge route-' + this.value.toLowerCase();">
                        <option value="oral" ${route === 'oral' ? 'selected' : ''}>ORAL</option>
                        <option value="nasal" ${route === 'nasal' ? 'selected' : ''}>NASAL</option>
                        <option value="topical" ${route === 'topical' ? 'selected' : ''}>TOPICAL</option>
                        <option value="ophthalmic" ${route === 'ophthalmic' ? 'selected' : ''}>OPHTHALMIC</option>
                        <option value="inhalation" ${route === 'inhalation' ? 'selected' : ''}>INHALATION</option>
                    </select>
                </td>
                <td><input type="text" class="table-input" value="${escapeHtml(r.instruction || 'NONE')}" onchange="updateRecord(${i}, 'instruction', this.value)"></td>
                <td><input type="text" class="table-input" value="${escapeHtml(r.additional_instruction || 'NONE')}" onchange="updateRecord(${i}, 'additional_instruction', this.value)"></td>
                <td class="actions-col">
                    <button class="btn-del-row" onclick="deleteRow(${i})" title="Delete row">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function updateRecord(index, field, value) {
    if (state.extractedRecords[index]) {
        state.extractedRecords[index][field] = value;
    }
}

function addNewRow() {
    state.extractedRecords.push({
        Drug_name: 'New Medicine',
        strength: 'NONE',
        frequency: 'once daily',
        duration: '5 days',
        route: 'oral',
        instruction: 'NONE',
        additional_instruction: 'NONE'
    });
    renderTable(state.extractedRecords);
    document.getElementById('metric-meds').innerHTML = `<strong>${state.extractedRecords.length}</strong> Medications`;
}

function deleteRow(index) {
    state.extractedRecords.splice(index, 1);
    renderTable(state.extractedRecords);
    document.getElementById('metric-meds').innerHTML = `<strong>${state.extractedRecords.length}</strong> Medications`;
}

/**
 * Export Helpers
 */
function exportToCSV() {
    if (!state.extractedRecords || state.extractedRecords.length === 0) {
        showToast('No records to export.');
        return;
    }
    const headers = ['Drug_name', 'strength', 'frequency', 'duration', 'route', 'instruction', 'additional_instruction'];
    const rows = state.extractedRecords.map(r => headers.map(h => `"${(r[h] || '').replace(/"/g, '""')}"`).join(','));
    const csvContent = [headers.join(','), ...rows].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `rx_extraction_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('CSV downloaded successfully.');
}

function copyTableJSON() {
    if (!state.extractedRecords || state.extractedRecords.length === 0) {
        showToast('No records to copy.');
        return;
    }
    navigator.clipboard.writeText(JSON.stringify(state.extractedRecords, null, 2));
    showToast('JSON copied to clipboard!');
}

/**
 * Load Prescription History from SQLite
 */
async function loadHistory() {
    const tbody = document.getElementById('history-table-body');
    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">Loading history records...</td></tr>';

    try {
        const res = await fetch('/api/history?limit=30');
        const data = await res.json();
        if (data.records && data.records.length > 0) {
            tbody.innerHTML = data.records.map(r => `
                <tr>
                    <td><strong>#${r.id}</strong></td>
                    <td>${r.timestamp || ''}</td>
                    <td>Process ${r.process_id}</td>
                    <td style="max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(r.input_text || '')}</td>
                    <td><strong>${(r.parsed_records || []).length}</strong> meds</td>
                    <td>${r.generation_time}s</td>
                    <td>
                        <button class="btn btn-sm btn-ghost" onclick="loadHistoryItem(${escapeHtml(JSON.stringify(r))})">Load to Studio</button>
                    </td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No prescription history found.</td></tr>';
        }
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-state">Failed to load history: ${e.message}</td></tr>`;
    }
}

function loadHistoryItem(item) {
    switchView('studio');
    document.getElementById('rx-input-text').value = item.input_text || '';
    state.extractedRecords = item.parsed_records || [];
    renderTable(state.extractedRecords);
    document.getElementById('metric-meds').innerHTML = `<strong>${state.extractedRecords.length}</strong> Medications`;
    document.getElementById('metric-latency').innerHTML = `<strong>${item.generation_time}s</strong> Latency`;
    showToast(`Loaded History Record #${item.id} into Studio.`);
}

function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.remove('hidden');
    setTimeout(() => { toast.classList.add('hidden'); }, 3000);
}

function escapeHtml(str) {
    if (typeof str !== 'string') return str;
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
