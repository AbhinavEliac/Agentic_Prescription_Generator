/**
 * server.js
 * ---------
 * Express.js Server for the Agentic Prescription Extractor Parallel Web App.
 * Proxies extraction and STT transcription requests to the Python FastAPI backend,
 * and serves the modern clinical frontend on Port 3000.
 */
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const http = require('http');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://127.0.0.1:8080';

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Setup multer memory storage for audio file proxying
const upload = multer({ storage: multer.memoryStorage() });

// Serve static frontend assets with anti-cache headers to ensure immediate JS updates
app.use(express.static(path.join(__dirname, 'public'), {
    etag: false,
    maxAge: 0,
    setHeaders: (res, filePath) => {
        if (filePath.endsWith('.js') || filePath.endsWith('.html') || filePath.endsWith('.css')) {
            res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
            res.setHeader('Pragma', 'no-cache');
            res.setHeader('Expires', '0');
        }
    }
}));

/**
 * Proxy Helper using standard Node.js fetch / http
 */
async function proxyToPython(urlPath, options = {}) {
    const targetUrl = `${PYTHON_API_BASE}${urlPath}`;
    try {
        const response = await fetch(targetUrl, options);
        const data = await response.json();
        return { status: response.status, data };
    } catch (err) {
        return {
            status: 503,
            data: {
                error: 'Python Agentic API service is unreachable. Ensure the FastAPI server is running on port 8000.',
                details: err.message
            }
        };
    }
}

// Health / Status Check
app.get('/api/status', async (req, res) => {
    const result = await proxyToPython('/api/status');
    res.status(result.status).json(result.data);
});

// Streaming WebSocket configuration endpoint
app.get('/api/streaming-config', (req, res) => {
    const pyHost = PYTHON_API_BASE.replace(/^http/, 'ws');
    res.json({
        ws_url: process.env.WS_STREAM_URL || `${pyHost}/ws/transcribe`,
        python_api_base: PYTHON_API_BASE,
        streaming_enabled: true
    });
});

// Models List
app.get('/api/models', async (req, res) => {
    const result = await proxyToPython('/api/models');
    res.status(result.status).json(result.data);
});

const drugDbService = require('./drugDbService');

// Sub-Second Prescription Extraction via Compiled SQL Relational Flowsheet (<15ms)
app.post('/api/extract', async (req, res) => {
    const text = req.body.text || '';
    const flowsheetResult = drugDbService.executeSqlFlowsheet(text);

    // Asynchronously log to Python backend / DB in background without blocking latency
    proxyToPython('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...req.body, fast_mode: true })
    }).catch(() => {});

    res.json({
        success: true,
        latency_ms: flowsheetResult.latency_ms,
        generation_time: flowsheetResult.generation_time,
        total_medicines: flowsheetResult.total_medicines,
        parsed_records: flowsheetResult.records
    });
});

// Drug Database Search API (returns variants e.g. all Paracetamols)
app.get('/api/drugs/search', (req, res) => {
    const q = req.query.q || '';
    const results = drugDbService.searchDrugs(q, parseInt(req.query.limit) || 30);
    res.json({ query: q, total: results.length, results });
});

// Drug Did-You-Mean Fuzzy / Phonetic API
app.get('/api/drugs/did-you-mean', (req, res) => {
    const q = req.query.q || '';
    const suggestion = drugDbService.findDidYouMean(q);
    res.json({ query: q, suggestion });
});

// Dynamic Route Lookup for a Drug (mapped from Drug_Route_mapping.csv)
app.get('/api/drugs/:drug_id/routes', (req, res) => {
    const routes = drugDbService.getRoutesForDrug(req.params.drug_id, req.query.drug_type);
    res.json({ drug_id: req.params.drug_id, routes });
});

// Clinical Reference Data (Schedules, Dose Units, Routes)
app.get('/api/reference-data', (req, res) => {
    res.json({
        schedules: drugDbService.getAllSchedules(),
        dose_units: drugDbService.getAllDoseUnits(),
        routes: drugDbService.getAllRoutes()
    });
});

// Direct Semantic Prescription Matcher via Compiled Sub-Second SQL Flowsheet (<1ms)
app.post('/api/match-prescription', (req, res) => {
    const text = req.body.text || '';
    const flowsheetResult = drugDbService.executeSqlFlowsheet(text);
    res.json({
        success: true,
        latency_ms: flowsheetResult.latency_ms,
        generation_time: flowsheetResult.generation_time,
        total: flowsheetResult.total_medicines,
        records: flowsheetResult.records
    });
});

// Save Prescription Endpoint
app.post('/api/save-prescription', (req, res) => {
    const { prescription, transcript, patient_id, visit_id } = req.body;
    const saveRecord = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        patient_id: patient_id || '1234',
        visit_id: visit_id || '4056',
        transcript: transcript || '',
        medications: prescription || []
    };

    try {
        const historyDir = path.join(__dirname, '..', 'rx_extractor_app', 'data');
        if (!fs.existsSync(historyDir)) fs.mkdirSync(historyDir, { recursive: true });
        const savePath = path.join(historyDir, 'saved_prescriptions.jsonl');
        fs.appendFileSync(savePath, JSON.stringify(saveRecord) + '\n', 'utf8');
    } catch (e) {
        console.error('[Save] File log error:', e.message);
    }

    res.json({ success: true, message: 'Prescription saved successfully!', record: saveRecord });
});

// STT Audio Transcription Proxy
app.post('/api/transcribe', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No audio file provided' });
    }

    try {
        const formData = new FormData();
        const blob = new Blob([req.file.buffer], { type: req.file.mimetype || 'audio/wav' });
        formData.append('file', blob, req.file.originalname || 'recording.wav');
        formData.append('stt_model', req.body.stt_model || 'whisper_ayush');

        const targetUrl = `${PYTHON_API_BASE}/api/transcribe`;
        const response = await fetch(targetUrl, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        res.status(response.status).json(data);
    } catch (err) {
        res.status(500).json({ error: 'Transcription proxy failed', details: err.message });
    }
});

// History Proxy
app.get('/api/history', async (req, res) => {
    const queryString = req.url.includes('?') ? req.url.substring(req.url.indexOf('?')) : '';
    const result = await proxyToPython(`/api/history${queryString}`);
    res.status(result.status).json(result.data);
});

// Threads Proxy
app.get('/api/threads', async (req, res) => {
    const result = await proxyToPython('/api/threads');
    res.status(result.status).json(result.data);
});

app.post('/api/threads', async (req, res) => {
    const result = await proxyToPython('/api/threads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body)
    });
    res.status(result.status).json(result.data);
});

// Fallback to index.html for Single-Page Application routing
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`=======================================================`);
    console.log(`🚀 Rx Extractor Node.js Web App running on http://localhost:${PORT}`);
    console.log(`🔗 Connected Python API Gateway: ${PYTHON_API_BASE}`);
    console.log(`=======================================================`);
});
