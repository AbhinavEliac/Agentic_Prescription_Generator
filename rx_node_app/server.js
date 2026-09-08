/**
 * server.js
 * ---------
 * Express.js Server for the Agentic Prescription Extractor Parallel Web App.
 * Proxies extraction and STT transcription requests to the Python FastAPI backend,
 * and serves the modern clinical frontend on Port 5000.
 */
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const http = require('http');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://127.0.0.1:8080';

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Setup multer memory storage for audio file proxying with 25MB limit and audio type validation
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 25 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        const allowedExts = ['.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac'];
        const ext = path.extname(file.originalname || '').toLowerCase();
        if ((file.mimetype && file.mimetype.startsWith('audio/')) || (ext && allowedExts.includes(ext))) {
            return cb(null, true);
        }
        cb(new Error(`Unsupported media type. Only audio files (${allowedExts.join(', ')}) are accepted.`));
    }
});

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

const crypto = require('crypto');

/**
 * Proxy Helper using standard Node.js fetch / http with Request ID propagation
 */
async function proxyToPython(urlPath, options = {}, req = null) {
    const targetUrl = `${PYTHON_API_BASE}${urlPath}`;
    const requestId = (req && req.headers['x-request-id']) || crypto.randomUUID();
    
    const headers = Object.assign({}, options.headers || {}, {
        'X-Request-ID': requestId
    });

    try {
        const response = await fetch(targetUrl, { ...options, headers });
        const data = await response.json();
        const respRequestId = response.headers.get('x-request-id') || requestId;
        return { status: response.status, data, requestId: respRequestId };
    } catch (err) {
        return {
            status: 503,
            data: {
                success: false,
                error_code: 'FASTAPI_SERVICE_UNAVAILABLE',
                message: 'FastAPI clinical extraction backend is unreachable. Ensure the backend is running on port 8080.',
                details: [{ field: 'python_api_base', code: 'connection_refused', message: err.message }],
                request_id: requestId
            },
            requestId
        };
    }
}

// Canonical Health Probe
app.get('/api/health', async (req, res) => {
    const result = await proxyToPython('/api/health', {}, req);
    res.setHeader('X-Request-ID', result.requestId);
    res.status(result.status).json(result.data);
});

app.get('/api/status', async (req, res) => {
    const result = await proxyToPython('/api/status', {}, req);
    res.setHeader('X-Request-ID', result.requestId);
    res.status(result.status).json(result.data);
});

// Streaming WebSocket configuration endpoint
app.get('/api/streaming-config', (req, res) => {
    const clientHost = req.hostname && req.hostname !== 'localhost' && req.hostname !== '127.0.0.1' 
        ? req.hostname 
        : '127.0.0.1';
    const pyHost = PYTHON_API_BASE.replace(/^http/, 'ws').replace(/127\.0\.0\.1|localhost/, clientHost);

    res.json({
        ws_url: process.env.WS_STREAM_URL || `${pyHost}/ws/transcribe`,
        python_api_base: PYTHON_API_BASE,
        streaming_enabled: true
    });
});

// Models List
app.get('/api/models', async (req, res) => {
    const result = await proxyToPython('/api/models', {}, req);
    res.setHeader('X-Request-ID', result.requestId);
    res.status(result.status).json(result.data);
});

// Canonical Prescription Extraction Proxy (Python FastAPI backend)
const handleExtract = async (req, res) => {
    const result = await proxyToPython('/api/prescription/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body)
    }, req);

    res.setHeader('X-Request-ID', result.requestId);
    if (result.status === 200 && result.data) {
        res.json({
            success: true,
            prescription: result.data.prescription || null,
            execution_time_ms: result.data.execution_time_ms || Math.round((result.data.generation_time || 0) * 1000),
            warnings: result.data.warnings || [],
            latency_ms: Math.round((result.data.generation_time || 0) * 1000),
            generation_time: result.data.generation_time || 0,
            total_medicines: result.data.total_medicines || (result.data.prescription ? result.data.prescription.items.length : 0),
            parsed_records: result.data.parsed_records || [],
            canonical: result.data.canonical || result.data.prescription || null,
            agent_logs: result.data.agent_logs || []
        });
    } else {
        res.status(result.status || 500).json(result.data || { error: 'Extraction service error' });
    }
};

app.post('/api/prescription/extract', handleExtract);
app.post('/api/extract', handleExtract);

// Canonical Prescription Validation Proxy
app.post('/api/prescription/validate', async (req, res) => {
    const result = await proxyToPython('/api/prescription/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body)
    }, req);
    res.setHeader('X-Request-ID', result.requestId);
    res.status(result.status).json(result.data);
});

// Canonical Drug Database Search API (proxied to Python FastAPI DrugRepository)
app.get('/api/drugs/search', async (req, res) => {
    const queryString = req.url.includes('?') ? req.url.substring(req.url.indexOf('?')) : '';
    const result = await proxyToPython(`/api/drugs/search${queryString}`, {}, req);
    res.setHeader('X-Request-ID', result.requestId);
    res.status(result.status).json(result.data);
});

// Drug Did-You-Mean Fuzzy / Phonetic API
app.get('/api/drugs/did-you-mean', async (req, res) => {
    const queryString = req.url.includes('?') ? req.url.substring(req.url.indexOf('?')) : '';
    const result = await proxyToPython(`/api/drugs/did-you-mean${queryString}`);
    res.status(result.status).json(result.data);
});

// Dynamic Route Lookup for a Drug
app.get('/api/drugs/:drug_id/routes', async (req, res) => {
    const result = await proxyToPython(`/api/drugs/${encodeURIComponent(req.params.drug_id)}/routes`);
    res.status(result.status).json(result.data);
});

// Clinical Reference Data (Schedules, Dose Units, Routes)
app.get('/api/reference-data', async (req, res) => {
    const result = await proxyToPython('/api/reference-data');
    res.status(result.status).json(result.data);
});

// [DEPRECATED] Semantic Prescription Matcher (retained as backwards-compatible alias to canonical extraction)
app.post('/api/match-prescription', async (req, res) => {
    res.setHeader('X-API-Deprecated', '/api/match-prescription is deprecated; use /api/prescription/extract');
    const result = await proxyToPython('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: req.body.text || '', fast_mode: true })
    }, req);

    res.setHeader('X-Request-ID', result.requestId);
    if (result.status === 200 && result.data) {
        res.json({
            success: true,
            latency_ms: Math.round((result.data.generation_time || 0) * 1000),
            generation_time: result.data.generation_time || 0,
            total: result.data.total_medicines || 0,
            records: result.data.parsed_records || [],
            canonical: result.data.canonical || null
        });
    } else {
        res.status(result.status || 500).json(result.data || { error: 'Extraction service error' });
    }
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
const handleTranscribe = async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No audio file provided' });
    }

    const requestId = req.headers['x-request-id'] || crypto.randomUUID();

    try {
        const formData = new FormData();
        const blob = new Blob([req.file.buffer], { type: req.file.mimetype || 'audio/wav' });
        formData.append('file', blob, req.file.originalname || 'recording.wav');
        formData.append('stt_model', req.body.stt_model || 'whisper_ayush');

        const targetUrl = `${PYTHON_API_BASE}/api/prescription/transcribe`;
        const response = await fetch(targetUrl, {
            method: 'POST',
            headers: {
                'X-Request-ID': requestId
            },
            body: formData
        });
        const data = await response.json();
        const respRequestId = response.headers.get('x-request-id') || requestId;
        res.setHeader('X-Request-ID', respRequestId);
        res.status(response.status).json(data);
    } catch (err) {
        res.setHeader('X-Request-ID', requestId);
        res.status(500).json({
            success: false,
            error_code: 'TRANSCRIPTION_PROXY_ERROR',
            message: 'Transcription proxy failed to connect to FastAPI.',
            details: [{ field: 'stt', code: 'proxy_failure', message: err.message }],
            request_id: requestId
        });
    }
};

app.post('/api/prescription/transcribe', upload.single('file'), handleTranscribe);
app.post('/api/transcribe', upload.single('file'), handleTranscribe);

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

// Centralized error-handling middleware for upload limits and validation
app.use((err, req, res, next) => {
    if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
            success: false,
            error_code: 'PAYLOAD_TOO_LARGE',
            message: 'Audio upload exceeds maximum limit of 25MB.',
            request_id: req.headers['x-request-id'] || null
        });
    }
    if (err.message && err.message.includes('Unsupported media type')) {
        return res.status(415).json({
            success: false,
            error_code: 'UNSUPPORTED_MEDIA_TYPE',
            message: err.message,
            request_id: req.headers['x-request-id'] || null
        });
    }
    console.error('[Express Error]', err.message);
    res.status(500).json({
        success: false,
        error_code: 'INTERNAL_SERVER_ERROR',
        message: 'An unexpected server error occurred.',
        request_id: req.headers['x-request-id'] || null
    });
});

app.listen(PORT, () => {
    console.log(`=======================================================`);
    console.log(`🚀 Rx Extractor Node.js Web App running on http://localhost:${PORT}`);
    console.log(`🔗 Connected Python API Gateway: ${PYTHON_API_BASE}`);
    console.log(`=======================================================`);
});
