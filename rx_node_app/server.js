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

// Serve static frontend assets
app.use(express.static(path.join(__dirname, 'public')));

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

// Models List
app.get('/api/models', async (req, res) => {
    const result = await proxyToPython('/api/models');
    res.status(result.status).json(result.data);
});

// Prescription Extraction Proxy
app.post('/api/extract', async (req, res) => {
    const result = await proxyToPython('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body)
    });
    res.status(result.status).json(result.data);
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
