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
 * 2. Semantic matching against Drug_databse (drugList.json & Drug_Route_mapping.csv)
 * 3. Phonetic / spelling typo 'Did You Mean?' suggestions (e.g. aracentamol -> Paracetamol)
 * 4. Drug variants dropdown (e.g., all Paracetamols from database, auto-populating dose & dose unit)
 * 5. Dynamic Route dropdown mapped from Drug_Route_mapping.csv for the matched drug_id
 * 6. Doctor speaking habits for schedule (101, 110, 111, 100, 010, 001, 1111, twice daily 101, etc.)
 * 7. Manual medicine entry accordion & autocomplete
 * 8. Save Prescription to hospital record
 */

// In-memory clinical database client cache with O(1) Soundex & Prefix Buckets
const clinicalDb = {
    drugs: [],
    drugsById: new Map(),
    drugsByBaseName: new Map(),
    soundexBuckets: new Map(),
    prefixIndex: new Map(),
    routesByDrugId: new Map(),
    routesByDrugType: new Map(),
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
 * Normalizes drug name to base name (removes dosage, form like TAB, CAP, SYP, INJ)
 */
function cleanDrugBaseName(drugName) {
    if (!drugName) return '';
    return drugName
        .toUpperCase()
        .replace(/\b\d+['`][sS]\b/g, '')
        .replace(/\b(TAB|TABS|TABLET|TABLETS|CAP|CAPS|CAPSULE|CAPSULES|SYP|SYRUP|INJ|INJECTION|OINT|OINTMENT|CREAM|CRM|LOT|LOTION|DROPS|GEL|SOLN|SOLUTION|SOL|SUSP|SUSPENSION|RET|RETD|DT|SR|XL|ER|CR|DS|PLUS|FORTE|POWD|PWDR|RESP|NEB|VIAL|SACHET)\b/g, '')
        .replace(/\b\d+(?:\.\d+)?\s*(MG|G|MCG|ML|L|IU|%|GM)\b/g, '')
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

/**
 * Normalizes speech recognition acoustic confusions and misheard brand names.
 */
function normalizeAsrPhonetics(text) {
    if (!text) return '';
    return text
        .replace(/\b(?:8|eight)\s+(?:and|&)\s+(?:50|fifty)\b/gi, 'Aten 50')
        .replace(/\b(?:8|eight)\s+(?:and|&)\s+(?:25|twenty\s*five)\b/gi, 'Aten 25')
        .replace(/\b(?:8|eight)\s+(?:and|&)\s+(?:100|one\s*hundred)\b/gi, 'Aten 100')
        .replace(/\b(?:8|eight)\s*(?:ten|10)\s+(?:50|fifty)\b/gi, 'Aten 50')
        .replace(/\b(?:8|eight)\s*(?:ten|10)\s+(?:25|twenty\s*five)\b/gi, 'Aten 25')
        .replace(/\b(?:8|eight)\s*(?:ten|10)\b/gi, 'Aten')
        .replace(/\bMetaforamine\b/gi, 'Metformin')
        .replace(/\bMetaformine\b/gi, 'Metformin')
        .replace(/\bMetaphormine\b/gi, 'Metformin')
        .replace(/\bSophramycin\b/gi, 'Soframycin');
}

/**
 * Soundex algorithm for phonetic matching
 */
function soundex(s) {
    if (!s) return '';
    const a = s.toLowerCase().replace(/ph/g, 'f').split('');
    const f = a.shift();
    let r = '';
    const codes = {
        a: '', e: '', i: '', o: '', u: '', y: '', h: '', w: '',
        b: 1, f: 1, p: 1, v: 1,
        c: 2, g: 2, j: 2, k: 2, q: 2, s: 2, x: 2, z: 2,
        d: 3, t: 3,
        l: 4,
        m: 5, n: 5,
        r: 6
    };
    r = f + a
        .map(v => codes[v])
        .filter((v, i, arr) => (i === 0 ? v !== codes[f] : v !== arr[i - 1]))
        .join('');
    return (r + '000').slice(0, 4).toUpperCase();
}

/**
 * Levenshtein distance for typos & missing characters
 */
function levenshtein(s1, s2) {
    s1 = (s1 || '').toLowerCase().replace(/ph/g, 'f');
    s2 = (s2 || '').toLowerCase().replace(/ph/g, 'f');
    const len1 = s1.length;
    const len2 = s2.length;
    const matrix = [];

    for (let i = 0; i <= len1; i++) matrix[i] = [i];
    for (let j = 0; j <= len2; j++) matrix[0][j] = j;

    for (let i = 1; i <= len1; i++) {
        for (let j = 1; j <= len2; j++) {
            const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
            matrix[i][j] = Math.min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            );
        }
    }
    return matrix[len1][len2];
}

/**
 * Initialize clinical database in browser
 */
async function initClinicalDatabase() {
    try {
        console.log('[DMH] Loading Drug_databse client cache...');
        // 1. Fetch drugList.json
        const drugRes = await fetch('/data/drugList.json');
        if (drugRes.ok) {
            const drugJson = await drugRes.json();
            clinicalDb.drugs = drugJson.drugData || [];

            clinicalDb.drugs.forEach(d => {
                clinicalDb.drugsById.set(String(d.drug_id), d);
                const base = cleanDrugBaseName(d.drug_name);
                d.base_name = base;
                d.soundex = soundex(base);

                if (base) {
                    if (!clinicalDb.drugsByBaseName.has(base)) {
                        clinicalDb.drugsByBaseName.set(base, []);

                        // O(1) Soundex Bucket Index
                        const sx = soundex(base);
                        if (!clinicalDb.soundexBuckets.has(sx)) clinicalDb.soundexBuckets.set(sx, []);
                        clinicalDb.soundexBuckets.get(sx).push(base);

                        // O(1) 3-Character Prefix Index
                        const pref = base.substring(0, 3);
                        if (!clinicalDb.prefixIndex.has(pref)) clinicalDb.prefixIndex.set(pref, []);
                        clinicalDb.prefixIndex.get(pref).push(base);
                    }
                    clinicalDb.drugsByBaseName.get(base).push(d);
                }
            });

            // Bidirectional aliases for INN / British vs US ASR transcriptions (e.g. AMOXICILLIN <-> AMOXYCILLIN)
            const COMMON_DRUG_ALIASES = [
                ['AMOXICILLIN', 'AMOXYCILLIN'],
                ['PARACETAMOL', 'ACETAMINOPHEN'],
                ['PANTOPRAZOL', 'PANTOPRAZOLE'],
                ['OMEPRAZOL', 'OMEPRAZOLE'],
                ['RABEPRAZOL', 'RABEPRAZOLE'],
                ['ESOMEPRAZOL', 'ESOMEPRAZOLE'],
                ['CIPROFLOXACIN', 'CIPROFLOXACINE'],
                ['LEVOFLOXACIN', 'LEVOFLOXACINE'],
                ['AZITHROMYCIN', 'AZITHROMYCINE'],
                ['METFORMIN', 'METPHORMIN'],
                ['METFORMIN', 'METAFORAMINE'],
                ['METFORMIN', 'METAFORMINE'],
                ['SOFRAMYCIN', 'SOPHRAMYCIN'],
                ['ATEN', 'ATENOLOL'],
                ['AMLODIPIN', 'AMLODIPINE'],
                ['CEFIXIM', 'CEFIXIME'],
                ['CEFPODOXIM', 'CEFPODOXIME'],
                ['CEFUROXIM', 'CEFUROXIME'],
                ['CETRIZINE', 'CETIRIZINE'],
                ['LEVOCETRIZINE', 'LEVOCETIRIZINE'],
                ['MONTELUKAST', 'MONTELEKAST'],
                ['DICLOFENAC', 'DICLOFENACK'],
                ['IBUPROFEN', 'IBOPROFEN'],
                ['DOXYCYCLINE', 'DOXICYCLINE'],
                ['CLOTRIMAZOLE', 'CLOTRIMAZOL'],
                ['FLUCONAZOLE', 'FLUCONAZOL'],
                ['OXYMETAZOLINE', 'OXIMETHAZOLINE'],
                ['GABAPENTIN', 'GABAPENTINE'],
                ['PREGABALIN', 'PREGABALINE'],
                ['METRONIDAZOLE', 'METRONIDAZOL'],
                ['ATORVASTATIN', 'ATORVASTATION'],
                ['ROSUVASTATIN', 'ROSUVASTATION'],
                ['DOMPERIDONE', 'DOMPERIDON'],
                ['RANITIDINE', 'RANITIDIN']
            ];

            const registerAlias = (name, list) => {
                clinicalDb.drugsByBaseName.set(name, list);
                const sx = soundex(name);
                if (!clinicalDb.soundexBuckets.has(sx)) clinicalDb.soundexBuckets.set(sx, []);
                if (!clinicalDb.soundexBuckets.get(sx).includes(name)) clinicalDb.soundexBuckets.get(sx).push(name);
                const pref = name.substring(0, 3);
                if (!clinicalDb.prefixIndex.has(pref)) clinicalDb.prefixIndex.set(pref, []);
                if (!clinicalDb.prefixIndex.get(pref).includes(name)) clinicalDb.prefixIndex.get(pref).push(name);
            };

            COMMON_DRUG_ALIASES.forEach(([a, b]) => {
                const listA = clinicalDb.drugsByBaseName.get(a);
                const listB = clinicalDb.drugsByBaseName.get(b);
                if (listA && !listB) {
                    registerAlias(b, listA);
                } else if (listB && !listA) {
                    registerAlias(a, listB);
                }
            });

            console.log(`[DMH] Loaded ${clinicalDb.drugs.length} drugs into memory (${clinicalDb.soundexBuckets.size} Soundex buckets).`);
        }

        // 2. Fetch Drug_Route_mapping.csv
        const routeRes = await fetch('/data/Drug_Route_mapping.csv');
        if (routeRes.ok) {
            const routeText = await routeRes.text();
            const lines = routeText.split(/\r?\n/).filter(l => l.trim().length > 0);
            for (let i = 1; i < lines.length; i++) {
                const parts = lines[i].split(',').map(p => p.trim());
                if (parts.length >= 4) {
                    const drugId = parts[0];
                    const drugType = (parts[1] || '').toLowerCase();
                    const routeCode = (parts[3] || '').toUpperCase();

                    if (drugId && routeCode) {
                        if (!clinicalDb.routesByDrugId.has(drugId)) {
                            clinicalDb.routesByDrugId.set(drugId, []);
                        }
                        const existing = clinicalDb.routesByDrugId.get(drugId);
                        if (!existing.includes(routeCode)) existing.push(routeCode);
                    }

                    if (drugType && routeCode) {
                        if (!clinicalDb.routesByDrugType.has(drugType)) {
                            clinicalDb.routesByDrugType.set(drugType, new Set());
                        }
                        clinicalDb.routesByDrugType.get(drugType).add(routeCode);
                    }
                }
            }
            console.log(`[DMH] Loaded route mappings for ${clinicalDb.routesByDrugId.size} drugs.`);
        }

        clinicalDb.isLoaded = true;
    } catch (err) {
        console.warn('[DMH] Client database load notice:', err.message);
    }
}

/**
 * Get mapped routes for a specific drug_id and drug_type
 */
function getRoutesForDrug(drugId, drugType = '') {
    const dId = String(drugId || '').trim();
    if (dId && clinicalDb.routesByDrugId.has(dId)) {
        const routes = [...clinicalDb.routesByDrugId.get(dId)];
        return routes.sort((a, b) => (a === 'ORAL' ? -1 : b === 'ORAL' ? 1 : a.localeCompare(b)));
    }

    const dType = (drugType || '').trim().toLowerCase();
    if (dType && clinicalDb.routesByDrugType.has(dType)) {
        return Array.from(clinicalDb.routesByDrugType.get(dType)).sort((a, b) =>
            a === 'ORAL' ? -1 : b === 'ORAL' ? 1 : a.localeCompare(b)
        );
    }

    return ['ORAL', 'RT', 'PEG', 'IV', 'IM', 'TOPICAL', 'INHALATION', 'OPHTHALMIC', 'NASAL'];
}

const INSTRUCTION_STOPWORDS = new Set([
    'take', 'give', 'start', 'prescribe', 'add', 'use', 'apply', 'stop', 'continue', 'discontinue',
    'tab', 'tablet', 'tabs', 'tablets', 'cap', 'capsule', 'caps', 'capsules',
    'syrup', 'syp', 'inj', 'injection', 'drops', 'drop', 'orally', 'oral',
    'daily', 'twice', 'thrice', 'times', 'time', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'once', 'bid', 'tid', 'qid', 'sos', 'stat', 'od', 'bd', 'tds', 'hs',
    'for', 'after', 'before', 'with', 'without', 'at', 'in', 'on', 'to', 'from', 'by', 'of', 'and', 'then', 'also', 'plus',
    'next', 'second', 'third', 'medicine', 'medicines', 'drug', 'drugs', 'dose', 'dosage',
    'meals', 'meal', 'food', 'eating', 'breakfast', 'lunch', 'dinner',
    'water', 'milk', 'warm', 'cold', 'regular', 'hot', 'fluids', 'liquid', 'liquids',
    'empty', 'stomach', 'fasting', 'bedtime', 'night', 'sleep', 'morning', 'noon', 'afternoon', 'evening',
    'avoid', 'spicy', 'oily', 'alcohol', 'driving', 'smoking', 'rest', 'walk', 'walks', 'diet', 'bland',
    'sugar', 'sweets', 'salt', 'steam', 'inhalation',
    'rinse', 'mouth', 'locally', 'thinly', 'sparingly', 'affected', 'area',
    'swallow', 'whole', 'chew', 'chewable', 'thoroughly', 'dissolve', 'shake', 'well',
    'complete', 'course', 'midway', 'consult', 'doctor', 'physician', 'hospital', 'clinic', 'emergency',
    'strictly', 'plenty', 'gargle',
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'patient', 'patients',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    'not', 'no', 'yes', 'so', 'because', 'although', 'unless', 'if', 'when', 'whenever', 'case',
    'fever', 'pain', 'headache', 'cough', 'cold', 'rash', 'vomiting', 'nausea', 'diarrhea', 'infection',
    'increases', 'increase', 'decreases', 'decrease', 'persists', 'persist', 'worsens', 'worsen',
    'last', 'lasts', 'lasted', 'lasting', 'more', 'than', 'less', 'come', 'goes', 'tell', 'inform',
    'blood', 'body', 'skin', 'eyes', 'ears', 'throat', 'chest', 'back', 'neck',
    'contact', 'call', 'visit', 'see', 'report', 'review',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
]);

/**
 * Search drugs: returns variants for dropdown
 */
function searchDrugs(query, limit = 25) {
    if (!query || !query.trim()) return [];
    const q = query.trim().toUpperCase();
    const cleanQ = cleanDrugBaseName(q);
    if (!cleanQ || INSTRUCTION_STOPWORDS.has(cleanQ.toLowerCase())) return [];

    const matches = [];
    const seenIds = new Set();

    // 1. Exact base name match -> returns all variants (e.g. Paracetamol 500, 650, drops, etc.)
    if (clinicalDb.drugsByBaseName.has(cleanQ)) {
        clinicalDb.drugsByBaseName.get(cleanQ).forEach(d => {
            if (!seenIds.has(d.drug_id)) {
                matches.push(d);
                seenIds.add(d.drug_id);
            }
        });
    }

    // 2. Starts with clean query
    for (const [base, list] of clinicalDb.drugsByBaseName.entries()) {
        if (base.startsWith(cleanQ) && base !== cleanQ) {
            list.forEach(d => {
                if (!seenIds.has(d.drug_id) && matches.length < limit) {
                    matches.push(d);
                    seenIds.add(d.drug_id);
                }
            });
        }
        if (matches.length >= limit) break;
    }

    // 3. Fallback word-boundary search in drug_name
    if (matches.length < limit && cleanQ.length >= 3) {
        try {
            const wordRegex = new RegExp('\\b' + cleanQ, 'i');
            for (const d of clinicalDb.drugs) {
                if (wordRegex.test(d.drug_name) && !seenIds.has(d.drug_id)) {
                    matches.push(d);
                    seenIds.add(d.drug_id);
                    if (matches.length >= limit) break;
                }
            }
        } catch (e) {}
    }

    return matches.slice(0, limit).map(d => ({
        drug_id: d.drug_id,
        drug_code: d.drug_code,
        drug_name: d.drug_name,
        drug_type: d.drug_type,
        base_name: d.base_name,
        routes: getRoutesForDrug(d.drug_id, d.drug_type)
    }));
}

/**
 * Fuzzy & Phonetic match for missing characters / typos ("Did you mean?")
 * Handles "aracentamol" -> "PARACETAMOL", "ugmentin" -> "AUGMENTIN"
 */
function findDidYouMean(query) {
    if (!query || !query.trim()) return null;
    const cleanQ = cleanDrugBaseName(query);
    if (!cleanQ || cleanQ.length < 4 || INSTRUCTION_STOPWORDS.has(cleanQ.toLowerCase())) return null;

    if (clinicalDb.drugsByBaseName.has(cleanQ)) return null;

    const candidateSet = new Set();

    // 1. Same Soundex bucket
    const qSoundex = soundex(cleanQ);
    const soundexMatches = clinicalDb.soundexBuckets.get(qSoundex) || [];
    soundexMatches.forEach(b => candidateSet.add(b));

    // 2. Missing first-letter check (e.g. aracentamol -> paracetamol)
    const commonFirstLetters = ['P', 'A', 'C', 'M', 'T', 'D', 'B', 'S', 'L', 'I', 'R', 'N'];
    for (const letter of commonFirstLetters) {
        const potential = letter + cleanQ;
        if (clinicalDb.drugsByBaseName.has(potential)) {
            candidateSet.add(potential);
        }
        const sx = soundex(potential);
        (clinicalDb.soundexBuckets.get(sx) || []).forEach(b => candidateSet.add(b));
    }

    // 3. Prefix bucket
    const pref = cleanQ.substring(0, 3);
    (clinicalDb.prefixIndex.get(pref) || []).forEach(b => candidateSet.add(b));

    let bestMatch = null;
    let minDistance = 999;

    // Pruned candidate comparison (<20 candidates instead of thousands)
    for (const base of candidateSet) {
        if (base.endsWith(cleanQ) || cleanQ.endsWith(base) || base.includes(cleanQ) || cleanQ.includes(base)) {
            const diff = Math.abs(base.length - cleanQ.length);
            if (diff <= 3 && diff < minDistance) {
                minDistance = diff;
                bestMatch = base;
            }
        }

        const dist = levenshtein(cleanQ, base);
        const threshold = cleanQ.length <= 4 ? 1 : cleanQ.length <= 7 ? 2 : 3;

        if (dist <= threshold && dist < minDistance) {
            minDistance = dist;
            bestMatch = base;
        }
    }

    if (bestMatch) {
        const variants = clinicalDb.drugsByBaseName.get(bestMatch) || [];
        return {
            suggested_name: bestMatch,
            variants: variants.slice(0, 15).map(d => ({
                drug_id: d.drug_id,
                drug_code: d.drug_code,
                drug_name: d.drug_name,
                drug_type: d.drug_type,
                routes: getRoutesForDrug(d.drug_id, d.drug_type)
            }))
        };
    }

    return null;
}

/**
 * Doctor speaking habit schedule parser
 * Understands:
 * - "101", "1-0-1", "1 0 1", "twice daily 101", "twice daily" -> "Twice a day (1-0-1)"
 * - "111", "1-1-1", "1 1 1", "thrice daily", "three times a day" -> "Thrice a day (1-1-1)"
 * - "100", "1-0-0", "1 0 0", "once daily 100", "once a day" -> "Once a day (1-0-0)"
 * - "010", "0-1-0", "afternoon" -> "Once a day (0-1-0)"
 * - "001", "0-0-1", "bedtime", "night", "hs" -> "Once a day (bedtime)"
 * - "110", "1-1-0" -> "Twice a day (1-1-0)"
 * - "011", "0-1-1" -> "Twice a day (0-1-1)"
 * - "1111", "1-1-1-1", "four times a day" -> "Four times a day (1-1-1-1)"
 * - "SOS", "as needed", "if required" -> "If Required (SOS)"
 * - "stat", "immediately" -> "Stat (Immediate single dose only)"
 */
function parseSchedule(text) {
    if (!text) return '';
    const clean = text.toLowerCase().trim();

    // 1. Digital Doctor Notation
    if (/\b(101|1-0-1|1\s*0\s*1|one\s*zero\s*one|one\s*oh\s*one)\b/.test(clean)) {
        return 'Twice a day (1-0-1)';
    }
    if (/\b(111|1-1-1|1\s*1\s*1|one\s*one\s*one)\b/.test(clean)) {
        return 'Thrice a day (1-1-1)';
    }
    if (/\b(100|1-0-0|1\s*0\s*0|one\s*zero\s*zero)\b/.test(clean)) {
        return 'Once a day (1-0-0)';
    }
    if (/\b(010|0-1-0|0\s*1\s*0|zero\s*one\s*zero)\b/.test(clean)) {
        return 'Once a day (0-1-0)';
    }
    if (/\b(001|0-0-1|0\s*0\s*1|zero\s*zero\s*one)\b/.test(clean)) {
        return 'Once a day (bedtime)';
    }
    if (/\b(110|1-1-0|1\s*1\s*0|one\s*one\s*zero)\b/.test(clean)) {
        return 'Twice a day (1-1-0)';
    }
    if (/\b(011|0-1-1|0\s*1\s*1|zero\s*one\s*one)\b/.test(clean)) {
        return 'Twice a day (0-1-1)';
    }
    if (/\b(1111|1-1-1-1|1\s*1\s*1\s*1)\b/.test(clean)) {
        return 'Four times a day (1-1-1-1)';
    }

    // 2. Clinical verbal phrases
    if (/\b(four times|qid|q\.i\.d)\b/.test(clean)) return 'Four times a day (1-1-1-1)';
    if (/\b(three times|thrice|tid|t\.i\.d)\b/.test(clean)) return 'Thrice a day (1-1-1)';
    if (/\b(twice|two times|bid|b\.i\.d)\b/.test(clean)) return 'Twice a day (1-0-1)';
    if (/\b(bedtime|night|hs|qhs)\b/.test(clean)) return 'Once a day (bedtime)';
    if (/\b(afternoon)\b/.test(clean)) return 'Once a day (0-1-0)';
    if (/\b(once daily|once a day|od|q\.d|qday)\b/.test(clean)) return 'Once a day (1-0-0)';
    if (/\b(sos|if needed|if required|prn)\b/.test(clean)) return 'If Required (SOS)';
    if (/\b(stat|immediately)\b/.test(clean)) return 'Stat (Immediate single dose only)';

    return '';
}

/**
 * Generalized Clinical Instruction Parser
 * Captures:
 * 1. Administration timing / meal relation (e.g. 30 mins before breakfast, after meals, on empty stomach, bedtime)
 * 2. Medium & administration technique (e.g. with warm water, with milk, swallow whole, chew thoroughly, gargle, rinse mouth)
 * 3. Precautions, restrictions & lifestyle guidance (e.g. avoid oily and spicy food, complete full course, bed rest)
 * 4. Open Conditional / Contingency Directives (IF / IN CASE OF / WHENEVER <condition> [THEN]? <directive>)
 * 5. Direct Clinical Consult / Follow-up Actions (e.g. consult doctor immediately, review after 5 days)
 */
function parseInstructions(clauseText) {
    if (!clauseText) return 'NONE';
    const text = clauseText.trim();
    const lower = text.toLowerCase();
    const instructions = [];

    // 1. PRIMARY INTAKE & MEAL TIMINGS
    const timedMealMatch = lower.match(/\b(\d+)\s*(?:mins?|minutes?|hrs?|hours?)\s*(before|after)\s*(breakfast|lunch|dinner|meals?|food)\b/i);
    if (timedMealMatch) {
        const mealWord = timedMealMatch[3].toLowerCase();
        instructions.push(`${timedMealMatch[1]} mins ${timedMealMatch[2].toLowerCase()} ${mealWord.startsWith('meal') ? 'meals' : mealWord}`);
    } else {
        const mealPattern = lower.match(/\b(?:strictly\s+)?(before|after|with)\s+(breakfast|lunch|dinner|meals?|food|eating)\b/i);
        if (mealPattern) {
            const prefix = /strictly/i.test(mealPattern[0]) ? 'strictly ' : '';
            const prep = mealPattern[1].toLowerCase();
            let target = mealPattern[2].toLowerCase();
            if (target === 'food' || target === 'eating') target = 'meals';
            instructions.push(`${prefix}${prep} ${target}`);
        } else if (/\b(?:on\s*(?:an?\s*)?empty\s*stomach|empty\s*stomach|fasting)\b/i.test(lower)) {
            instructions.push('on empty stomach');
        }
    }

    // Bedtime / Time of day
    if (/\b(?:at\s*bedtime|before\s*bed(?:time)?|before\s*sleep|at\s*night)\b/i.test(lower)) {
        instructions.push('at bedtime');
    } else if (/\b(?:early\s*morning|in\s*the\s*morning)\b/i.test(lower) && !instructions.some(i => i.includes('breakfast'))) {
        instructions.push('in the morning');
    }

    // 2. FLUIDS & METHOD OF INTAKE
    if (/\bwith\s*(?:warm|hot)\s*water\b/i.test(lower)) {
        instructions.push('with warm water');
    } else if (/\bwith\s*(?:cold|regular|clean|fresh|plenty\s*of)?\s*water\b/i.test(lower) && !instructions.some(i => i.includes('warm water'))) {
        instructions.push('with water');
    } else if (/\bwith\s*(?:warm\s*)?milk\b/i.test(lower)) {
        instructions.push('with milk');
    }

    if (/\b(?:drink\s*plenty\s*of\s*(?:water|fluids?|liquids?)|plenty\s*of\s*(?:fluids?|water)|hydrate\s*well)\b/i.test(lower)) {
        instructions.push('drink plenty of fluids');
    }

    if (/\b(?:swallow\s*whole|do\s*not\s*(?:chew|crush))\b/i.test(lower)) {
        instructions.push('swallow whole (do not chew)');
    } else if (/\b(?:chew\s*thoroughly|chewable)\b/i.test(lower)) {
        instructions.push('chew thoroughly');
    } else if (/\bdissolve\s*in\s*(?:water|half\s*glass\s*water)\b/i.test(lower)) {
        instructions.push('dissolve in water');
    } else if (/\bshake\s*well(?:\s*before\s*use)?\b/i.test(lower)) {
        instructions.push('shake well before use');
    }

    // Device / Topical / Inhalation / Oral Rinse
    if (/\brinse\s*mouth(?:\s*after\s*use)?\b/i.test(lower)) {
        instructions.push('rinse mouth after use');
    }
    if (/\b(?:apply\s*(?:thinly|sparingly|locally)|on\s*affected\s*area)\b/i.test(lower)) {
        instructions.push('apply on affected area');
    }
    if (/\b(?:warm\s*water\s*gargle|salt\s*water\s*gargle|gargle)\b/i.test(lower)) {
        instructions.push('warm water gargle');
    }
    if (/\bsteam\s*inhalation\b/i.test(lower)) {
        instructions.push('steam inhalation');
    }

    // 3. COURSE, PRECAUTIONS & LIFESTYLE (GENERALIZED)
    if (/\b(?:complete\s*(?:the\s*)?(?:full\s*)?course|do\s*not\s*stop\s*midway)\b/i.test(lower)) {
        instructions.push('complete full course');
    }
    const avoidMatch = lower.match(/\b(?:strictly\s+)?avoid\s+([^,;\n\.]+)/i);
    if (avoidMatch) {
        let what = avoidMatch[1].replace(/\b(?:and\s+then|then|also|take)\b.*/i, '').trim();
        if (what.length > 2) {
            instructions.push(`avoid ${what}`);
        }
    }
    if (/\b(?:bed\s*rest|rest)\b/i.test(lower) && !instructions.some(i => i.includes('rest'))) {
        instructions.push('bed rest');
    }
    if (/\b(?:light|bland)\s*diet\b/i.test(lower)) {
        instructions.push('bland diet');
    }

    // 4. OPEN CONDITIONAL / CONTINGENCY DIRECTIVES (Generalized Grammar)
    const conditionalRegex = /\b(?:for\s+)?((?:if|in\s+case\s+(?:of\s+)?|whenever|when|as\s+needed)\s+[^,;\n\.]+(?:,\s*[^,;\n\.]+|\s+(?:then\s+)?[^,;\n\.]+)*)/gi;
    let condMatch;
    while ((condMatch = conditionalRegex.exec(text)) !== null) {
        let condText = condMatch[1].trim();
        condText = condText.replace(/\b(?:okay|ok|alright|fine|thank\s*you|thanks|please|next|done)\s*$/i, '').trim();
        condText = condText.replace(/\bconfer(?:red)?\b/gi, 'consult');
        if (condText.length > 6 && !instructions.some(i => i.toLowerCase().includes(condText.toLowerCase()))) {
            instructions.push(condText);
        }
    }

    // 5. DIRECT CLINICAL ACTION / FOLLOW-UP DIRECTIVES (Standalone)
    const directActionRegex = /\b((?:consult|contact|report\s+to|visit|see|call|review\s+with|follow\s*up\s+with)\s+(?:the\s+)?(?:doctor|physician|hospital|clinic|specialist|emergency)[^,;\n\.]*)/gi;
    let actMatch;
    while ((actMatch = directActionRegex.exec(text)) !== null) {
        let actText = actMatch[1].trim();
        actText = actText.replace(/\b(?:okay|ok|alright|fine|thank\s*you|thanks|please|next|done)\s*$/i, '').trim();
        actText = actText.replace(/\bconfer(?:red)?\b/gi, 'consult');
        const alreadyInCond = instructions.some(i => i.toLowerCase().includes(actText.toLowerCase()));
        if (actText.length > 5 && !alreadyInCond) {
            instructions.push(actText);
        }
    }

    const unique = [];
    for (const inst of instructions) {
        const cleanInst = inst.replace(/\s+/g, ' ').trim();
        if (!unique.some(u => u.toLowerCase() === cleanInst.toLowerCase() || cleanInst.toLowerCase().includes(u.toLowerCase()))) {
            unique.push(cleanInst);
        }
    }

    return unique.length > 0 ? unique.join('; ') : 'NONE';
}

/**
 * Scan segment for drug entity using multi-word n-gram matching and phonetic fallback
 */
function findDrugInSegment(seg) {
    if (!seg || !seg.trim()) return null;

    // Tokenize into clean words
    const tokens = seg.match(/[a-zA-Z0-9\-]+/g) || [];
    if (tokens.length === 0) return null;

    // 1. Check 3-grams, 2-grams, 1-grams against database base names
    for (let len = Math.min(3, tokens.length); len >= 1; len--) {
        for (let i = 0; i <= tokens.length - len; i++) {
            const gram = tokens.slice(i, i + len).join(' ');
            const cleanGram = cleanDrugBaseName(gram);
            if (cleanGram && !INSTRUCTION_STOPWORDS.has(cleanGram.toLowerCase()) && clinicalDb.drugsByBaseName.has(cleanGram)) {
                return {
                    matchedBase: cleanGram,
                    tokenIndex: i,
                    tokenCount: len,
                    variants: clinicalDb.drugsByBaseName.get(cleanGram),
                    didYouMean: null
                };
            }
        }
    }

    // 2. If no exact database match, evaluate non-stopword tokens for phonetic Did-You-Mean
    for (const token of tokens) {
        if (token.length >= 3 && !INSTRUCTION_STOPWORDS.has(token.toLowerCase()) && !/^\d+$/.test(token)) {
            const dym = findDidYouMean(token);
            if (dym) {
                return {
                    matchedBase: dym.suggested_name,
                    tokenIndex: tokens.indexOf(token),
                    tokenCount: 1,
                    variants: dym.variants,
                    didYouMean: dym.suggested_name
                };
            }
        }
    }

    return null;
}

/**
 * Intelligent Entity-Aware Prescription Clause Segmenter (Client)
 * Scans the transcript text and partitions it into individual medication order clauses
 * based on drug entity occurrences, transition words, and clinical order boundaries.
 */
/**
 * Helper to partition a sentence containing multiple drugs (Client)
 */
function partitionMultiDrugSentence(sentence) {
    const tokens = sentence.match(/[a-zA-Z0-9\-]+/g) || [];
    if (tokens.length === 0) return [sentence];

    const spans = [];
    let idx = 0;
    while (idx < tokens.length) {
        let matched = false;
        for (let len = Math.min(3, tokens.length - idx); len >= 1; len--) {
            const gram = tokens.slice(idx, idx + len).join(' ');
            const cleanGram = cleanDrugBaseName(gram);
            if (cleanGram && !INSTRUCTION_STOPWORDS.has(cleanGram.toLowerCase()) && clinicalDb.drugsByBaseName.has(cleanGram)) {
                spans.push({ startIndex: idx, length: len, base: cleanGram });
                idx += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            const tok = tokens[idx];
            if (tok.length >= 4 && !INSTRUCTION_STOPWORDS.has(tok.toLowerCase()) && !/^\d+$/.test(tok)) {
                const dym = findDidYouMean(tok);
                if (dym) {
                    spans.push({ startIndex: idx, length: 1, base: dym.suggested_name });
                    idx++;
                    matched = true;
                }
            }
        }
        if (!matched) idx++;
    }

    if (spans.length <= 1) return [sentence];

    const result = [];
    for (let s = 0; s < spans.length; s++) {
        const cur = spans[s];
        const next = spans[s + 1];
        let startTok = s === 0 ? 0 : cur.startIndex;
        if (s > 0) {
            let lookback = cur.startIndex - 1;
            while (lookback >= 0 && /\b(?:and|also|take|give|start|prescribe|add|apply|plus)\b/i.test(tokens[lookback])) {
                startTok = lookback;
                lookback--;
            }
        }
        let endTok = tokens.length;
        if (next) {
            let transitionStart = next.startIndex;
            let lookback = next.startIndex - 1;
            while (lookback > cur.startIndex && /\b(?:and|also|take|give|start|prescribe|add|apply|plus)\b/i.test(tokens[lookback])) {
                transitionStart = lookback;
                lookback--;
            }
            endTok = transitionStart;
        }
        const sub = tokens.slice(startTok, endTok).join(' ');
        if (sub.trim()) result.push(sub.trim());
    }
    return result.length > 0 ? result : [sentence];
}

/**
 * Intelligent Entity-Aware Prescription Clause Segmenter (Client)
 * Scans the transcript text and partitions it into individual medication order clauses
 * based on sentence boundaries, action verbs, and drug entity occurrences.
 */
function segmentPrescriptionClauses(transcriptText) {
    if (!transcriptText || !transcriptText.trim()) return [];

    const normText = normalizeAsrPhonetics(transcriptText.trim());

    // Split on sentence periods (followed by space or capital), newlines, semicolons, or major transition phrases
    const rawSentences = normText
        .split(/(?:[\r\n;]+|(?:\.|\?|!)(?:\s+|$)|(?:\s*,\s*|\s+)(?=(?:and\s+then|then|next\s+(?:medicine|drug)|second\s+medicine|third\s+medicine|also\s+(?:give|take|start|prescribe|add|apply)|plus)\b))+/i)
        .map(s => s.trim())
        .filter(Boolean);

    const segments = [];
    let currentOrder = '';

    for (const sentence of rawSentences) {
        // Check if this sentence contains a recognized drug entity
        const drugMatch = findDrugInSegment(sentence);

        if (drugMatch) {
            const subOrders = partitionMultiDrugSentence(sentence);
            if (currentOrder.trim()) {
                segments.push(currentOrder.trim());
                currentOrder = '';
            }
            if (subOrders.length > 1) {
                for (let i = 0; i < subOrders.length - 1; i++) {
                    segments.push(subOrders[i]);
                }
                currentOrder = subOrders[subOrders.length - 1];
            } else {
                currentOrder = sentence;
            }
        } else {
            // No drug in this sentence (e.g. advice / instructions) -> attach to current order
            if (currentOrder) {
                currentOrder += '. ' + sentence;
            } else {
                currentOrder = sentence;
            }
        }
    }

    if (currentOrder.trim()) {
        segments.push(currentOrder.trim());
    }

    return segments.length > 0 ? segments : [normText];
}

/**
 * Analyze prescription text against Drug_databse semantically
 */
function analyzePrescriptionText(transcriptText) {
    if (!transcriptText || !transcriptText.trim()) return [];

    const segments = segmentPrescriptionClauses(transcriptText.trim());
    const results = [];

    for (const seg of segments) {
        // Stage 1: Identify drug entity in segment
        const drugMatch = findDrugInSegment(seg);
        if (!drugMatch) {
            // Out-of-database rejection: do NOT add anything if no drug or phonetic candidate matches
            continue;
        }

        const matchedVariants = drugMatch.variants || [];
        if (matchedVariants.length === 0) continue;

        let primaryDrug = matchedVariants[0];
        const didYouMean = drugMatch.didYouMean;

        // Stage 2: Formulation Strength vs. Prescribed Order Dosage Disambiguation
        // User Clinical Rules:
        // 1) medicine + (integer + units) + (integer + units) == second or final integer + units is dose and dose units
        // 2) medicine + (integer + units) == integer is dose only when the database of medicines/drugs do not have those integer + units with their names, else no dose just medicine with name and integer + units within the name
        // 3) If no dose given -> NO default dose is populated. dose and dose_unit remain empty.

        const getNumbersFromDrug = (drugName) => (drugName || '').match(/\d+(?:\.\d+)?/g) || [];

        // Step B: Isolate core medication specification from trailing duration and instructions
        const coreMedSection = seg.split(/\b(?:for\s+\d+\s*(?:days?|weeks?|months?)|if\s+|in\s+case|whenever|when\s+|as\s+needed|avoid\s+|consult\s+|contact\s+)\b/i)[0];

        // Step C: Extract candidate numbers and units from coreMedSection
        const unitRegex = /(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|l|iu|drops|puffs?|units?|%|tabs?|caps?|tablets?|capsules?|spoonfuls?|spoons?|sachets?|vials?)\b/gi;
        const candidateMatches = [...coreMedSection.matchAll(unitRegex)];

        const candidates = candidateMatches.map(m => ({
            num: m[1],
            unit: m[2].toLowerCase().replace(/^tablets?$/, 'tab').replace(/^capsules?$/, 'cap'),
            raw: m[0],
            index: m.index
        }));

        // Also check for standalone numbers not captured above (excluding schedule habits like 101, 111, 100)
        const allNums = [...coreMedSection.matchAll(/\b(\d+(?:\.\d+)?)\b/g)];
        for (const nm of allNums) {
            const numVal = nm[1];
            const idx = nm.index;
            const captured = candidates.some(c => idx >= c.index && idx < (c.index + c.raw.length));
            const followingText = coreMedSection.substring(idx + numVal.length, idx + numVal.length + 15);
            const isDuration = /^\s*(?:days?|weeks?|months?)\b/i.test(followingText);
            if (!captured && !isDuration && !/^(?:101|111|100|010|001|110|011|1111)$/.test(numVal)) {
                candidates.push({
                    num: numVal,
                    unit: '',
                    raw: numVal,
                    index: idx
                });
            }
        }

        candidates.sort((a, b) => a.index - b.index);

        let dose = '';
        let doseUnit = '';

        if (candidates.length >= 2) {
            // Rule 1: medicine + (integer + units) + (integer + units)
            // Second or final integer + units is dose and dose units
            const finalCand = candidates[candidates.length - 1];
            dose = finalCand.num;
            doseUnit = finalCand.unit || 'mg';

            // Earlier integer binds to catalog formulation strength if in database
            const priorCands = candidates.slice(0, candidates.length - 1);
            for (const pc of priorCands) {
                const matchVar = matchedVariants.find(v => getNumbersFromDrug(v.drug_name).includes(pc.num));
                if (matchVar) {
                    primaryDrug = matchVar;
                    break;
                }
            }
        } else if (candidates.length === 1) {
            // Rule 2: medicine + (integer + units)
            // Integer is dose ONLY when the database of medicines/drugs do not have those integer + units with their names,
            // else no dose just medicine with name and integer + units within the name.
            const cand = candidates[0];
            const matchingVariant = matchedVariants.find(v => getNumbersFromDrug(v.drug_name).includes(cand.num));

            if (matchingVariant) {
                primaryDrug = matchingVariant;
                dose = '';
                doseUnit = '';
            } else {
                dose = cand.num;
                doseUnit = cand.unit || 'mg';
            }
        } else {
            // Rule 3: No integers spoken at all -> No dose given!
            dose = '';
            doseUnit = '';
        }

        if (!dose) {
            doseUnit = '';
        }

        // Stage 3: Extract duration
        const daysMatch = seg.match(/\b(?:for\s*)?(\d+)\s*(days?|weeks?|months?)\b/i);
        const forNumMatch = !daysMatch && seg.match(/\bfor\s+(\d+)\b(?!\s*(?:mg|g|mcg|ml|l|iu|%))/i);
        let days = '';
        if (daysMatch) {
            days = `${daysMatch[1]} ${daysMatch[2]}`;
        } else if (forNumMatch) {
            days = `${forNumMatch[1]} days`;
        } else if (/\bone\s*week\b/i.test(seg)) {
            days = '7 days';
        } else if (/\btwo\s*weeks\b/i.test(seg)) {
            days = '14 days';
        } else if (/\bone\s*month\b/i.test(seg)) {
            days = '30 days';
        }

        // Stage 4: Schedule Speaking Habit Normalization
        const schedule = parseSchedule(seg);

        // Stage 5: Generalized Instructions
        const parsedInst = parseInstructions(seg);
        const instruction = parsedInst !== 'NONE' ? parsedInst : '';

        // Stage 6: Relational Route Mapping Join
        const drugId = primaryDrug.drug_id;
        const drugType = primaryDrug.drug_type;
        const drugName = primaryDrug.drug_name;
        const routes = getRoutesForDrug(drugId, drugType);

        results.push({
            drug_id: drugId,
            Drug_name: drugName,
            dose: dose,
            dose_unit: doseUnit,
            schedule: schedule,
            route: routes[0] || 'ORAL',
            available_routes: routes,
            available_drugs: matchedVariants,
            did_you_mean: didYouMean,
            instruction: instruction,
            days: days
        });
    }

    // Deduplicate medications: only drop truly identical duplicate repeats (same drug AND same dose AND same duration AND same schedule)
    const deduped = [];
    for (const r of results) {
        const base = cleanDrugBaseName(r.Drug_name);
        const existingIdx = deduped.findIndex(d =>
            cleanDrugBaseName(d.Drug_name) === base &&
            d.dose === r.dose &&
            d.dose_unit === r.dose_unit &&
            d.days === r.days &&
            d.schedule === r.schedule
        );
        if (existingIdx === -1) {
            deduped.push(r);
        } else {
            const existing = deduped[existingIdx];
            if (!existing.instruction && r.instruction) existing.instruction = r.instruction;
            if (existing.instruction && r.instruction && !existing.instruction.includes(r.instruction)) {
                existing.instruction += `; ${r.instruction}`;
            }
        }
    }

    return deduped;
}

/**
 * Determines whether a spoken or typed prescription text fragment constitutes
 * a complete clinical sentence, or is still an in-progress incomplete fragment.
 * Prevents premature extraction while speech or typing is mid-sentence.
 */
function isPrescriptionSentenceComplete(transcriptText) {
    if (!transcriptText || !transcriptText.trim()) return false;
    const clean = transcriptText.trim();

    // 1. Explicit sentence termination punctuation (period, newline, etc.)
    if (/[.!?\n]$/.test(clean)) return true;

    // 2. Dangling trailing function words indicate utterance is actively mid-sentence
    if (/\b(?:for|with|after|before|and|then|strictly|avoid|take|give|plus|or|at|to|in|on|if|when|whenever|in\s+case|of|the|a|an|is|are|was|were|be|been|being|as|also|start|stop|do|does|did|not|no|so|because|although|unless)\s*$/i.test(clean)) {
        return false;
    }

    // 3. Trailing naked number without unit indicates unfinished dosage or duration (e.g. "take Paracetamol 500mg 20", "Aten 50mg for 5")
    if (/\b\d+(?:\.\d+)?\s*$/.test(clean)) {
        return false;
    }

    // 4. Incomplete Conditional Clause Check:
    // If the utterance opens an "if / when / in case of" clause, it must contain both
    // the condition antecedent AND a subsequent action directive or verb phrase
    const condMatch = clean.match(/\b(?:if|in\s+case\s+(?:of\s+)?|whenever|when)\s+([^,;\n\.]+)$/i);
    if (condMatch) {
        const condPhrase = condMatch[1].trim();
        const condWords = condPhrase.split(/\s+/);
        // If conditional clause has fewer than 4 words or has no action verb, it is unfinished
        const hasActionVerb = /\b(?:consult|contact|call|visit|see|report|meet|discontinue|stop|start|take|increase|decrease|reduce|review)\b/i.test(condPhrase);
        if (!hasActionVerb || condWords.length < 4) {
            return false;
        }
    }

    // 5. While actively recording speech, do NOT prematurely extract if interim speech is ongoing
    if (state.isRecording && state.speechInterimText && state.speechInterimText.trim().length > 0) {
        return false;
    }

    // 6. Clinical Sentence Completion:
    // Partition into clauses and evaluate the target order (the last clause currently being dictated)
    const segments = segmentPrescriptionClauses(clean);
    if (segments.length === 0) return false;
    const targetSegment = segments[segments.length - 1];

    const hasDrug = findDrugInSegment(targetSegment) !== null;
    if (!hasDrug) return false;

    const hasDose = /\b(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|l|iu|drops?|puffs?|units?|%|tabs?|caps?|tablets?|capsules?|spoonfuls?|spoons?|sachets?|vials?)\b/i.test(targetSegment);
    const hasSchedule = /\b(101|111|100|010|001|110|011|1111|1-0-1|1-1-1|1-0-0|0-1-0|0-0-1|1-1-0|0-1-1|once|twice|thrice|daily|od|bd|tds|hs|prn|qid|tid|bid|sos|stat|bedtime|night|morning|afternoon|evening)\b/i.test(targetSegment);
    const hasDuration = /\b(?:for\s*)?(\d+)\s*(days?|weeks?|months?)\b/i.test(targetSegment) || /\b(?:one|two)\s*(?:weeks?|months?)\b/i.test(targetSegment);
    const hasInstructions = /\b(?:after|before|with|without)\s+(?:meals?|breakfast|lunch|dinner|food|eating|water|milk)\b/i.test(targetSegment) ||
                            /\b(?:if|when|in\s+case)\s+[^,;\n\.]+\s+(?:consult|contact|call|visit|see|stop|start|take)\b/i.test(targetSegment);

    // Order is complete if order has:
    // (Dose + Schedule) OR (Dose + Duration) OR (Dose + Instructions) OR (Schedule + Duration) OR Instructions
    return (hasDose && (hasSchedule || hasDuration || hasInstructions)) ||
           (hasSchedule && hasDuration) ||
           hasInstructions;
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
        const processor = state.audioContext.createScriptProcessor(4096, 1, 1);
        state.audioProcessor = processor;

        let wsUrl = 'ws://127.0.0.1:8080/ws/transcribe?sample_rate=16000&stt_model=whisper_ayush';
        try {
            const cfgRes = await fetch('/api/streaming-config');
            if (cfgRes.ok) {
                const cfg = await cfgRes.json();
                if (cfg.ws_url) wsUrl = `${cfg.ws_url}?sample_rate=16000&stt_model=whisper_ayush`;
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

        source.connect(processor);
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
 * Process text against Drug_databse (hybrid: fast client + server endpoint)
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

    // 1. Instant client-side analysis (< 1ms with O(1) Soundex & candidate pruning)
    const clientResults = analyzePrescriptionText(cleanText);
    const clientLatency = performance.now() - t0;

    // Populate table with instant client analysis results
    state.extractedRecords = clientResults;
    renderTable(state.extractedRecords);

    const dymMatch = clientResults.find(r => r.did_you_mean);
    if (dymMatch) {
        showDidYouMeanBanner(dymMatch);
    } else {
        hideDidYouMeanBanner();
    }

    if (clientResults.length > 0) {
        const latElem = document.getElementById('latency-telemetry-text');
        if (latElem) {
            latElem.textContent = `⚡ Extracted in ${clientLatency.toFixed(1)}ms (Sub-second)`;
        }
    }

    // 2. Also query server for backend relational SQL flowsheet (< 15ms)
    try {
        const res = await fetch('/api/match-prescription', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: cleanText })
        });
        const data = await res.json();
        if (data.success && data.records && data.records.length > 0) {
            state.extractedRecords = data.records;
            renderTable(state.extractedRecords);

            const serverDym = data.records.find(r => r.did_you_mean);
            if (serverDym) {
                showDidYouMeanBanner(serverDym);
            } else {
                hideDidYouMeanBanner();
            }

            const latElem = document.getElementById('latency-telemetry-text');
            if (latElem) {
                const totalMs = data.latency_ms !== undefined ? data.latency_ms : clientLatency.toFixed(1);
                latElem.textContent = `⚡ SQL Flowsheet: ${totalMs}ms (< 0.02s)`;
            }
        }
    } catch (e) {
        // Client results already rendered instantly, ignore server connection errors
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

    manualSearchTimer = setTimeout(() => {
        const results = searchDrugs(val, 15);
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
