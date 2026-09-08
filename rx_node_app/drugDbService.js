/**
 * drugDbService.js [DEPRECATED - PHASE 8 CONSOLIDATION]
 * -----------------------------------------------------
 * All independent clinical parsing, drug matching, Soundex indexing,
 * and route resolution have been consolidated into the canonical Python
 * backend (DrugRepository and FastAPI /api/drugs/*).
 *
 * Node.js acts strictly as a thin proxy gateway; it does not maintain
 * independent drug datasets or clinical interpretations.
 */

const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://127.0.0.1:8080';

module.exports = {
    isDeprecated: true,
    message: 'Drug querying is canonically served by FastAPI DrugRepository via reverse proxy.',
    pythonApiBase: PYTHON_API_BASE
};
