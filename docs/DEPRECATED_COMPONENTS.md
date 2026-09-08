# Deprecated Components Register: Agentic Prescription Generator

**Document**: `docs/DEPRECATED_COMPONENTS.md`  
**Date**: September 8, 2026  
**Status**: Active Deprecation Tracking  

This document inventories all legacy modules, endpoints, and wrappers intentionally retained in the codebase as lightweight backwards-compatibility stubs, along with their canonical migration targets and deprecation policy.

---

## 1. Summary of Retained Deprecation Stubs

| Deprecated Component | Type | Canonical Replacement | Retained Rationale | Scheduled Removal |
| :--- | :--- | :--- | :--- | :--- |
| [`app/drug_db/repository.py`](file:///c:/Users/ADMIN/Downloads/rx_extractor_app_agentic/app/drug_db/repository.py) | Python Module Alias | [`app/drugs/repository.py`](file:///c:/Users/ADMIN/Downloads/rx_extractor_app_agentic/app/drugs/repository.py) | Re-exports canonical `DrugRepository`, `soundex`, etc. All internal pipeline code has been repointed to `app.drugs`. | Next major release (v4.0.0) |
| [`rx_node_app/drugDbService.js`](file:///c:/Users/ADMIN/Downloads/rx_extractor_app_agentic/rx_node_app/drugDbService.js) | Node.js Stub | FastAPI `/api/drugs/*` & `app/drugs/repository.py` | 19-line notice stub informing developers that drug queries are served by FastAPI. | Next major release (v4.0.0) |
| [`rx_extractor_app/streaming_transcriber.py`](file:///c:/Users/ADMIN/Downloads/rx_extractor_app_agentic/rx_extractor_app/streaming_transcriber.py) | Python Wrapper | [`app/stt/streaming.py`](file:///c:/Users/ADMIN/Downloads/rx_extractor_app_agentic/app/stt/streaming.py) & `fast_streaming_transcriber.py` | Preserved for legacy CLI benchmark `test_streaming_engine.py`. | Upon test unification |
| `POST /api/extract` | API Route Alias | `POST /api/prescription/extract` | Backwards compatibility for external clients expecting legacy route. | End of Phase 10 |
| `POST /api/transcribe` | API Route Alias | `POST /api/prescription/transcribe` | Backwards compatibility for legacy audio transcription consumers. | End of Phase 10 |
| `POST /api/match-prescription` | Node Route Alias | `POST /api/prescription/extract` | Emits `X-API-Deprecated` header and transparently delegates to canonical extraction. | Next major release (v4.0.0) |

---

## 2. Component Details & Migration Guidance

### 2.1 `app/drug_db/repository.py`
- **Location**: `app/drug_db/repository.py`
- **Current Role**: Pure re-export shim.
- **Migration Code**:
  ```python
  # Old Import (Deprecated)
  from app.drug_db.repository import DrugRepository, get_drug_repository

  # New Import (Canonical)
  from app.drugs.repository import DrugRepository
  from app.drugs import get_drug_repository
  ```
- **Internal Status**: 100% of internal imports across `app/prescription/` and `tests/` have been migrated to `app.drugs.repository`. Zero internal modules import from `app.drug_db`.

### 2.2 `rx_node_app/drugDbService.js`
- **Location**: `rx_node_app/drugDbService.js`
- **Current Role**: Lightweight 19-line informational object.
- **Migration Guidance**: Node.js clients must not run independent database queries or maintain local Soundex indexes. Use standard HTTP reverse proxy calls:
  ```javascript
  // Old (Deprecated Local Call)
  const results = drugDbService.executeSqlFlowsheet(text);

  // New (Canonical REST Query)
  const res = await fetch(`/api/drugs/search?q=${encodeURIComponent(query)}`);
  const data = await res.json();
  ```

### 2.3 `rx_extractor_app/streaming_transcriber.py`
- **Location**: `rx_extractor_app/streaming_transcriber.py`
- **Current Role**: Legacy audio streaming coordinator wrapping `VADDetector` and `SlidingWindowStreamingDecoder`.
- **Migration Guidance**: Use the decoupled, framework-agnostic streaming pipeline in `app.stt.streaming`:
  ```python
  # Old (Deprecated)
  from streaming_transcriber import LiveStreamingTranscriber

  # New (Canonical)
  from app.stt.streaming import StreamingTranscriber
  from app.stt import get_stt_manager
  ```

### 2.4 API Route Aliases (`/api/extract`, `/api/transcribe`, `/api/match-prescription`)
- **FastAPI Endpoints**:
  - `POST /api/extract` maps identically to `POST /api/prescription/extract`.
  - `POST /api/transcribe` maps identically to `POST /api/prescription/transcribe`.
- **Node.js Endpoint**:
  - `POST /api/match-prescription` returns the response header `X-API-Deprecated: /api/match-prescription is deprecated; use /api/prescription/extract`.

---

## 3. Permanently Decommissioned & Deleted Components

The following obsolete components have been permanently deleted from the repository during the Phase 10 cleanup:

1. **`rx_extractor_app/requirements.txt`**: Deleted (redundant duplicate of root `requirements.txt`).
2. **`Whisper_Ayush/`**: Deleted (dead storage containing only tokenizer JSONs with 0 weight files; active weights reside in `Whisper_Ayush_ct2/`).
3. **`rx_node_app/public/app.js` (lines 70–504)**: Deleted (~430 lines of dead client-side regexes, dead Soundex/Levenshtein algorithms, and malformed syntax error).
4. **Duplicate `/api/health` in `server.js`**: Deleted (redundant duplicate route registration).
5. **Dead Cached Loaders in `api_server.py`**: Deleted (`_CACHED_MODELS`, `_CACHED_STORE`, `get_cached_chat`, `get_cached_vector_store`, and obsolete imports `vectorstore`, `pipeline`, `run_graph_extraction`).
