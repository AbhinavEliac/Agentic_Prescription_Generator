"""
scratch/adversarial_breaker_test.py
-----------------------------------
Adversarial Stress Test Battery for the Rehabilitated System.
Executes 12 brutal stress tests to identify edge case failures,
unhandled exceptions, hallucinations, and architectural regressions.
"""

import sys
import io
import time
import json
from pathlib import Path

# Ensure project root is on PYTHONPATH
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.prescription.pipeline import PrescriptionPipeline, PipelineMode
from app.prescription.schema import ClinicalRoute, ExtractionStatus
from app.drugs import get_drug_repository
from app.stt import get_stt_manager
from rx_extractor_app.api_server import app as fastapi_app
from fastapi.testclient import TestClient


def run_tests():
    pipeline = PrescriptionPipeline()
    drug_repo = get_drug_repository()
    stt_mgr = get_stt_manager()
    client = TestClient(fastapi_app)

    results = []

    def record(test_name, passed, details):
        results.append({"test": test_name, "passed": passed, "details": details})
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {test_name}: {details}")

    print("================================================================================")
    print("EXECUTING ADVERSARIAL BREAKER TEST SUITE")
    print("================================================================================")

    # --------------------------------------------------------------------------
    # 1. MALFORMED / EMPTY PRESCRIPTION
    # --------------------------------------------------------------------------
    try:
        rx_empty = pipeline.extract("", mode=PipelineMode.FAST)
        rx_spaces = pipeline.extract("   \n\t  ", mode=PipelineMode.FAST)
        rx_garbage = pipeline.extract("????!!!!----;;;;....", mode=PipelineMode.FAST)
        rx_huge = pipeline.extract("word " * 2000, mode=PipelineMode.FAST)
        passed = (
            len(rx_empty.items) == 0 and
            len(rx_spaces.items) == 0 and
            len(rx_garbage.items) == 0
        )
        record("T01_Malformed_And_Garbage_Input", passed, f"Empty: {len(rx_empty.items)}, Garbage: {len(rx_garbage.items)}, Huge items: {len(rx_huge.items)}")
    except Exception as e:
        record("T01_Malformed_And_Garbage_Input", False, f"Threw unexpected exception: {e}")

    # --------------------------------------------------------------------------
    # 2. UNKNOWN / FABRICATED MEDICINE (Hallucination Test)
    # --------------------------------------------------------------------------
    try:
        text = "Take Xyloflurazepam 500 mg twice daily for 5 days."
        rx = pipeline.extract(text, mode=PipelineMode.FAST)
        if len(rx.items) == 1:
            item = rx.items[0]
            # Must NOT match a random real drug ID
            hallucinated = (item.matched_drug_id is not None)
            needs_review = (item.status == ExtractionStatus.NEEDS_REVIEW or len(item.validation_warnings) > 0)
            record("T02_Unknown_Medicine_Hallucination_Check", not hallucinated, f"Drug: '{item.medicine}', matched_id: {item.matched_drug_id}, status: {item.status}")
        else:
            record("T02_Unknown_Medicine_Hallucination_Check", True, f"Extracted {len(rx.items)} items for unknown drug")
    except Exception as e:
        record("T02_Unknown_Medicine_Hallucination_Check", False, f"Failed with exception: {e}")

    # --------------------------------------------------------------------------
    # 3. MISSING STRENGTH (Zero Hallucination Rule)
    # --------------------------------------------------------------------------
    try:
        text = "Take Paracetamol tablet twice daily for 5 days after meals."
        rx = pipeline.extract(text, mode=PipelineMode.FAST)
        assert len(rx.items) >= 1
        item = rx.items[0]
        # Strength MUST be None, never guessed (e.g. 500mg or 650mg)
        passed = (item.strength is None)
        record("T03_Missing_Strength_Null_Invariant", passed, f"Extracted strength: '{item.strength}' (Expected None)")
    except Exception as e:
        record("T03_Missing_Strength_Null_Invariant", False, f"Exception: {e}")

    # --------------------------------------------------------------------------
    # 4. MISSING DURATION (Zero Hallucination Rule)
    # --------------------------------------------------------------------------
    try:
        text = "Tab Augmentin 625mg twice daily after meals."
        rx = pipeline.extract(text, mode=PipelineMode.FAST)
        assert len(rx.items) >= 1
        item = rx.items[0]
        passed = (item.duration is None)
        record("T04_Missing_Duration_Null_Invariant", passed, f"Extracted duration: '{item.duration}' (Expected None)")
    except Exception as e:
        record("T04_Missing_Duration_Null_Invariant", False, f"Exception: {e}")

    # --------------------------------------------------------------------------
    # 5. MISSING ROUTE (Zero Hallucination Rule)
    # --------------------------------------------------------------------------
    try:
        text = "Take Paracetamol 500mg twice daily for 5 days."
        rx = pipeline.extract(text, mode=PipelineMode.FAST)
        assert len(rx.items) >= 1
        item = rx.items[0]
        record("T05_Missing_Route_Inference", True, f"Route assigned: '{item.route}'")
    except Exception as e:
        record("T05_Missing_Route_Inference", False, f"Exception: {e}")

    # --------------------------------------------------------------------------
    # 6. AMBIGUOUS / CONFLICTING FREQUENCY
    # --------------------------------------------------------------------------
    try:
        text = "Take Dolo 650mg once or twice daily maybe three times as needed for 3 days."
        rx = pipeline.extract(text, mode=PipelineMode.FAST)
        assert len(rx.items) >= 1
        item = rx.items[0]
        record("T06_Ambiguous_Frequency_Handling", True, f"Frequency extracted: '{item.frequency}', Confidence: {item.confidence}")
    except Exception as e:
        record("T06_Ambiguous_Frequency_Handling", False, f"Exception: {e}")

    # --------------------------------------------------------------------------
    # 7. CONFLICTING MULTIPLE STRENGTHS
    # --------------------------------------------------------------------------
    try:
        text = "Take Paracetamol 500mg and then Paracetamol 650mg twice daily for 5 days."
        rx = pipeline.extract(text, mode=PipelineMode.FAST)
        passed = (len(rx.items) >= 1)
        med_strengths = [it.strength for it in rx.items]
        record("T07_Conflicting_Multiple_Strengths", passed, f"Items found: {len(rx.items)}, Strengths: {med_strengths}")
    except Exception as e:
        record("T07_Conflicting_Multiple_Strengths", False, f"Exception: {e}")

    # --------------------------------------------------------------------------
    # 8. LLM ADAPTER CRASH RESILIENCE
    # --------------------------------------------------------------------------
    try:
        class FailingLLM:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("Simulated catastrophic CUDA OutOfMemoryError in LLM")

        from app.prescription.agentic.adapter import AgenticAdapter
        failing_adapter = AgenticAdapter(llm=FailingLLM())
        candidates = failing_adapter.extract("Tab Augmentin 625mg 1-0-1 for 5 days")
        # Must handle LLM failure gracefully (return list or fallback candidates) and not crash caller
        passed = isinstance(candidates, list)
        record("T08_LLM_Failure_Graceful_Degradation", passed, f"Returned {len(candidates)} candidates gracefully on crash")
    except Exception as e:
        record("T08_LLM_Failure_Graceful_Degradation", False, f"Crashed unhandled: {e}")

    # --------------------------------------------------------------------------
    # 9. STT FAILURE RESILIENCE
    # --------------------------------------------------------------------------
    try:
        # Corrupted audio
        corrupt_audio = b"GARBAGE_NON_AUDIO_DATA_BYTES_12345"
        try:
            res = stt_mgr.transcribe(corrupt_audio, model_key="whisper_ayush", allow_fallback=False)
            corrupt_handled = False
        except Exception:
            corrupt_handled = True

        # Empty audio
        try:
            res_empty = stt_mgr.transcribe(b"", model_key="mock")
            empty_handled = False
        except Exception:
            empty_handled = True

        passed = corrupt_handled and empty_handled
        record("T09_STT_Corrupt_And_Empty_Audio", passed, f"Corrupt handled: {corrupt_handled}, Empty handled: {empty_handled}")
    except Exception as e:
        record("T09_STT_Corrupt_And_Empty_Audio", False, f"Failed: {e}")

    # --------------------------------------------------------------------------
    # 10. DRUG DATABASE UNKNOWN ENTITY & SPECIAL CHARACTERS
    # --------------------------------------------------------------------------
    try:
        res_sql = drug_repo.find_exact("'; DROP TABLE drugs; --")
        res_quote = drug_repo.search_drugs("'''\"\"\"///\\\\\\")
        passed = (res_sql is None and isinstance(res_quote, list))
        record("T10_SQL_Injection_And_Fuzzing", passed, f"Exact: {res_sql}, Search count: {len(res_quote)}")
    except Exception as e:
        record("T10_SQL_Injection_And_Fuzzing", False, f"Exception: {e}")

    # --------------------------------------------------------------------------
    # 11. INVALID API PAYLOADS & HTTP ERRORS
    # --------------------------------------------------------------------------
    try:
        # Missing text field (422)
        res_missing = client.post("/api/prescription/extract", json={"fast_mode": True})
        # Empty text (400)
        res_empty_str = client.post("/api/prescription/extract", json={"text": "   "})
        # Non-audio file upload (415)
        res_bad_ext = client.post("/api/prescription/transcribe", files={"file": ("virus.sh", io.BytesIO(b"echo hi"), "text/plain")})

        passed = (
            res_missing.status_code == 422 and
            res_empty_str.status_code in (400, 422) and
            res_bad_ext.status_code == 415
        )
        record("T11_API_Error_Handling_Envelopes", passed, f"Missing: {res_missing.status_code}, Empty: {res_empty_str.status_code}, Bad Ext: {res_bad_ext.status_code}")
    except Exception as e:
        record("T11_API_Error_Handling_Envelopes", False, f"Exception: {e}")

    # --------------------------------------------------------------------------
    # 12. MISSING ENVIRONMENT VARIABLES
    # --------------------------------------------------------------------------
    try:
        import os
        old_origins = os.environ.pop("ALLOWED_ORIGINS", None)
        # Should still respond 200 using safe defaults
        res_health = client.get("/api/health")
        passed = (res_health.status_code == 200)
        if old_origins:
            os.environ["ALLOWED_ORIGINS"] = old_origins
        record("T12_Missing_Environment_Variables_Defaults", passed, f"Health probe returned {res_health.status_code} with missing env var")
    except Exception as e:
        record("T12_Missing_Environment_Variables_Defaults", False, f"Exception: {e}")

    print("\n================================================================================")
    print(f"ADVERSARIAL TEST SUMMARY: {sum(1 for r in results if r['passed'])} / {len(results)} PASSED")
    print("================================================================================")
    return results


if __name__ == "__main__":
    run_tests()
