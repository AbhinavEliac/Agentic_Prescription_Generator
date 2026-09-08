"""
tests/clinical/evaluate_pipeline.py
-----------------------------------
Automated Evaluation and Benchmarking Engine for the Golden Dataset.
Tests both FAST and STANDARD modes against tests/clinical/golden_dataset.json.
Evaluates field accuracy, FP/FN rates, safety invariants, and STT-to-extraction fidelity.
Outputs empirical results and structured data for docs/CLINICAL_QA_REPORT.md.
"""

import os
import sys
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_APP_DIR = os.path.join(_PROJECT_ROOT, "rx_extractor_app")
if _APP_DIR not in sys.path:
    sys.path.append(_APP_DIR)
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from app.prescription.pipeline import PrescriptionPipeline, PipelineMode
from app.prescription.schema import ExtractionStatus
from app.stt import get_stt_manager


def _normalize_string(s: Optional[str]) -> str:
    """Normalizes string for clinical comparison."""
    if s is None:
        return ""
    # remove punctuation and extra spaces
    cleaned = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(cleaned.split())


def _matches_expected(extracted: Optional[str], expected: Optional[str]) -> bool:
    """Checks if extracted field matches expected clinical ground truth."""
    if expected is None:
        return extracted is None
    if extracted is None:
        return False
    norm_ext = _normalize_string(extracted)
    norm_exp = _normalize_string(expected)
    if norm_exp in norm_ext or norm_ext in norm_exp:
        return True
    return False


def evaluate_dataset(mode: PipelineMode) -> Dict[str, Any]:
    """Runs evaluation across all cases in golden_dataset.json for a given pipeline mode."""
    dataset_path = os.path.join(_CURRENT_DIR, "golden_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    pipeline = PrescriptionPipeline()

    total_cases = len(cases)
    passed_cases = 0
    failed_cases = 0

    field_stats = {
        "medicine": {"tp": 0, "fp": 0, "fn": 0, "total_expected": 0},
        "strength": {"tp": 0, "fp": 0, "fn": 0, "total_expected": 0},
        "frequency": {"tp": 0, "fp": 0, "fn": 0, "total_expected": 0},
        "duration": {"tp": 0, "fp": 0, "fn": 0, "total_expected": 0},
        "route": {"tp": 0, "fp": 0, "fn": 0, "total_expected": 0},
        "status": {"tp": 0, "fp": 0, "fn": 0, "total_expected": 0},
    }

    hallucinations = {
        "fabricated_strength": 0,
        "fabricated_duration": 0,
        "fabricated_route": 0,
        "failed_unknown_med_review": 0,
        "failed_ambiguous_freq_review": 0,
    }

    category_results: Dict[str, Dict[str, int]] = {}
    latencies: List[float] = []
    case_details: List[Dict[str, Any]] = []

    for case in cases:
        c_id = case["id"]
        cat = case["category"]
        raw_text = case["input_text"]
        expected_items = case["expected_items"]
        invariants = case.get("safety_invariants", {})

        if cat not in category_results:
            category_results[cat] = {"total": 0, "passed": 0}
        category_results[cat]["total"] += 1

        t0 = time.perf_counter()
        result = pipeline.extract(raw_text, mode=mode)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(elapsed_ms)

        ext_items = result.items
        case_passed = True
        case_failures: List[str] = []

        # Check item counts
        if len(ext_items) != len(expected_items):
            case_passed = False
            case_failures.append(f"Count mismatch: expected {len(expected_items)}, got {len(ext_items)}")

        # Evaluate pairs
        for idx, exp in enumerate(expected_items):
            field_stats["medicine"]["total_expected"] += 1
            if exp.get("strength") is not None:
                field_stats["strength"]["total_expected"] += 1
            if exp.get("frequency") is not None:
                field_stats["frequency"]["total_expected"] += 1
            if exp.get("duration") is not None:
                field_stats["duration"]["total_expected"] += 1
            if exp.get("route") is not None:
                field_stats["route"]["total_expected"] += 1
            field_stats["status"]["total_expected"] += 1

            if idx >= len(ext_items):
                case_passed = False
                case_failures.append(f"Missing item {idx+1}: {exp.get('medicine_name')}")
                field_stats["medicine"]["fn"] += 1
                if exp.get("strength") is not None:
                    field_stats["strength"]["fn"] += 1
                if exp.get("frequency") is not None:
                    field_stats["frequency"]["fn"] += 1
                if exp.get("duration") is not None:
                    field_stats["duration"]["fn"] += 1
                if exp.get("route") is not None:
                    field_stats["route"]["fn"] += 1
                field_stats["status"]["fn"] += 1
                continue

            act = ext_items[idx]

            # 1. Medicine Name Check
            if _matches_expected(act.medicine_name, exp.get("medicine_name")):
                field_stats["medicine"]["tp"] += 1
            else:
                field_stats["medicine"]["fp"] += 1
                case_passed = False
                case_failures.append(f"Medicine mismatch: expected '{exp.get('medicine_name')}', got '{act.medicine_name}'")

            # 2. Strength Check
            exp_str = exp.get("strength")
            act_str = act.strength
            if exp_str is None:
                if act_str is not None and not invariants.get("allow_null_strength", True):
                    field_stats["strength"]["fp"] += 1
                    hallucinations["fabricated_strength"] += 1
                    case_passed = False
                    case_failures.append(f"Fabricated strength: expected null, got '{act_str}'")
            else:
                if _matches_expected(act_str, exp_str):
                    field_stats["strength"]["tp"] += 1
                else:
                    if act_str is None:
                        field_stats["strength"]["fn"] += 1
                    else:
                        field_stats["strength"]["fp"] += 1
                    case_passed = False
                    case_failures.append(f"Strength mismatch: expected '{exp_str}', got '{act_str}'")

            # 3. Frequency Check
            exp_freq = exp.get("frequency")
            act_freq = act.frequency
            if exp_freq is None:
                if act_freq is not None:
                    field_stats["frequency"]["fp"] += 1
            else:
                if _matches_expected(act_freq, exp_freq):
                    field_stats["frequency"]["tp"] += 1
                else:
                    if act_freq is None:
                        field_stats["frequency"]["fn"] += 1
                    else:
                        field_stats["frequency"]["fp"] += 1
                    case_passed = False
                    case_failures.append(f"Frequency mismatch: expected '{exp_freq}', got '{act_freq}'")

            # 4. Duration Check
            exp_dur = exp.get("duration")
            act_dur = act.duration
            if exp_dur is None:
                if act_dur is not None and not invariants.get("allow_null_duration", True):
                    field_stats["duration"]["fp"] += 1
                    hallucinations["fabricated_duration"] += 1
                    case_passed = False
                    case_failures.append(f"Fabricated duration: expected null, got '{act_dur}'")
            else:
                if _matches_expected(act_dur, exp_dur):
                    field_stats["duration"]["tp"] += 1
                else:
                    if act_dur is None:
                        field_stats["duration"]["fn"] += 1
                    else:
                        field_stats["duration"]["fp"] += 1
                    case_passed = False
                    case_failures.append(f"Duration mismatch: expected '{exp_dur}', got '{act_dur}'")

            # 5. Route Check
            exp_route = exp.get("route")
            act_route = act.route
            if exp_route is None:
                if act_route is not None and not invariants.get("allow_null_route", True):
                    field_stats["route"]["fp"] += 1
                    hallucinations["fabricated_route"] += 1
                    case_passed = False
                    case_failures.append(f"Fabricated route: expected null, got '{act_route}'")
            else:
                if _matches_expected(act_route, exp_route):
                    field_stats["route"]["tp"] += 1
                else:
                    if act_route is None:
                        field_stats["route"]["fn"] += 1
                    else:
                        field_stats["route"]["fp"] += 1
                    case_passed = False
                    case_failures.append(f"Route mismatch: expected '{exp_route}', got '{act_route}'")

            # 6. Safety Status Check
            exp_status = exp.get("expected_status")
            act_status = act.status.value if hasattr(act.status, "value") else str(act.status)
            if exp_status:
                if act_status == exp_status:
                    field_stats["status"]["tp"] += 1
                else:
                    field_stats["status"]["fp"] += 1
                    case_passed = False
                    case_failures.append(f"Status mismatch: expected '{exp_status}', got '{act_status}'")
                    if exp_status == "NEEDS_REVIEW":
                        if cat == "unknown_medicine_trap":
                            hallucinations["failed_unknown_med_review"] += 1
                        elif cat == "ambiguous_instructions":
                            hallucinations["failed_ambiguous_freq_review"] += 1

        # Check safety invariant: requires_review
        if invariants.get("requires_review"):
            has_review_status = (result.overall_status == ExtractionStatus.NEEDS_REVIEW) or any(
                it.status == ExtractionStatus.NEEDS_REVIEW or it.review_reasons for it in ext_items
            )
            if not has_review_status:
                case_passed = False
                case_failures.append("Safety violation: case requires_review was True, but status was not NEEDS_REVIEW")

        if case_passed:
            passed_cases += 1
            category_results[cat]["passed"] += 1
        else:
            failed_cases += 1

        case_details.append({
            "id": c_id,
            "category": cat,
            "passed": case_passed,
            "failures": case_failures,
            "latency_ms": elapsed_ms,
        })

    # Calculate field metrics
    field_metrics = {}
    for fld, counts in field_stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        total_exp = counts["total_expected"]
        precision = round(tp / max(1, tp + fp), 4)
        recall = round(tp / max(1, total_exp), 4)
        fpr = round(fp / max(1, tp + fp), 4)
        fnr = round(fn / max(1, total_exp), 4)
        accuracy = round(tp / max(1, tp + fp + fn), 4)
        field_metrics[fld] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_expected": total_exp,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        }

    avg_latency = round(sum(latencies) / max(1, len(latencies)), 2)
    p95_latency = round(sorted(latencies)[int(0.95 * len(latencies))], 2) if latencies else 0.0

    return {
        "mode": mode.value,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": round(passed_cases / max(1, total_cases) * 100, 2),
        "field_metrics": field_metrics,
        "hallucinations": hallucinations,
        "category_results": category_results,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "case_details": case_details,
    }


def evaluate_stt_pipeline() -> Dict[str, Any]:
    """Evaluates STT audio transcription feeding directly into PrescriptionPipeline."""
    mgr = get_stt_manager()
    pipeline = PrescriptionPipeline()

    audio_file = os.path.join(_PROJECT_ROOT, "rx_extractor_app", "data", "audio_recordings", "proc_70_20260828_134911.wav")
    has_real_audio = os.path.exists(audio_file)

    test_audio = audio_file if has_real_audio else None

    results = []

    # 1. Real / mock speech note audio test
    if test_audio:
        t0 = time.perf_counter()
        asr_res = mgr.transcribe(test_audio, model_key="whisper_ayush")
        asr_dur = round((time.perf_counter() - t0) * 1000, 2)

        rx_res = pipeline.extract(asr_res.text, mode=PipelineMode.FAST)
        results.append({
            "test_type": "end_to_end_audio",
            "audio_source": os.path.basename(test_audio),
            "transcript": asr_res.text,
            "model_used": asr_res.model,
            "asr_latency_ms": asr_dur,
            "extracted_items": [it.model_dump() for it in rx_res.items],
            "success": len(rx_res.items) > 0,
        })

    # 2. Mock engine test
    wav_header = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00"
        b"\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    mock_asr = mgr.transcribe(wav_header, model_key="mock")
    rx_res_mock = pipeline.extract(mock_asr.text, mode=PipelineMode.FAST)
    results.append({
        "test_type": "mock_stt_audio",
        "transcript": mock_asr.text,
        "model_used": "mock",
        "extracted_items": [it.model_dump() for it in rx_res_mock.items],
        "success": len(rx_res_mock.items) == 1 and "Paracetamol" in rx_res_mock.items[0].medicine_name,
    })

    return {
        "total_stt_tests": len(results),
        "tests": results,
    }


def main():
    print("=================================================================")
    print(" CLINICAL REGRESSION & SAFETY TEST SYSTEM: EVALUATION HARNESS")
    print("=================================================================\n")

    print("[1/3] Running FAST mode evaluation on 60 golden test cases...")
    fast_results = evaluate_dataset(PipelineMode.FAST)
    print(f"      FAST Pass Rate: {fast_results['pass_rate']}% ({fast_results['passed_cases']}/{fast_results['total_cases']})")
    print(f"      FAST Avg Latency: {fast_results['avg_latency_ms']} ms | P95: {fast_results['p95_latency_ms']} ms")

    print("\n[2/3] Running STANDARD mode evaluation on 60 golden test cases...")
    std_results = evaluate_dataset(PipelineMode.STANDARD)
    print(f"      STANDARD Pass Rate: {std_results['pass_rate']}% ({std_results['passed_cases']}/{std_results['total_cases']})")
    print(f"      STANDARD Avg Latency: {std_results['avg_latency_ms']} ms | P95: {std_results['p95_latency_ms']} ms")

    print("\n[3/3] Running STT -> Extraction pipeline evaluation...")
    stt_results = evaluate_stt_pipeline()
    print(f"      STT Tests Executed: {stt_results['total_stt_tests']}")

    out_file = os.path.join(_CURRENT_DIR, "evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fast_mode": fast_results,
            "standard_mode": std_results,
            "stt_pipeline": stt_results,
        }, f, indent=2)

    print(f"\n[Done] Detailed evaluation results saved to: {out_file}\n")

    # Print Category Breakdown for FAST mode
    print("=================================================================")
    print(" CATEGORY BREAKDOWN (FAST MODE)")
    print("=================================================================")
    for cat, res in fast_results["category_results"].items():
        status_str = "PASS" if res["passed"] == res["total"] else "FAIL"
        print(f"  {cat:<30} : {res['passed']}/{res['total']} passed [{status_str}]")

    print("\n=================================================================")
    print(" FIELD-LEVEL METRICS (FAST MODE)")
    print("=================================================================")
    print(f"  {'Field':<12} | {'Prec':<7} | {'Recall':<7} | {'FPR':<7} | {'FNR':<7} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("  " + "-" * 62)
    for fld, m in fast_results["field_metrics"].items():
        print(f"  {fld:<12} | {m['precision']:<7.2f} | {m['recall']:<7.2f} | {m['false_positive_rate']:<7.2f} | {m['false_negative_rate']:<7.2f} | {m['tp']:<4} | {m['fp']:<4} | {m['fn']:<4}")

    print("\n=================================================================")
    print(" SAFETY INVARIANTS & HALLUCINATIONS (FAST MODE)")
    print("=================================================================")
    for k, v in fast_results["hallucinations"].items():
        print(f"  {k:<32} : {v}")

    # Print failed cases if any
    failed = [c for c in fast_results["case_details"] if not c["passed"]]
    if failed:
        print("\n=================================================================")
        print(f" FAILED CASES ({len(failed)})")
        print("=================================================================")
        for f_case in failed:
            print(f"  [{f_case['id']}] {f_case['category']}:")
            for fail in f_case["failures"]:
                print(f"    - {fail}")


if __name__ == "__main__":
    main()
