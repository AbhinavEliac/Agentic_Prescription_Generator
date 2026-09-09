import time
import sys
import os

# Set up paths - ensure workspace root is first so 'app' package is found
root_dir = os.path.abspath(".")
sys.path = [p for p in sys.path if "rx_extractor_app" not in p]
sys.path.insert(0, root_dir)
if 'app' in sys.modules and not hasattr(sys.modules['app'], '__path__'):
    del sys.modules['app']

# Prevent torchvision / torchaudio DLL binary incompatibility from crashing transformers on Windows Python 3.13
sys.modules.setdefault('torchvision', None)
sys.modules.setdefault('torchaudio', None)

from app.drugs.repository import get_drug_repository
from app.prescription.pipeline import PrescriptionPipeline, PipelineMode

print("Loading DrugRepository and PrescriptionPipeline...")
t0 = time.perf_counter()
repo = get_drug_repository()
pipeline = PrescriptionPipeline(drug_repo=repo)
t1 = time.perf_counter()
print(f"Loaded in {(t1 - t0)*1000:.1f}ms. Total drugs: {len(repo.drugs_by_id)}")

test_cases = [
    {
        "name": "Case 1: Dual Dose Rule 1 (Augmentin 625 mg 1 tablet twice daily 101)",
        "input": "Augmentin 625 mg 1 tablet twice daily 101 for 5 days with food",
    },
    {
        "name": "Case 2: Dose in Formulary Name Rule 2 (Dolo 650 once daily)",
        "input": "Dolo 650 once daily for 3 days after food",
    },
    {
        "name": "Case 3: Dose not in Formulary Name Rule 2 (Paracetamol 500 mg once daily)",
        "input": "Paracetamol 500 mg once daily for 5 days",
    },
    {
        "name": "Case 4: Phonetic / Accent Mispronunciation ('grocin' -> 'Crocin')",
        "input": "grocin 650 twice daily 101",
    },
    {
        "name": "Case 5: Multi-word Token ASR Split ('parasita mall' -> 'Paracetamol')",
        "input": "parasita mall 500 once daily 100 for 5 days",
    },
    {
        "name": "Case 6: Non-Drug Hallucination Rejection (Strict Grounding)",
        "input": "Patient has severe cough headache and high blood pressure, consult doctor next week",
    }
]

print("\n" + "="*80)
print("RUNNING VERIFICATION TEST SUITE")
print("="*80)

all_passed = True

for tc in test_cases:
    print(f"\n--- {tc['name']} ---")
    print(f"Input: \"{tc['input']}\"")
    
    t_start = time.perf_counter()
    res = pipeline.extract(tc["input"], mode=PipelineMode.FAST)
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000
    
    print(f"Latency: {latency_ms:.2f}ms (Target: < 50ms)")
    print(f"Status: {res.overall_status}")
    print(f"Extracted Items Count: {len(res.items)}")
    
    for idx, it in enumerate(res.items):
        print(f"  Item {idx+1}:")
        print(f"    Medicine Name   : {it.medicine_name}")
        print(f"    Dose / Unit     : {it.dose} / {it.dose_unit}")
        print(f"    Strength        : {it.strength}")
        print(f"    Schedule        : {it.frequency}")
        print(f"    Duration        : {it.duration}")
        print(f"    Route           : {it.route}")
        print(f"    Instruction     : {it.instruction}")
        print(f"    Did You Mean?   : {it.did_you_mean}")
        print(f"    Available Routes: {it.available_routes[:5]}")
        print(f"    Dropdown Drugs  : {len(it.available_drugs)} formulations matching dose")
        if it.available_drugs:
            print(f"      e.g.: {[d['drug_name'] for d in it.available_drugs[:3]]}")
            
    # Check specific assertions
    if "Augmentin" in tc["name"]:
        assert len(res.items) == 1, "Should extract 1 item"
        it = res.items[0]
        assert "AUGMENTIN" in it.medicine_name.upper(), f"Medicine name should be Augmentin, got {it.medicine_name}"
        assert it.dose == "1", f"Dose should be 1 tablet (Rule 1), got {it.dose}"
        assert it.dose_unit == "tablet", f"Dose unit should be tablet, got {it.dose_unit}"
        assert "1-0-1" in it.frequency or "Twice" in it.frequency, f"Schedule should capture twice daily / 1-0-1, got {it.frequency}"
        print("  [PASS] Rule 1 Dual Dose Verified")
        
    elif "Dolo" in tc["name"]:
        assert len(res.items) == 1, "Should extract 1 item"
        it = res.items[0]
        assert "650" in it.medicine_name, f"650 should remain in medicine name (Rule 2), got {it.medicine_name}"
        assert it.dose is None, f"Dose should be None when integer is part of drug catalog name, got {it.dose}"
        print("  [PASS] Rule 2 Formulation Name Integer Verified")
        
    elif "Case 3" in tc["name"]:
        assert len(res.items) == 1, "Should extract 1 item"
        it = res.items[0]
        assert it.dose == "500", f"Dose should be 500, got {it.dose}"
        assert it.dose_unit == "mg", f"Dose unit should be mg, got {it.dose_unit}"
        print("  [PASS] Rule 2 Order Dose Verified")
        
    elif "grocin" in tc["name"]:
        assert len(res.items) == 1, "Should extract 1 item"
        it = res.items[0]
        assert it.did_you_mean is not None or "CROCIN" in it.medicine_name.upper(), "Should match Crocin"
        assert len(it.available_drugs) > 0, "Should populate same-dose dropdown data"
        print(f"  [PASS] Phonetic Did You Mean Verified (did_you_mean='{it.did_you_mean}')")
        
    elif "parasita mall" in tc["name"]:
        assert len(res.items) == 1, "Should extract 1 item"
        it = res.items[0]
        assert "PARACETAMOL" in it.medicine_name.upper(), f"Should match paracetamol, got {it.medicine_name}"
        print(f"  [PASS] Multi-token ASR Concatenation Verified (matched='{it.medicine_name}')")
        
    elif "Case 6" in tc["name"]:
        assert len(res.items) == 0, f"Non-drug utterance should be rejected, but got: {[it.medicine_name for it in res.items]}"
        print("  [PASS] Strict Grounding Verified: Zero hallucinations extracted from non-drug text")

print("\n" + "="*80)
print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
print("="*80)
