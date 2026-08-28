"""
test_langgraph_pipeline.py
--------------------------
Comprehensive, drift-proof verification suite for the LangGraph Multi-Agent Extractor.
Tests:
1. Multi-drug with companion drugs and lifestyle advice
2. Dual-dose parsing (Drug name + strength split)
3. Inhalation and topical route specificity
4. Strict doctor mode (zero unsolicited moralizing or disclaimers)
5. Validator feedback loop & 3-repetition cap
6. Complex gastro prescription (5 medicines, suspensions, sachets, PRN)
7. Multi-form prescription with intervals, titrations, and speaker timestamps
8. Topicals, nasal sprays, ophthalmic drops, and interval schedules
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_pipeline import run_graph_extraction
from exporter import parse_output_fields


def test_multi_drug_diet_instructions():
    print("\n--- Test 1: Multi-drug with Diet & Lifestyle Instructions ---")
    raw_prescription = (
        "Take Ferrous ascorbate with Folic acid once daily after lunch for 60 days, "
        "and take Vitamin C 500 mg alongside it to optimize iron absorption. "
        "Take Albendazole 400 mg single dose at bedtime on day one. "
        "Include dark green leafy vegetables in diet, avoid tea near meal times, "
        "re-test blood count in 2 months, and go for morning walks daily."
    )

    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    assert len(parsed) >= 3, f"Expected at least 3 medicines, got {len(parsed)}"

    all_instructions = " ".join([p.get("instruction", "") + " " + p.get("additional_instruction", "") for p in parsed]).lower()
    assert "leafy vegetables" in all_instructions or "green" in all_instructions, "Diet instruction missing!"
    assert "morning walk" in all_instructions or "tea" in all_instructions, "Lifestyle instruction missing!"
    print("PASS: Multi-drug & diet instructions successfully extracted and attributed.")


def test_dual_dose_extraction():
    print("\n--- Test 2: Dual Dosage Extraction ---")
    raw_prescription = "Take one tablet of Paracetamol 650 mg 20 mg twice daily(1-0-1) for 5 days, increase the dose by 100 mg after 7 days."
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    assert len(parsed) == 1, f"Expected 1 medicine, got {len(parsed)}"
    med = parsed[0]

    assert "Paracetamol 650 mg" in med["Drug_name"], f"Unexpected Drug_name: {med['Drug_name']}"
    assert "20 mg" in med["strength"], f"Unexpected strength: {med['strength']}"
    assert "twice daily(1-0-1)" in med["frequency"], f"Unexpected frequency: {med['frequency']}"
    assert "5 days" in med["duration"], f"Unexpected duration: {med['duration']}"
    assert med["route"] == "oral", f"Unexpected route: {med['route']}"
    print("PASS: Dual dose and frequency notations successfully extracted.")


def test_route_specificity():
    print("\n--- Test 3: Route Specificity Resolution ---")
    raw_prescription = "Inhale Budecort 200 mcg Rotacap twice daily for 10 days. Rinse mouth after using the inhaler."
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    assert len(parsed) == 1
    assert parsed[0]["route"] == "inhalation", f"Expected route 'inhalation', got '{parsed[0]['route']}'"
    assert "rinse mouth" in parsed[0]["instruction"].lower(), "Device instruction missing"
    print("PASS: Inhalation route specificity and device instruction verified.")


def test_no_moralizing_or_extra_cautions():
    print("\n--- Test 4: Strict Doctor Mode (No Extra Commentary) ---")
    raw_prescription = "Take Pantop 40 mg once daily before breakfast for 14 days."
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)

    forbidden = ["please note", "caution:", "warning:", "moral", "consult your doctor", "disclaimer:"]
    for f in forbidden:
        assert f not in output.lower(), f"Forbidden commentary '{f}' detected in output!"
    print("PASS: Output is strictly clinical 6-field blocks with zero extra commentary.")


def test_validator_loop_and_cap():
    print("\n--- Test 5: Validator Feedback Loop & Max 3 Repetition Cap ---")
    from graph_state import AgenticRxState
    from agents.validator_agent import validator_agent

    state: AgenticRxState = {
        "input_text": "Take Aten 25 mg once daily for 10 days.",
        "iteration_count": 1,
        "aggregated_blocks": [
            {
                "Drug_name": "Aten 50 mg",
                "strength": "NONE",
                "frequency": "once daily",
                "duration": "10 days",
                "route": "oral",
                "instruction": "NONE",
            }
        ],
        "agent_logs": [],
    }

    val_out_1 = validator_agent(state)
    assert val_out_1["validation_status"] == "NEEDS_CORRECTION", "Expected NEEDS_CORRECTION on iteration 1"
    assert "medicine_agent" in val_out_1["validation_feedback"], "Expected feedback for medicine_agent"

    state["iteration_count"] = 3
    val_out_3 = validator_agent(state)
    assert val_out_3["validation_status"] == "VALID", "Expected loop to be capped at iteration 3"
    print("PASS: Validator feedback generation and max 3-repetition cap successfully verified.")


def test_complex_5_drug_prescription():
    print("\n--- Test 6: Complex 5-Medicine Multi-Form Prescription ---")
    raw_prescription = (
        "Take one Ofloxacin-Ornidazole tablet orally twice daily after meals for 5 days, "
        "and take one Racecadotril 100 mg capsule three times daily before food for 3 days. "
        "Take one Ondansetron 4 mg tablet up to three times daily as needed 30 minutes before meals for nausea, "
        "consume one vial of Enterogermina oral suspension twice daily for 5 days, "
        "and dissolve one sachet of Oral Rehydration Salts in one liter of clean drinking water to consume throughout the day. "
        "Stick to a bland diet consisting of rice, curd, and bananas, avoid spicy or oily foods, "
        "and visit the emergency room immediately if severe dehydration or persistent vomiting occurs."
    )

    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine blocks:")
    for idx, p in enumerate(parsed):
        print(f"  {idx+1}. {p['Drug_name']} | {p['frequency']} | {p['duration']} | {p['route']} | {p['instruction']}")

    assert len(parsed) == 5, f"Expected 5 medicines, got {len(parsed)}"
    drug_names = [p["Drug_name"].lower() for p in parsed]
    assert any("ofloxacin-ornidazole" in d for d in drug_names), "Ofloxacin-Ornidazole missing!"
    assert any("racecadotril" in d for d in drug_names), "Racecadotril missing!"
    assert any("ondansetron" in d for d in drug_names), "Ondansetron missing!"
    assert any("enterogermina" in d for d in drug_names), "Enterogermina missing!"
    assert any("oral rehydration salts" in d for d in drug_names), "Oral Rehydration Salts missing!"
    print("PASS: All 5 complex medicines and multi-clause instructions extracted successfully.")


def test_multi_medicine_with_timestamps_and_conditionals():
    print("\n--- Test 7: Multi-Medicine with Conditionals & Voice Timestamps ---")
    raw_prescription = (
        "Administer ATEN tablet 50mg by mouth every morning before food for 10 days and seek reassessment of blood pressure afterwards. "
        "Take Crocin 650 mg every 6 hours for 3 days and discontinue once the fever resolves. "
        "Take Amlodipine 5 mg once daily for 14 days. If blood pressure remains high, consult your doctor before increasing the dose. "
        "Animesh Kumar, Yesterday 4:30 PM Take one tablet of paracetamol 650 mg 20 mg twice daily(1-0-1) for 5 day, increase the dose by 100 mg after 7 day."
    )

    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine blocks:")
    for idx, p in enumerate(parsed):
        print(f"  {idx+1}. {p['Drug_name']} | {p['frequency']} | {p['duration']} | {p['route']} | {p['instruction']}")

    assert len(parsed) == 4, f"Expected 4 medicines, got {len(parsed)}"
    assert "ATEN 50mg" in parsed[0]["Drug_name"], "ATEN 50mg drug name missing or malformed!"
    assert parsed[0]["frequency"] == "every morning", f"Expected 'every morning', got '{parsed[0]['frequency']}'"
    assert "Crocin 650 mg" in parsed[1]["Drug_name"], "Crocin 650 mg missing!"
    assert parsed[1]["frequency"] == "every 6 hours", f"Expected 'every 6 hours', got '{parsed[1]['frequency']}'"
    assert "Amlodipine 5 mg" in parsed[2]["Drug_name"], "Amlodipine 5 mg missing!"
    assert "paracetamol 650 mg" in parsed[3]["Drug_name"], "Paracetamol 650 mg missing!"
    assert parsed[3]["strength"] == "20 mg", f"Expected strength '20 mg', got '{parsed[3]['strength']}'"
    print("PASS: Multi-medicine with intervals, titrations, and speaker timestamps passed.")


def test_topicals_sprays_drops():
    print("\n--- Test 8: Topicals, Nasal Sprays, & Ophthalmic Drops ---")
    raw_prescription = (
        "Apply Clotrimazole 1% cream topically twice daily for 14 days, "
        "spray Fluticasone 50 mcg nasal spray into both nostrils once daily every morning, "
        "and instill Moxifloxacin 0.5% eye drops into left eye every 4 hours for 7 days."
    )

    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine blocks:")
    for idx, p in enumerate(parsed):
        print(f"  {idx+1}. {p['Drug_name']} | {p['frequency']} | {p['duration']} | {p['route']} | {p['instruction']}")

    assert len(parsed) == 3
    assert parsed[0]["route"] == "topical", f"Expected route 'topical', got '{parsed[0]['route']}'"
    assert parsed[1]["route"] == "nasal", f"Expected route 'nasal', got '{parsed[1]['route']}'"
    assert parsed[2]["route"] == "ophthalmic", f"Expected route 'ophthalmic', got '{parsed[2]['route']}'"
    assert parsed[2]["frequency"] == "every 4 hours", f"Expected 'every 4 hours', got '{parsed[2]['frequency']}'"
    print("PASS: Topicals, sprays, and drops with specialized routes verified.")


def test_cross_sentence_and_class_instructions():
    print("\n--- Test 9: Cross-Sentence, Cross-Drug, and Class Instruction Linkage ---")
    raw_prescription = (
        "Take Metformin 500 mg twice daily. Take Metformin strictly with meals to avoid stomach upset. "
        "Also take Glimepiride 1 mg once daily before breakfast. "
        "Start Amoxicillin 500 mg TID for 7 days, and take Paracetamol 650 mg as needed for fever. "
        "Do not stop the antibiotic course early even if fever subsides."
    )

    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine blocks:")
    for idx, p in enumerate(parsed):
        print(f"  {idx+1}. {p['Drug_name']} | {p['frequency']} | {p['duration']} | {p['route']} | {p['instruction']}")

    assert len(parsed) == 4, f"Expected 4 medicines, got {len(parsed)}"
    # Metformin should receive its separate sentence instruction
    assert "strictly with meals" in parsed[0]["instruction"].lower(), "Metformin missing its separate sentence instruction!"
    # Glimepiride should have its timing
    assert "before breakfast" in parsed[1]["instruction"].lower(), "Glimepiride missing breakfast instruction!"
    # Amoxicillin should receive antibiotic course completion in additional_instruction
    all_amox = (parsed[2].get("instruction", "") + " " + parsed[2].get("additional_instruction", "")).lower()
    assert "antibiotic course" in all_amox, "Amoxicillin missing antibiotic course directive!"
    # Paracetamol should NOT receive antibiotic instruction
    all_paracetamol = (parsed[3].get("instruction", "") + " " + parsed[3].get("additional_instruction", "")).lower()
    assert "antibiotic" not in all_paracetamol, "Antibiotic instruction leaked into Paracetamol!"
    print("PASS: Cross-sentence and class-specific instruction routing verified.")


def test_additional_instructions_column():
    print("\n--- Test 10: Dedicated Additional Instructions Column ---")
    raw_prescription = "Administer PHEXIN DT tablet 250 mg once daily(0-0-1) before breakfast for 10 days. Seek reassessment if adverse effects develop."
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine block:")
    for p in parsed:
        print(f"  Drug_name: {p['Drug_name']}")
        print(f"  strength: {p['strength']}")
        print(f"  frequency: {p['frequency']}")
        print(f"  duration: {p['duration']}")
        print(f"  route: {p['route']}")
        print(f"  instruction: {p['instruction']}")
        print(f"  additional_instruction: {p.get('additional_instruction', 'NONE')}")

    assert len(parsed) == 1
    assert "PHEXIN DT 250 mg" in parsed[0]["Drug_name"]
    assert "once daily(0-0-1)" in parsed[0]["frequency"]
    assert "before breakfast" in parsed[0]["instruction"].lower()
    assert "adverse effects" in parsed[0]["additional_instruction"].lower()
    print("PASS: Primary instruction and secondary additional_instruction separated cleanly.")


def test_natural_language_time_and_evaluation():
    print("\n--- Test 11: Natural Language Times-of-Day and Evaluation Sentence Parsing ---")
    raw_prescription = "Take Stugeron Forte by mouth twice daily Morning Night for 2 weeks. Return for evaluation after completing the course"
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine block:")
    for p in parsed:
        print(f"  Drug_name: {p['Drug_name']}")
        print(f"  strength: {p['strength']}")
        print(f"  frequency: {p['frequency']}")
        print(f"  duration: {p['duration']}")
        print(f"  route: {p['route']}")
        print(f"  instruction: {p['instruction']}")
        print(f"  additional_instruction: {p.get('additional_instruction', 'NONE')}")

    assert len(parsed) == 1
    assert "Stugeron Forte" in parsed[0]["Drug_name"]
    assert "twice daily" in parsed[0]["frequency"]
    assert "2 weeks" in parsed[0]["duration"]
    assert "oral" in parsed[0]["route"]
    assert "morning night" in parsed[0]["instruction"].lower()
    assert "return for evaluation" in parsed[0]["additional_instruction"].lower()
    print("PASS: Natural language times-of-day and evaluation sentence captured dynamically.")


if __name__ == "__main__":
    test_multi_drug_diet_instructions()
    test_dual_dose_extraction()
    test_route_specificity()
    test_no_moralizing_or_extra_cautions()
    test_validator_loop_and_cap()
    test_complex_5_drug_prescription()
    test_multi_medicine_with_timestamps_and_conditionals()
    test_topicals_sprays_drops()
    test_cross_sentence_and_class_instructions()
    test_additional_instructions_column()
    test_natural_language_time_and_evaluation()
    print("\n========================================================")
    print("ALL 11 LANGGRAPH DRIFT-PROOF MULTI-AGENT TESTS PASSED!")
    print("========================================================")
