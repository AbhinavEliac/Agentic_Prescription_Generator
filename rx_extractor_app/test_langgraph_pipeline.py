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


def test_punctuation_free_continuous_voice_speech():
    print("\n--- Test 12: Continuous Unpunctuated Voice Speech Extraction ---")
    raw_prescription = "Take this print 500mg 20mg tablets after breakfast if headache does not go away Meet the doctor also take walks after dinner and it will reduce your headaches Please see me after 7 days"
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
    assert "this print 500mg" in parsed[0]["Drug_name"]
    assert "20mg" in parsed[0]["strength"]
    assert "oral" in parsed[0]["route"]
    assert "after breakfast" in parsed[0]["instruction"].lower()
    assert "after dinner" not in parsed[0]["instruction"].lower()
    assert "doctor" in parsed[0]["additional_instruction"].lower()
    assert "walks" in parsed[0]["additional_instruction"].lower()
    print("PASS: Unpunctuated continuous voice speech and walking advice cleanly decoupled.")


def test_complex_multidrug_decimal_and_advice_guards():
    print("\n--- Test 13: Complex Multi-Drug Decimal Preservation & Non-Drug Guards ---")
    raw_prescription = (
        "Take Wallach Clover 1000 MCG Tablets, Overly 3 times Daily for 7 days and take 1 pre-gab ball in 75 MG capsule. "
        "Once daily at bedtime for 14 days, take 1 parasitamol 500 MG with Phradamol 37.5 MG tablet twice daily after meals for severe pain for 5 days. "
        "Apply Kalamine lotion gently over the close rash areas 3 times daily and take 1 methello glogamine 150,000 MG tablet daily after lunch for 30 days "
        "keep the blistered area clean and dry avoid close physical contact with pregnant individuals or non-immune persons and return if the rash involves the eye region."
    )
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine blocks:")
    for i, p in enumerate(parsed):
        print(f"  {i+1}. {p['Drug_name']} | {p['frequency']} | {p['duration']} | {p['route']} | {p['instruction']}")

    assert len(parsed) == 6
    assert any("37.5" in p["Drug_name"] for p in parsed), "Decimal 37.5 MG was truncated!"
    assert not any("pregnant" in p["Drug_name"].lower() for p in parsed), "Non-drug advice misclassified as medication!"
    assert "clean and dry" in parsed[-1]["additional_instruction"].lower()
    print("PASS: Decimals preserved, phonetic routes normalized, and non-drug advice guarded.")


def test_noisy_transcript_and_conversational_chatter_filtering():
    print("\n--- Test 14: Noisy Transcript & Conversational Chatter Filtering ---")
    raw_prescription = (
        "Good morning doctor. Hello Mr. Sharma, how are you feeling today? "
        "I have severe fever and throat pain since yesterday. Let me check your vitals. "
        "Temperature is 101 F, BP is 130/80 mmHg, chest is clear. "
        "Take Augmentin 625mg twice daily for 5 days and take Paracetamol 650mg SOS for fever. "
        "Gargle with warm salt water thrice daily and drink plenty of fluids. "
        "Thank you doctor, I will take care. Have a great day."
    )
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine blocks:")
    for i, p in enumerate(parsed):
        print(f"  {i+1}. {p['Drug_name']} | {p['frequency']} | {p['duration']} | {p['route']} | {p['instruction']}")

    assert len(parsed) == 2
    assert "Augmentin 625mg" in parsed[0]["Drug_name"]
    assert "Paracetamol 650mg" in parsed[1]["Drug_name"]
    for p in parsed:
        assert not any(noise in p["Drug_name"].lower() for noise in ("good morning", "sharma", "fever", "vitals", "130/80", "thank you", "great day"))
    assert "gargle" in parsed[-1]["additional_instruction"].lower()
def test_single_med_hydration_and_visit_doctor_advice():
    print("\n--- Test 15: Single Medicine with Hydration & Doctor Visit Advice ---")
    raw_prescription = "Take Disprin 500 mg tablets. If the fever does not go away, come visit the doctor. Take regular water."
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_prescription)
    parsed = parse_output_fields(output, query=raw_prescription)
    print(f"Extracted {len(parsed)} medicine blocks:")
    for i, p in enumerate(parsed):
        print(f"  {i+1}. {p['Drug_name']} | {p['route']} | {p['instruction']} | {p['additional_instruction']}")

    assert len(parsed) == 1
    assert "Disprin 500 mg" in parsed[0]["Drug_name"]
    assert "oral" in parsed[0]["route"]
    assert "visit the doctor" in parsed[0]["additional_instruction"].lower()
    assert "regular water" in parsed[0]["additional_instruction"].lower()
    print("PASS: Single medicine with hydration and doctor visit advice verified.")


def test_cross_sentence_coreference_frequency_resolution():
    print("\n--- Test 16: Cross-Sentence Coreference Frequency Resolution ---")
    # 1. Singular coreference ("It should be taken 4 times a day")
    raw_prescription_1 = "Take parasitamol 500 mg for 3 days. If the fever does not go away, come visit the doctor. It should be taken 4 times a day."
    output_1, _, _, _ = run_graph_extraction(None, raw_prescription_1)
    parsed_1 = parse_output_fields(output_1, query=raw_prescription_1)
    assert len(parsed_1) == 1
    assert "parasitamol 500 mg" in parsed_1[0]["Drug_name"]
    assert "4 times a day" in parsed_1[0]["frequency"].lower()
    assert "4 times a day" not in parsed_1[0]["additional_instruction"].lower()
    assert "visit the doctor" in parsed_1[0]["additional_instruction"].lower()

    # 2. Plural coreference ("Both should be taken twice daily after meals")
    raw_prescription_2 = "Take Pan 40 mg and Paracetamol 650 mg for 5 days. Both should be taken twice daily after meals. Drink plenty of water."
    output_2, _, _, _ = run_graph_extraction(None, raw_prescription_2)
    parsed_2 = parse_output_fields(output_2, query=raw_prescription_2)
    assert len(parsed_2) == 2
    assert "twice daily" in parsed_2[0]["frequency"].lower()
    assert "twice daily" in parsed_2[1]["frequency"].lower()
    assert "after meals" in parsed_2[0]["instruction"].lower()
    assert "after meals" in parsed_2[1]["instruction"].lower()
    print("PASS: Singular and plural cross-sentence coreference frequencies verified.")


def test_dosage_titration_instruction_capture():
    print("\n--- Test 17: Dosage Titration Instruction Capture ---")
    # 1. Unpunctuated speech with conditional titration
    raw_1 = "Take parasitamol teplis 500 mg for 3 days if the fever does not go away increase the dosage by 100 mg"
    output_1, _, _, _ = run_graph_extraction(None, raw_1)
    parsed_1 = parse_output_fields(output_1, query=raw_1)
    assert len(parsed_1) == 1
    assert "parasitamol" in parsed_1[0]["Drug_name"].lower()
    assert "increase the dosage by 100 mg" in parsed_1[0]["additional_instruction"].lower()
    assert "if the fever does not go away" in parsed_1[0]["additional_instruction"].lower()

    # 2. Punctuated speech with leading titration clause
    raw_2 = "Take parasitamol tablets 500 mg for 3 days, increase the dosage by 100 mg if the fever does not go away."
    output_2, _, _, _ = run_graph_extraction(None, raw_2)
    parsed_2 = parse_output_fields(output_2, query=raw_2)
    assert len(parsed_2) == 1
    assert "increase the dosage by 100 mg" in parsed_2[0]["additional_instruction"].lower()
    print("PASS: Dosage titration instructions cleanly captured without truncation.")


def test_faulty_grammar_duration_and_comma_titration():
    print("\n--- Test 18: Faulty Grammar Duration & Comma Titration ---")
    raw = "Take parasita mode, tablets 500 mg, 3 times a day, till 7 days, if the fever does not go away, increase the dosage by 20 mgs."
    output, _, _, _ = run_graph_extraction(None, raw)
    parsed = parse_output_fields(output, query=raw)
    assert len(parsed) == 1
    assert "parasita mode 500 mg" in parsed[0]["Drug_name"]
    assert "3 times a day" in parsed[0]["frequency"].lower()
    assert "7 days" in parsed[0]["duration"].lower()
    assert "oral" in parsed[0]["route"].lower()
    assert "if the fever does not go away, increase the dosage by 20 mgs" in parsed[0]["additional_instruction"].lower()
    print("PASS: Faulty grammar duration and unified comma titration instruction verified.")


def test_complex_multidrug_with_nasal_irrigations_and_precautions():
    print("\n--- Test 19: Complex Multi-Drug with Nasal Irrigations & Precautions ---")
    raw = (
        "Take one Cefpodoxime proxetil 200 mg tablet orally twice daily after meals for 7 days, and take one Levocetirizine 5 mg with Montelukast 10 mg tablet once daily at bedtime for 10 days. "
        "Take one Paracetamol 650 mg tablet up to three times daily after food for pain or fever, take one Pantoprazole 40 mg tablet once daily before breakfast for 7 days, and administer two sprays of Oxymetazoline 0.05% nasal spray into each nostril twice daily for a strict maximum of 3 days. "
        "Use saline nasal irrigations twice daily, perform steam inhalation, and seek reassessment if eye swelling or severe headaches develop."
    )
    output, _, _, _ = run_graph_extraction(None, raw)
    parsed = parse_output_fields(output, query=raw)
    assert len(parsed) == 6
    assert "Oxymetazoline 0.05%" in parsed[-1]["Drug_name"]
    assert "nasal" in parsed[-1]["route"]
    assert "into each nostril" in parsed[-1]["instruction"].lower()
    assert "saline nasal irrigations" in parsed[-1]["additional_instruction"].lower()
    assert "strict maximum of 3 days" in parsed[-1]["additional_instruction"].lower()
def test_sentence_punctuation_correction_agent():
    print("\n--- Test 20: Sentence & Punctuation Correction Agent ---")
    raw_unpunctuated = (
        "take disprin 500 mg tablets if the fever does not go away come visit the doctor take regular water "
        "and take amoxicillin 500 mg tid for 7 days and take pantoprazole 40 mg once daily before breakfast for 14 days"
    )
    output, gen_time, agent_logs, blocks = run_graph_extraction(None, raw_unpunctuated)
    
    # 1. Verify Punctuation Agent log exists and has normalized sentences
    punct_logs = [log for log in agent_logs if "Punctuation" in log.get("agent", "")]
    assert len(punct_logs) > 0, "Punctuation & Sentence Correction Agent log missing!"
    punctuated_text = punct_logs[0].get("punctuated_text", "")
    assert "Take disprin 500 mg tablets." in punctuated_text or "Take Disprin 500 mg tablets." in punctuated_text
    assert "Take amoxicillin 500 mg" in punctuated_text or "Take Amoxicillin 500 mg" in punctuated_text
    assert "Take pantoprazole 40 mg" in punctuated_text or "Take Pantoprazole 40 mg" in punctuated_text
    
    # 2. Verify accurate 3-drug extraction
    parsed = parse_output_fields(output, query=raw_unpunctuated)
    assert len(parsed) == 3, f"Expected 3 medicines, got {len(parsed)}"
    assert "disprin 500 mg" in parsed[0]["Drug_name"].lower()
    assert "amoxicillin 500 mg" in parsed[1]["Drug_name"].lower()
    assert "tid" in parsed[1]["frequency"].lower()
    assert "7 days" in parsed[1]["duration"].lower()
    assert "before breakfast" in parsed[2]["instruction"].lower()
    assert "come visit the doctor" in parsed[0]["additional_instruction"].lower() or any("come visit the doctor" in p["additional_instruction"].lower() for p in parsed)
    print("PASS: Sentence & Punctuation Correction Agent successfully segmented, punctuated, and forwarded clinical statements.")


def test_chronological_instruction_order_and_conditional_punctuation():
    print("\n--- Test 21: Chronological Instruction Order & Multi-Conditional Punctuation ---")
    raw = "Take parasitamol 400 mg 100 mg tablets for 5 days every 4 hours if the fever does not go away consult the doctor and if it is still does not go away start eating 3-3 little"
    output, _, logs, _ = run_graph_extraction(None, raw)
    
    # Verify punctuation
    punct_logs = [l for l in logs if "Punctuation" in l.get("agent", "")]
    assert len(punct_logs) > 0
    punc_text = punct_logs[0].get("punctuated_text", "")
    assert "If the fever does not go away, consult the doctor." in punc_text
    assert "If it is still does not go away, start eating 3-3 little." in punc_text
    
    # Verify exact chronological ordering in additional_instruction
    parsed = parse_output_fields(output, query=raw)
    assert len(parsed) == 1
    add_inst = parsed[0]["additional_instruction"]
    idx_fever = add_inst.find("If the fever does not go away, consult the doctor")
    idx_still = add_inst.find("If it is still does not go away, start eating 3-3 little")
    assert idx_fever != -1, "First conditional clause missing!"
    assert idx_still != -1, "Second conditional clause missing!"
    assert idx_fever < idx_still, "Instructions were not ordered chronologically!"
    print("PASS: Multi-conditional sentence punctuation and strictly chronological instruction ordering verified.")


def test_5_drug_sequential_conditional_advice_attribution():
    print("\n--- Test 22: 5-Drug Sequential Conditional Advice Attribution ---")
    raw = (
        "Take paracetamol 400 mg for 30 days. If the fever does not go away, consult the doctor. "
        "Take disprin 300 mg for 30 days. If the headache does not go away, consult the doctor. "
        "Take cellulose 50 grams for your body build up. If the body does not build up, start eating more protein and consult the doctor. "
        "take ibroughin for 60 days if the fever does not build up start eating more protein and consult the doctor "
        "take I brew fill for 60 days if the fever does not go away consult the doctor "
        "take all the medicines in the above in the liquid form and your result should start stowing if it does not show consult the doctor"
    )
    output, _, logs, blocks = run_graph_extraction(None, raw)
    parsed = parse_output_fields(output, query=raw)
    assert len(parsed) == 5, f"Expected 5 medicines, got {len(parsed)}"
    
    # 1. Paracetamol
    assert "paracetamol 400 mg" in parsed[0]["Drug_name"].lower()
    assert "30 days" in parsed[0]["duration"].lower()
    assert "fever does not go away" in parsed[0]["additional_instruction"].lower()
    
    # 2. Disprin
    assert "disprin 300 mg" in parsed[1]["Drug_name"].lower()
    assert "30 days" in parsed[1]["duration"].lower()
    assert "headache does not go away" in parsed[1]["additional_instruction"].lower()
    
    # 3. Cellulose
    assert "cellulose 50 g" in parsed[2]["Drug_name"].lower()
    assert "body does not build up" in parsed[2]["additional_instruction"].lower()
    
    # 4. Ibroughin
    assert "ibroughin" in parsed[3]["Drug_name"].lower()
    assert "60 days" in parsed[3]["duration"].lower()
    assert "fever does not build up" in parsed[3]["additional_instruction"].lower()
    
    # 5. I brew fill
    assert "i brew fill" in parsed[4]["Drug_name"].lower()
    assert "60 days" in parsed[4]["duration"].lower()
    assert "fever does not go away" in parsed[4]["additional_instruction"].lower()
    print("PASS: 5-drug sequential conditional advice attribution successfully verified.")


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
    test_punctuation_free_continuous_voice_speech()
    test_complex_multidrug_decimal_and_advice_guards()
    test_noisy_transcript_and_conversational_chatter_filtering()
    test_single_med_hydration_and_visit_doctor_advice()
    test_cross_sentence_coreference_frequency_resolution()
    test_dosage_titration_instruction_capture()
    test_faulty_grammar_duration_and_comma_titration()
    test_complex_multidrug_with_nasal_irrigations_and_precautions()
    test_sentence_punctuation_correction_agent()
    test_chronological_instruction_order_and_conditional_punctuation()
    test_5_drug_sequential_conditional_advice_attribution()
    print("\n========================================================")
    print("ALL 22 LANGGRAPH DRIFT-PROOF MULTI-AGENT TESTS PASSED!")
    print("========================================================")

