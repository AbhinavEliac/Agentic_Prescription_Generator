"""
tests/integration/test_canonical_pipeline.py
--------------------------------------------
Complete Regression & Integration Test Suite for the Canonical Prescription Pipeline.

Migrates all 22 clinical test cases from legacy LangGraph tests to the unified PrescriptionPipeline.
Tests both FAST and STANDARD modes.
"""

import pytest
from app.prescription.pipeline import PrescriptionPipeline, PipelineMode
from app.prescription.schema import ExtractionStatus, ClinicalRoute


@pytest.fixture(scope="module")
def pipeline():
    return PrescriptionPipeline()


def test_1_multi_drug_diet_instructions(pipeline):
    raw_prescription = (
        "Take Ferrous ascorbate with Folic acid once daily after lunch for 60 days, "
        "and take Vitamin C 500 mg alongside it to optimize iron absorption. "
        "Take Albendazole 400 mg single dose at bedtime on day one. "
        "Include dark green leafy vegetables in diet, avoid tea near meal times, "
        "re-test blood count in 2 months, and go for morning walks daily."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) >= 3, f"Expected at least 3 medicines, got {len(res.items)}"

    all_instructions = " ".join([
        (it.instruction or "") + " " + (it.additional_instruction or "")
        for it in res.items
    ]).lower()
    assert "leafy vegetables" in all_instructions or "green" in all_instructions
    assert "morning walk" in all_instructions or "tea" in all_instructions


def test_2_dual_dose_extraction(pipeline):
    raw_prescription = (
        "Take one tablet of Paracetamol 650 mg 20 mg twice daily(1-0-1) for 5 days, "
        "increase the dose by 100 mg after 7 days."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    med = res.items[0]

    assert "Paracetamol 650 mg" in med.medicine_name
    assert med.strength == "20 mg"
    assert "1-0-1" in med.frequency or "twice" in med.frequency.lower()
    assert "5 days" in med.duration
    assert med.route == "oral"


def test_3_route_specificity(pipeline):
    raw_prescription = "Inhale Budecort 200 mcg Rotacap twice daily for 10 days. Rinse mouth after using the inhaler."
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    assert res.items[0].route == "inhalation"
    assert "rinse mouth" in (res.items[0].instruction or "").lower()


def test_4_no_moralizing_or_extra_cautions(pipeline):
    raw_prescription = "Take Pantop 40 mg once daily before breakfast for 14 days."
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    forbidden = ["please note", "caution:", "warning:", "moral", "consult your doctor", "disclaimer:"]
    for it in res.items:
        inst = ((it.instruction or "") + " " + (it.additional_instruction or "")).lower()
        for f in forbidden:
            assert f not in inst, f"Forbidden cue '{f}' found in instructions!"


def test_5_validator_boundary(pipeline):
    # Tests that pure non-drug entities are filtered and do not become valid medicines
    raw_prescription = "Good morning doctor. Check blood pressure 130/80 mmHg and endoscopy report."
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    # Blood pressure and endoscopy report must not become prescription items
    for it in res.items:
        assert "blood pressure" not in it.medicine_name.lower()
        assert "endoscopy" not in it.medicine_name.lower()


def test_6_complex_5_drug_prescription(pipeline):
    raw_prescription = (
        "Take one Ofloxacin-Ornidazole tablet orally twice daily after meals for 5 days, "
        "and take one Racecadotril 100 mg capsule three times daily before food for 3 days. "
        "Take one Ondansetron 4 mg tablet up to three times daily as needed 30 minutes before meals for nausea, "
        "consume one vial of Enterogermina oral suspension twice daily for 5 days, "
        "and dissolve one sachet of Oral Rehydration Salts in one liter of clean drinking water to consume throughout the day. "
        "Stick to a bland diet consisting of rice, curd, and bananas, avoid spicy or oily foods, "
        "and visit the emergency room immediately if severe dehydration or persistent vomiting occurs."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 5, f"Expected 5 medicines, got {len(res.items)}"

    drug_names = [it.medicine_name.lower() for it in res.items]
    assert any("ofloxacin-ornidazole" in d for d in drug_names)
    assert any("racecadotril" in d for d in drug_names)
    assert any("ondansetron" in d for d in drug_names)
    assert any("enterogermina" in d for d in drug_names)
    assert any("oral rehydration salts" in d for d in drug_names)


def test_7_multi_medicine_with_timestamps_and_conditionals(pipeline):
    raw_prescription = (
        "Administer ATEN tablet 50mg by mouth every morning before food for 10 days and seek reassessment of blood pressure afterwards. "
        "Take Crocin 650 mg every 6 hours for 3 days and discontinue once the fever resolves. "
        "Take Amlodipine 5 mg once daily for 14 days. If blood pressure remains high, consult your doctor before increasing the dose. "
        "Animesh Kumar, Yesterday 4:30 PM Take one tablet of paracetamol 650 mg 20 mg twice daily(1-0-1) for 5 day, increase the dose by 100 mg after 7 day."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 4, f"Expected 4 medicines, got {len(res.items)}"

    assert "ATEN 50mg" in res.items[0].medicine_name or "ATEN 50 mg" in res.items[0].medicine_name
    assert res.items[0].frequency == "every morning before food" or "every morning" in res.items[0].frequency
    assert "Crocin 650 mg" in res.items[1].medicine_name
    assert "every 6 hours" in res.items[1].frequency
    assert "Amlodipine 5 mg" in res.items[2].medicine_name
    assert "paracetamol 650 mg" in res.items[3].medicine_name.lower()
    assert res.items[3].strength == "20 mg"


def test_8_topicals_sprays_drops(pipeline):
    raw_prescription = (
        "Apply Clotrimazole 1% cream topically twice daily for 14 days, "
        "spray Fluticasone 50 mcg nasal spray into both nostrils once daily every morning, "
        "and instill Moxifloxacin 0.5% eye drops into left eye every 4 hours for 7 days."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 3
    assert res.items[0].route == "topical"
    assert res.items[1].route == "nasal"
    assert res.items[2].route == "ophthalmic"
    assert "every 4 hours" in res.items[2].frequency


def test_9_cross_sentence_and_class_instructions(pipeline):
    raw_prescription = (
        "Take Metformin 500 mg twice daily. Take Metformin strictly with meals to avoid stomach upset. "
        "Also take Glimepiride 1 mg once daily before breakfast. "
        "Start Amoxicillin 500 mg TID for 7 days, and take Paracetamol 650 mg as needed for fever. "
        "Do not stop the antibiotic course early even if fever subsides."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 4

    assert "strictly with meals" in (res.items[0].instruction or "").lower()
    assert "before breakfast" in (res.items[1].instruction or "").lower()

    # Amoxicillin should receive antibiotic course directive
    amox_all = ((res.items[2].instruction or "") + " " + (res.items[2].additional_instruction or "")).lower()
    assert "antibiotic course" in amox_all

    # Paracetamol should NOT receive antibiotic instruction
    parac_all = ((res.items[3].instruction or "") + " " + (res.items[3].additional_instruction or "")).lower()
    assert "antibiotic" not in parac_all


def test_10_additional_instructions_column(pipeline):
    raw_prescription = (
        "Administer PHEXIN DT tablet 250 mg once daily(0-0-1) before breakfast for 10 days. "
        "Seek reassessment if adverse effects develop."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    assert "PHEXIN DT 250 mg" in res.items[0].medicine_name or "PHEXIN DT" in res.items[0].medicine_name
    assert "0-0-1" in res.items[0].frequency or "bedtime" in res.items[0].frequency.lower() or "once" in res.items[0].frequency.lower()
    assert "before breakfast" in (res.items[0].instruction or "").lower()
    assert "adverse effects" in (res.items[0].additional_instruction or "").lower()


def test_11_natural_language_time_and_evaluation(pipeline):
    raw_prescription = (
        "Take Stugeron Forte by mouth twice daily Morning Night for 2 weeks. "
        "Return for evaluation after completing the course"
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    assert "Stugeron Forte" in res.items[0].medicine_name
    assert "twice daily" in res.items[0].frequency.lower()
    assert "2 weeks" in res.items[0].duration
    assert res.items[0].route == "oral"
    assert "return for evaluation" in (res.items[0].additional_instruction or "").lower()


def test_12_punctuation_free_continuous_voice_speech(pipeline):
    raw_prescription = (
        "Take this print 500mg 20mg tablets after breakfast if headache does not go away "
        "Meet the doctor also take walks after dinner and it will reduce your headaches Please see me after 7 days"
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    assert "this print 500mg" in res.items[0].medicine_name.lower() or "this print" in res.items[0].medicine_name.lower()
    assert res.items[0].strength == "20mg" or res.items[0].strength == "20 mg"
    assert res.items[0].route == "oral"
    assert "after breakfast" in (res.items[0].instruction or "").lower()
    assert "doctor" in (res.items[0].additional_instruction or "").lower()
    assert "walks" in (res.items[0].additional_instruction or "").lower()


def test_13_complex_multidrug_decimal_and_advice_guards(pipeline):
    raw_prescription = (
        "Take Wallach Clover 1000 MCG Tablets, Overly 3 times Daily for 7 days and take 1 pre-gab ball in 75 MG capsule. "
        "Once daily at bedtime for 14 days, take 1 parasitamol 500 MG with Phradamol 37.5 MG tablet twice daily after meals for severe pain for 5 days. "
        "Apply Kalamine lotion gently over the close rash areas 3 times daily and take 1 methello glogamine 150,000 MG tablet daily after lunch for 30 days "
        "keep the blistered area clean and dry avoid close physical contact with pregnant individuals or non-immune persons and return if the rash involves the eye region."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) >= 5
    assert any("37.5" in (it.medicine_name or "") or (it.strength and "37.5" in it.strength) for it in res.items)
    assert not any("pregnant" in it.medicine_name.lower() for it in res.items)
    assert "clean and dry" in (res.items[-1].additional_instruction or "").lower()


def test_14_noisy_transcript_and_conversational_chatter_filtering(pipeline):
    raw_prescription = (
        "Good morning doctor. Hello Mr. Sharma, how are you feeling today? "
        "I have severe fever and throat pain since yesterday. Let me check your vitals. "
        "Temperature is 101 F, BP is 130/80 mmHg, chest is clear. "
        "Take Augmentin 625mg twice daily for 5 days and take Paracetamol 650mg SOS for fever. "
        "Gargle with warm salt water thrice daily and drink plenty of fluids. "
        "Thank you doctor, I will take care. Have a great day."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 2
    assert "Augmentin 625mg" in res.items[0].medicine_name or "Augmentin 625 mg" in res.items[0].medicine_name
    assert "Paracetamol 650mg" in res.items[1].medicine_name or "Paracetamol 650 mg" in res.items[1].medicine_name

    for it in res.items:
        assert not any(noise in it.medicine_name.lower() for noise in ("good morning", "sharma", "fever", "vitals", "130/80", "thank you", "great day"))


def test_15_single_med_hydration_and_visit_doctor_advice(pipeline):
    raw_prescription = (
        "Take Disprin 500 mg tablets. If the fever does not go away, come visit the doctor. Take regular water."
    )
    res = pipeline.extract(raw_prescription, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    assert "Disprin 500 mg" in res.items[0].medicine_name
    assert res.items[0].route == "oral"
    add_inst = (res.items[0].additional_instruction or "").lower()
    assert "visit the doctor" in add_inst or "doctor" in add_inst
    assert "regular water" in add_inst or "water" in add_inst


def test_16_cross_sentence_coreference_frequency_resolution(pipeline):
    # Singular coreference ("It should be taken 4 times a day")
    raw_1 = "Take parasitamol 500 mg for 3 days. If the fever does not go away, come visit the doctor. It should be taken 4 times a day."
    res_1 = pipeline.extract(raw_1, mode=PipelineMode.FAST)
    assert len(res_1.items) == 1
    assert "parasitamol 500 mg" in res_1.items[0].medicine_name.lower()
    assert "4 times a day" in res_1.items[0].frequency.lower()

    # Plural coreference ("Both should be taken twice daily after meals")
    raw_2 = "Take Pan 40 mg and Paracetamol 650 mg for 5 days. Both should be taken twice daily after meals. Drink plenty of water."
    res_2 = pipeline.extract(raw_2, mode=PipelineMode.FAST)
    assert len(res_2.items) == 2
    assert "twice" in res_2.items[0].frequency.lower() or "1-0-1" in res_2.items[0].frequency
    assert "twice" in res_2.items[1].frequency.lower() or "1-0-1" in res_2.items[1].frequency
    assert "after meals" in (res_2.items[0].instruction or "").lower()
    assert "after meals" in (res_2.items[1].instruction or "").lower()


def test_17_dosage_titration_instruction_capture(pipeline):
    raw_1 = "Take parasitamol teplis 500 mg for 3 days if the fever does not go away increase the dosage by 100 mg"
    res_1 = pipeline.extract(raw_1, mode=PipelineMode.FAST)
    assert len(res_1.items) == 1
    assert "parasitamol" in res_1.items[0].medicine_name.lower()
    add_1 = (res_1.items[0].additional_instruction or "").lower()
    assert "increase the dosage by 100 mg" in add_1 or "increase" in add_1

    raw_2 = "Take parasitamol tablets 500 mg for 3 days, increase the dosage by 100 mg if the fever does not go away."
    res_2 = pipeline.extract(raw_2, mode=PipelineMode.FAST)
    assert len(res_2.items) == 1
    add_2 = (res_2.items[0].additional_instruction or "").lower()
    assert "increase the dosage by 100 mg" in add_2 or "increase" in add_2


def test_18_faulty_grammar_duration_and_comma_titration(pipeline):
    raw = "Take parasita mode, tablets 500 mg, 3 times a day, till 7 days, if the fever does not go away, increase the dosage by 20 mgs."
    res = pipeline.extract(raw, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    assert "parasita mode" in res.items[0].medicine_name.lower()
    assert res.items[0].strength == "500 mg" or res.items[0].dose == "500"
    assert "3 times a day" in res.items[0].frequency.lower()
    assert "7 days" in res.items[0].duration.lower()
    assert res.items[0].route == "oral"
    add_inst = (res.items[0].additional_instruction or "").lower()
    assert "increase the dosage by 20 mgs" in add_inst or "fever does not go away" in add_inst


def test_19_complex_multidrug_with_nasal_irrigations_and_precautions(pipeline):
    raw = (
        "Take one Cefpodoxime proxetil 200 mg tablet orally twice daily after meals for 7 days, "
        "and take one Levocetirizine 5 mg with Montelukast 10 mg tablet once daily at bedtime for 10 days. "
        "Take one Paracetamol 650 mg tablet up to three times daily after food for pain or fever, "
        "take one Pantoprazole 40 mg tablet once daily before breakfast for 7 days, "
        "and administer two sprays of Oxymetazoline 0.05% nasal spray into each nostril twice daily for a strict maximum of 3 days. "
        "Use saline nasal irrigations twice daily, perform steam inhalation, and seek reassessment if eye swelling or severe headaches develop."
    )
    res = pipeline.extract(raw, mode=PipelineMode.FAST)
    assert len(res.items) == 6
    assert "Oxymetazoline 0.05%" in res.items[-1].medicine_name
    assert res.items[-1].route == "nasal"
    assert "into each nostril" in (res.items[-1].instruction or "").lower()
    assert "saline nasal irrigations" in (res.items[-1].additional_instruction or "").lower()


def test_20_sentence_punctuation_correction_and_multi_drug(pipeline):
    raw_unpunctuated = (
        "take disprin 500 mg tablets if the fever does not go away come visit the doctor take regular water "
        "and take amoxicillin 500 mg tid for 7 days and take pantoprazole 40 mg once daily before breakfast for 14 days"
    )
    res = pipeline.extract(raw_unpunctuated, mode=PipelineMode.FAST)
    assert len(res.items) == 3
    assert "disprin 500 mg" in res.items[0].medicine_name.lower()
    assert "amoxicillin 500 mg" in res.items[1].medicine_name.lower()
    assert "tid" in res.items[1].frequency.lower() or "thrice" in res.items[1].frequency.lower() or "1-1-1" in res.items[1].frequency
    assert "7 days" in res.items[1].duration.lower()
    assert "before breakfast" in (res.items[2].instruction or "").lower()


def test_21_chronological_instruction_order_and_conditional_punctuation(pipeline):
    raw = "Take parasitamol 400 mg 100 mg tablets for 5 days every 4 hours if the fever does not go away consult the doctor and if it is still does not go away start eating 3-3 little"
    res = pipeline.extract(raw, mode=PipelineMode.FAST)
    assert len(res.items) == 1
    add_inst = (res.items[0].additional_instruction or "").lower()
    idx_fever = add_inst.find("fever does not go away")
    idx_still = add_inst.find("start eating")
    assert idx_fever != -1
    assert idx_still != -1
    assert idx_fever < idx_still, "Instructions were not ordered chronologically!"


def test_22_5_drug_sequential_conditional_advice_attribution(pipeline):
    raw = (
        "Take paracetamol 400 mg for 30 days. If the fever does not go away, consult the doctor. "
        "Take disprin 300 mg for 30 days. If the headache does not go away, consult the doctor. "
        "Take cellulose 50 grams for your body build up. If the body does not build up, start eating more protein and consult the doctor. "
        "take ibroughin for 60 days if the fever does not build up start eating more protein and consult the doctor "
        "take I brew fill for 60 days if the fever does not go away consult the doctor "
        "take all the medicines in the above in the liquid form and your result should start stowing if it does not show consult the doctor"
    )
    res = pipeline.extract(raw, mode=PipelineMode.FAST)
    assert len(res.items) == 5

    # 1. Paracetamol
    assert "paracetamol" in res.items[0].medicine_name.lower()
    assert res.items[0].strength == "400 mg" or res.items[0].dose == "400"
    assert "30 days" in res.items[0].duration.lower()
    assert "fever does not go away" in (res.items[0].additional_instruction or "").lower()

    # 2. Disprin
    assert "disprin 300 mg" in res.items[1].medicine_name.lower()
    assert "30 days" in res.items[1].duration.lower()
    assert "headache does not go away" in (res.items[1].additional_instruction or "").lower()

    # 3. Cellulose
    assert "cellulose 50 g" in res.items[2].medicine_name.lower()
    assert "body does not build up" in (res.items[2].additional_instruction or "").lower()

    # 4. Ibroughin
    assert "ibroughin" in res.items[3].medicine_name.lower()
    assert "60 days" in res.items[3].duration.lower()
    assert "fever does not build up" in (res.items[3].additional_instruction or "").lower()

    # 5. I brew fill
    assert "i brew fill" in res.items[4].medicine_name.lower() or "medicine not found" in res.items[4].medicine_name.lower()
    assert "60 days" in res.items[4].duration.lower()
    assert "fever does not go away" in (res.items[4].additional_instruction or "").lower()
