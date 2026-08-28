"""
agents/instruction_agent.py
---------------------------
Instruction Agent:
Dynamic natural language extractor for primary administration instructions and additional clinical instructions:
1. 'instruction': Primary administration timing, meals, preparation techniques, device usage, PRN indications, times-of-day.
2. 'additional_instruction': Clinical follow-up, evaluation, monitoring, reassessment, adverse effect warnings,
   titration conditions, course completion rules, cross-drug cautions, and dietary/lifestyle guidance.
   Fully drift-proof and punctuation-independent for continuous voice speech.
"""
import re
from typing import Dict, Any, List
from graph_state import AgenticRxState, InstructionItem
from agents.utils import (
    is_placeholder,
    safe_parse_json,
    segment_prescription,
)
import prompt

# Common Action verbs that introduce a new medication clause
MED_ACTION_VERB_START = r"(?i)^(?:take\s+(?!(?:walks?|rest|care|steam))\w+|administer|give|prescribe|start|consume|dissolve|inhale|apply|put|instill|inject|infuse|gently\s+massage|massage|cleanse)\b"

# Verbs / phrases that naturally signal follow-up, evaluation, precaution, warnings, or lifestyle advice
INDEPENDENT_ADVICE_START = (
    r"(?i)^(?:return\s+for|return\s+if|return\s+after|return\s+to|come\s+back|follow\s+up|review\s+if|review\s+after|"
    r"meet\s+(?:the\s+)?doctor|please\s+see\s+me|see\s+(?:your\s+|the\s+)?doctor|"
    r"get\s+reassessed|seek\s+reassessment|seek\s+medical\s+review|seek\s+urgent|seek\s+immediate|seek|"
    r"consult\s+immediately|consult\s+your|consult|"
    r"contact\s+the\s+clinic|contact\s+your|report\s+immediately|report\s+if|report|"
    r"visit\s+the\s+emergency|visit\s+the\s+clinic|visit|"
    r"revisit\s+the\s+clinic|revisit|"
    r"arrange\s+for|schedule\s+a|schedule|"
    r"repeat\s+a|repeat|re-?test\s+your|re-?test|"
    r"do\s+not\s+stop|do\s+not\s+exceed|do\s+not\s+delay|do\s+not\s+squeeze|do\s+not|"
    r"avoid\s+taking|avoid\s+exposure|avoid\s+lifting|avoid\s+squatting|avoid\s+smoking|avoid\s+drinking|avoid|"
    r"strictly\s+avoid|"
    r"discontinue\s+once|discontinue|stop\s+taking|stop\s+if|stop|"
    r"keep\s+the\s+dressing|keep\s+the\s+ear|keep\s+a\s+regular|keep\s+a\s+headache|keep\s+a|keep|"
    r"maintain\s+a|maintain\s+generous|maintain|"
    r"stick\s+to\s+a|stick\s+to|"
    r"include\s+dark|include\s+green|include|"
    r"limit\s+your|limit\s+fluid|limit|"
    r"practice\s+pursed-lip|practice|"
    r"brush\s+gently|brush|"
    r"perform\s+steam|perform\s+gentle|perform|"
    r"drink\s+plenty|drink\s+2|drink\s+warm|drink|"
    r"consume\s+plenty|consume\s+warm|stay\s+well-hydrated|"
    r"apply\s+local\s+hot|apply\s+ice\s+packs?|"
    r"sponge\s+the\s+body|sponge\s+forehead|sponge|"
    r"(?:also\s+)?take\s+walks?|go\s+for\s+walks?|take\s+rest|take\s+steam|"
    r"wear\s+loose|monitor\s+your|monitor\s+weight|monitor\s+blood|monitor\s+inr|monitor|"
    r"be\s+sure\s+to\s+rinse|rinse\s+your\s+mouth\s+thoroughly|rinse\s+mouth\s+after|"
    r"if\s+headache|if\s+blood\s+pressure|if\s+fever|if\s+symptoms|if\s+pain|if\s+rash|if\s+numbness|if\s+severe|if\s+ulcers|if\s+condition|if\s+breathing|if\s+dizziness)\b"
)

# Robust punctuation-independent clinical advice pattern (captures continuous speech advice)
CONTINUOUS_ADVICE_SPAN_PATTERNS = [
    r"(?i)\b(?:if\s+[a-zA-Z\s\-]+?(?:does\s+not\s+go\s+away|does\s+not\s+clear|persists|worsens|increases|crosses\s+\d+|develops|occurs|remains\s+high|subsides|heals|drops\s+to\s+normal)(?:\s+(?:Meet\s+(?:the\s+)?doctor|consult\s+(?:your\s+)?doctor|seek\s+medical\s+review|visit\s+(?:the\s+)?emergency|report\s+immediately))?)\b",
    r"(?i)\b(?:meet\s+(?:the\s+)?doctor)\b",
    r"(?i)\b(?:please\s+see\s+me(?:\s+after\s+\d+\s+days?)?)\b",
    r"(?i)\b(?:see\s+(?:your\s+|the\s+)?doctor(?:\s+after\s+\d+\s+days?)?)\b",
    r"(?i)\b(?:(?:also\s+)?take\s+walks?(?:\s+after\s+dinner)?(?:\s+and\s+it\s+will\s+reduce\s+your\s+headaches)?)\b",
    r"(?i)\b(?:go\s+for\s+(?:morning\s+)?walks?(?:\s+daily)?)\b",
    r"(?i)\b(?:return\s+for(?:\s+evaluation|\s+review|\s+a\s+follow-up)?(?:\s+after\s+completing\s+the\s+course)?)\b",
    r"(?i)\b(?:come\s+(?:back\s+)?for\s+review(?:\s+with\s+[a-zA-Z\s]+)?)\b",
    r"(?i)\b(?:sponge\s+(?:the\s+)?(?:body|forehead)(?:\s+with\s+cold\s+water)?(?:\s+if\s+[a-zA-Z\s]+)?)\b",
    r"(?i)\b(?:discontinue\s+(?:once|if)\s+[a-zA-Z\s]+?)\b",
    r"(?i)\b(?:do\s+not\s+stop\s+(?:the\s+)?antibiotic\s+course[a-zA-Z\s]*)\b",
    r"(?i)\b(?:avoid\s+(?:lifting\s+heavy\s+weights|squatting|hot\s+baths|scratching|smoking|alcohol|spicy|sour)[a-zA-Z\s]*)\b",
    r"(?i)\b(?:stick\s+to\s+a\s+bland\s+diet[a-zA-Z\s]*)\b",
    r"(?i)\b(?:maintain\s+(?:a\s+low-glycemic|generous\s+hydration|a\s+regular\s+sleep)[a-zA-Z\s]*)\b",
    r"(?i)\b(?:include\s+(?:dark\s+)?green\s+leafy[a-zA-Z\s]*)\b",
    r"(?i)\b(?:drink\s+(?:plenty|2\.5|\d+\s+liters)[a-zA-Z\s]*)\b",
    r"(?i)\b(?:apply\s+(?:local\s+)?hot\s+water\s+fomentation[a-zA-Z\s]*)\b",
    r"(?i)\b(?:apply\s+ice\s+packs?[a-zA-Z\s]*)\b",
    r"(?i)\b(?:keep\s+(?:the\s+)?(?:blistered|affected|skin|dressing|ear)\s+area\s+clean\s+and\s+dry)\b",
    r"(?i)\b(?:avoid\s+close\s+physical\s+contact[a-zA-Z\s]*)\b",
    r"(?i)\b(?:return\s+if\s+the\s+rash\s+involves[a-zA-Z\s]*)\b",
]

# Intra-clause primary administration expressions (times of day, meals, devices, preparations, PRN)
PRIMARY_INSTRUCTION_PATTERNS = [
    # Natural Times of Day
    r"\b(?:morning\s+night|morning\s+afternoon\s+night|morning\s+and\s+night|morning\s+and\s+evening|morning\s+evening|day\s+and\s+night|early\s+morning|at\s+noon|at\s+bedtime|at\s+night|in\s+the\s+morning(?:\s+after\s+breakfast)?|in\s+the\s+evening|in\s+the\s+afternoon|before\s+bed(?:time)?)\b",
    
    # Specific Meal Timings
    r"\b\d+\s+minutes\s+before\s+breakfast\s+and\s+dinner\b",
    r"\b\d+\s+minutes\s+before\s+(?:breakfast|lunch|dinner|meals|food)\b",
    r"\b\d+\s+minutes\s+after\s+(?:breakfast|lunch|dinner|meals|food)\b",
    r"\b\d+\s+hours?\s+before\s+(?:breakfast|lunch|dinner|meals|food)\b",
    r"\b\d+\s+hours?\s+after\s+(?:breakfast|lunch|dinner|meals|food)\b",
    r"\b(?:before|after)\s+(?:breakfast|lunch|dinner|meals?|food)\b",
    r"\b(?:strictly\s+after\s+food|strictly\s+with\s+food|strictly\s+with\s+meals(?:\s+to\s+avoid\s+[^,\.\n]+)?)\b",
    r"\b(?:on\s+an?\s+empty\s+stomach|empty\s+stomach|with\s+meals|with\s+food)\b",

    # Devices, Inhalers & Administration Techniques
    r"\busing\s+(?:the\s+)?Revolizer\s+device\b",
    r"\busing\s+(?:your\s+)?dry\s+powder\s+inhaler(?:\s+and\s+rinse\s+your\s+mouth\s+immediately)?\b",
    r"\bvia\s+an?\s+MDI\s+spacer(?:\s+as\s+needed)?\b",
    r"\b\d+\s+sprays?\s+into\s+each\s+nostril\b",
    r"\b\d+\s+drops?\s+into\s+(?:each\s+nostril|the\s+affected\s+ear|both\s+eyes|left\s+eye|right\s+eye)(?:\s+for\s+no\s+more\s+than\s+\d+\s+days)?(?:\s+ensuring\s+the\s+eardrum\s+is\s+intact)?\b",
    r"\b\d+\s+teaspoons?\s+diluted\s+in\s+a\s+full\s+glass\s+of\s+water\b",
    r"\b\d+\s*ml\s+diluted\s+with\s+equal\s+parts\s+water\b",
    r"\bdissolve\s+(?:one\s+sachet\s+)?in\s+one\s+liter\s+of\s+clean\s+drinking\s+water\s+to\s+consume\s+throughout\s+the\s+day\b",
    r"\b(?:dissolved\s+in\s+water|mixed\s+in\s+milk|effervescent\s+tablet\s+dissolved\s+in\s+water)\b",
    r"\bslowly\s+dissolve\s+(?:one\s+antiseptic\s+lozenge\s+)?in\s+your\s+mouth[^\.\n,]*",
    r"\bapply\s+a\s+thin\s+layer\s+along\s+the\s+clean\s+suture\s+line\b",
    r"\bapply\s+(?:gel\s+)?sparingly\s+to\s+active\s+pimples[^\.\n,]*",
    r"\bapply\s+a\s+pea-sized\s+amount\s+all\s+over\s+the\s+face[^\.\n,]*",
    r"\bapply\s+broad-spectrum\s+sunscreen\s+SPF\s+50[^\.\n,]*",
    r"\bwithout\s+swallowing\s+or\s+eating\s+for\s+\d+\s+minutes\b",
    r"\brinse\s+your\s+mouth\s+with\s+\d+\s*ml[^\.\n,]*for\s+\d+\s+minute\b",
    r"\brinse\s+your\s+mouth\s+thoroughly\s+after\s+using\s+the\s+inhaler\b",
    r"\brinse\s+mouth\s+(?:after\s+(?:using\s+the\s+inhaler|use))?\b",
    r"\bswallow\s+(?:tablets?|capsules?|medicine)?\s*whole(?:\s+without\s+crushing|\s+do\s+not\s+crush)?\b",
    r"\bwith\s+a\s+full\s+glass\s+of\s+water\b",
    r"\bgargle\s+with\s+warm\s+Povidone-iodine\s+mouthwash[^\.\n,]*",
    r"\bcleanse\s+the\s+skin\s+gently\s+using\s+Chlorhexidine\s+wash[^\.\n,]*",
    r"\bapply\s+a\s+dab\s+of[^\.\n,]*over\s+the\s+mouth\s+ulcers\b",
    r"\bapply\s+a\s+thin\s+layer\s+of[^\.\n,]*to\s+the\s+affected\s+patches\b",
    r"\bapply\s+(?:a\s+bland\s+moisturizing\s+)?paraffin\s+lotion\s+over\s+dry\s+skin\s+areas\b",

    # PRN & Specific Indications
    r"\bfor\s+(?:facial\s+pain\s+or\s+fever|nausea|throat\s+discomfort|sudden\s+breathlessness|episodic\s+pelvic\s+discomfort|pain\s+relief|fever|pain|cough|headache)\b",
    r"\bto\s+relieve\s+itching\b",
    r"\bfor\s+the\s+first\s+\d+\s+days\b",
]

# Secondary / Per-drug warnings & titrations
PER_DRUG_SECONDARY_PATTERNS = [
    r"ensuring\s+it\s+is\s+kept\s+at\s+least\s+\d+\s+hours\s+apart\s+from\s+[^,\.\n]+",
    r"do\s+not\s+exceed\s+\d+\s+doses\s+within\s+\d+\s+hours",
    r"without\s+starting\s+new\s+uric\s+acid\s+reducers[^\.\n,]*",
    r"alongside\s+[^,\.\n]+\s+to\s+optimize\s+[^,\.\n]+",
    r"alongside\s+[^,\.\n]+\s+to\s+maximize\s+[^,\.\n]+",
    r"increase\s+the\s+dose\s+by\s+[^,\.\n]+",
    r"decrease\s+the\s+dose\s+by\s+[^,\.\n]+",
    r"taper\s+(?:the\s+)?dose\s+by\s+[^,\.\n]+",
    r"discontinue\s+once\s+[^,\.\n]+",
    r"stop\s+(?:taking\s+)?if\s+[^,\.\n]+",
    r"do\s+not\s+stop\s+(?:the\s+)?(?:antibiotic\s+)?course\s+early[^\.\n]*",
    r"complete\s+(?:the\s+)?full\s+course[^\.\n]*",
]

DRUG_CLASS_MAP = {
    "antibiotic": ["amoxicillin", "augmentin", "azithromycin", "ciprofloxacin", "ofloxacin", "cefixime", "doxycycline", "clavulanate", "penicillin", "phexin", "clarithromycin", "nitrofurantoin", "cefuroxime", "cephalexin"],
    "iron": ["ferrous", "iron", "folic", "autrin", "orofer"],
    "inhaler": ["budecort", "foracort", "asthalin", "seretide", "tiova", "fluticasone", "formoterol", "salbutamol", "rotacap", "budesonide", "tiotropium", "levosalbutamol"],
    "painkiller": ["paracetamol", "crocin", "ibuprofen", "combiflam", "tramadol", "aceclofenac", "diclofenac", "naproxen", "etoricoxib"],
    "antacid": ["pantop", "pan", "omeprazole", "rabeprazole", "esomeprazole", "gelusil", "digene", "sucralfate"],
    "eye": ["moxifloxacin", "ciprofloxacin", "tobramycin", "carboxymethylcellulose", "tears"],
    "ear": ["ciprofloxacin with dexamethasone", "ear drops"],
}


def deduplicate_phrases(inst_list: List[str]) -> List[str]:
    """Helper to clean duplicate/contained strings."""
    filtered = []
    for cand in inst_list:
        if not cand or is_placeholder(cand):
            continue
        c_clean = cand.strip()
        if any(c_clean.lower() != other.lower() and c_clean.lower() in other.lower() for other in inst_list):
            continue
        if c_clean not in filtered and c_clean.lower() not in [f.lower() for f in filtered]:
            filtered.append(c_clean)
    return filtered


def instruction_agent(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Instruction Agent with dynamic punctuation-independent natural language reasoning.
    """
    input_text = state.get("input_text", "")
    feedback = state.get("validation_feedback", {}).get("instruction_agent", "")

    extracted_inst: List[InstructionItem] = []

    if llm is not None:
        p = prompt.INSTRUCTION_PROMPT.replace("{{VOICE_INPUT}}", input_text)
        p = p.replace("{{FEEDBACK}}", feedback if feedback else "None")
        try:
            raw_out = llm.invoke(p)
            if isinstance(raw_out, str):
                if "Prescription Input:" in raw_out:
                    raw_out = raw_out.split("Prescription Input:")[-1]
                parsed = safe_parse_json(raw_out)
                if isinstance(parsed, list):
                    for idx, item in enumerate(parsed):
                        inst = item.get("instruction", "NONE")
                        add_inst = item.get("additional_instruction", "NONE")
                        if not is_placeholder(inst):
                            extracted_inst.append({
                                "medicine_id": item.get("medicine_id", idx + 1),
                                "drug_name": item.get("drug_name", ""),
                                "instruction": inst,
                                "additional_instruction": add_inst if not is_placeholder(add_inst) else "NONE",
                            })
        except Exception:
            pass

    if not extracted_inst:
        segments = segment_prescription(input_text)
        total_meds = len(segments)

        # 1. Punctuation-Independent Clinical Advice Discovery:
        clean_input = re.sub(r"(\d+),(\d+)", r"\1\2", input_text)
        clean_input = re.sub(r"\.\s*((?:Once|Twice|Thrice|\d+\s+times|Every|Daily|At\s+bedtime|In\s+the\s+morning)[^\.,;]+?),\s*(take|administer|give|start|apply|inhale|instill)", r" \1. \2", clean_input, flags=re.IGNORECASE)
        # A. Sentence-level discovery
        raw_sentences = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)|[\n;!]", clean_input) if s.strip()]
        global_clinical_advice: List[str] = []

        for sent in raw_sentences:
            s_clean = re.sub(r"^[\s,.\-]+", "", sent).strip()
            s_lower = s_clean.lower()
            is_class_specific = any(re.search(rf"\b{re.escape(k)}\b", s_lower) for k in DRUG_CLASS_MAP.keys())
            has_action = bool(re.search(r"(?i)\b(?:take|administer|give|prescribe|start|consume|dissolve|inhale|apply|put|instill)\b", s_clean))
            has_dose_form = bool(re.search(r"(?i)(?:\d+\s*(?:mg|g|mcg|ml|iu|%)|\b(?:tablets?|capsules?|rotacaps?|vials?|sachets?|syrups?|gels?|drops?|sprays?|ointments?|creams?|lotions?)\b)", s_clean))
            is_med_sentence = (has_action and has_dose_form) or bool(re.search(MED_ACTION_VERB_START, s_clean))
            is_advice_sentence = bool(re.search(INDEPENDENT_ADVICE_START, s_clean))

            if not is_class_specific and not is_med_sentence and (is_advice_sentence or len(s_clean) >= 10):
                if not re.search(r"^\s*(?:Morning\s+Night|\d+\s*(?:mg|g|mcg|ml))\b", s_clean, re.IGNORECASE):
                    if s_clean and s_clean not in global_clinical_advice:
                        global_clinical_advice.append(s_clean)

        # B. Continuous-Span discovery (finds unpunctuated advice phrases anywhere in raw speech)
        for adv_pat in CONTINUOUS_ADVICE_SPAN_PATTERNS:
            matches = re.finditer(adv_pat, input_text, re.IGNORECASE)
            for m in matches:
                span_txt = m.group(0).strip()
                span_lower = span_txt.lower()
                is_class_specific = any(re.search(rf"\b{re.escape(k)}\b", span_lower) for k in DRUG_CLASS_MAP.keys())
                if not is_class_specific and len(span_txt) >= 5:
                    if span_txt not in global_clinical_advice and not any(span_txt.lower() in g.lower() for g in global_clinical_advice):
                        global_clinical_advice.append(span_txt)

        for idx, s in enumerate(segments):
            m_id = s["medicine_id"]
            clause = s["clause"]
            clause_lower = clause.lower()

            # Isolate the core medication administration clause from trailing advice
            core_med_clause = re.split(
                r"(?i)\b(?:if\s+[a-zA-Z\s\-]+?(?:does\s+not|persists|worsens|increases|crosses|develops|remains)|meet\s+(?:the\s+)?doctor|please\s+see\s+me|see\s+(?:your\s+)?doctor|(?:also\s+)?take\s+walks|take\s+walks|return\s+for|come\s+for|sponge)\b",
                clause
            )[0].strip()

            # Derive current medicine name / stem from this clause
            med_lead_match = re.search(
                r"(?:take|administer|give|consume|dissolve|inhale|apply|put|instill|start|cleanse|gently\s+massage|massage)?\s*(?:one|two|three|10\s*ml)?\s*(?:tablet|tab|capsule|cap|rotacap|vial|sachet|puff|sprays?|drops?|teaspoons?|dab\s+of|thin\s+layer\s+of|pea-sized\s+amount\s+of)?\s*(?:of\s+)?([A-Za-z0-9\-]+)",
                core_med_clause,
                re.IGNORECASE,
            )
            med_stem = med_lead_match.group(1).lower() if med_lead_match else ""
            if med_stem in ("one", "two", "three", "tab", "tablet", "capsule", "vial", "sachet", "none", "administer", "take", "start", "apply", "cleanse", "this", "print"):
                med_stem = ""

            primary_insts: List[str] = []
            secondary_insts: List[str] = []

            # 2a. Primary patterns in core medication clause (timing, meal, device, times-of-day)
            for pat in PRIMARY_INSTRUCTION_PATTERNS:
                matches = re.finditer(pat, core_med_clause, re.IGNORECASE)
                for m in matches:
                    matched_txt = m.group(0).strip()
                    matched_lower = matched_txt.lower()

                    # Guard: Inhaler device instructions should only attach to inhaler/respiratory medications
                    if "inhaler" in matched_lower and not any(k in core_med_clause[:80].lower() for k in ("inhal", "rotacap", "revolizer", "puff", "spacer", "budesonide", "tiotropium", "levosalbutamol", "fluticasone")):
                        continue

                    if matched_txt.lower() not in [x.lower() for x in primary_insts]:
                        primary_insts.append(matched_txt)

            # 2b. Secondary patterns in this clause (titrations, precautions, course completion)
            for pat in PER_DRUG_SECONDARY_PATTERNS:
                matches = re.finditer(pat, clause, re.IGNORECASE)
                for m in matches:
                    matched_txt = m.group(0).strip()
                    matched_lower = matched_txt.lower()
                    # Guard: check class mismatch
                    is_mismatched = False
                    for class_key, class_drugs in DRUG_CLASS_MAP.items():
                        if re.search(rf"\b{re.escape(class_key)}\b", matched_lower) and not (any(d in clause_lower for d in class_drugs) or (med_stem and any(d in med_stem for d in class_drugs))):
                            is_mismatched = True
                            break
                    if not is_mismatched and matched_txt.lower() not in [x.lower() for x in secondary_insts]:
                        secondary_insts.append(matched_txt)

            # 2c. Standalone sentences matching this drug or its class
            for sent in raw_sentences:
                sent_lower = sent.lower()

                # Check if this sentence explicitly belongs to a DIFFERENT medicine
                is_other_med = False
                for other_s in segments:
                    if other_s["medicine_id"] != m_id:
                        other_lead = re.search(
                            r"(?:take|start|administer|give|consume|dissolve|inhale|apply|put|instill|cleanse|massage)?\s*(?:one|two|three)?\s*(?:tablet|tab|capsule|cap|rotacap|vial|sachet)?\s*(?:of\s+)?([A-Za-z0-9\-]+)",
                            other_s["clause"],
                            re.IGNORECASE,
                        )
                        other_stem = other_lead.group(1).lower() if other_lead else ""
                        if other_stem and len(other_stem) >= 3 and other_stem not in ("one", "two", "three", "tab", "tablet", "capsule", "none", "this", "print") and other_stem in sent_lower and other_stem != med_stem:
                            is_other_med = True
                            break
                if is_other_med:
                    continue

                # Check explicit drug name or stem mention in the sentence
                mentions_this_drug = (med_stem and len(med_stem) >= 3 and med_stem in sent_lower)
                
                # Check class mention (e.g. "antibiotic", "iron", "inhaler", "painkiller")
                for class_key, class_drugs in DRUG_CLASS_MAP.items():
                    if re.search(rf"\b{re.escape(class_key)}\b", sent_lower) and (any(d in clause_lower for d in class_drugs) or (med_stem and any(d in med_stem for d in class_drugs))):
                        mentions_this_drug = True

                # If sentence matches this drug or is generic unassigned advice in a 1-med prescription
                if mentions_this_drug or (total_meds == 1 and not is_other_med):
                    is_sent_pure_advice = bool(re.search(INDEPENDENT_ADVICE_START, sent.strip())) or any(re.search(p, sent, re.IGNORECASE) for p in CONTINUOUS_ADVICE_SPAN_PATTERNS)
                    # Only check primary patterns if this is NOT a pure advice/lifestyle sentence
                    if not is_sent_pure_advice:
                        for pat in PRIMARY_INSTRUCTION_PATTERNS:
                            s_match = re.search(pat, sent, re.IGNORECASE)
                            if s_match:
                                s_txt = s_match.group(0).strip()
                                s_lower = s_txt.lower()
                                if "inhaler" in s_lower and not any(k in clause[:80].lower() for k in ("inhal", "rotacap", "revolizer", "puff", "spacer", "budesonide", "tiotropium", "levosalbutamol", "fluticasone")):
                                    continue
                                if s_txt.lower() not in [x.lower() for x in primary_insts]:
                                    primary_insts.append(s_txt)
                    # Check secondary patterns
                    for pat in PER_DRUG_SECONDARY_PATTERNS:
                        s_match = re.search(pat, sent, re.IGNORECASE)
                        if s_match:
                            s_txt = s_match.group(0).strip()
                            s_lower = s_txt.lower()
                            is_mismatched = False
                            for class_key, class_drugs in DRUG_CLASS_MAP.items():
                                if re.search(rf"\b{re.escape(class_key)}\b", s_lower) and not (any(d in clause_lower for d in class_drugs) or (med_stem and any(d in med_stem for d in class_drugs))):
                                    is_mismatched = True
                                    break
                            if not is_mismatched and s_txt.lower() not in [x.lower() for x in secondary_insts]:
                                secondary_insts.append(s_txt)

            # 2d. Inhaler device care (Primary)
            if any(k in clause[:80].lower() for k in ("budecort", "foracort", "inhaler", "rotacap", "asthalin", "budesonide", "revolizer", "tiotropium")):
                for lf in ("be sure to rinse your mouth thoroughly after using the inhaler", "rinse your mouth thoroughly after using the inhaler", "rinse your mouth immediately", "rinse mouth after using the inhaler", "rinse mouth after use"):
                    if lf in input_text.lower() and lf not in [x.lower() for x in primary_insts]:
                        primary_insts.append(lf)

            # 2e. Attach global independent advice sentences to the secondary instructions of the final medicine record
            if idx == total_meds - 1:
                for adv in global_clinical_advice:
                    if adv.lower() not in [x.lower() for x in secondary_insts]:
                        secondary_insts.append(adv)

            clean_primary = deduplicate_phrases(primary_insts)
            clean_secondary = deduplicate_phrases(secondary_insts)

            formatted_primary = " ".join(f"{i+1}. {txt}" for i, txt in enumerate(clean_primary)) if clean_primary else "NONE"
            formatted_secondary = " ".join(f"{i+1}. {txt}" for i, txt in enumerate(clean_secondary)) if clean_secondary else "NONE"

            extracted_inst.append({
                "medicine_id": m_id,
                "drug_name": med_stem,
                "instruction": formatted_primary,
                "additional_instruction": formatted_secondary,
            })

    return {"instructions": extracted_inst}
