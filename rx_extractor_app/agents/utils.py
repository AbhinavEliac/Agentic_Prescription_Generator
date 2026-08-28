"""
agents/utils.py
---------------
Shared helper utilities, JSON parsers, noise filters, and multi-medicine clause segmenters.
"""
import re
import json
from typing import Any, Optional, List, Dict

FORM_PATTERN = r"(?i)\b(tablets?|tabs?|capsules?|caps?|rotacaps?|pills?|syrups?|gels?|drops?|sprays?|ointments?|creams?|sachets?|lozenges?|puffs?|respules?|suspensions?|solutions?|vials?|lotions?|patches?|suppositor(?:y|ies)|pastes?|mouthwash(?:es)?|washes?)\b"

ACTION_VERBS_PATTERN = (
    r"(?i)^(?:administer\s+one\s+tablet\s+of|administer\s+one\s+tab\s+of|administer\s+one\s+capsule\s+of|"
    r"administer\s+two\s+sprays\s+of|administer\s+one\s+spray\s+of|administer\s+two\s+drops\s+of|"
    r"administer\s+three\s+drops\s+of|administer\s+one\s+of|administer\s+one|administer\s+tablet\s+of|"
    r"administer\s+tab\s+of|administer|"
    r"give\s+one\s+tablet\s+of|give\s+one|give|"
    r"prescribe\s+one|prescribe|"
    r"start\s+one|start|"
    r"take\s+one\s+combination\s+tablet\s+of|take\s+one\s+tablet\s+of|take\s+one\s+tab\s+of|take\s+one\s+capsule\s+of|"
    r"take\s+one\s+vial\s+of|take\s+one\s+sachet\s+of|take\s+one\s+single\s+dose\s+of|take\s+single\s+dose\s+of|single\s+dose\s+of|single\s+dose|"
    r"take\s+one\s+of|take\s+one|"
    r"take\s+two\s+teaspoons\s+of|take\s+two\s+teaspoon\s+of|take\s+10\s*ml\s+of|take\s+two\s+tablets\s+of|take\s+two|"
    r"take\s+tablet\s+of|take\s+tab\s+of|take\s+capsule\s+of|"
    r"take\s+(?!(?:walks?|a\s+walk|rest|care|steam|bath|deep\s+breaths?|precautions?|fomentation|ice|hot\s+water|cold\s+water))\b|"
    r"consume\s+one\s+vial\s+of|consume\s+one\s+sachet\s+of|consume\s+one|consume|"
    r"dissolve\s+one\s+sachet\s+of|dissolve\s+one|dissolve|"
    r"slowly\s+dissolve\s+one|slowly\s+dissolve|"
    r"inhale\s+one\s+rotacap\s+of|inhale\s+one\s+puff\s+of|inhale\s+two\s+puffs\s+of|inhale\s+one\s+capsule\s+of|inhale\s+one|inhale\s+two|inhale|"
    r"apply\s+a\s+thin\s+layer\s+of|apply\s+a\s+pea-sized\s+amount\s+of|apply\s+a\s+dab\s+of|apply\s+a\s+bland\s+moisturizing|apply\s+broad-spectrum|apply\s+one|apply|"
    r"gently\s+massage|massage|"
    r"rub\s+one|rub|"
    r"spray\s+one|spray|"
    r"cleanse\s+the\s+skin\s+gently\s+using|cleanse|"
    r"put\s+one|put|instill\s+two\s+drops\s+of|instill\s+three\s+drops\s+of|instill\s+two|instill\s+three|instill\s+one|instill|inject\s+one|inject|infuse\s+one|infuse|"
    r"thin\s+layer\s+of|layer\s+of|layer|pea-sized\s+amount\s+of|amount\s+of|amount|dab\s+of|dab|combined|skin\s+gently\s+using)\s+"
)

SPEAKER_HEADER_PATTERN = r"(?i)\b[A-Za-z\s\.\-]{2,30},\s*(?:yesterday|today|tomorrow|\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?)[\s\u202f\xa0]*\d{1,2}:\d{2}[\s\u202f\xa0]*(?:[ap]\.?m\.?)?\s*"

# Conversational chatter, greetings, examination findings, and filler patterns
CONVERSATIONAL_NOISE_PATTERNS = [
    r"(?i)\b(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening)\s*(?:doctor|dr\.|mr\.|mrs\.|ms\.)?(?:\s+[A-Za-z]+)?\b[^\.\n;!?]*[\.\n;!?]?",
    r"(?i)\bhow\s+are\s+you\s+(?:feeling|doing)?(?:\s+today)?\b[^\.\n;!?]*[\.\n;!?]?",
    r"(?i)\b(?:i\s+have|patient\s+complains\s+of|complaining\s+of|suffering\s+from|history\s+of|diagnosed\s+with|patient\s+has)\s+[^\.\n;!?]+[\.\n;!?]?",
    r"(?i)\b(?:let\s+me\s+check\s+your\s+[^\.\n;!?]+)[\.\n;!?]?",
    r"(?i)\b(?:throat\s+is\s+congested|chest\s+is\s+clear|tonsils\s+are\s+swollen)[\.\n;!?]?",
    r"(?i)\b(?:bp\s+is|pulse\s+is|temperature\s+is|spo2\s+is|saturation\s+is|weight\s+is)\s+[^,\.\n;!]+[\.\n;!?]?",
    r"(?i)\b(?:thank\s+you(?:\s+very\s+much)?(?:\s+doctor)?|thanks\s+doctor|thanks)\b[^\.\n;!?]*[\.\n;!?]?",
    r"(?i)\b(?:have\s+a\s+(?:nice|great|good)\s+day|take\s+care(?:\s+bye)?|goodbye|see\s+you\s+(?:next\s+time|soon|after\s+a\s+week))\b[^\.\n;!?]*[\.\n;!?]?",
    r"(?i)\b(?:i\s+will\s+follow\s+this|sure\s+doctor|okay\s+doctor|understood\s+doctor|alright\s+doctor)\b[^\.\n;!?]*[\.\n;!?]?",
    r"(?i)\b(?:patient|name|age|gender|sex|recorded\s+on|appointment\s+id):\s*[^\.\n,]+[,\.]?",
]

# Non-medication entity blacklist (prevents noisy terms from becoming drug names)
INVALID_DRUG_NAMES_LOWER = {
    "doctor", "dr", "patient", "individual", "individuals", "person", "persons", "people",
    "mr", "mrs", "ms", "this", "that", "none", "take", "give", "start", "apply", "cleanse", "massage",
    "ice packs", "ice pack", "fomentation", "hot water fomentation", "water", "drinking water",
    "hot water", "cold water", "tap water", "salt water", "saline water", "cloth", "towel", "bandage", "tape",
    "endoscopy", "endoscopy report", "x-ray", "ultrasound", "mri", "ecg", "blood test", "blood test results",
    "urine culture", "biopsy", "examination", "vitals", "report", "results", "appointment", "consultation",
    "blood pressure", "bp", "pulse", "temperature", "saturation", "spo2", "weight", "chest", "throat", "abdomen",
    "good morning", "thank you", "thanks", "how are you", "have a nice day", "goodbye", "bye", "okay", "alright",
    "walks", "morning walks", "rest", "steam", "steam inhalation", "pregnant", "pregnant individuals"
}


def is_placeholder(val: Any) -> bool:
    """Checks if a string is a template placeholder copied by smaller LLMs."""
    if not isinstance(val, str):
        return False
    s = val.strip().lower()
    if s.startswith("<") and s.endswith(">"):
        return True
    placeholder_signatures = [
        "<exact drug name",
        "<second dose",
        "<verbatim",
        "<oral|",
        "<instruction",
        "primary dose>",
        "second dose or none",
        "verbatim frequency",
        "verbatim duration",
        "<drug name>",
    ]
    return any(sig in s for sig in placeholder_signatures)


def is_valid_medication_entity(drug_name: str) -> bool:
    """
    Noise-proofing filter: Validates whether a candidate string represents a genuine pharmaceutical entity.
    Rejects conversational chatter, diagnostic tests, vital signs, physical items, and non-drug terms.
    """
    if not drug_name or is_placeholder(drug_name):
        return False
    name_clean = drug_name.strip()
    name_lower = name_clean.lower()

    if name_lower in ("none", "", "null", "n/a"):
        return False

    # Check exact blacklist matches
    if name_lower in INVALID_DRUG_NAMES_LOWER:
        return False

    # Check substring blacklist matches
    for invalid in ("pregnant individuals", "endoscopy report", "blood pressure", "urine culture", "blood test", "ice packs", "fomentation", "thank you", "good morning"):
        if invalid in name_lower and not any(unit in name_lower for unit in ("mg", "mcg", "ml", "iu", "%", "g")):
            return False

    # Must contain at least one word character of length >= 2
    words = re.findall(r"[A-Za-z0-9\-]+", name_clean)
    if not words:
        return False

    # Reject if it consists purely of noise words
    meaningful_words = [w for w in words if w.lower() not in ("take", "one", "two", "three", "tablet", "capsule", "syrup", "pill", "none", "of", "and", "with", "the", "a", "an")]
    if not meaningful_words:
        return False

    return True


def clean_noise_and_chatter(input_text: str) -> str:
    """
    Noise-proofing cleaner: Filters conversational chatter, doctor-patient greetings,
    examination findings, and audio recording timestamps from raw transcripts.
    """
    if not input_text:
        return ""
    
    cleaned = re.sub(SPEAKER_HEADER_PATTERN, ". Take ", input_text).strip()
    for pat in CONVERSATIONAL_NOISE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned)
    
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_instruction_noise(inst_text: str) -> str:
    """
    Filters conversational banter and greeting residues from instructions.
    """
    if not inst_text or inst_text.upper() == "NONE":
        return "NONE"
    
    cleaned = inst_text
    for pat in CONVERSATIONAL_NOISE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned)
    
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "NONE"


def safe_parse_json(text: str) -> Optional[Any]:
    """Attempts to extract and parse JSON array or object from LLM response."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass

    arr_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0).strip())
        except Exception:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0).strip())
        except Exception:
            pass

    return None


def segment_prescription(input_text: str) -> List[Dict[str, Any]]:
    """
    Robustly segments multi-medicine prescriptions into discrete clauses.
    Recognizes all transition points, companion drugs, and filters non-drug advice sentences.
    Works seamlessly with or without punctuation.
    """
    clean_text = clean_noise_and_chatter(input_text)
    clean_text = re.sub(r"(\d+),(\d+)", r"\1\2", clean_text)
    # Bridge schedule/duration clauses split across periods (e.g. "capsule. Once daily for 14 days, take...")
    clean_text = re.sub(r"\.\s*((?:Once|Twice|Thrice|\d+\s+times|Every|Daily|At\s+bedtime|In\s+the\s+morning)[^\.,;]+?),\s*(take|administer|give|start|apply|inhale|instill)", r" \1. \2", clean_text, flags=re.IGNORECASE)

    # 1. Break into sentences by standard sentence punctuation (decimal-safe)
    raw_sentences = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)|[\n;!]", clean_text) if s.strip()]

    # 2. Split intra-sentence transition points (e.g. "Take A, and take B, also start C")
    split_pattern = (
        r"(?i)(?:,\s*(?:and\s+take|also\s+take|additionally(?:,\s*take)?|then\s+take|take\s+one|take\s+two|take\s+10\s*ml|take\s+\d+|take\s+(?!(?:walks?|a\s+walk|rest|care|steam|bath))\b|"
        r"and\s+start|also\s+start|start\s+one|start|"
        r"and\s+administer|also\s+administer|administer\s+one|administer\s+two|administer|"
        r"and\s+give|also\s+give|give\s+one|give|"
        r"and\s+consume|also\s+consume|consume\s+one|consume|"
        r"and\s+dissolve|also\s+dissolve|dissolve\s+one|dissolve|slowly\s+dissolve|"
        r"and\s+inhale|also\s+inhale|inhale\s+one|inhale\s+two|inhale|"
        r"and\s+apply|also\s+apply|apply\s+a\s+thin|apply\s+a\s+pea-sized|apply\s+a\s+dab|apply\s+broad-spectrum|apply\s+one|apply|"
        r"and\s+gently\s+massage|gently\s+massage|and\s+massage|massage|"
        r"and\s+rub|also\s+rub|rub\s+one|rub|"
        r"and\s+spray|also\s+spray|spray\s+one|spray|"
        r"and\s+instill|also\s+instill|instill\s+two|instill\s+three|instill\s+one|instill|"
        r"and\s+cleanse|cleanse\s+the\s+skin|cleanse|"
        r"and\s+put|also\s+put|put\s+one|put|inject\s+one|inject|infuse\s+one|infuse)|"
        r"\band\s+take\s+\d+\b|\band\s+take\s+(?!(?:walks?|a\s+walk|rest|care|steam|bath))\b|"
        r"\balso\s+take\s+(?!(?:walks?|a\s+walk|rest|care|steam|bath))\b|"
        r"\badditionally\b|\bthen\s+take\b|"
        r"\band\s+start\b|\band\s+administer\b|\band\s+consume\b|\band\s+dissolve\b|\band\s+spray\b|\band\s+instill\b|\binhale\s+one\b|\binhale\s+two\b|\bapply\s+one\b|\bapply\s+a\b|"
        r"\bkeep\s+the\s+blistered\s+area\b|\bavoid\s+close\s+physical\s+contact\b|\breturn\s+if\s+the\s+rash\b)"
    )

    clauses: List[str] = []
    for sent in raw_sentences:
        sub_clauses = [c.strip() for c in re.split(split_pattern, sent) if c.strip()]
        clauses.extend(sub_clauses if sub_clauses else [sent])

    if not clauses:
        clauses = [clean_text]

    segments: List[Dict[str, Any]] = []
    med_id = 1

    # Non-drug trigger keywords that indicate a clause is advice/follow-up, not a medicine
    PURE_ADVICE_TRIGGERS = [
        r"^(?:return|come\s+back|follow\s+up|review|get\s+reassessed|seek|consult|contact|report|visit|revisit|arrange|schedule|repeat|re-?test)\b",
        r"^(?:meet\s+(?:the\s+)?doctor|please\s+see\s+me|see\s+(?:your\s+|the\s+)?doctor)\b",
        r"^(?:do\s+not|avoid|strictly\s+avoid|discontinue|stop|keep|maintain|stick\s+to|include|limit|practice|brush|perform)\b",
        r"^(?:drink|consume|stay\s+well-hydrated|apply\s+local|apply\s+ice|sponge|wear|monitor|be\s+sure\s+to|if\s+)\b",
        r"(?:also\s+)?take\s+walks?",
        r"go\s+for\s+(?:morning\s+)?walks?",
        r"(?:apply\s+)?(?:local\s+)?hot\s+water\s+fomentation",
        r"(?:apply\s+)?ice\s+packs?",
        r"sponge\s+(?:the\s+)?(?:body|forehead)",
        r"keep\s+(?:the\s+)?(?:blistered|affected|skin|dressing|ear)\s+area\s+clean\s+and\s+dry",
        r"avoid\s+close\s+physical\s+contact",
    ]

    NON_DRUG_WITH = (
        "food", "meals", "meal", "water", "milk", "breakfast", "lunch", "dinner", "juice",
        "caution", "care", "warm", "cold", "tap", "equal", "a", "an", "the", "endoscopy", "report",
        "doctor", "review", "evaluation", "prescription", "consultation", "food", "stomach", "walks",
        "pregnant", "individuals", "persons", "children", "people"
    )

    for clause in clauses:
        clause_clean = clause.strip()
        # If this clause is purely advice without medication, skip creating a medicine block
        is_advice = any(re.search(pat, clause_clean, re.IGNORECASE) for pat in PURE_ADVICE_TRIGGERS)
        if is_advice:
            continue

        # Check if the clause has any medication indicator (action verb, dosage, or form word)
        has_med_indicator = bool(
            re.search(r"(?i)(?:\b(?:take\s+(?!(?:walks?|rest|care|steam))\w+|administer|give|prescribe|start|consume|dissolve|inhale|apply|put|instill|inject|infuse|tablet|tab|capsule|cap|rotacap|vial|sachet|syrup|gel|drops?|spray|ointment|cream|lotion)\b|\d+\s*(?:mg|g|mcg|ml|iu|%)\b)", clause_clean)
        )
        if not has_med_indicator and len(segments) > 0:
            continue

        # Check companion drug "combination of A and B" or "A with B [dose]"
        # 1. "combination tablet/of A and B"
        comb_match = re.search(r"combination\s+(?:tablet\s+of\s+|of\s+)?([A-Za-z0-9\s\.\-]+?\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|IU)?)\s+and\s+([A-Za-z0-9\s\.\-]+?\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|IU)?)", clause_clean, re.IGNORECASE)
        if comb_match:
            d1_raw = comb_match.group(1).strip()
            d2_raw = comb_match.group(2).strip()
            d1 = re.split(r"(?i)\b(?:once|twice|thrice|daily|od|bd|tid|qid|for|after|before|every)\b", d1_raw)[0].strip()
            d2 = re.split(r"(?i)\b(?:once|twice|thrice|daily|od|bd|tid|qid|for|after|before|every)\b", d2_raw)[0].strip()
            segments.append({"medicine_id": med_id, "clause": clause_clean, "seed_name": d1})
            med_id += 1
            segments.append({"medicine_id": med_id, "clause": clause_clean, "seed_name": d2})
            med_id += 1
            continue

        # 2. "A with B [dose]"
        comp_match = re.search(r"([A-Za-z0-9\s\.\-]+?)\s+with\s+([A-Za-z0-9\s\.\-]+)", clause_clean, re.IGNORECASE)
        if comp_match:
            d1_raw = comp_match.group(1).strip()
            d2_raw = comp_match.group(2).strip()
            d2_first_word = d2_raw.split()[0].lower() if d2_raw else ""
            has_dose_or_form = bool(re.search(r"(?i)\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|%)|\b(?:tablets?|capsules?|gel|lotion|drops?|solution|cream|ointment)\b", d2_raw))
            if d2_first_word not in NON_DRUG_WITH and len(d2_first_word) >= 3 and has_dose_or_form:
                d1 = re.split(r"(?i)\b(?:once|twice|thrice|three\s+times|four\s+times|daily|od|bd|tid|qid|for|after|before|every)\b", d1_raw)[0].strip()
                d2 = re.split(r"(?i)\b(?:locally|once|twice|thrice|three\s+times|four\s+times|daily|od|bd|tid|qid|for|after|before|every|without|using|at|into|in|on|to|along|over|around|strictly|ensuring|diluted|dissolved|mixed|slowly)\b", d2_raw)[0].strip()
                segments.append({"medicine_id": med_id, "clause": clause_clean, "seed_name": d1})
                med_id += 1
                segments.append({"medicine_id": med_id, "clause": clause_clean, "seed_name": d2})
                med_id += 1
            else:
                segments.append({"medicine_id": med_id, "clause": clause_clean, "seed_name": ""})
                med_id += 1
        else:
            segments.append({"medicine_id": med_id, "clause": clause_clean, "seed_name": ""})
            med_id += 1

    if not segments:
        segments.append({"medicine_id": 1, "clause": clean_text, "seed_name": ""})

    return segments
