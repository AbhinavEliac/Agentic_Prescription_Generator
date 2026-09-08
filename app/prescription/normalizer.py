"""
app/prescription/normalizer.py
------------------------------
Canonical Text Normalization Engine for the Prescription Pipeline.

Consolidates:
1. ASR acoustic phonetic error corrections (misheard Indian brand names, drug homophones)
2. Audio dictation timestamp and speaker header elimination
3. Conversational chatter and examination findings filtering
4. Numeric and dosage unit standardization
"""

import re

# Audio recording speaker headers (e.g., "Animesh Kumar, Yesterday 4:30 PM", "Dr. Sharma, 12/04 10:15 AM")
SPEAKER_HEADER_PATTERN = (
    r"(?i)\b[A-Za-z\s\.\-]{2,30},\s*(?:yesterday|today|tomorrow|\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?)[\s\u202f\xa0]*"
    r"\d{1,2}:\d{2}[\s\u202f\xa0]*(?:[ap]\.?m\.?)?\s*"
)

# Conversational chatter, examination findings, vitals, and non-prescription banter
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
    r"(?i)\b(?:apply\s+)?(?:hot|cold)?\s*water\s+fomentation(?:\s+and\s+ice\s+packs?)?(?:\s+to\s+[a-zA-Z\s]+)?\b[^\.\n;!?]*[\.\n;!?]?",
    r"(?i)\b(?:apply\s+)?ice\s+packs?(?:\s+to\s+[a-zA-Z\s]+)?\b[^\.\n;!?]*[\.\n;!?]?",
]


def normalize_asr_phonetics(text: str) -> str:
    """
    Normalizes speech recognition acoustic confusions and misheard brand names.
    Consolidates rules from Fast Mode, Node drugDbService, and LangGraph utils.
    """
    if not text:
        return ""
    
    t = text
    # Atenolol / Aten variations
    t = re.sub(r"\b(?:8|eight)\s+(?:and|&)\s+(?:50|fifty)\b", "Aten 50", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:8|eight)\s+(?:and|&)\s+(?:25|twenty\s*five)\b", "Aten 25", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:8|eight)\s+(?:and|&)\s+(?:100|one\s*hundred)\b", "Aten 100", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:8|eight)\s*(?:ten|10)\s+(?:50|fifty)\b", "Aten 50", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:8|eight)\s*(?:ten|10)\s+(?:25|twenty\s*five)\b", "Aten 25", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:8|eight)\s*(?:ten|10)\b", "Aten", t, flags=re.IGNORECASE)
    
    # Metformin variations & ASR errors
    t = re.sub(r"\bMetaforamine\b", "Metformin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bMetaformine\b", "Metformin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bMetaphormine\b", "Metformin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bMetphormin\b", "Metformin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmet\s+for\s+men\b", "Metformin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmet\s+form\s+in\b", "Metformin", t, flags=re.IGNORECASE)
    
    # Soframycin variations
    t = re.sub(r"\bSophramycin\b", "Soframycin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bSophramicine\b", "Soframycin", t, flags=re.IGNORECASE)
    
    # Paracetamol / Dolo / Crocin ASR errors
    t = re.sub(r"\bParacip\b", "Paracetamol", t, flags=re.IGNORECASE)
    t = re.sub(r"\bParacitamol\b", "Paracetamol", t, flags=re.IGNORECASE)
    t = re.sub(r"\bAzithromicin\b", "Azithromycin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bMetforrmin\b", "Metformin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdollar\s+650\b", "Dolo 650", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdollar\s+six\s+fifty\b", "Dolo 650", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdolo\s+six\s+fifty\b", "Dolo 650", t, flags=re.IGNORECASE)
    t = re.sub(r"\bcrocin\s+six\s+fifty\b", "Crocin 650", t, flags=re.IGNORECASE)
    t = re.sub(r"\bamoxie\s+klav\b", "Amoxyclav", t, flags=re.IGNORECASE)
    t = re.sub(r"\bamoxi\s+clav\b", "Amoxyclav", t, flags=re.IGNORECASE)
    t = re.sub(r"\bpantocid\s+forty\b", "Pantocid 40", t, flags=re.IGNORECASE)
    t = re.sub(r"\bpantop\s+forty\b", "Pantop 40", t, flags=re.IGNORECASE)
    
    return t


def normalize_hinglish_expressions(text: str) -> str:
    """
    Normalizes mixed Hindi-English (Hinglish) clinical phrasing commonly spoken in Indian clinical dictations.
    Converts Hindi frequency, timing, meals, and duration words to canonical English terms.
    """
    if not text:
        return ""
    
    t = text
    # Fever SOS condition
    t = re.sub(r"\bjab\s+bukhar\s+ho(?:\s+tab\s+lena)?(?:\s+zarurat\s+padne\s+par)?\b", "SOS for fever", t, flags=re.IGNORECASE)

    # Frequency: din me / din mein
    t = re.sub(r"\bdin\s+me(?:in)?\s+do\s+baar\b", "twice daily", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdin\s+me(?:in)?\s+ek\s+baar\b", "once daily", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdin\s+me(?:in)?\s+teen\s+baar\b", "three times daily", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdin\s+me(?:in)?\s+chaar\s+baar\b", "four times daily", t, flags=re.IGNORECASE)
    t = re.sub(r"\broz(?:ana)?\s+ek\s+baar\b", "once daily", t, flags=re.IGNORECASE)
    t = re.sub(r"\broz(?:ana)?\s+do\s+baar\b", "twice daily", t, flags=re.IGNORECASE)
    t = re.sub(r"\broz(?:ana)?\b", "daily", t, flags=re.IGNORECASE)
    t = re.sub(r"\bsubah\s+(?:aur\s+)?shaam\b", "morning and evening (twice daily)", t, flags=re.IGNORECASE)

    # Meals & Timing
    t = re.sub(r"\bsubah\s+khali\s+pet(?:\s+lo|\s+lena)?\b", "daily on an empty stomach", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkhali\s+pet(?:\s+lo|\s+lena)?\b", "on an empty stomach", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkhana\s+khane\s+ke\s+baad(?:\s+lo|\s+lena)?\b", "after meals", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkhane\s+ke\s+baad(?:\s+lo|\s+lena)?\b", "after meals", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkhana\s+khane\s+se\s+pehle(?:\s+lo|\s+lena)?\b", "before meals", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkhane\s+se\s+pehle(?:\s+lo|\s+lena)?\b", "before meals", t, flags=re.IGNORECASE)
    t = re.sub(r"\braat\s+ko\s+sone\s+se\s+pehle(?:\s+lo|\s+lena)?\b", "at bedtime", t, flags=re.IGNORECASE)
    t = re.sub(r"\bsone\s+se\s+pehle(?:\s+lo|\s+lena)?\b", "at bedtime", t, flags=re.IGNORECASE)
    t = re.sub(r"\bzarurat\s+padne\s+par\b", "as needed (SOS)", t, flags=re.IGNORECASE)
    t = re.sub(r"\bjab\s+zarurat\s+ho\b", "as needed (SOS)", t, flags=re.IGNORECASE)
    t = re.sub(r"\bjab\s+dard\s+ho\b", "as needed for pain", t, flags=re.IGNORECASE)

    # Dosage forms & numbers
    t = re.sub(r"\bek\s+goli\b", "one tablet", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdo\s+goli\b", "two tablets", t, flags=re.IGNORECASE)
    t = re.sub(r"\bteen\s+goli\b", "three tablets", t, flags=re.IGNORECASE)
    t = re.sub(r"\bek\s+chammach\b", "one teaspoon", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdo\s+chammach\b", "two teaspoons", t, flags=re.IGNORECASE)
    t = re.sub(r"\bgoli\b", "tablet", t, flags=re.IGNORECASE)

    # Duration: din tak / hafte tak
    t = re.sub(r"\b(\d+)\s+din\s+tak\b", r"for \1 days", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\d+)\s+hafte\s+tak\b", r"for \1 weeks", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\d+)\s+mahine\s+tak\b", r"for \1 months", t, flags=re.IGNORECASE)
    t = re.sub(r"\bpaani\s+ke\s+saath(?:\s+lo|\s+lena)?\b", "with water", t, flags=re.IGNORECASE)

    # Strip standalone Hindi imperative verb tokens
    t = re.sub(r"\b(?:lo|le\s+lena|lena)\b", "", t, flags=re.IGNORECASE)

    return t


def clean_conversational_noise(text: str) -> str:
    """
    Strips speaker metadata, audio timestamps, clinical vitals dictation,
    and doctor-patient conversational greetings/banter.
    """
    if not text:
        return ""
    
    # 1. Transform speaker headers into sentence transitions
    cleaned = re.sub(SPEAKER_HEADER_PATTERN, ". Take ", text).strip()
    
    # 2. Filter conversational filler sentences
    for pat in CONVERSATIONAL_NOISE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned)
    
    return cleaned


def normalize_numbers_and_units(text: str) -> str:
    """
    Standardizes numbers, commas in integers, and decimal formulations.
    """
    if not text:
        return ""
    
    # Remove commas between digits (e.g., 1,000 -> 1000)
    cleaned = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    
    # Lead-in zero for naked decimal fractions (e.g., .5% -> 0.5%, .25 mg -> 0.25 mg)
    cleaned = re.sub(r"(?<=\s)\.(\d+)\s*(%|mg|g|mcg|ml)", r"0.\1 \2", cleaned, flags=re.IGNORECASE)
    
    # Collapse irregular whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def correct_sentence_punctuation(text: str) -> str:
    """
    Normalizes sentence boundaries, periods, commas, and capitalizations
    without modifying spelling or numeric values.
    """
    if not text:
        return ""

    s = text.strip()

    # 1. Clean repetitive whitespace and formatting without touching numbers/decimals
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<!\d)\s*,\s*(?!\d)", ", ", s)
    s = re.sub(r"(?<!\d)\s*\.\s*(?!\d)", ". ", s)
    s = re.sub(r"[\s,\-]+(?=\.(?!\d))", "", s)

    # 2. Insert comma after condition clauses before consequence action verbs (consult, start, increase, meet, visit, etc.)
    s = re.sub(
        r"(?i)\b(if\s+[a-zA-Z0-9\s\-]+?(?:does\s+not\s+go\s+away|does\s+not\s+clear|does\s+not\s+improve|persists|worsens|increases|crosses\s+\d+|develops|occurs|remains\s+high|subsides|heals|build\s+up))\s+(increase|decrease|reduce|double|taper|meet|come\s+visit|visit|consult|seek|start|stop|discontinue)\b",
        r"\1, \2",
        s,
    )

    # 3. Bridge schedule/duration clauses split across periods (e.g. "capsule. Once daily for 14 days, take...")
    s = re.sub(
        r"\.\s*((?:Once|Twice|Thrice|\d+\s+times|Every|Daily|At\s+bedtime|In\s+the\s+morning)[^\.,;]+?),\s*(take|administer|give|start|apply|inhale|instill)",
        r" \1. \2",
        s,
        flags=re.IGNORECASE,
    )

    # 4. Insert sentence breaks before medication action verbs when preceded by completed medication instructions
    med_verb_pattern = (
        r"(?<=[a-zA-Z0-9%\)])(?<!\bto)(?:\s*,\s*|\s+)(?:and\s+)?"
        r"(take\s+(?!(?:walks?|rest|care|steam))\w+|take\s+regular\s+water|administer|inhale|apply|instill|consume|dissolve|inject|infuse)\b"
    )
    s = re.sub(med_verb_pattern, r". \1", s, flags=re.IGNORECASE)

    # 5. Insert sentence breaks before clinical follow-up, advice, and warnings
    advice_pattern = (
        r"(?<=[a-zA-Z0-9%\)])(?<!\bto)(?:\s*,\s*|\s+)(?:and\s+)?"
        r"(if\s+(?:the\s+|it\s+(?:is\s+)?(?:still\s+)?|they\s+|there\s+is\s+)?(?:fever|headache|pain|symptoms?|rash|condition|severe|swelling|ulcers|blood\s+pressure|breathing|does\s+not|persists|not\s+relieved|body\s+does\s+not)|"
        r"meet\s+(?:the\s+)?doctor|please\s+see\s+me|see\s+(?:your\s+|the\s+)?doctor|seek\s+reassessment|seek\s+medical|"
        r"return\s+for|return\s+if|return\s+after|use\s+saline\s+nasal|perform\s+steam|(?:also\s+)?take\s+walks|"
        r"drink\s+plenty|keep\s+the\s+blistered|avoid\s+close)\b"
    )
    s = re.sub(advice_pattern, r". \1", s, flags=re.IGNORECASE)

    # 6. Clean up multiple periods without touching decimals
    s = re.sub(r"(?<!\d)\.\s*\.+(?!\d)", ".", s)
    s = re.sub(r"(?<!\d)\s*\.\s*(?!\d)", ". ", s).strip()
    if s and not s.endswith((".", "!", "?")):
        s += "."

    # 7. Sentence segmentation and start-of-sentence capitalization
    sentences = [sent.strip() for sent in re.split(r"(?<=[a-zA-Z%!?\)])\.\s+(?=[A-Za-z0-9])", s) if sent.strip()]
    capitalized = []
    for sent in sentences:
        if sent:
            sent_cap = sent[0].upper() + sent[1:]
            capitalized.append(sent_cap)

    punctuated_result = ". ".join(capitalized) if capitalized else s
    if punctuated_result and not punctuated_result.endswith((".", "!", "?")):
        punctuated_result += "."

    return punctuated_result


def normalize_prescription_text(raw_text: str) -> str:
    """
    Master normalization entry point for raw prescription transcripts.
    Executes phonetics -> noise filtering -> sentence punctuation -> number/unit sanitation.
    """
    if not raw_text or not raw_text.strip():
        return ""
    
    step1 = normalize_asr_phonetics(raw_text)
    step2 = normalize_hinglish_expressions(step1)
    step3 = clean_conversational_noise(step2)
    step4 = correct_sentence_punctuation(step3)
    step5 = normalize_numbers_and_units(step4)
    return step5
