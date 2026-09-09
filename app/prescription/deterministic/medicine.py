"""
app/prescription/deterministic/medicine.py
------------------------------------------
Deterministic Drug Name Extractor.

Extracts genuine pharmaceutical entities from prescription clauses:
- Matches against DrugRepository using exact base name, Soundex phonetic index, and prefix tree
- Filters out non-medication entities (examination findings, diagnostics, physical care items)
- Removes formulation noise words (TAB, CAP, SYRUP, ROTACAP, INJ) and action verbs
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, List, Set
from app.drugs.repository import DrugRepository

FORM_PATTERN = (
    r"(?i)\b(tablets?|tabs?|capsules?|caps?|rotacaps?|pills?|syrups?|gels?|drops?|sprays?|"
    r"ointments?|creams?|sachets?|lozenges?|puffs?|respules?|suspensions?|solutions?|vials?|"
    r"lotions?|patches?|suppositor(?:y|ies)|pastes?|mouthwash(?:es)?|washes?)\b"
)

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

# Blacklist of non-medication clinical entities, body parts, diagnostic terms, physical care
INVALID_DRUG_NAMES_LOWER: Set[str] = {
    "doctor", "dr", "patient", "individual", "individuals", "person", "persons", "people",
    "mr", "mrs", "ms", "this", "that", "none", "take", "give", "start", "apply", "cleanse", "massage",
    "ice packs", "ice pack", "fomentation", "hot water fomentation", "water", "drinking water",
    "hot water", "cold water", "tap water", "salt water", "saline water", "cloth", "towel", "bandage", "tape",
    "endoscopy", "endoscopy report", "x-ray", "ultrasound", "mri", "ecg", "blood test", "blood test results",
    "urine culture", "biopsy", "examination", "vitals", "report", "results", "appointment", "consultation",
    "blood pressure", "bp", "pulse", "temperature", "saturation", "spo2", "weight", "chest", "throat", "abdomen",
    "good morning", "thank you", "thanks", "how are you", "have a nice day", "goodbye", "bye", "okay", "alright",
    "walks", "morning walks", "rest", "steam", "steam inhalation", "pregnant", "pregnant individuals",
    "fever", "pain", "headache", "cough", "cold", "rash", "vomiting", "nausea",
    "daily", "once", "twice", "thrice", "every", "before", "after", "morning", "night", "evening",
    "afternoon", "breakfast", "lunch", "dinner", "food", "meals", "meal", "days", "day", "weeks",
    "week", "months", "month", "hours", "hour", "times", "time", "little", "more", "still", "water",
    "protein", "eating", "form", "liquid", "stowing", "show", "build", "up", "body",
    "into", "each", "both", "nostril", "nostrils", "ear", "ears", "eye", "eyes",
    "into each nostril", "into both nostrils", "affected ear",
    "knee", "leg", "arm", "shoulder", "back", "joint", "skin", "packs", "pack", "fomentation", "ice", "hot water", "cold water",
    "ice pack", "ice packs", "water fomentation", "hot water fomentation", "affected knee", "affected joint",
    "for", "and", "also", "then", "with", "from", "till", "until", "upto", "along"
}


def is_valid_medication_name(candidate: str) -> bool:
    """Noise-proofing filter: ensures candidate string represents a genuine drug name."""
    if not candidate:
        return False
    cand_clean = candidate.strip()
    cand_lower = cand_clean.lower()

    if cand_lower in ("none", "", "null", "n/a", "unknown"):
        return False

    if cand_lower in INVALID_DRUG_NAMES_LOWER:
        return False

    for invalid in ("pregnant individuals", "endoscopy report", "blood pressure", "urine culture", "blood test", "ice packs", "hot water", "fomentation", "into each nostril", "affected knee", "packs to"):
        if invalid in cand_lower and not any(unit in cand_lower for unit in ("mg", "mcg", "ml", "iu", "%", "g")):
            return False

    words = re.findall(r"[A-Za-z0-9\-]+", cand_clean)
    if not words:
        return False

    meaningful = [
        w for w in words
        if w.lower() not in ("take", "one", "two", "three", "tablet", "capsule", "syrup", "pill", "none", "of", "and", "with", "the", "a", "an", "into", "each", "both", "nostril", "nostrils", "for", "also", "then", "from", "till", "until", "upto")
        and w.lower() not in INVALID_DRUG_NAMES_LOWER
    ]
    return bool(meaningful)


def clean_candidate_name(raw_name: str) -> str:
    """Strips leading verbs, articles, and formulation words from a raw candidate name."""
    cleaned = re.sub(FORM_PATTERN, "", raw_name).strip()
    cleaned = re.sub(ACTION_VERBS_PATTERN, "", cleaned).strip()
    cleaned = re.sub(r"(?i)^(?:take|administer|give|start|prescribe|consume|dissolve|inhale|apply|put|instill|inject|infuse)\s+", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^(?:one|two|three|four|five|\d+)\s+", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^(?:of\s+|a\s+|an\s+|the\s+)", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\b(?:of|one|vial|sachet|combination)\b", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\b(?:nasal|topical)\s+(?=spray|drops?|suspension|cream|ointment|gel|lotion)", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\boral\s+(?=suspension|solution|drops?|tablets?|capsules?|syrup)", "", cleaned).strip()
    cleaned = re.sub(r"[\s,;\-]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_medicine_candidate(
    clause_text: str,
    seed_drug_name: Optional[str] = None,
    drug_repo: Optional[DrugRepository] = None,
) -> Tuple[Optional[str], Optional[Tuple[int, int]], float, Optional[str], Optional[str], List[Dict[str, Any]]]:
    """
    Deterministically extracts and grounds the candidate medicine name against Drug_database.
    Returns:
      (matched_base_name, span, confidence, did_you_mean, matched_drug_id, did_you_mean_options)
    Strictly returns None if the candidate does not exist in Drug_database (neither exact, normalized, nor phonetic).
    """
    if not clause_text or not clause_text.strip():
        return None, None, 0.0, None, None, []

    # Helper to test and ground a candidate string against Drug_database
    def _test_grounding(cand: str) -> Optional[Tuple[str, float, Optional[str], Optional[str], List[Dict[str, Any]]]]:
        if not cand or not is_valid_medication_name(cand):
            return None
        cleaned = clean_candidate_name(cand)
        if not cleaned or not is_valid_medication_name(cleaned):
            return None

        if not drug_repo:
            return cleaned, 0.85, None, None, []

        # 1. Exact or normalized match in Drug_database
        exact = drug_repo.find_exact(cleaned) or drug_repo.find_normalized(cleaned)
        if exact:
            return cleaned, 0.95, None, exact.drug_id, []

        # Also check concatenated multi-word token (e.g. "parasita mall" -> "parasitamall")
        if " " in cleaned:
            concat = re.sub(r"\s+", "", cleaned)
            exact_c = drug_repo.find_exact(concat) or drug_repo.find_normalized(concat)
            if exact_c:
                return cleaned, 0.95, None, exact_c.drug_id, []

        # 2. Phonetic sound-alike / Did-you-mean match (e.g. "grocin" -> "Crocin", "parasita mall" -> "Paracetamol")
        top_dym = drug_repo.find_top_did_you_mean(cleaned, limit=3, min_confidence=0.55)
        if top_dym:
            best = top_dym[0]
            did_you_mean = best["drug_name"]
            return cleaned, best["confidence"], did_you_mean, best["drug_id"], top_dym

        # Check first token for brand formulations with trailing descriptors
        tokens = cleaned.split()
        if len(tokens) >= 2:
            first_t = tokens[0]
            if len(first_t) >= 4 and is_valid_medication_name(first_t):
                exact_t = drug_repo.find_exact(first_t) or drug_repo.find_normalized(first_t)
                if exact_t:
                    return cleaned, 0.90, None, exact_t.drug_id, []
                top_dym_t = drug_repo.find_top_did_you_mean(first_t, limit=3, min_confidence=0.60)
                if top_dym_t:
                    best_t = top_dym_t[0]
                    return cleaned, best_t["confidence"], best_t["drug_name"], best_t["drug_id"], top_dym_t

        return None

    # Step 1: Check seed_drug_name if provided by clause segmenter
    if seed_drug_name:
        res = _test_grounding(seed_drug_name)
        if res:
            name, conf, dym, did, dym_opts = res
            idx = clause_text.lower().find(seed_drug_name.lower())
            span = (idx, idx + len(seed_drug_name)) if idx >= 0 else (0, len(seed_drug_name))
            return name, span, conf, dym, did, dym_opts

    # Step 2: Extract candidate preceding any dosage pattern (with units or unitless formulation numbers)
    dosage_pattern = (
        r"\b\d+(?:\.\d+)?\s*(?:mg(?:\/ml|\/g)?|grams?|gm|g|mcg|µg|ml|l|iu|units?|%|meq|puffs?|drops?|tablets?|capsules?|sachets?|vials?)(?!\w)"
        r"|\b(?<!\w)(?!(?:101|111|100|010|001|110|011|1111)\b)\d{2,4}\b"
    )
    dose_match = re.search(dosage_pattern, clause_text, re.IGNORECASE)
    if dose_match:
        lead_text = clause_text[:dose_match.start()].strip()
        res = _test_grounding(lead_text)
        if res:
            name, conf, dym, did, dym_opts = res
            idx = clause_text.find(lead_text)
            span = (idx, idx + len(lead_text)) if idx >= 0 else (0, len(lead_text))
            return name, span, conf, dym, did, dym_opts

        cleaned_lead = clean_candidate_name(lead_text)
        if cleaned_lead and is_valid_medication_name(cleaned_lead) and len(cleaned_lead) >= 3:
            idx = clause_text.find(cleaned_lead)
            span = (idx, idx + len(cleaned_lead)) if idx >= 0 else (0, len(cleaned_lead))
            return "Medicine not found", span, 0.20, None, None, []

    # Step 3: Extract candidate from administration verbs
    nodose_match = re.search(
        r"(?:take|administer|give|consume|dissolve|inhale|apply|put|instill|gently\s+massage|massage|cleanse)?\s*"
        r"(?:one|two|three)?\s*(?:tablet|tab|capsule|cap|rotacap|pill|vial|sachet|puff)?\s*(?:of\s+)?"
        r"([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,3}?)\s*"
        r"(?:tablet|capsule|oral\s+suspension|suspension|sachet|vial|cream|ointment|gel|drops|rotacap|puff|paste|wash|orally|topically|by\s+mouth|before|after|twice|once|three|up\s+to|in\s+one\s+liter|every|onto|along|to\s+the|over|for|till|upto|until|daily|SOS|HS|STAT)\b",
        clause_text,
        re.IGNORECASE,
    )
    if nodose_match:
        raw = nodose_match.group(1).strip()
        res = _test_grounding(raw)
        if res:
            name, conf, dym, did, dym_opts = res
            idx = clause_text.find(raw)
            span = (idx, idx + len(raw)) if idx >= 0 else (nodose_match.start(1), nodose_match.end(1))
            return name, span, conf, dym, did, dym_opts

    # Step 4: Multi-token n-gram search across the clause
    words = re.findall(r"[A-Za-z0-9\-]{3,}", clause_text)
    # Check 2-word combinations first (e.g. "parasita mall")
    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        res = _test_grounding(two_word)
        if res:
            name, conf, dym, did, dym_opts = res
            idx = clause_text.lower().find(two_word.lower())
            span = (idx, idx + len(two_word)) if idx >= 0 else (0, len(two_word))
            return name, span, conf, dym, did, dym_opts

    # Check individual words
    for w in words:
        res = _test_grounding(w)
        if res:
            name, conf, dym, did, dym_opts = res
            idx = clause_text.lower().find(w.lower())
            span = (idx, idx + len(w)) if idx >= 0 else (0, len(w))
            return name, span, conf, dym, did, dym_opts
    # Fallback: if a candidate string preceded dosage/administration but could not be grounded
    if dose_match:
        lead_text = clause_text[:dose_match.start()].strip()
        cleaned_lead = clean_candidate_name(lead_text)
        if cleaned_lead and is_valid_medication_name(cleaned_lead) and len(cleaned_lead) >= 3:
            idx = clause_text.find(cleaned_lead)
            span = (idx, idx + len(cleaned_lead)) if idx >= 0 else (0, len(cleaned_lead))
            return "Medicine not found", span, 0.20, None, None, []

    if seed_drug_name:
        cleaned_seed = clean_candidate_name(seed_drug_name)
        if cleaned_seed and is_valid_medication_name(cleaned_seed) and len(cleaned_seed) >= 3:
            idx = clause_text.lower().find(seed_drug_name.lower())
            span = (idx, idx + len(seed_drug_name)) if idx >= 0 else (0, len(seed_drug_name))
            return "Medicine not found", span, 0.20, None, None, []

    return None, None, 0.0, None, None, []
