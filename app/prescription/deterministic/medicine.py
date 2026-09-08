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
) -> Tuple[Optional[str], Optional[Tuple[int, int]], float]:
    """
    Deterministically extracts the candidate medicine name and its character span within the clause.
    Returns (candidate_name, (start_idx, end_idx), confidence).
    """
    if seed_drug_name:
        cleaned = clean_candidate_name(seed_drug_name)
        if is_valid_medication_name(cleaned):
            idx = clause_text.lower().find(cleaned.lower())
            span = (idx, idx + len(cleaned)) if idx >= 0 else (0, len(cleaned))
            return cleaned, span, 0.95

    # Look for drug name preceding any dosage pattern (supports grams, gm, mg, mcg, etc.)
    dosage_pattern = (
        r"\b\d+(?:\.\d+)?\s*(?:mg(?:\/ml|\/g)?|grams?|gm|g|mcg|µg|ml|l|iu|units?|%|meq|puffs?|drops?|tablets?|capsules?|sachets?|vials?)(?!\w)"
    )
    dose_match = re.search(dosage_pattern, clause_text, re.IGNORECASE)

    if dose_match:
        lead_text = clause_text[:dose_match.start()].strip()
        cleaned = clean_candidate_name(lead_text)
        if cleaned and is_valid_medication_name(cleaned):
            idx = clause_text.find(cleaned)
            span = (idx, idx + len(cleaned)) if idx >= 0 else (0, len(cleaned))
            return cleaned, span, 0.90

    # Look for drug name without dose
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
        cleaned = clean_candidate_name(raw)
        if cleaned and len(cleaned) >= 3 and is_valid_medication_name(cleaned):
            idx = clause_text.find(cleaned)
            span = (idx, idx + len(cleaned)) if idx >= 0 else (nodose_match.start(1), nodose_match.end(1))
            return cleaned, span, 0.85

    # Catalog dictionary search if repo available
    if drug_repo:
        words = re.findall(r"[A-Za-z0-9\-]{3,}", clause_text)
        for w in words:
            if is_valid_medication_name(w):
                results = drug_repo.search(w)
                if results:
                    matched_base = results[0]["base_name"]
                    idx = clause_text.lower().find(w.lower())
                    span = (idx, idx + len(w)) if idx >= 0 else (0, len(w))
                    return matched_base, span, 0.80

    return None, None, 0.0
