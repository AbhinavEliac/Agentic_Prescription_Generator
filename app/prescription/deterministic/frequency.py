"""
app/prescription/deterministic/frequency.py
-------------------------------------------
Deterministic Administration Frequency Extractor.

Consolidates:
1. Doctor speaking habit numeric schedules (101, 1-0-1, 111, 100, 010, 001, 110, 1111)
2. Latin and international medical codes (OD, BD, BID, TID, QID, PRN, SOS, STAT, HS)
3. Natural language intervals ("every 4 hours", "every morning", "twice daily Morning Night")
4. Global plural broadcast coreference ("Both medicines should be taken twice daily")
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, List

FREQUENCY_PATTERNS = [
    # Ambiguous frequency patterns requiring clinical review
    r"(?i)\b(?:once|twice|thrice|\d+\s+times)\s+or\s+(?:once|twice|thrice|\d+\s+times)\s+(?:daily|a\s+day)\b",
    r"(?i)\bas\s+needed\s+but\s+not\s+more\s+than\s+(?:once|twice|thrice|\w+)(?:\s+times)?\s+daily\b",
    r"(?i)\bnot\s+more\s+than\s+(?:once|twice|thrice|\w+)(?:\s+times)?\s+daily\b",
    # Up to N times daily as needed
    r"(?i)\bup\s+to\s+\w+\s+times\s+daily(?:\s+as\s+needed)?\b",
    # Specific doctor schedule patterns with preceding descriptor: e.g. "twice daily(1-0-1)", "once daily(0-0-1)"
    r"(?i)\b(?:twice|once|thrice|three\s+times|four\s+times|\d+\s+times)\s+(?:daily|a\s+day)\s*(?:\([0-9\-\/]+\))",
    # Hour intervals
    r"(?i)\bevery\s+\d+(?:\s+to\s+\d+)?\s*(?:hours?|hrs?|days?|weeks?|months?)\b",
    # Time of day / daily intervals
    r"(?i)\bevery\s+(?:morning|night|evening|afternoon|bedtime|other\s+day)\b",
    # Natural combined times of day: e.g. "twice daily Morning Night", "twice daily morning and evening"
    r"(?i)\b(?:twice|thrice|three\s+times)\s+daily\s+(?:Morning\s+Night|Morning\s+and\s+Night|morning\s+evening)\b",
    # Numeric schedules
    r"\b\d+-\d+-\d+(?:-\d+)?\b",
    r"\b(?:101|111|100|010|001|110|011|1111)\b(?!\s*(?:mg|g|mcg|µg|ml|l|iu|units?|%|tablets?|capsules?|pills?|goli))",
    # Standard times per day / week
    r"(?i)\b(?:twice|once|thrice|three\s+times|four\s+times|\d+\s+times)\s+(?:daily|a\s+day|per\s+day|a\s+week|weekly|a\s+month|monthly)\b",
    r"(?i)\b(?:once|twice|thrice)\s+weekly\b",
    r"(?i)\bthree\s+times\s+daily\b",
    r"(?i)\bfour\s+times\s+daily\b",
    r"(?i)\btwice\s+daily\b",
    r"(?i)\bonce\s+daily\b",
    r"(?i)\bthrice\s+daily\b",
    r"(?i)\bthroughout\s+the\s+day\b",
    r"(?i)\bsingle\s+dose\b",
    r"(?i)\bstat\s+dose\b",
    r"(?i)\bas\s+needed\s*\([A-Za-z]+\)\b",
    r"(?i)\bas\s+needed(?:\s+for\s+[a-zA-Z\s]+)?\b",
    # Specific time of day directives (standalone frequency fallback)
    r"(?i)\b(?:in\s+the\s+)?(?:morning|evening|afternoon)\b",
    r"(?i)\bearly\s+morning\b",
    r"(?i)\bat\s+bedtime\b",
    r"(?i)\bdaily\b",
    r"(?i)\b(?:OD|BD|TID|QID|QDS|TDS|BID|PRN|SOS|HS|STAT|q\d+h|q\d+-\d+h)\b",
]

NUMERIC_SCHEDULE_NORMALIZATION = {
    "101": "twice daily(1-0-1)",
    "1-0-1": "twice daily(1-0-1)",
    "111": "thrice daily(1-1-1)",
    "1-1-1": "thrice daily(1-1-1)",
    "100": "once daily(1-0-0)",
    "1-0-0": "once daily(1-0-0)",
    "010": "once daily(0-1-0)",
    "0-1-0": "once daily(0-1-0)",
    "001": "once daily(0-0-1)",
    "0-0-1": "once daily(0-0-1)",
    "110": "twice daily(1-1-0)",
    "1-1-0": "twice daily(1-1-0)",
    "011": "twice daily(0-1-1)",
    "0-1-1": "twice daily(0-1-1)",
    "1111": "four times daily(1-1-1-1)",
    "1-1-1-1": "four times daily(1-1-1-1)",
}


def extract_frequency(
    clause_text: str,
    full_prescription_text: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Tuple[int, int]], float]:
    """
    Extracts dosage administration frequency from clause or via cross-sentence coreference.
    Returns:
      (frequency_string, (start_idx, end_idx), confidence)
    """
    # 1. Search in first 100 characters of the clause first to avoid matching trailing advice
    core_clause = clause_text[:120]
    for pat in FREQUENCY_PATTERNS:
        m = re.search(pat, core_clause)
        if m:
            raw_freq = m.group(0).strip()
            # Clean off any meal words trailing the frequency match
            clean_freq = re.sub(r"(?i)\s+(?:before|after)\s+(?:food|breakfast|meals|lunch|dinner)$", "", raw_freq).strip()
            # Standardize numeric schedule if standalone
            if clean_freq in NUMERIC_SCHEDULE_NORMALIZATION:
                clean_freq = NUMERIC_SCHEDULE_NORMALIZATION[clean_freq]
            return clean_freq, (m.start(), m.start() + len(clean_freq)), 0.95

    # 2. Search anywhere in the clause
    for pat in FREQUENCY_PATTERNS:
        m = re.search(pat, clause_text)
        if m:
            raw_freq = m.group(0).strip()
            clean_freq = re.sub(r"(?i)\s+(?:before|after)\s+(?:food|breakfast|meals|lunch|dinner)$", "", raw_freq).strip()
            if clean_freq in NUMERIC_SCHEDULE_NORMALIZATION:
                clean_freq = NUMERIC_SCHEDULE_NORMALIZATION[clean_freq]
            return clean_freq, (m.start(), m.start() + len(clean_freq)), 0.90

    # 3. Check broadcast/plural coreference if full prescription text provided
    if full_prescription_text:
        plural_match = re.search(
            r"(?i)\b(?:both(?:\s+of\s+them|\s+medicines|\s+drugs|\s+tablets|\s+capsules)?|"
            r"all(?:\s+these|\s+of\s+them)?(?:\s+medicines|\s+drugs|\s+tablets)?|each(?:\s+of\s+them)?)\s+"
            r"(?:should\s+be\s+taken|are\s+to\s+be\s+taken|must\s+be\s+taken|to\s+be\s+taken|should\s+be\s+given|should\s+be|are|must\s+be)\s+([^,\.\n;!]+)",
            full_prescription_text,
        )
        if plural_match:
            cand = plural_match.group(1).strip()
            for pat in FREQUENCY_PATTERNS:
                fm = re.search(pat, cand)
                if fm:
                    raw_freq = fm.group(0).strip()
                    clean_freq = re.sub(r"(?i)\s+(?:before|after)\s+(?:food|breakfast|meals|lunch|dinner)$", "", raw_freq).strip()
                    if clean_freq in NUMERIC_SCHEDULE_NORMALIZATION:
                        clean_freq = NUMERIC_SCHEDULE_NORMALIZATION[clean_freq]
                    return clean_freq, None, 0.85

        # Check singular coreference: e.g. "Take this medicine twice daily"
        singular_match = re.search(
            r"(?i)\b(?:it\s+(?:should\s+be\s+taken|is\s+to\s+be\s+taken|must\s+be\s+taken|to\s+be\s+taken|is\s+taken)|take\s+(?:it|this(?:\s+medicine)?))\s+([^,\.\n;!]+)",
            full_prescription_text,
        )
        if singular_match:
            cand = singular_match.group(1).strip()
            for pat in FREQUENCY_PATTERNS:
                fm = re.search(pat, cand)
                if fm:
                    raw_freq = fm.group(0).strip()
                    clean_freq = re.sub(r"(?i)\s+(?:before|after)\s+(?:food|breakfast|meals|lunch|dinner)$", "", raw_freq).strip()
                    if clean_freq in NUMERIC_SCHEDULE_NORMALIZATION:
                        clean_freq = NUMERIC_SCHEDULE_NORMALIZATION[clean_freq]
                    return clean_freq, None, 0.80

    return None, None, 0.0
