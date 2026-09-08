"""
app/prescription/deterministic/duration.py
-----------------------------------------
Deterministic Prescription Duration Extractor.

Extracts duration spans:
- Numeric day/week/month/year periods ("for 5 days", "till 7 days", "for 2 weeks", "in 2 months")
- Single-day or start-day directives ("on day one", "single day")
- Plural broadcast durations across companion drugs
"""

from __future__ import annotations
import re
from typing import Optional, Tuple

DURATION_PATTERNS = [
    # Explicit prefix patterns: "for 5 days", "for 5 day", "till 7 days", "in 2 months"
    r"(?i)\b(?:for\s+|duration\s+of\s+|till\s+|until\s+|upto\s+|up\s+to\s+|for\s+upto\s+|for\s+up\s+to\s+|for\s+next\s+|next\s+|about\s+|around\s+|approx(?:\s+)?|for\s+around\s+|for\s+about\s+|x\s*|in\s+)(\d+\s*(?:days?|d|weeks?|wks?|months?|mo|years?|hrs?|hours?)|day\s+one|single\s+day|\d+\s*to\s*\d+\s*(?:days?|weeks?|months?)|no\s+more\s+than\s+\d+\s+days?)\b",
    # Specific single day indicator
    r"(?i)\b(?:on\s+)?(day\s+one|single\s+day)\b",
    # Until event
    r"(?i)\b(until\s+[a-zA-Z\s]{3,30})\b",
    # Naked duration
    r"(?i)\b(\d+\s*(?:days?|weeks?|months?))\b",
]


def extract_duration(
    clause_text: str,
    full_prescription_text: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Tuple[int, int]], float]:
    """
    Extracts duration span from clause or global broadcast context.
    Returns:
      (duration_string, (start_idx, end_idx), confidence)
    """
    # 1. Search in the clause
    for pat in DURATION_PATTERNS:
        m = re.search(pat, clause_text)
        if m:
            dur = m.group(1).strip()
            # Normalize single day/grammar if necessary (e.g. "5 day" -> "5 days")
            day_fix = re.match(r"^(\d+)\s+day$", dur, re.IGNORECASE)
            if day_fix and int(day_fix.group(1)) > 1:
                dur = f"{day_fix.group(1)} days"
            return dur, (m.start(1), m.end(1)), 0.95

    # 2. Check plural broadcast duration in full prescription text
    if full_prescription_text:
        plural_match = re.search(
            r"(?i)\b(?:both|all|each)\s+(?:of\s+them\s+)?(?:for\s+|till\s+|upto\s+)(\d+\s*(?:days?|weeks?|months?))\b",
            full_prescription_text,
        )
        if plural_match:
            dur = plural_match.group(1).strip()
            return dur, None, 0.85

    return None, None, 0.0
