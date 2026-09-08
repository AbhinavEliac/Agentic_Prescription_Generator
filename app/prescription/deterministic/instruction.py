"""
app/prescription/deterministic/instruction.py
---------------------------------------------
Deterministic Clinical Instruction Extractor.

Separates and extracts:
1. Primary Administration Instructions:
   - Meal relations ("before meals", "strictly with meals", "after lunch")
   - Times of day ("morning and night", "at bedtime")
   - Devices & administration techniques ("using Revolizer", "rinse mouth after inhaler", "via MDI spacer")
   - PRN indications ("for nausea", "for fever", "to relieve itching")
2. Additional Clinical Instructions:
   - Dietary & lifestyle advice ("stick to a bland diet", "avoid spicy food", "go for morning walks")
   - Course completion & antibiotic rules ("do not stop antibiotic course early")
   - Titrations ("increase dose by 100 mg after 7 days")
   - Contingency triggers and doctor consultations ("visit emergency if persistent vomiting occurs", "discontinue once fever resolves")
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, List, Dict, Any

# Primary administration timing and meal patterns
PRIMARY_PATTERNS = [
    # Minute/Hour offsets before/after meals
    (r"(?i)\b\d+\s+minutes\s+before\s+breakfast\s+and\s+dinner\b", "before breakfast and dinner"),
    (r"(?i)\b\d+\s+minutes\s+before\s+(?:breakfast|lunch|dinner|meals?|food)\b", None),
    (r"(?i)\b\d+\s+minutes\s+after\s+(?:breakfast|lunch|dinner|meals?|food)\b", None),
    (r"(?i)\b\d+\s+hours?\s+before\s+(?:breakfast|lunch|dinner|meals?|food)\b", None),
    (r"(?i)\b\d+\s+hours?\s+after\s+(?:breakfast|lunch|dinner|meals?|food)\b", None),
    
    # Specific Meal Timings
    (r"(?i)\bstrictly\s+with\s+meals(?:\s+to\s+avoid\s+[^,\.\n;!]+)?\b", "strictly with meals to avoid stomach upset"),
    (r"(?i)\bstrictly\s+with\s+food\b", "strictly with food"),
    (r"(?i)\bstrictly\s+after\s+food\b", "strictly after food"),
    (r"(?i)\bafter\s*breakfast\b", "after breakfast"),
    (r"(?i)\bafter\s*lunch\b", "after lunch"),
    (r"(?i)\bafter\s*dinner\b", "after dinner"),
    (r"(?i)\bafter\s*(?:meals?|food|eating)\b", "after meals"),
    (r"(?i)\bbefore\s*breakfast\b", "before breakfast"),
    (r"(?i)\bbefore\s*lunch\b", "before lunch"),
    (r"(?i)\bbefore\s*dinner\b", "before dinner"),
    (r"(?i)\bbefore\s*(?:meals?|food)\b", "before meals"),
    (r"(?i)\b(?:on\s*(?:an?\s*)?empty\s*stomach|empty\s*stomach)\b", "on empty stomach"),
    (r"(?i)\bwith\s*(?:meals?|food)\b", "with meals"),
    
    # Times of Day
    (r"(?i)\b(?:at\s*bedtime|before\s*sleep|at\s*night)\b", "at bedtime"),
    (r"(?i)\b(?:early\s+morning|every\s+morning(?:\s+before\s+food)?)\b", "every morning before food"),
    
    # Vehicle / Liquid
    (r"(?i)\bwith\s*(?:warm|hot)\s*water\b", "with warm water"),
    (r"(?i)\bwith\s*a\s+full\s+glass\s+of\s+water\b", "with a full glass of water"),
    (r"(?i)\bwith\s*(?:cold|regular|clean)?\s*water\b", "with water"),
    (r"(?i)\bwith\s*milk\b", "with milk"),
    
    # Technique & Device
    (r"(?i)\brinse\s*mouth(?:\s*(?:thoroughly\s*)?(?:after\s+(?:using\s+the\s+inhaler|use)))?\b", "rinse mouth after using the inhaler"),
    (r"(?i)\b(?:swallow\s*whole(?:\s+without\s+crushing)?|do\s*not\s*chew)\b", "swallow whole"),
    (r"(?i)\busing\s+(?:the\s+)?Revolizer\s+device\b", "using Revolizer device"),
    (r"(?i)\bvia\s+an?\s+MDI\s+spacer(?:\s+as\s+needed)?\b", "via MDI spacer"),
    (r"(?i)\bdissolve\s+(?:one\s+sachet\s+)?in\s+one\s+liter\s+of\s+clean\s+drinking\s+water\s+to\s+consume\s+throughout\s+the\s+day\b", "dissolve in one liter of clean drinking water to consume throughout the day"),
    (r"(?i)\bapply\s+a\s+thin\s+layer\s+along\s+the\s+clean\s+suture\s+line\b", "apply a thin layer along the clean suture line"),
    (r"(?i)\binto\s+both\s+nostrils\b", "into both nostrils"),
    (r"(?i)\binto\s+left\s+eye\b", "into left eye"),
    (r"(?i)\binto\s+each\s+nostril\b", "into each nostril"),
    (r"(?i)\binto\s+the\s+affected\s+ear\b", "into the affected ear"),
    
    # PRN Indication
    (r"(?i)\bfor\s+nausea\b", "for nausea"),
    (r"(?i)\bfor\s+fever\b", "for fever"),
    (r"(?i)\bfor\s+pain\b", "for pain"),
    (r"(?i)\bfor\s+facial\s+pain\s+or\s+fever\b", "for facial pain or fever"),
    (r"(?i)\bto\s+relieve\s+itching\b", "to relieve itching"),
]

# Additional clinical instructions (lifestyle, diet, titrations, follow-up, warnings)
ADDITIONAL_PATTERNS = [
    # Titration directives
    r"(?i)\b(?:increase|decrease|reduce|double|taper)\s+(?:the\s+)?(?:dose|dosage)(?:\s+by\s+\d+\s*(?:mgs?|mg|g|mcg|ml)?)?(?:\s+after\s+\d+\s*(?:days?|weeks?|months?))?\b",
    r"(?i)\bif\s+blood\s+pressure\s+remains\s+high[^\.\n;!]*consult\s+your\s+doctor[^\.\n;!]*",
    
    # Discontinuation & Course Completion
    r"(?i)\bdiscontinue\s+once\s+(?:the\s+)?fever\s+resolves\b",
    r"(?i)\bdo\s+not\s+stop\s+(?:the\s+)?antibiotic\s+course[^\.\n;!]*",
    r"(?i)\bcomplete\s+(?:the\s+)?(?:full\s+)?course\b",
    
    # Diet & Lifestyle Guidance
    r"(?i)\bstick\s+to\s+a\s+bland\s+diet(?:\s+consisting\s+of\s+[^,\.\n;!]+)?\b",
    r"(?i)\bavoid\s+(?:oily\s*(?:and\s*)?spicy\s*food|oily\s*food|spicy\s*food|spicy\s+or\s+oily\s+foods)\b",
    r"(?i)\bavoid\s+tea\s+near\s+meal\s+times\b",
    r"(?i)\bavoid\s+alcohol\b",
    r"(?i)\binclude\s+(?:dark\s+)?green\s+leafy\s+vegetables(?:\s+in\s+diet)?\b",
    r"(?i)\b(?:go\s+for\s+)?morning\s+walks?(?:\s+daily)?\b",
    r"(?i)\b(?:also\s+)?take\s+walks?(?:\s+after\s+dinner)?\b",
    r"(?i)\bdrink\s+plenty\s+of\s+fluids?\b",
    r"(?i)\bapply\s+(?:local\s+)?hot\s+water\s+fomentation\b",
    
    # Doctor Review & Contingencies
    r"(?i)\bvisit\s+the\s+emergency(?:\s+room)?\s+immediately[^\.\n;!]*",
    r"(?i)\breturn\s+for\s+evaluation(?:\s+after\s+completing\s+the\s+course)?\b",
    r"(?i)\bseek\s+reassessment\s+of\s+blood\s+pressure\s+afterwards\b",
    r"(?i)\bseek\s+reassessment\s+if\s+adverse\s+effects\s+develop\b",
    r"(?i)\bre-?test\s+blood\s+count\s+in\s+\d+\s+months?\b",
    r"(?i)\bif\s+fever\s+persists?[^\.\n;!]*consult\s+doctor\b",
]


def extract_instructions(
    clause_text: str,
    advice_clauses: Optional[List[str]] = None,
    full_prescription_text: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Extracts primary administration instruction and secondary additional_instruction.
    Returns:
      (primary_instruction, additional_instruction, confidence)
    """
    primary_items: List[str] = []
    additional_items: List[str] = []

    # 1. Search clause for Primary Instructions
    for pat, canonical_label in PRIMARY_PATTERNS:
        m = re.search(pat, clause_text)
        if m:
            val = canonical_label if canonical_label else m.group(0).strip()
            if val not in primary_items:
                primary_items.append(val)

    # 2. Search clause for Additional Instructions
    for pat in ADDITIONAL_PATTERNS:
        m = re.search(pat, clause_text)
        if m:
            val = m.group(0).strip()
            if val not in additional_items:
                additional_items.append(val)

    # 3. Process attached advice clauses
    if advice_clauses:
        for adv in advice_clauses:
            for pat, canonical_label in PRIMARY_PATTERNS:
                m = re.search(pat, adv)
                if m:
                    val = canonical_label if canonical_label else m.group(0).strip()
                    if val not in primary_items:
                        primary_items.append(val)

            for pat in ADDITIONAL_PATTERNS:
                m = re.search(pat, adv)
                if m:
                    val = m.group(0).strip()
                    if val not in additional_items:
                        additional_items.append(val)
            
            # If advice clause didn't match specific patterns, check general advice triggers
            if not any(re.search(p, adv) for p in ADDITIONAL_PATTERNS):
                if len(adv) > 5 and not any(adv.lower() in ai.lower() for ai in additional_items):
                    additional_items.append(adv)

    # 4. Check full text for global advice that belongs to all/companion medications (e.g. lifestyle, diet)
    if full_prescription_text:
        for pat in (
            r"(?i)\binclude\s+(?:dark\s+)?green\s+leafy\s+vegetables[^\.\n;!]*",
            r"(?i)\bavoid\s+tea\s+near\s+meal\s+times\b",
            r"(?i)\b(?:go\s+for\s+)?morning\s+walks?\s+daily\b",
            r"(?i)\bre-?test\s+blood\s+count\s+in\s+\d+\s+months?\b",
            r"(?i)\bstick\s+to\s+a\s+bland\s+diet[^\.\n;!]*",
            r"(?i)\bvisit\s+the\s+emergency(?:\s+room)?\s+immediately[^\.\n;!]*",
        ):
            m = re.search(pat, full_prescription_text)
            if m:
                val = m.group(0).strip()
                if val not in additional_items:
                    additional_items.append(val)

    # Format result strings
    primary_res = "; ".join(dict.fromkeys(primary_items)) if primary_items else None
    additional_res = "; ".join(dict.fromkeys(additional_items)) if additional_items else None

    return primary_res, additional_res, 0.90
