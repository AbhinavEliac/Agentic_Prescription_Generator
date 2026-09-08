"""
app/prescription/segmenter.py
-----------------------------
Canonical Prescription Clause Segmenter.

Consolidates multi-medicine partitioning rules from Fast Mode, LangGraph, and Node flowsheet:
1. Punctuation and transition-based boundary identification
2. Companion medication handling ("combination tablet of A and B", "A with B")
3. Independent clinical advice and lifestyle clause routing to parent medications
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PrescriptionClause(BaseModel):
    """Represents a discrete medication order clause within a prescription."""
    clause_id: int
    text: str
    seed_drug_name: Optional[str] = None
    advice_clauses: List[str] = Field(default_factory=list)
    start_char: int = 0
    end_char: int = 0


# Non-drug trigger keywords that indicate a clause is advice, caution, or lifestyle instruction
PURE_ADVICE_TRIGGERS = [
    r"^(?:take\s+all(?:\s+the)?\s+medicines|take\s+both(?:\s+of\s+them)?)\b",
    r"^(?:return|come\s+back|follow\s+up|review|get\s+reassessed|seek|consult|contact|report|visit|revisit|arrange|schedule|repeat|re-?test)\b",
    r"^(?:meet\s+(?:the\s+)?doctor|please\s+see\s+me|see\s+(?:your\s+|the\s+)?doctor|come\s+(?:and\s+)?visit\s+(?:the\s+|your\s+)?doctor|visit\s+(?:the\s+|your\s+)?doctor)\b",
    r"^(?:do\s+not|avoid|strictly\s+avoid|discontinue|stop|keep|maintain|stick\s+to|include|limit|practice|brush|perform)\b",
    r"^(?:drink|stay\s+well-hydrated|apply\s+local|apply\s+ice|sponge|wear|monitor|be\s+sure\s+to|if\s+)\b",
    r"^(?:consume\s+(?:plenty|clean|regular|warm|fluids?|water|food|meals?))\b",
    r"^(?:take\s+(?:regular\s+|warm\s+|cold\s+|plenty\s+of\s+|sufficient\s+|clean\s+)?(?:water|fluids?))\b",
    r"(?:also\s+)?take\s+walks?",
    r"go\s+for\s+(?:morning\s+)?walks?",
    r"(?:apply\s+)?(?:local\s+)?hot\s+water\s+fomentation",
    r"(?:apply\s+)?ice\s+packs?",
    r"sponge\s+(?:the\s+)?(?:body|forehead)",
    r"keep\s+(?:the\s+)?(?:blistered|affected|skin|dressing|ear)\s+area\s+clean\s+and\s+dry",
    r"avoid\s+close\s+physical\s+contact",
]

# Words that follow "with" which are food/drink/instructions, NOT companion drugs
NON_DRUG_WITH = {
    "food", "meals", "meal", "water", "milk", "breakfast", "lunch", "dinner", "juice",
    "caution", "care", "warm", "cold", "tap", "equal", "a", "an", "the", "endoscopy", "report",
    "doctor", "review", "evaluation", "prescription", "consultation", "food", "stomach", "walks",
    "pregnant", "individuals", "persons", "children", "people"
}

DRUG_CLASS_MAP: Dict[str, List[str]] = {
    "antibiotic": ["amoxicillin", "augmentin", "azithromycin", "ciprofloxacin", "ofloxacin", "cefixime", "doxycycline", "clavulanate", "penicillin", "phexin", "clarithromycin", "nitrofurantoin", "cefuroxime", "cephalexin", "cefpodoxime"],
    "iron": ["ferrous", "iron", "folic", "autrin", "orofer"],
    "inhaler": ["budecort", "foracort", "asthalin", "seretide", "tiova", "fluticasone", "formoterol", "salbutamol", "rotacap", "budesonide", "tiotropium", "levosalbutamol"],
    "painkiller": ["paracetamol", "crocin", "ibuprofen", "combiflam", "tramadol", "aceclofenac", "diclofenac", "naproxen", "etoricoxib"],
    "antacid": ["pantop", "pan", "omeprazole", "rabeprazole", "esomeprazole", "gelusil", "digene", "sucralfate", "pantoprazole"],
}

SPLIT_TRANSITION_PATTERN = (
    r"(?i)(?:,\s*(?:and\s+|also\s+|additionally\s+|then\s+)?(?:take(?!\s+(?:walks?|a\s+walk|rest|care|steam|bath|water|fluids?|food|meals?|all\s+the\s+medicines))|start(?!\s+(?:eating|diet|walks?|exercis|hydrat))|administer|give|consume|dissolve|inhale|apply|gently\s+massage|massage|rub|spray|instill|cleanse|put|inject|infuse)\b|"
    r"\band\s+take\s+\d+\b|\band\s+take\s+(?!(?:walks?|a\s+walk|rest|care|steam|bath|water|regular\s+water|warm\s+water|cold\s+water|fluids?|food|meals?|all\s+the\s+medicines))\w+|"
    r"\balso\s+take\s+(?!(?:walks?|a\s+walk|rest|care|steam|bath|water|regular\s+water|warm\s+water|cold\s+water|fluids?|food|meals?))\w+|"
    r"\bthen\s+take\b|"
    r"\band\s+start(?!\s+(?:eating|diet|walks?|exercis|hydrat))\b|\band\s+administer\b|\band\s+consume\b|\band\s+dissolve\b|\band\s+spray\b|\band\s+instill\b|\binhale\s+one\b|\binhale\s+two\b|\bapply\s+one\b|"
    r"\bkeep\s+the\s+blistered\s+area\b|\bavoid\s+close\s+physical\s+contact\b|\breturn\s+if\s+the\s+rash\b)"
)


def route_advice_clause(advice_text: str, current_clauses: List[PrescriptionClause]) -> None:
    """Routes an advice or lifestyle clause to its clinically appropriate parent medication."""
    if not current_clauses:
        return
    adv_lower = advice_text.lower()

    # 1. Plural / Broadcast advice (e.g. "Both should be taken twice daily after meals", "Take all the medicines in liquid form")
    is_broadcast = bool(re.search(r"(?i)\b(?:both(?:\s+of\s+them|\s+medicines|\s+drugs|\s+tablets|\s+capsules)?|all(?:\s+the)?\s+medicines|all(?:\s+these|\s+of\s+them)?(?:\s+medicines|\s+drugs|\s+tablets)?|each(?:\s+of\s+them)?)\b", advice_text))
    if is_broadcast:
        for cl in current_clauses:
            cl.advice_clauses.append(advice_text)
        return

    # 2. Drug Class Routing (e.g. "Do not stop the antibiotic course early even if fever subsides")
    for class_key, class_drugs in DRUG_CLASS_MAP.items():
        if re.search(rf"\b{re.escape(class_key)}\b", adv_lower):
            for cl in current_clauses:
                cl_lower = cl.text.lower()
                if any(d in cl_lower for d in class_drugs):
                    cl.advice_clauses.append(advice_text)
                    return

    # 3. Direct Drug Name Mention in Advice
    for cl in current_clauses:
        if cl.seed_drug_name and cl.seed_drug_name.lower() in adv_lower:
            cl.advice_clauses.append(advice_text)
            return

    # 4. Fallback: attach to preceding clause
    current_clauses[-1].advice_clauses.append(advice_text)


def is_pure_advice_clause(clause: str) -> bool:
    """Checks if a clause represents standalone clinical, lifestyle, or contingency advice."""
    c = clause.strip()
    return any(re.search(pat, c, re.IGNORECASE) for pat in PURE_ADVICE_TRIGGERS)


def segment_prescription_clauses(text: str) -> List[PrescriptionClause]:
    """
    Segments a normalized prescription string into discrete medication clauses.
    Preserves companion medications and groups advice clauses to appropriate medication parents.
    """
    if not text or not text.strip():
        return []

    # Adjust spoken period transitions before common frequency directives
    clean_text = re.sub(
        r"\.\s*((?:Once|Twice|Thrice|\d+\s+times|Every|Daily|At\s+bedtime|In\s+the\s+morning)[^\.,;]+?),\s*(take|administer|give|start|apply|inhale|instill)",
        r" \1. \2",
        text,
        flags=re.IGNORECASE,
    )

    # 1. Split on sentence boundaries, newlines, and semicolons
    raw_sentences = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)|[\n;!]", clean_text) if s.strip()]

    # 2. Further split on action verb transitions
    raw_clauses: List[str] = []
    for sent in raw_sentences:
        sent_clean = re.sub(r"^(?:additionally|also)\s*,\s*", "", sent, flags=re.IGNORECASE)
        sub = [c.strip() for c in re.split(SPLIT_TRANSITION_PATTERN, sent_clean) if c.strip()]
        raw_clauses.extend(sub if sub else [sent_clean])

    if not raw_clauses:
        raw_clauses = [clean_text]

    # 3. Partition into medication clauses vs advice clauses
    clauses: List[PrescriptionClause] = []
    clause_counter = 1

    for item in raw_clauses:
        c_clean = item.strip()
        if not c_clean:
            continue

        if is_pure_advice_clause(c_clean):
            route_advice_clause(c_clean, clauses)
            continue

        # Indicator of medication clause: contains action verbs, forms, or dosage units
        has_med_indicator = bool(
            re.search(
                r"(?i)(?:\b(?:take\s+(?!(?:walks?|rest|care|steam|water|fluids?|all\s+the\s+medicines))\w+|"
                r"administer|give|prescribe|start|consume|dissolve|inhale|apply|put|instill|inject|infuse|"
                r"tablet|tab|capsule|cap|rotacap|vial|sachet|syrup|gel|drops?|spray|ointment|cream|lotion)\b|"
                r"\d+\s*(?:mg|g|mcg|ml|iu|%)\b|\bfor\s+\d+\s+days\b|\btill\s+\d+\s+days\b|\bfor\s+\d+\s+weeks\b)",
                c_clean,
            )
        )

        if not has_med_indicator:
            route_advice_clause(c_clean, clauses)
            continue

        # Check if clause is an instruction follow-up for an existing clause without a new dosage/schedule
        # e.g. "Take Metformin strictly with meals to avoid stomach upset"
        has_dose_or_sched = bool(
            re.search(r"\d+\s*(?:mg|g|mcg|ml|iu|%|grams?|gm)\b|\b(?:daily|once|twice|thrice|\d+\s+days?|\d+\s+weeks?)\b", c_clean, re.IGNORECASE)
        )
        if not has_dose_or_sched and clauses:
            matched_prev = False
            for cl in clauses:
                cand_med = cl.seed_drug_name
                if not cand_med:
                    m_lead = re.search(r"(?i)\b(?:take|administer|give|start)\s+(?:one\s+|two\s+)?(?:tablet\s+of\s+|capsule\s+of\s+)?([A-Za-z0-9\-]{4,})", cl.text)
                    if m_lead:
                        cand_med = m_lead.group(1)
                if cand_med and len(cand_med) >= 4:
                    cand_clean = re.sub(r"(?i)\b(take|tablet|capsule|one|two|three|oral|and|with)\b", "", cand_med).strip().lower()
                    if cand_clean and re.search(rf"\b{re.escape(cand_clean)}\b", c_clean, re.IGNORECASE):
                        cl.advice_clauses.append(c_clean)
                        matched_prev = True
                        break
            if matched_prev:
                continue

        # 4. Check companion drug formulations (e.g. "combination tablet of A 500mg and B 10mg")
        comb_match = re.search(
            r"(?:combination\s+(?:tablet\s+of\s+|of\s+)?|take\s+|administer\s+|give\s+)?"
            r"([A-Za-z0-9\s\.\-]+?\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu))\s+and\s+"
            r"([A-Za-z0-9\s\.\-]+?\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu)?)",
            c_clean,
            re.IGNORECASE,
        )
        if comb_match:
            d1_raw = comb_match.group(1).strip()
            d2_raw = comb_match.group(2).strip()
            d1 = re.split(r"(?i)\b(?:once|twice|thrice|daily|od|bd|tid|qid|for|after|before|every)\b", d1_raw)[0].strip()
            d2 = re.split(r"(?i)\b(?:once|twice|thrice|daily|od|bd|tid|qid|for|after|before|every)\b", d2_raw)[0].strip()
            
            idx = text.find(c_clean) if c_clean in text else 0
            clauses.append(
                PrescriptionClause(
                    clause_id=clause_counter,
                    text=c_clean,
                    seed_drug_name=d1,
                    start_char=max(0, idx),
                    end_char=max(0, idx + len(c_clean)),
                )
            )
            clause_counter += 1
            clauses.append(
                PrescriptionClause(
                    clause_id=clause_counter,
                    text=c_clean,
                    seed_drug_name=d2,
                    start_char=max(0, idx),
                    end_char=max(0, idx + len(c_clean)),
                )
            )
            clause_counter += 1
            continue

        # Check "Drug A with Drug B"
        comp_match = re.search(
            r"([A-Za-z0-9\s\.\-]+?)\s+with\s+([A-Za-z0-9\s\.\-]+)",
            c_clean,
            re.IGNORECASE,
        )
        if comp_match:
            d1_raw = comp_match.group(1).strip()
            d2_raw = comp_match.group(2).strip()
            d2_first_word = d2_raw.split()[0].lower() if d2_raw else ""
            has_dose_or_form_1 = bool(
                re.search(r"(?i)\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|%)|\b(?:tablets?|capsules?|gel|lotion|drops?|solution|cream|ointment)\b", d1_raw)
            )
            has_dose_or_form_2 = bool(
                re.search(r"(?i)\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|%)|\b(?:tablets?|capsules?|gel|lotion|drops?|solution|cream|ointment)\b", d2_raw)
            )
            if d2_first_word not in NON_DRUG_WITH and len(d2_first_word) >= 3 and has_dose_or_form_1 and has_dose_or_form_2:
                d1 = re.split(r"(?i)\b(?:once|twice|thrice|three\s+times|four\s+times|daily|od|bd|tid|qid|for|after|before|every)\b", d1_raw)[0].strip()
                d2 = re.split(
                    r"(?i)\b(?:locally|once|twice|thrice|three\s+times|four\s+times|daily|od|bd|tid|qid|for|after|before|every|without|using|at|into|in|on|to|along|over|around|strictly|ensuring|diluted|dissolved|mixed|slowly)\b",
                    d2_raw,
                )[0].strip()
                idx = text.find(c_clean) if c_clean in text else 0
                clauses.append(
                    PrescriptionClause(
                        clause_id=clause_counter,
                        text=c_clean,
                        seed_drug_name=d1,
                        start_char=max(0, idx),
                        end_char=max(0, idx + len(c_clean)),
                    )
                )
                clause_counter += 1
                clauses.append(
                    PrescriptionClause(
                        clause_id=clause_counter,
                        text=c_clean,
                        seed_drug_name=d2,
                        start_char=max(0, idx),
                        end_char=max(0, idx + len(c_clean)),
                    )
                )
                clause_counter += 1
                continue

        # Standard discrete medication clause
        idx = text.find(c_clean) if c_clean in text else 0
        clauses.append(
            PrescriptionClause(
                clause_id=clause_counter,
                text=c_clean,
                start_char=max(0, idx),
                end_char=max(0, idx + len(c_clean)),
            )
        )
        clause_counter += 1

    if not clauses:
        clauses.append(
            PrescriptionClause(
                clause_id=1,
                text=clean_text,
                start_char=0,
                end_char=len(clean_text),
            )
        )

    return clauses
