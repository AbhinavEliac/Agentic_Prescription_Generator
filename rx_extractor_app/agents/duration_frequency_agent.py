"""
agents/duration_frequency_agent.py
----------------------------------
Duration & Frequency Agent:
Drift-proof extractor for administration frequencies and duration spans across arbitrary clinical expressions.
"""
import re
from typing import Dict, Any, List
from graph_state import AgenticRxState, DurationFrequencyItem
from agents.utils import (
    is_placeholder,
    safe_parse_json,
    segment_prescription,
)
import prompt

GENERAL_FREQUENCY_PATTERNS = [
    r"up\s+to\s+\w+\s+times\s+daily(?:\s+as\s+needed)?",
    r"every\s+\d+(?:\s+to\s+\d+)?\s*(?:hours?|hrs?|days?|weeks?|months?)",
    r"every\s+(?:morning|night|evening|afternoon|bedtime|other\s+day)",
    r"(?:twice|once|thrice|three\s+times|four\s+times|\d+\s+times)\s+daily\s*\([0-9\-\/]+\)",
    r"\b\d+-\d+-\d+(?:-\d+)?\b",
    r"(?:twice|once|thrice|three\s+times|four\s+times|\d+\s+times)\s+(?:daily|a\s+day|per\s+day|a\s+week|weekly|a\s+month|monthly)",
    r"(?:once|twice|thrice)\s+weekly",
    r"three\s+times\s+daily",
    r"four\s+times\s+daily",
    r"twice\s+daily",
    r"once\s+daily",
    r"thrice\s+daily",
    r"throughout\s+the\s+day",
    r"single\s+dose",
    r"stat\s+dose",
    r"as\s+needed\s*\([A-Za-z]+\)",
    r"as\s+needed(?:\s+for\s+[a-zA-Z\s]+)?",
    r"early\s+morning",
    r"at\s+bedtime",
    r"\bdaily\b",
    r"\b(?:OD|BD|TID|QID|QDS|TDS|BID|PRN|SOS|HS|STAT|q\d+h|q\d+-\d+h)\b",
]

GENERAL_DURATION_PATTERNS = [
    r"\b(?:for\s+|duration\s+of\s+|till\s+|until\s+|upto\s+|up\s+to\s+|for\s+upto\s+|for\s+up\s+to\s+|for\s+next\s+|next\s+|about\s+|around\s+|approx(?:\s+)?|for\s+around\s+|for\s+about\s+|x\s*)(\d+\s*(?:days?|d|weeks?|wks?|months?|mo|years?|hrs?|hours?)|day\s+one|single\s+day|\d+\s*to\s*\d+\s*(?:days?|weeks?|months?)|no\s+more\s+than\s+\d+\s+days?)\b",
    r"\b(until\s+[a-zA-Z\s]{3,30})\b",
    r"\b(\d+\s*(?:days?|weeks?|months?))\b",
]


def duration_frequency_agent(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Duration & Frequency Agent.
    """
    input_text = state.get("input_text", "")
    feedback = state.get("validation_feedback", {}).get("duration_frequency_agent", "")

    extracted_dur_freq: List[DurationFrequencyItem] = []

    if llm is not None:
        p = prompt.DURATION_FREQUENCY_PROMPT.replace("{{VOICE_INPUT}}", input_text)
        p = p.replace("{{FEEDBACK}}", feedback if feedback else "None")
        try:
            raw_out = llm.invoke(p)
            if isinstance(raw_out, str):
                if "Prescription Input:" in raw_out:
                    raw_out = raw_out.split("Prescription Input:")[-1]
                parsed = safe_parse_json(raw_out)
                if isinstance(parsed, list):
                    for idx, item in enumerate(parsed):
                        freq = item.get("frequency", "NONE")
                        dur = item.get("duration", "NONE")
                        if not is_placeholder(freq) and not is_placeholder(dur):
                            extracted_dur_freq.append({
                                "medicine_id": item.get("medicine_id", idx + 1),
                                "drug_name": item.get("drug_name", ""),
                                "frequency": freq,
                                "duration": dur,
                            })
        except Exception:
            pass

    if not extracted_dur_freq:
        segments = segment_prescription(input_text)

        # Check global plural/broadcast coreference (e.g. "Both should be taken twice daily", "All medicines are once daily")
        global_plural_freq = "NONE"
        global_plural_dur = "NONE"
        plural_match = re.search(r"(?i)\b(?:both(?:\s+of\s+them|\s+medicines|\s+drugs|\s+tablets|\s+capsules)?|all(?:\s+these|\s+of\s+them)?(?:\s+medicines|\s+drugs|\s+tablets)?|each(?:\s+of\s+them)?)\s+(?:should\s+be\s+taken|are\s+to\s+be\s+taken|must\s+be\s+taken|to\s+be\s+taken|should\s+be\s+given|should\s+be|are|must\s+be)\s+([^,\.\n;!]+)", input_text)
        if plural_match:
            cand_p = plural_match.group(1).strip()
            for fp in GENERAL_FREQUENCY_PATTERNS:
                fm = re.search(fp, cand_p, re.IGNORECASE)
                if fm:
                    global_plural_freq = fm.group(0).strip()
                    break
            for dp in GENERAL_DURATION_PATTERNS:
                dm = re.search(dp, input_text, re.IGNORECASE)
                if dm:
                    global_plural_dur = dm.group(1).strip()
                    break

        # Check singular coreference (e.g. "It should be taken 4 times a day", "This medicine is to be taken twice daily")
        global_singular_freq = "NONE"
        singular_match = re.search(r"(?i)\b(?:it\s+(?:should\s+be\s+taken|is\s+to\s+be\s+taken|must\s+be\s+taken|to\s+be\s+taken|is\s+taken)|take\s+(?:it|this(?:\s+medicine)?))\s+([^,\.\n;!]+)", input_text)
        if singular_match:
            cand_s = singular_match.group(1).strip()
            for fp in GENERAL_FREQUENCY_PATTERNS:
                fm = re.search(fp, cand_s, re.IGNORECASE)
                if fm:
                    global_singular_freq = fm.group(0).strip()
                    break

        for s in segments:
            m_id = s["medicine_id"]
            clause = s["clause"]

            # 1. Frequency (search in first 100 chars of clause first to avoid matching trailing advice)
            core_clause = clause[:100]
            freq = "NONE"
            for fp in GENERAL_FREQUENCY_PATTERNS:
                f_match = re.search(fp, core_clause, re.IGNORECASE)
                if f_match:
                    matched_freq = f_match.group(0).strip()
                    matched_freq = re.sub(r"(?i)\s+(?:before|after)\s+(?:food|breakfast|meals|lunch|dinner)$", "", matched_freq).strip()
                    freq = matched_freq
                    break
            
            if freq == "NONE":
                for fp in GENERAL_FREQUENCY_PATTERNS:
                    f_match = re.search(fp, clause, re.IGNORECASE)
                    if f_match:
                        matched_freq = f_match.group(0).strip()
                        matched_freq = re.sub(r"(?i)\s+(?:before|after)\s+(?:food|breakfast|meals|lunch|dinner)$", "", matched_freq).strip()
                        freq = matched_freq
                        break

            if freq == "NONE":
                if global_plural_freq != "NONE":
                    freq = global_plural_freq
                elif global_singular_freq != "NONE" and (len(segments) == 1 or m_id == len(segments)):
                    freq = global_singular_freq

            # 2. Duration
            dur = "NONE"
            for dp in GENERAL_DURATION_PATTERNS:
                d_match = re.search(dp, clause, re.IGNORECASE)
                if d_match:
                    dur = d_match.group(1).strip()
                    break

            if dur == "NONE" and global_plural_dur != "NONE":
                dur = global_plural_dur

            extracted_dur_freq.append({
                "medicine_id": m_id,
                "drug_name": "",
                "frequency": freq,
                "duration": dur,
            })

    return {"durations_frequencies": extracted_dur_freq}
