"""
punctuation_agent.py
--------------------
Punctuation & Sentence Correction Agent for clinical transcripts.

This agent operates directly on raw clinical transcripts or unpunctuated voice speech.
It performs:
1. Intelligent sentence boundary demarcation (separating continuous speech into logical clinical statements).
2. Medication clause separation (e.g. separating multi-drug instructions with periods).
3. Conditional titration clause formatting (e.g. inserting commas in 'if fever persists, increase dose by...').
4. Follow-up and warning sentence separation.
5. Capitalization of sentence start tokens.

STRICT INVARIANT:
- Does NOT alter spelling, names, numbers, decimal values (e.g., 37.5 mg, 0.05%), or clinical units.
"""
import re
from typing import Any, Dict
from graph_state import AgenticRxState


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
        r"(?i)\b(if\s+[a-zA-Z0-9\s\-]+?(?:does\s+not\s+go\s+away|does\s+not\s+clear|does\s+not\s+improve|persists|worsens|increases|crosses\s+\d+|develops|occurs|remains\s+high|subsides|heals))\s+(increase|decrease|reduce|double|taper|meet|come\s+visit|visit|consult|seek|start|stop|discontinue)\b",
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
    # Note: Guard against infinitives ('to consume', 'to take', 'to apply')
    med_verb_pattern = (
        r"(?<=[a-zA-Z0-9%\)])(?<!\bto)(?:\s*,\s*|\s+)(?:and\s+)?"
        r"(take\s+(?!(?:walks?|rest|care|steam))\w+|take\s+regular\s+water|administer|inhale|apply|instill|consume|dissolve|inject|infuse)\b"
    )
    s = re.sub(med_verb_pattern, r". \1", s, flags=re.IGNORECASE)

    # 4. Insert sentence breaks before clinical follow-up, advice, and warnings
    advice_pattern = (
        r"(?<=[a-zA-Z0-9%\)])(?<!\bto)(?:\s*,\s*|\s+)(?:and\s+)?"
        r"(if\s+(?:the\s+|it\s+(?:is\s+)?(?:still\s+)?|they\s+|there\s+is\s+)?(?:fever|headache|pain|symptoms?|rash|condition|severe|swelling|ulcers|blood\s+pressure|breathing|does\s+not|persists|not\s+relieved)|"
        r"meet\s+(?:the\s+)?doctor|please\s+see\s+me|see\s+(?:your\s+|the\s+)?doctor|seek\s+reassessment|seek\s+medical|"
        r"return\s+for|return\s+if|return\s+after|use\s+saline\s+nasal|perform\s+steam|(?:also\s+)?take\s+walks|"
        r"drink\s+plenty|keep\s+the\s+blistered|avoid\s+close)\b"
    )
    s = re.sub(advice_pattern, r". \1", s, flags=re.IGNORECASE)

    # 5. Clean up multiple periods without touching decimals
    s = re.sub(r"(?<!\d)\.\s*\.+(?!\d)", ".", s)
    s = re.sub(r"(?<!\d)\s*\.\s*(?!\d)", ". ", s).strip()
    if s and not s.endswith((".", "!", "?")):
        s += "."

    # 6. Sentence segmentation and start-of-sentence capitalization
    sentences = [sent.strip() for sent in re.split(r"(?<=[a-zA-Z%!?\)])\.\s+(?=[A-Za-z0-9])", s) if sent.strip()]
    capitalized = []
    for sent in sentences:
        if sent:
            # Capitalize only the first character to preserve existing brand case
            sent_cap = sent[0].upper() + sent[1:]
            capitalized.append(sent_cap)

    punctuated_result = ". ".join(capitalized) if capitalized else s
    if punctuated_result and not punctuated_result.endswith((".", "!", "?")):
        punctuated_result += "."

    return punctuated_result


def punctuation_agent(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    LangGraph Node: Punctuation & Sentence Correction Agent.
    Ingests input_text, performs sentence boundary and punctuation correction,
    and forwards punctuated_text to the Supervisor Node.
    """
    raw_text = state.get("raw_input_text") or state.get("input_text", "")
    punctuated = correct_sentence_punctuation(raw_text)

    log_entry = {
        "agent": "Punctuation & Sentence Correction Agent",
        "action": "Sentence segmentation and clinical punctuation normalization",
        "raw_text": raw_text,
        "punctuated_text": punctuated,
    }

    current_logs = list(state.get("agent_logs", []))
    current_logs.append(log_entry)

    return {
        "raw_input_text": raw_text,
        "punctuated_text": punctuated,
        "input_text": punctuated,  # Downstream agents receive well-punctuated text
        "agent_logs": current_logs,
    }
