"""
app/prescription/deterministic/route.py
--------------------------------------
Deterministic Route of Administration Extractor.

Resolves clinical route based on formulation type, administration verbs, and anatomical targets:
- otic (ear drops, affected ear)
- ophthalmic (eye drops, into left eye)
- nasal (nasal spray, both nostrils)
- inhalation (rotacap, puff, inhaler, revolizer, spacer)
- topical (cream, ointment, gel, lotion, suture line, patches)
- oral (tablet, capsule, syrup, oral suspension, sachet, swallow)
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, List
from app.drugs.repository import DrugRepository


def extract_route(
    clause_text: str,
    base_drug_name: Optional[str] = None,
    drug_repo: Optional[DrugRepository] = None,
) -> Tuple[Optional[str], Optional[Tuple[int, int]], float]:
    """
    Extracts route of administration with formulation precedence.
    Strictly returns None if route cannot be determined from text or formulary.
    Never fabricates 'oral' simply because it is medically common.
    """
    clause_lower = clause_text.lower()
    core_directive = clause_lower[:100]

    # 1. Otic / Ear
    if any(w in core_directive or w in clause_lower for w in ("ear drops", "affected ear", "eardrum", "into each ear", "ear drop", "otic")):
        return "otic", None, 0.95

    # 2. Ophthalmic / Eye
    if any(w in core_directive or w in clause_lower for w in ("eye drops", "into both eyes", "into left eye", "into right eye", "ophthalmic", "eye drop")):
        return "ophthalmic", None, 0.95

    # 3. Nasal
    if any(w in core_directive or w in clause_lower for w in ("nasal spray", "nasal drops", "into each nostril", "both nostrils", "nasal drop")):
        return "nasal", None, 0.95

    # 4. Inhalation
    if any(w in core_directive or w in clause_lower for w in ("inhale", "rotacap", "rotacaps", "puff", "puffs", "revolizer", "spacer", "turbuhaler", "respule", "respules", "inhaler", "inhalation")):
        return "inhalation", None, 0.95

    # 5. Topical
    if any(w in core_directive or w in clause_lower for w in ("apply", "gel", "ointment", "cream", "lotion", "massage", "sunscreen", "cleanse", "topically", "topical", "patch", "patches", "onto your lower back", "along the clean suture line", "to the affected patches", "over dry skin")):
        return "topical", None, 0.95

    # 6. Sublingual / Buccal
    if any(w in core_directive or w in clause_lower for w in ("sublingual", "sublingually", "under the tongue", "under tongue", "buccal")):
        return "sublingual", None, 0.95

    # 7. Intravenous / Injection
    if any(w in core_directive or w in clause_lower for w in ("intravenous", "intramuscular", "iv infusion", "iv injection", "im injection", "subcutaneous", "\biv\b", "\bim\b", "\bsc\b")):
        if any(w in clause_lower for w in ("\bim\b", "intramuscular")):
            return "intramuscular", None, 0.95
        return "intravenous", None, 0.95

    # 8. Rectal
    if any(w in core_directive or w in clause_lower for w in ("suppository", "suppositories", "rectally", "per rectum")):
        return "rectal", None, 0.95

    # 9. Oral (Strictly requires oral dosage forms, verbs, or explicit delivery descriptors)
    oral_indicators = (
        "tablet", "tablets", "tab", "tabs",
        "capsule", "capsules", "cap", "caps",
        "syrup", "sachet", "sachets", "pill", "pills",
        "oral suspension", "suspension",
        "gargle", "mouthwash", "lozenge", "lozenges", "paste",
        "orally", "by mouth", "per os", "swallow"
    )
    if any(re.search(r"\b" + re.escape(w) + r"\b", clause_lower) for w in oral_indicators):
        return "oral", None, 0.95

    # 10. Formulary route check if drug repo available
    if drug_repo and base_drug_name:
        routes = drug_repo.get_routes_for_drug(base_drug_name)
        if routes and len(routes) == 1 and routes[0].lower() in ("oral", "topical", "nasal", "inhalation", "ophthalmic", "otic"):
            # Only use formulary route if unambiguous
            return routes[0].lower(), None, 0.80

    # If unstated and cannot be definitively determined, return None (never fabricate)
    return None, None, 0.0
