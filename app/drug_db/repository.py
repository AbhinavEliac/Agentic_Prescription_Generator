"""
app/drug_db/repository.py
-------------------------
Backwards compatibility alias for app.drugs.repository.
Canonical implementation has moved to app.drugs.repository.DrugRepository.
"""

from app.drugs.repository import (
    DrugRepository,
    get_drug_repository,
    soundex,
    clean_drug_base_name,
    levenshtein_distance,
    WELL_KNOWN_CLINICAL_DRUGS,
    COMMON_DRUG_ALIASES,
)

__all__ = [
    "DrugRepository",
    "get_drug_repository",
    "soundex",
    "clean_drug_base_name",
    "levenshtein_distance",
    "WELL_KNOWN_CLINICAL_DRUGS",
    "COMMON_DRUG_ALIASES",
]
