"""
app.drugs package
-----------------
Canonical drug formulary, domain models, and repository interface.
"""

from app.drugs.schemas import DrugEntry, DrugMatch, DrugType, ProvenanceSource, ReferenceDataResponse
from app.drugs.repository import DrugRepository, get_drug_repository, soundex, clean_drug_base_name

__all__ = [
    "DrugRepository",
    "get_drug_repository",
    "DrugEntry",
    "DrugMatch",
    "DrugType",
    "ProvenanceSource",
    "ReferenceDataResponse",
    "soundex",
    "clean_drug_base_name",
]
