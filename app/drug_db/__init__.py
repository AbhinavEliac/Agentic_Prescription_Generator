"""
app.drug_db package
-------------------
Compatibility wrapper around canonical app.drugs.
"""
from app.drugs import DrugRepository, get_drug_repository

__all__ = ["DrugRepository", "get_drug_repository"]
