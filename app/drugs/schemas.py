"""
app/drugs/schemas.py
--------------------
Canonical domain models and data contracts for pharmaceutical data.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field, ConfigDict


class DrugType(str, Enum):
    """Classification of drug entry."""
    BRAND = "b"
    GENERIC = "g"
    UNKNOWN = "u"


class ProvenanceSource(str, Enum):
    """Origin and authority level of drug data."""
    MASTER_FORMULARY = "master_formulary"
    CLINICAL_CURATION = "clinical_curation"
    OFFICIAL_ALIAS = "official_alias"


class DrugEntry(BaseModel):
    """Canonical model for a single drug formulation in the repository."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    drug_id: str
    drug_code: str
    drug_name: str
    base_name: str
    drug_type: DrugType = DrugType.BRAND
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    dosage_form: Optional[str] = None
    strength_values: Set[str] = Field(default_factory=set)
    routes: List[str] = Field(default_factory=list)
    provenance: ProvenanceSource = ProvenanceSource.MASTER_FORMULARY
    raw_data: Dict[str, Any] = Field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Returns concise dictionary for API consumers."""
        return {
            "drug_id": self.drug_id,
            "drug_code": self.drug_code,
            "drug_name": self.drug_name,
            "base_name": self.base_name,
            "drug_type": self.drug_type.value,
            "generic_name": self.generic_name,
            "brand_name": self.brand_name,
            "dosage_form": self.dosage_form,
            "strengths": sorted(list(self.strength_values)),
            "routes": self.routes,
            "provenance": self.provenance.value,
        }


class DrugMatch(BaseModel):
    """Result of a search or fuzzy recommendation query with provenance & confidence."""
    drug: DrugEntry
    similarity: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    matched_via: str = "EXACT"  # EXACT, NORMALIZED, SOUNDEX, LEVENSHTEIN, ALIAS, BRAND_MAP, PREFIX

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.drug.to_summary_dict(),
            "similarity": round(self.similarity, 3),
            "confidence": round(self.confidence, 3),
            "matched_via": self.matched_via,
        }


class ReferenceDataResponse(BaseModel):
    """Reference data container for schedules, dose units, and routes."""
    schedules: List[Dict[str, str]] = Field(default_factory=list)
    dose_units: List[Dict[str, str]] = Field(default_factory=list)
    routes: List[Dict[str, str]] = Field(default_factory=list)
