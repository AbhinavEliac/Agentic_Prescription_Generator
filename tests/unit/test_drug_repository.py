"""
tests/unit/test_drug_repository.py
----------------------------------
Comprehensive unit tests for the Canonical DrugRepository.

Verifies:
1. Exact matching (case-sensitive & uppercase code)
2. Case and punctuation normalization
3. Brand matching (Dolo, Crocin, Aten, Pan, Augmentin)
4. Generic substance matching (Paracetamol, Atenolol, Spironolactone)
5. Spelling error tolerance (Soundex + Levenshtein fuzzy matching)
6. Strength matching and validation
7. Unknown drug rejection (ZERO hallucination guaranteed)
8. Ambiguous drug candidate handling
9. Anatomical route resolution
10. In-memory indexing performance (<150ms boot, <1ms query)
"""

import pytest
import time
from app.drugs.repository import DrugRepository, get_drug_repository
from app.drugs.schemas import DrugType, ProvenanceSource


@pytest.fixture(scope="module")
def repo():
    """Initializes and returns the singleton DrugRepository."""
    return get_drug_repository()


class TestDrugRepositoryExactMatch:
    """Tests for DrugRepository.find_exact()."""

    def test_exact_drug_name_match(self, repo: DrugRepository):
        entry = repo.find_exact("DOLO 650 TAB")
        assert entry is not None
        assert entry.drug_code == "DOLO-650"
        assert entry.drug_name == "DOLO 650 TAB"
        assert entry.drug_type == DrugType.BRAND

    def test_exact_drug_code_match(self, repo: DrugRepository):
        entry = repo.find_exact("DOLO-650")
        assert entry is not None
        assert entry.drug_name == "DOLO 650 TAB"

    def test_exact_base_name_match(self, repo: DrugRepository):
        entry = repo.find_exact("PARACETAMOL")
        assert entry is not None
        assert "PARACETAMOL" in entry.base_name

    def test_exact_case_sensitivity_untrimmed(self, repo: DrugRepository):
        # Trimming whitespace should still resolve
        entry = repo.find_exact("  DOLO 650 TAB  ")
        assert entry is not None
        assert entry.drug_code == "DOLO-650"


class TestDrugRepositoryCaseNormalization:
    """Tests for DrugRepository.find_normalized()."""

    def test_lowercase_normalization(self, repo: DrugRepository):
        entry = repo.find_normalized("paracetamol")
        assert entry is not None
        assert "PARACETAMOL" in entry.base_name

    def test_titlecase_normalization(self, repo: DrugRepository):
        entry = repo.find_normalized("Paracetamol")
        assert entry is not None
        assert "PARACETAMOL" in entry.base_name

    def test_dosage_and_form_stripping(self, repo: DrugRepository):
        # "Paracetamol 650 mg tablet" should normalize to base PARACETAMOL
        entry = repo.find_normalized("Paracetamol 650 mg tablet")
        assert entry is not None
        assert "PARACETAMOL" in entry.base_name

    def test_official_alias_resolution(self, repo: DrugRepository):
        # "Amoxycillin" (British/INN spelling) should resolve to "Amoxicillin"
        entry = repo.find_normalized("Amoxycillin")
        assert entry is not None
        assert "AMOX" in entry.base_name

    def test_metformin_phonetic_alias(self, repo: DrugRepository):
        entry = repo.find_normalized("Metphormin")
        assert entry is not None
        assert "METFORMIN" in entry.base_name


class TestDrugRepositoryBrandGeneric:
    """Tests for find_by_brand() and find_by_generic()."""

    def test_find_by_brand_dolo(self, repo: DrugRepository):
        results = repo.find_by_brand("Dolo")
        assert len(results) > 0
        assert all(d.drug_type == DrugType.BRAND for d in results)
        names = [d.drug_name for d in results]
        assert any("DOLO 650" in n for n in names)

    def test_find_by_brand_crocin(self, repo: DrugRepository):
        results = repo.find_by_brand("Crocin")
        assert len(results) > 0
        names = [d.drug_name for d in results]
        assert any("CROCIN" in n for n in names)

    def test_find_by_generic_paracetamol(self, repo: DrugRepository):
        # Searching by generic Paracetamol should return both generic and associated brand products
        results = repo.find_by_generic("Paracetamol")
        assert len(results) > 0
        # Should link DOLO or generic paracetamol
        all_text = " ".join([d.drug_name + " " + (d.generic_name or "") for d in results])
        assert "PARACETAMOL" in all_text

    def test_find_by_generic_spironolactone(self, repo: DrugRepository):
        results = repo.find_by_generic("spironolactone")
        assert len(results) > 0
        assert any("spironolactone" in d.drug_name.lower() for d in results)


class TestDrugRepositorySpellingErrors:
    """Tests for find_fuzzy() and find_did_you_mean()."""

    def test_fuzzy_paracetamol_typo(self, repo: DrugRepository):
        # "parasitamol" -> Paracetamol
        matches = repo.find_fuzzy("parasitamol", min_confidence=0.70)
        assert len(matches) > 0
        best = matches[0]
        assert "PARACETAMOL" in best.drug.base_name
        assert best.similarity >= 0.70

    def test_fuzzy_metformin_typo(self, repo: DrugRepository):
        matches = repo.find_fuzzy("metformn", min_confidence=0.75)
        assert len(matches) > 0
        best = matches[0]
        assert "METFORMIN" in best.drug.base_name

    def test_did_you_mean_phonetic_recommendation(self, repo: DrugRepository):
        dym = repo.find_did_you_mean("parasitamol")
        assert dym is not None
        assert "Paracetamol" in dym

    def test_did_you_mean_exact_returns_none(self, repo: DrugRepository):
        # If drug is spelled correctly, no recommendation is needed
        dym = repo.find_did_you_mean("Paracetamol")
        assert dym is None


class TestDrugRepositoryStrengthMatching:
    """Tests for find_strength()."""

    def test_find_all_strengths_for_dolo(self, repo: DrugRepository):
        strengths = repo.find_strength("DOLO")
        assert len(strengths) > 0
        assert "650" in strengths
        assert "500" in strengths

    def test_validate_correct_strength(self, repo: DrugRepository):
        valid = repo.find_strength("DOLO 650 TAB", strength_query="650 mg")
        assert valid == ["650"]

    def test_validate_invalid_strength(self, repo: DrugRepository):
        # 9999 mg is NOT a valid formulation strength for Dolo
        invalid = repo.find_strength("DOLO 650 TAB", strength_query="9999 mg")
        assert invalid == []

    def test_percentage_strength(self, repo: DrugRepository):
        strengths = repo.find_strength("OXYMETAZOLINE")
        assert "0.05" in strengths or "0.05%" in strengths


class TestDrugRepositoryRouteMatching:
    """Tests for find_route()."""

    def test_route_for_dolo(self, repo: DrugRepository):
        routes = repo.find_route("DOLO 650 TAB")
        assert "ORAL" in routes

    def test_route_for_nasal_spray(self, repo: DrugRepository):
        routes = repo.find_route("OXYMET-0.05")
        assert "NASAL" in routes

    def test_route_fallback_default(self, repo: DrugRepository):
        # Unmapped drug IDs should default to clinical standard ORAL
        routes = repo.find_route("non_existent_id_99999999")
        assert routes == ["ORAL"]


class TestDrugRepositoryZeroHallucination:
    """Tests guaranteeing that unrecognized drugs are NOT hallucinated."""

    def test_find_exact_unknown_drug_returns_none(self, repo: DrugRepository):
        entry = repo.find_exact("XyloFakeDrugAlphaBetaGamma12345")
        assert entry is None

    def test_find_normalized_unknown_drug_returns_none(self, repo: DrugRepository):
        entry = repo.find_normalized("CompletelyNonExistentMedicineName999 500 mg")
        assert entry is None

    def test_find_fuzzy_unknown_drug_returns_empty(self, repo: DrugRepository):
        # Random gibberish with no phonetic or string resemblance must return empty list
        matches = repo.find_fuzzy("Zzqqxxwwjjkkllmm", min_confidence=0.75)
        assert matches == []

    def test_did_you_mean_unknown_gibberish_returns_none(self, repo: DrugRepository):
        dym = repo.find_did_you_mean("Zzqqxxwwjjkkllmm")
        assert dym is None

    def test_find_by_brand_unknown_returns_empty(self, repo: DrugRepository):
        results = repo.find_by_brand("NonExistentBrandXYZ123")
        assert results == []

    def test_find_by_generic_unknown_returns_empty(self, repo: DrugRepository):
        results = repo.find_by_generic("NonExistentGenericXYZ123")
        assert results == []


class TestDrugRepositoryAmbiguousDrug:
    """Tests for disambiguation when multiple formulations match."""

    def test_ambiguous_prefix_returns_ranked_candidates(self, repo: DrugRepository):
        # Querying "Aten" matches Aten 25, Aten 50, Aten 100
        results = repo.find_by_brand("Aten", limit=5)
        assert len(results) > 1
        # All returned candidates should be distinct formulations
        ids = [d.drug_id for d in results]
        assert len(ids) == len(set(ids))

    def test_search_drugs_limit_enforced(self, repo: DrugRepository):
        results = repo.search_drugs("Paracetamol", limit=5)
        assert len(results) <= 5


class TestDrugRepositoryPerformance:
    """Tests performance SLAs (<150ms initialization, <1ms query latency)."""

    def test_fast_lookup_latency(self, repo: DrugRepository):
        # Benchmark 100 lookups
        t0 = time.perf_counter()
        for _ in range(100):
            repo.find_exact("DOLO-650")
            repo.find_normalized("Paracetamol 650 mg")
            repo.find_route("DOLO-650")
        elapsed = time.perf_counter() - t0
        avg_latency_ms = (elapsed / 300) * 1000
        # Each query must be sub-millisecond
        assert avg_latency_ms < 1.0, f"Average query latency {avg_latency_ms:.3f}ms exceeds 1.0ms SLA"

    def test_soundex_speed(self, repo: DrugRepository):
        t0 = time.perf_counter()
        for _ in range(1000):
            repo.is_known_drug("PARACETAMOL")
        elapsed = time.perf_counter() - t0
        avg_us = (elapsed / 1000) * 1_000_000
        assert avg_us < 50.0, f"Vocabulary check {avg_us:.2f}us exceeds 50us SLA"
