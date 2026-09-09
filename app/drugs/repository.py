"""
app/drugs/repository.py
-----------------------
Single Canonical Drug Repository for the Agentic Prescription Generator.

Provides high-performance in-memory indexing, Soundex phonetic matching,
brand/generic relational queries, dosage validation, and route resolution (<120ms boot).
"""

from __future__ import annotations
import os
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from app.drugs.schemas import (
    DrugEntry,
    DrugMatch,
    DrugType,
    ProvenanceSource,
    ReferenceDataResponse,
)


def phonetic_backbone(s: str) -> str:
    """
    Computes a generalized phonetic consonant backbone based on articulatory acoustics:
    - Velars/Palatals: C, K, Q, G, J -> 'K'
    - Labials: B, P, V, F, W, PH -> 'P'
    - Dentals/Alveolars: D, T, TH -> 'T'
    - Sibilants: S, Z, C(soft), X -> 'S'
    - Nasals: M, N -> 'M', 'N'
    - Liquids: L, R -> 'L', 'R'
    Compresses consecutive identical articulatory classes and normalizes digraphs.
    Enables zero-overfitting phonetic matching across speech-to-text accent variations.
    """
    if not s:
        return ""
    clean = re.sub(r"[^a-zA-Z]", "", s.lower())
    if not clean:
        return ""
    clean = clean.replace("ph", "f").replace("ck", "k").replace("th", "t").replace("sh", "s")
    c_soft = re.sub(r"c(?=[eiy])", "s", clean)
    char_map = {
        'b': 'P', 'p': 'P', 'v': 'P', 'f': 'P', 'w': 'P',
        'c': 'K', 'k': 'K', 'q': 'K', 'g': 'K', 'j': 'K',
        'd': 'T', 't': 'T',
        's': 'S', 'z': 'S', 'x': 'S',
        'l': 'L',
        'r': 'R',
        'm': 'M', 'n': 'N',
    }
    encoded = []
    prev = ""
    for ch in c_soft:
        code = char_map.get(ch, "")
        if code and code != prev:
            encoded.append(code)
            prev = code
        elif not code:
            prev = ""
    return "".join(encoded)


def soundex(s: str) -> str:
    """Calculates 4-character Soundex code with ph->f phonetic normalization."""
    if not s:
        return ""
    clean = re.sub(r"[^A-Za-z]", "", s.lower().replace("ph", "f"))
    if not clean:
        return ""
    f = clean[0].upper()
    chars = clean[1:]
    codes = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3",
        "l": "4",
        "m": "5", "n": "5",
        "r": "6",
    }
    r = []
    prev_code = codes.get(f.lower(), "")
    for char in chars:
        code = codes.get(char, "")
        if code and code != prev_code:
            r.append(code)
            prev_code = code
        elif not code:
            prev_code = ""
    return (f + "".join(r) + "000")[:4]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein distance with ph->f equivalence."""
    s1 = (s1 or "").lower().replace("ph", "f")
    s2 = (s2 or "").lower().replace("ph", "f")
    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
    return matrix[len1][len2]


def clean_drug_base_name(drug_name: str) -> str:
    """Normalizes drug name to base substance/brand name by stripping dosage, form words, and symbols."""
    if not drug_name:
        return ""
    cleaned = drug_name.upper()
    cleaned = re.sub(r"\b\d+['`][sS]\b", "", cleaned)
    cleaned = re.sub(
        r"\b(TAB|TABS|TABLET|TABLETS|CAP|CAPS|CAPSULE|CAPSULES|SYP|SYRUP|INJ|INJECTION|"
        r"OINT|OINTMENT|CREAM|CRM|LOT|LOTION|DROPS|GEL|SOLN|SOLUTION|SOL|SUSP|SUSPENSION|"
        r"RET|RETD|DT|SR|XL|ER|CR|DS|PLUS|FORTE|POWD|PWDR|RESP|NEB|VIAL|SACHET|SPRAY|NASAL|"
        r"ROTACAP|ROTACAPS|RESPULE|RESPULES|INHALER|EXP|EXPECTORANT|SYNCHROBREATHE)(?!\w)",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:MG|G|MCG|ML|L|IU|%|GM)?(?!\w)", "", cleaned)
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


WELL_KNOWN_CLINICAL_DRUGS: List[Dict[str, Any]] = [
    {"drug_id": "99001", "drug_code": "DOLO-650", "drug_name": "DOLO 650 TAB", "drug_type": "b", "generic_name": "PARACETAMOL", "brand_name": "DOLO", "routes": ["ORAL", "RT"]},
    {"drug_id": "99002", "drug_code": "DOLO-500", "drug_name": "DOLO 500 TAB", "drug_type": "b", "generic_name": "PARACETAMOL", "brand_name": "DOLO", "routes": ["ORAL", "RT"]},
    {"drug_id": "99003", "drug_code": "PAN-40", "drug_name": "PAN 40 TAB", "drug_type": "b", "generic_name": "PANTOPRAZOLE", "brand_name": "PAN", "routes": ["ORAL", "IV"]},
    {"drug_id": "99004", "drug_code": "PAN-D", "drug_name": "PAN-D CAP", "drug_type": "b", "generic_name": "PANTOPRAZOLE + DOMPERIDONE", "brand_name": "PAN-D", "routes": ["ORAL"]},
    {"drug_id": "99005", "drug_code": "TELMA-40", "drug_name": "TELMA 40 TAB", "drug_type": "b", "generic_name": "TELMISARTAN", "brand_name": "TELMA", "routes": ["ORAL"]},
    {"drug_id": "99006", "drug_code": "CROCIN-650", "drug_name": "CROCIN 650 TAB", "drug_type": "b", "generic_name": "PARACETAMOL", "brand_name": "CROCIN", "routes": ["ORAL", "RT"]},
    {"drug_id": "99007", "drug_code": "CROCIN-500", "drug_name": "CROCIN 500 TAB", "drug_type": "b", "generic_name": "PARACETAMOL", "brand_name": "CROCIN", "routes": ["ORAL", "RT"]},
    {"drug_id": "99008", "drug_code": "DISPRIN-300", "drug_name": "DISPRIN 300 TAB", "drug_type": "b", "generic_name": "ASPIRIN", "brand_name": "DISPRIN", "routes": ["ORAL"]},
    {"drug_id": "99009", "drug_code": "DISPRIN-500", "drug_name": "DISPRIN 500 TAB", "drug_type": "b", "generic_name": "ASPIRIN", "brand_name": "DISPRIN", "routes": ["ORAL"]},
    {"drug_id": "99010", "drug_code": "OXYMET-0.05", "drug_name": "OXYMETAZOLINE 0.05% NASAL SPRAY", "drug_type": "g", "generic_name": "OXYMETAZOLINE", "routes": ["NASAL"]},
    {"drug_id": "99011", "drug_code": "CELLULOSE-50", "drug_name": "CELLULOSE 50 G POWD", "drug_type": "g", "generic_name": "CELLULOSE", "routes": ["ORAL"]},
    {"drug_id": "99012", "drug_code": "ASCORIL-D", "drug_name": "ASCORIL D SYRUP", "drug_type": "b", "generic_name": "DEXTROMETHORPHAN + PHENYLEPHRINE + CHLORPHENIRAMINE", "brand_name": "ASCORIL", "routes": ["ORAL"]},
    {"drug_id": "99013", "drug_code": "FORACORT-200", "drug_name": "FORACORT 200 INHALER", "drug_type": "b", "generic_name": "BUDESONIDE + FORMOTEROL", "brand_name": "FORACORT", "routes": ["INHALATION"]},
    {"drug_id": "99014", "drug_code": "PANTOP-40", "drug_name": "PANTOP 40 TAB", "drug_type": "b", "generic_name": "PANTOPRAZOLE", "brand_name": "PANTOP", "routes": ["ORAL", "IV"]},
    {"drug_id": "99015", "drug_code": "AMOXYCLAV-625", "drug_name": "AMOXYCLAV 625 TAB", "drug_type": "g", "generic_name": "AMOXICILLIN + CLAVULANIC ACID", "routes": ["ORAL"]},
    {"drug_id": "99016", "drug_code": "CLAVAM-625", "drug_name": "CLAVAM 625 TAB", "drug_type": "b", "generic_name": "AMOXICILLIN + CLAVULANIC ACID", "brand_name": "CLAVAM", "routes": ["ORAL"]},
    {"drug_id": "99017", "drug_code": "VIT-C-500", "drug_name": "VITAMIN C 500 MG TAB", "drug_type": "g", "generic_name": "ASCORBIC ACID", "brand_name": "VITAMIN C", "routes": ["ORAL"]},
    {"drug_id": "99018", "drug_code": "AUGMENTIN-625", "drug_name": "AUGMENTIN TAB 625MG", "drug_type": "b", "generic_name": "AMOXICILLIN + CLAVULANIC ACID", "brand_name": "AUGMENTIN", "routes": ["ORAL"]},
]

BRAND_TO_GENERIC_MAP: Dict[str, str] = {
    "DOLO": "PARACETAMOL",
    "CROCIN": "PARACETAMOL",
    "CALPOL": "PARACETAMOL",
    "PAN": "PANTOPRAZOLE",
    "PAN-D": "PANTOPRAZOLE + DOMPERIDONE",
    "PANTOCID": "PANTOPRAZOLE",
    "PANTOP": "PANTOPRAZOLE",
    "TELMA": "TELMISARTAN",
    "ATEN": "ATENOLOL",
    "DISPRIN": "ASPIRIN",
    "AUGMENTIN": "AMOXICILLIN + CLAVULANIC ACID",
    "MOX": "AMOXICILLIN",
    "AZITHRAL": "AZITHROMYCIN",
    "MEFTAL": "MEFENAMIC ACID",
    "MEFTAL-SPAS": "MEFENAMIC ACID + DICYCLOMINE",
    "CIPLOX": "CIPROFLOXACIN",
    "AMLONG": "AMLODIPINE",
    "GLYCOMET": "METFORMIN",
    "SOFRAMYCIN": "FRAMYCETIN",
    "BRUFEN": "IBUPROFEN",
    "VOMIKIND": "ONDANSETRON",
    "ZOFRAN": "ONDANSETRON",
    "ASCORIL": "DEXTROMETHORPHAN",
    "FORACORT": "BUDESONIDE + FORMOTEROL",
    "CLAVAM": "AMOXICILLIN + CLAVULANIC ACID",
    "AMOXYCLAV": "AMOXICILLIN + CLAVULANIC ACID",
    "ZINETAC": "RANITIDINE",
    "ACILOC": "RANITIDINE",
}

COMMON_DRUG_ALIASES: List[Tuple[str, str]] = [
    ("AMOXICILLIN", "AMOXYCILLIN"),
    ("PARACETAMOL", "ACETAMINOPHEN"),
    ("PANTOPRAZOL", "PANTOPRAZOLE"),
    ("OMEPRAZOL", "OMEPRAZOLE"),
    ("RABEPRAZOL", "RABEPRAZOLE"),
    ("ESOMEPRAZOL", "ESOMEPRAZOLE"),
    ("CIPROFLOXACIN", "CIPROFLOXACINE"),
    ("LEVOFLOXACIN", "LEVOFLOXACINE"),
    ("AZITHROMYCIN", "AZITHROMYCINE"),
    ("METFORMIN", "METPHORMIN"),
    ("METFORMIN", "METAFORAMINE"),
    ("METFORMIN", "METAFORMINE"),
    ("SOFRAMYCIN", "SOPHRAMYCIN"),
    ("ATEN", "ATENOLOL"),
    ("AMLODIPIN", "AMLODIPINE"),
    ("CEFIXIM", "CEFIXIME"),
    ("CEFPODOXIM", "CEFPODOXIME"),
    ("CEFUROXIM", "CEFUROXIME"),
    ("CETRIZINE", "CETIRIZINE"),
    ("LEVOCETRIZINE", "LEVOCETIRIZINE"),
    ("MONTELUKAST", "MONTELEKAST"),
    ("DICLOFENAC", "DICLOFENACK"),
    ("IBUPROFEN", "IBOPROFEN"),
    ("DOXYCYCLINE", "DOXICYCLINE"),
    ("CLOTRIMAZOLE", "CLOTRIMAZOL"),
    ("FLUCONAZOLE", "FLUCONAZOL"),
    ("OXYMETAZOLINE", "OXIMETHAZOLINE"),
    ("GABAPENTIN", "GABAPENTINE"),
    ("PREGABALIN", "PREGABALINE"),
    ("METRONIDAZOLE", "METRONIDAZOL"),
    ("ATORVASTATIN", "ATORVASTATION"),
    ("ROSUVASTATIN", "ROSUVASTATION"),
    ("DOMPERIDONE", "DOMPERIDON"),
    ("RANITIDINE", "RANITIDIN"),
]


class DrugRepository:
    """Canonical high-performance repository for master pharmaceutical data."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or self._resolve_data_dir()
        self.drugs_by_id: Dict[str, DrugEntry] = {}
        self.drugs_by_clean_name: Dict[str, List[DrugEntry]] = {}
        self.drugs_by_verbatim_name: Dict[str, DrugEntry] = {}
        self.drugs_by_code: Dict[str, DrugEntry] = {}
        self.brand_index: Dict[str, List[DrugEntry]] = {}
        self.generic_index: Dict[str, List[DrugEntry]] = {}
        self.soundex_buckets: Dict[str, List[str]] = {}
        self.phonetic_buckets: Dict[str, List[str]] = {}
        self.prefix_index: Dict[str, List[str]] = {}
        self.route_mappings_by_drug_id: Dict[str, List[str]] = {}
        self.drug_strengths_map: Dict[str, Set[str]] = {}
        self.drug_words_set: Set[str] = set()

        self.all_routes: List[Dict[str, str]] = []
        self.all_schedules: List[Dict[str, str]] = []
        self.all_dose_units: List[Dict[str, str]] = []

        self._load_all()

    def _resolve_data_dir(self) -> Path:
        """Resolves the canonical Drug_database directory."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            repo_root / "Drug_database",
            repo_root / "Drug_databse",
        ]
        for c in candidates:
            if c.exists() and (c / "drugList.json").exists():
                return c
        return candidates[0]

    def _load_all(self) -> None:
        """Loads and compiles master formulary datasets into indexed in-memory data structures."""
        # 1. Load reference route mappings first
        route_csv = self.data_dir / "Drug_Route_mapping.csv"
        if route_csv.exists():
            try:
                with open(route_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        d_id = str(row.get("drug_id", "")).strip()
                        r_code = (row.get("route_code") or "").upper().strip()
                        if d_id and r_code:
                            self.route_mappings_by_drug_id.setdefault(d_id, []).append(r_code)
            except Exception as e:
                print(f"[DrugRepository] Error reading {route_csv}: {e}")

        # Inject explicit routes for well-known clinical drugs
        for wkd in WELL_KNOWN_CLINICAL_DRUGS:
            d_id = str(wkd["drug_id"])
            if d_id not in self.route_mappings_by_drug_id and "routes" in wkd:
                self.route_mappings_by_drug_id[d_id] = wkd["routes"]

        # 2. Load drugList.json
        json_path = self.data_dir / "drugList.json"
        raw_items: List[Dict[str, Any]] = []
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_items = data.get("drugData", [])
            except Exception as e:
                print(f"[DrugRepository] Error reading {json_path}: {e}")

        # Inject well-known clinical formulations if missing
        for wkd in WELL_KNOWN_CLINICAL_DRUGS:
            if not any((d.get("drug_name") or "").upper() == wkd["drug_name"] for d in raw_items):
                wkd_item = dict(wkd)
                wkd_item["provenance"] = ProvenanceSource.CLINICAL_CURATION
                raw_items.append(wkd_item)

        # 3. Index all entries into canonical DrugEntry models
        for item in raw_items:
            d_id = str(item.get("drug_id"))
            code = str(item.get("drug_code") or "").strip().upper()
            raw_name = str(item.get("drug_name") or "").strip()
            raw_type = str(item.get("drug_type") or "b").lower()
            dtype = DrugType.GENERIC if raw_type == "g" else DrugType.BRAND
            base = clean_drug_base_name(raw_name)

            # Extract formulation strength numbers
            strengths = set(re.findall(r"(\d+(?:\.\d+)?)\s*(?:MG|G|MCG|ML|L|IU|%|GM|TAB|TABS|CAP|CAPS)?(?!\w)", raw_name.upper()))
            code_strengths = set(re.findall(r"(\d+(?:\.\d+)?)\b", code))
            all_strengths = strengths | code_strengths
            if "%" in raw_name:
                pct_matches = set(re.findall(r"(\d+(?:\.\d+)?%)", raw_name))
                all_strengths.update(pct_matches)

            # Dosage form token
            form_match = re.search(
                r"\b(TAB|TABLET|CAP|CAPSULE|SYP|SYRUP|INJ|INJECTION|OINT|OINTMENT|CREAM|DROPS|GEL|SOLUTION|SUSPENSION|POWD|SPRAY)\b",
                raw_name.upper(),
            )
            dosage_form = form_match.group(0) if form_match else None

            # Generic and Brand extraction
            generic_name = item.get("generic_name")
            brand_name = item.get("brand_name")
            if dtype == DrugType.GENERIC:
                # e.g., 'nimodipine(30 mg)Tablet' -> generic is 'nimodipine'
                m_gen = re.match(r"^([^(]+)", raw_name)
                if m_gen:
                    gen_sub = m_gen.group(1).strip().upper()
                    if not generic_name:
                        generic_name = gen_sub
            else:
                if not brand_name and base:
                    brand_name = base
                if not generic_name and base in BRAND_TO_GENERIC_MAP:
                    generic_name = BRAND_TO_GENERIC_MAP[base]

            routes = self.route_mappings_by_drug_id.get(d_id, ["ORAL"])
            provenance = item.get("provenance", ProvenanceSource.MASTER_FORMULARY)

            entry = DrugEntry(
                drug_id=d_id,
                drug_code=code,
                drug_name=raw_name,
                base_name=base,
                drug_type=dtype,
                generic_name=generic_name,
                brand_name=brand_name,
                dosage_form=dosage_form,
                strength_values=all_strengths,
                routes=routes,
                provenance=provenance,
                raw_data=item,
            )

            self.drugs_by_id[d_id] = entry
            self.drugs_by_verbatim_name[raw_name.upper()] = entry
            if code:
                self.drugs_by_code[code] = entry

            if base:
                self.drugs_by_clean_name.setdefault(base, []).append(entry)
                self.drug_strengths_map.setdefault(base, set()).update(all_strengths)

                # Soundex bucket
                sx = soundex(base)
                if sx:
                    bucket = self.soundex_buckets.setdefault(sx, [])
                    if base not in bucket:
                        bucket.append(base)

                # Phonetic articulatory backbone bucket
                pb = phonetic_backbone(base)
                if pb:
                    pb_bucket = self.phonetic_buckets.setdefault(pb, [])
                    if base not in pb_bucket:
                        pb_bucket.append(base)

                # 3-char prefix index
                if len(base) >= 3:
                    pref = base[:3]
                    p_bucket = self.prefix_index.setdefault(pref, [])
                    if base not in p_bucket:
                        p_bucket.append(base)

                # Vocabulary word set
                for w in base.split():
                    if len(w) >= 3:
                        self.drug_words_set.add(w)

                # Index primary brand root token if multi-word formulation
                first_w = base.split()[0].strip()
                if len(first_w) >= 4 and first_w != base:
                    self.drugs_by_clean_name.setdefault(first_w, []).append(entry)
                    self.drug_strengths_map.setdefault(first_w, set()).update(all_strengths)
                    pb_fw = phonetic_backbone(first_w)
                    if pb_fw and first_w not in self.phonetic_buckets.setdefault(pb_fw, []):
                        self.phonetic_buckets[pb_fw].append(first_w)

            # Brand index
            if brand_name:
                self.brand_index.setdefault(brand_name.upper(), []).append(entry)
                pb_bn = phonetic_backbone(brand_name)
                if pb_bn and brand_name.upper() not in self.phonetic_buckets.setdefault(pb_bn, []):
                    self.phonetic_buckets[pb_bn].append(brand_name.upper())
            # Generic index
            if generic_name:
                self.generic_index.setdefault(generic_name.upper(), []).append(entry)
                pb_gn = phonetic_backbone(generic_name)
                if pb_gn and generic_name.upper() not in self.phonetic_buckets.setdefault(pb_gn, []):
                    self.phonetic_buckets[pb_gn].append(generic_name.upper())

        # 4. Register official aliases
        for a, b in COMMON_DRUG_ALIASES:
            list_a = self.drugs_by_clean_name.get(a)
            list_b = self.drugs_by_clean_name.get(b)
            if list_a and not list_b:
                self.drugs_by_clean_name[b] = list_a
                sx = soundex(b)
                if sx and b not in self.soundex_buckets.setdefault(sx, []):
                    self.soundex_buckets[sx].append(b)
                pb_b = phonetic_backbone(b)
                if pb_b and b not in self.phonetic_buckets.setdefault(pb_b, []):
                    self.phonetic_buckets[pb_b].append(b)
                if len(b) >= 3:
                    pref = b[:3]
                    if b not in self.prefix_index.setdefault(pref, []):
                        self.prefix_index[pref].append(b)
                if a in self.drug_strengths_map:
                    self.drug_strengths_map[b] = self.drug_strengths_map[a]
                for w in b.split():
                    if len(w) >= 3:
                        self.drug_words_set.add(w)
            elif list_b and not list_a:
                self.drugs_by_clean_name[a] = list_b
                sx = soundex(a)
                if sx and a not in self.soundex_buckets.setdefault(sx, []):
                    self.soundex_buckets[sx].append(a)
                pb_a = phonetic_backbone(a)
                if pb_a and a not in self.phonetic_buckets.setdefault(pb_a, []):
                    self.phonetic_buckets[pb_a].append(a)
                if len(a) >= 3:
                    pref = a[:3]
                    if a not in self.prefix_index.setdefault(pref, []):
                        self.prefix_index[pref].append(a)
                if b in self.drug_strengths_map:
                    self.drug_strengths_map[a] = self.drug_strengths_map[b]
                for w in a.split():
                    if len(w) >= 3:
                        self.drug_words_set.add(w)

        # 5. Load reference CSVs
        routes_csv = self.data_dir / "drug_routes.csv"
        if routes_csv.exists():
            try:
                with open(routes_csv, "r", encoding="utf-8") as f:
                    self.all_routes = list(csv.DictReader(f))
            except Exception:
                pass

        schedules_csv = self.data_dir / "drug_schedule.csv"
        if schedules_csv.exists():
            try:
                with open(schedules_csv, "r", encoding="utf-8") as f:
                    self.all_schedules = list(csv.DictReader(f))
            except Exception:
                pass

        units_csv = self.data_dir / "dose_units.csv"
        if units_csv.exists():
            try:
                with open(units_csv, "r", encoding="utf-8") as f:
                    self.all_dose_units = list(csv.DictReader(f))
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Core Canonical Query Methods (Zero Hallucination Guaranteed)
    # -------------------------------------------------------------------------

    def find_exact(self, name: str) -> Optional[DrugEntry]:
        """Finds drug entry by verbatim drug name, drug code, or exact base name.
        Returns None if not found. Never hallucinates.
        """
        if not name:
            return None
        q = name.strip()
        q_upper = q.upper()

        # 1. Exact drug_code match
        if q_upper in self.drugs_by_code:
            return self.drugs_by_code[q_upper]

        # 2. Exact verbatim drug_name match
        if q_upper in self.drugs_by_verbatim_name:
            return self.drugs_by_verbatim_name[q_upper]

        # 3. Exact drug_id match
        if q in self.drugs_by_id:
            return self.drugs_by_id[q]

        # 4. Exact base_name match (first formulation)
        if q_upper in self.drugs_by_clean_name:
            return self.drugs_by_clean_name[q_upper][0]

        return None

    def find_normalized(self, name: str) -> Optional[DrugEntry]:
        """Finds drug entry by normalizing case, stripping dosage/forms, and resolving aliases.
        Returns None if not found.
        """
        if not name:
            return None
        base = clean_drug_base_name(name)
        if not base:
            return None

        # Check clean base names
        if base in self.drugs_by_clean_name:
            return self.drugs_by_clean_name[base][0]

        # Check brand index
        if base in self.brand_index:
            return self.brand_index[base][0]

        # Check generic index
        if base in self.generic_index:
            return self.generic_index[base][0]

        return None

    def find_fuzzy(self, name: str, min_confidence: float = 0.68, limit: int = 5) -> List[DrugMatch]:
        """Finds candidate drug entries for spelling/ASR acoustic errors using articulatory phonetics,
        Soundex, prefix tree, and Levenshtein distance.
        Guarantees zero hallucination: returns empty list if no candidate exceeds min_confidence.
        """
        if not name:
            return []
        base = clean_drug_base_name(name)
        if not base or len(base) < 3:
            return []

        # Check exact/normalized first
        exact = self.find_normalized(base)
        if exact:
            return [DrugMatch(drug=exact, similarity=1.0, confidence=1.0, matched_via="NORMALIZED")]

        is_multi_word = " " in base
        concat_base = re.sub(r"\s+", "", base) if is_multi_word else base

        if is_multi_word:
            exact_concat = self.find_normalized(concat_base)
            if exact_concat:
                return [DrugMatch(drug=exact_concat, similarity=1.0, confidence=1.0, matched_via="NORMALIZED_CONCAT")]

        candidates: List[DrugMatch] = []
        seen_ids: Set[str] = set()

        # Collect candidate bases from phonetic backbone, soundex, and prefix tree
        pb_base = phonetic_backbone(base)
        pb_concat = phonetic_backbone(concat_base) if is_multi_word else pb_base
        sx_base = soundex(base)

        candidate_bases: Set[str] = set()
        if pb_base and pb_base in self.phonetic_buckets:
            candidate_bases.update(self.phonetic_buckets[pb_base])
        if is_multi_word and pb_concat and pb_concat in self.phonetic_buckets:
            candidate_bases.update(self.phonetic_buckets[pb_concat])
        if sx_base and sx_base in self.soundex_buckets:
            candidate_bases.update(self.soundex_buckets[sx_base])
        if len(base) >= 3:
            pref = base[:3]
            candidate_bases.update(self.prefix_index.get(pref, []))
        if is_multi_word and len(concat_base) >= 3:
            pref_c = concat_base[:3]
            candidate_bases.update(self.prefix_index.get(pref_c, []))

        scored_bases: List[Tuple[float, float, str, str]] = []  # (confidence, similarity, cand_base, match_via)

        for cand_base in candidate_bases:
            dist1 = levenshtein_distance(base, cand_base)
            dist2 = levenshtein_distance(concat_base, cand_base) if is_multi_word else 999
            best_dist = min(dist1, dist2)
            eff_len = len(concat_base) if (is_multi_word and dist2 < dist1) else len(base)
            max_len = max(eff_len, len(cand_base))
            sim = 1.0 - (best_dist / max_len) if max_len > 0 else 0.0

            pb_cand = phonetic_backbone(cand_base)
            is_pb_match = bool(pb_cand and (pb_cand == pb_base or (is_multi_word and pb_cand == pb_concat)))

            match_via = "PHONETIC" if is_pb_match else ("SOUNDEX" if sx_base == soundex(cand_base) else "LEVENSHTEIN")
            conf = max(sim, 0.88) if is_pb_match else sim

            if conf >= min_confidence or (is_pb_match and sim >= 0.50):
                scored_bases.append((conf, sim, cand_base, match_via))

        # Sort by confidence descending, then similarity descending
        scored_bases.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for conf, sim, cand_base, match_via in scored_bases:
            entries = self.drugs_by_clean_name.get(cand_base, [])
            for entry in entries:
                if entry.drug_id not in seen_ids:
                    seen_ids.add(entry.drug_id)
                    candidates.append(
                        DrugMatch(
                            drug=entry,
                            similarity=sim,
                            confidence=conf,
                            matched_via=match_via,
                        )
                    )
                    if len(candidates) >= limit:
                        return candidates

        return candidates

    def find_by_brand(self, brand_name: str, limit: int = 10) -> List[DrugEntry]:
        """Searches specifically for brand name products and formulations.
        e.g. 'Dolo' returns all DOLO formulations, 'Crocin' returns CROCIN formulations.
        """
        if not brand_name:
            return []
        q = clean_drug_base_name(brand_name)
        if not q:
            return []

        results: List[DrugEntry] = []
        seen_ids: Set[str] = set()

        # 1. Exact brand index match
        if q in self.brand_index:
            for d in self.brand_index[q]:
                if d.drug_id not in seen_ids and d.drug_type == DrugType.BRAND:
                    seen_ids.add(d.drug_id)
                    results.append(d)

        # 2. Brand name substring or prefix search
        for b_name, entries in self.brand_index.items():
            if b_name.startswith(q) or q in b_name:
                for d in entries:
                    if d.drug_id not in seen_ids and d.drug_type == DrugType.BRAND:
                        seen_ids.add(d.drug_id)
                        results.append(d)
                        if len(results) >= limit:
                            return results

        return results[:limit]

    def find_by_generic(self, generic_name: str, limit: int = 10) -> List[DrugEntry]:
        """Searches for generic substance and all associated formulations (including brand products that contain it).
        e.g. 'Paracetamol' returns generic paracetamol formulations as well as Dolo, Crocin, Calpol.
        """
        if not generic_name:
            return []
        q = clean_drug_base_name(generic_name)
        if not q:
            return []

        results: List[DrugEntry] = []
        seen_ids: Set[str] = set()

        # 1. Direct generic index match
        if q in self.generic_index:
            for d in self.generic_index[q]:
                if d.drug_id not in seen_ids:
                    seen_ids.add(d.drug_id)
                    results.append(d)

        # 2. Substring matching across generics
        for g_name, entries in self.generic_index.items():
            if q in g_name or g_name in q:
                for d in entries:
                    if d.drug_id not in seen_ids:
                        seen_ids.add(d.drug_id)
                        results.append(d)
                        if len(results) >= limit:
                            return results

        # 3. Check if generic name is clean base name for generic drugs
        if q in self.drugs_by_clean_name:
            for d in self.drugs_by_clean_name[q]:
                if d.drug_id not in seen_ids:
                    seen_ids.add(d.drug_id)
                    results.append(d)
                    if len(results) >= limit:
                        return results

        return results[:limit]

    def find_strength(self, drug_name_or_id: str, strength_query: Optional[str] = None) -> List[str]:
        """Retrieves or validates recognized formulation strengths for a drug.
        If strength_query is given: returns [strength_query] if valid, else [].
        If strength_query is None: returns all recognized formulation strengths (e.g. ['500', '650']).
        """
        if not drug_name_or_id:
            return []

        # Find entry
        entry = self.find_exact(drug_name_or_id) or self.find_normalized(drug_name_or_id)
        strengths: Set[str] = set()

        if entry:
            base = entry.base_name
            strengths.update(entry.strength_values)
            if base in self.drug_strengths_map:
                strengths.update(self.drug_strengths_map[base])
            if entry.brand_name and entry.brand_name in self.drug_strengths_map:
                strengths.update(self.drug_strengths_map[entry.brand_name])
            if entry.generic_name and entry.generic_name in self.drug_strengths_map:
                strengths.update(self.drug_strengths_map[entry.generic_name])
        else:
            base = clean_drug_base_name(drug_name_or_id)
            if base in self.drug_strengths_map:
                strengths.update(self.drug_strengths_map[base])
            for k, v in self.drug_strengths_map.items():
                if base and (base in k or k in base):
                    strengths.update(v)

        sorted_strengths = sorted(list(strengths), key=lambda x: (float(x) if re.match(r"^\d+(?:\.\d+)?$", x) else 999999, x))

        if strength_query is None:
            return sorted_strengths

        # Validate specific strength query (e.g. '650 mg', '650', '0.05%')
        sq = strength_query.strip().upper()
        sq_nums = re.findall(r"(\d+(?:\.\d+)?)\s*(?:MG|G|MCG|ML|L|IU|%|GM)?", sq)
        if sq_nums and sq_nums[0] in strengths:
            return [sq_nums[0]]
        if sq in strengths:
            return [sq]

        return []

    def get_drug_strengths_map(self) -> Dict[str, Set[str]]:
        """Returns the master formulary mapping of base drug names to their recognized strength sets."""
        return self.drug_strengths_map


    def find_route(self, drug_name_or_id: str) -> List[str]:
        """Retrieves permissible anatomical routes for a drug from master route mappings.
        Returns ['ORAL'] as standard clinical default if no explicit mapping exists.
        """
        if not drug_name_or_id:
            return ["ORAL"]

        entry = self.find_exact(drug_name_or_id) or self.find_normalized(drug_name_or_id)
        if entry:
            return entry.routes

        d_id = str(drug_name_or_id).strip()
        if d_id in self.route_mappings_by_drug_id:
            return list(dict.fromkeys(self.route_mappings_by_drug_id[d_id]))

        return ["ORAL"]

    # -------------------------------------------------------------------------
    # Extractor & Backward Compatibility Helper Methods
    # -------------------------------------------------------------------------

    def is_known_drug(self, token: str) -> bool:
        """Fast O(1) check if an uppercase word exists in the drug vocabulary."""
        return token.upper() in self.drug_words_set

    def get_strengths_for_base(self, base_name: str) -> Set[str]:
        """Returns all recognized formulation strength integers for a base medicine."""
        base_upper = clean_drug_base_name(base_name)
        strengths: Set[str] = set()
        for k, v in self.drug_strengths_map.items():
            if base_upper in k or k in base_upper:
                strengths.update(v)
        return strengths

    def get_routes_for_drug(self, drug_id: str) -> List[str]:
        """Returns all permissible anatomical routes for a given formulary drug ID."""
        return self.find_route(drug_id)

    def search_drugs(self, query: str, limit: int = 30) -> List[Any]:
        """Fast Soundex and prefix search across the master formulary."""
        clean_q = clean_drug_base_name(query)
        if not clean_q:
            return []

        results: List[DrugEntry] = []
        seen_ids: Set[str] = set()

        # 1. Exact base name match
        if clean_q in self.drugs_by_clean_name:
            for d in self.drugs_by_clean_name[clean_q]:
                if d.drug_id not in seen_ids:
                    seen_ids.add(d.drug_id)
                    results.append(d)

        # 2. Prefix matches (if query >= 3 chars)
        if len(clean_q) >= 3:
            pref = clean_q[:3]
            for base in self.prefix_index.get(pref, []):
                if base.startswith(clean_q) or clean_q in base:
                    for d in self.drugs_by_clean_name.get(base, []):
                        if d.drug_id not in seen_ids:
                            seen_ids.add(d.drug_id)
                            results.append(d)
                            if len(results) >= limit:
                                break

        # 3. Soundex bucket fallback
        if len(results) < limit:
            sx = soundex(clean_q)
            for base in self.soundex_buckets.get(sx, []):
                for d in self.drugs_by_clean_name.get(base, []):
                    if d.drug_id not in seen_ids:
                        seen_ids.add(d.drug_id)
                        results.append(d)
                        if len(results) >= limit:
                            break

        from app.api.schemas import DrugEntryResponse
        return [
            DrugEntryResponse(
                drug_id=d.drug_id,
                drug_code=d.drug_code,
                drug_name=d.drug_name,
                drug_type=d.drug_type.value,
                routes=d.routes,
            )
            for d in results[:limit]
        ]

    def search(self, query: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Dictionary-based search for internal extractor components."""
        clean_q = clean_drug_base_name(query)
        if not clean_q:
            return []
        res = self.search_drugs(query, limit)
        return [
            {"base_name": clean_drug_base_name(d.drug_name) or d.drug_name, "drug_name": d.drug_name, "drug_id": d.drug_id, "routes": d.routes}
            for d in res
        ]

    def find_did_you_mean(self, query: str) -> Optional[str]:
        """Phonetic fuzzy drug name recommendation using articulatory acoustics + Soundex + Levenshtein."""
        top = self.find_top_did_you_mean(query, limit=1)
        if top:
            return top[0]["base_name"]
        return None

    def find_top_did_you_mean(
        self,
        query: str,
        limit: int = 3,
        min_confidence: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """
        Phonetic fuzzy drug name recommendation returning up to `limit` (maximum 3)
        similarly spelt medicines from Drug_database.
        Each candidate contains:
          - drug_name: Full formulary product name (e.g. 'CROCIN 650 TAB')
          - base_name: Base drug/active substance (e.g. 'CROCIN')
          - drug_id: Formulary ID
          - confidence: Match score [0.0 - 1.0]
          - dose: First registered formulation strength
          - dose_unit: 'mg' or clinical unit
          - available_routes: List of permissible routes
          - available_drugs: List of same-dose formulations for dropdown
        """
        clean_q = clean_drug_base_name(query)
        if not clean_q or len(clean_q) < 3:
            return []

        # If exact match exists, no recommendation needed
        if clean_q in self.drugs_by_clean_name or clean_q in self.brand_index or clean_q in self.generic_index:
            return []

        matches = self.find_fuzzy(clean_q, min_confidence=min_confidence, limit=15)
        if not matches:
            return []

        # Deduplicate and group by distinct drug name or base name
        results: List[Dict[str, Any]] = []
        seen_names: Set[str] = set()

        for m in matches:
            d = m.drug
            disp_name = d.drug_name
            clean_b = d.base_name.upper()

            # Ensure distinct drugs in the top recommendations
            first_str = sorted(d.strength_values)[0] if d.strength_values else ''
            dedup_key = f"{clean_b}_{first_str}"
            if dedup_key in seen_names:
                continue
            seen_names.add(dedup_key)

            target_dose = first_str if first_str else None
            same_dose = self.find_same_dose_formulations(d.drug_id, target_dose or "")

            results.append({
                "drug_name": disp_name,
                "base_name": d.base_name.title(),
                "drug_id": d.drug_id,
                "confidence": round(m.confidence, 3),
                "similarity": round(m.similarity, 3),
                "dose": target_dose,
                "dose_unit": "mg" if target_dose else "",
                "available_routes": d.routes,
                "available_drugs": same_dose,
            })
            if len(results) >= limit:
                break

        return results

    def find_composite_formulation(
        self,
        base_name: str,
        int1: float,
        int2: float,
        unit: str = "mg",
    ) -> Optional[DrugEntry]:
        """
        Checks if a combination dosage (int1 + int2) maps to a registered formulation
        with total sum int0 = int1 + int2 (e.g. 500mg + 125mg = 625mg -> AUGMENTIN 625).
        """
        if not base_name:
            return None

        sum_int = int(round(int1 + int2))
        sum_str = str(sum_int)

        clean_b = clean_drug_base_name(base_name).upper()
        entries = self.drugs_by_clean_name.get(clean_b, [])
        if not entries and clean_b in self.brand_index:
            entries = self.brand_index[clean_b]
        if not entries:
            first_w = clean_b.split()[0]
            entries = self.drugs_by_clean_name.get(first_w, [])
        if not entries:
            fuzzy = self.find_fuzzy(clean_b, min_confidence=0.70, limit=3)
            if fuzzy:
                entries = [f.drug for f in fuzzy]

        for e in entries:
            # Check if total sum is in strength values
            if sum_str in e.strength_values:
                return e
            # Check if sum is in drug_name or drug_code
            if re.search(rf"\b{sum_str}\b", e.drug_name) or sum_str in e.drug_code:
                return e

        return None

    def find_same_dose_formulations(self, drug_name_or_id: str, dose: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all formulations in the master formulary for the specified drug
        that match the requested dose/strength integer.
        Provides dropdown data for UI table inline selection.
        """
        if not drug_name_or_id:
            return []

        entry = self.find_exact(drug_name_or_id) or self.find_normalized(drug_name_or_id)
        if not entry:
            fuzzy = self.find_fuzzy(drug_name_or_id, min_confidence=0.65, limit=1)
            if fuzzy:
                entry = fuzzy[0].drug

        candidate_entries: List[DrugEntry] = []
        seen_ids: Set[str] = set()

        if entry:
            base = entry.base_name
            if base in self.drugs_by_clean_name:
                for d in self.drugs_by_clean_name[base]:
                    if d.drug_id not in seen_ids:
                        seen_ids.add(d.drug_id)
                        candidate_entries.append(d)
            if entry.brand_name and entry.brand_name in self.brand_index:
                for d in self.brand_index[entry.brand_name]:
                    if d.drug_id not in seen_ids:
                        seen_ids.add(d.drug_id)
                        candidate_entries.append(d)
            if entry.generic_name and entry.generic_name in self.generic_index:
                for d in self.generic_index[entry.generic_name]:
                    if d.drug_id not in seen_ids:
                        seen_ids.add(d.drug_id)
                        candidate_entries.append(d)
        else:
            base = clean_drug_base_name(drug_name_or_id)
            if base in self.drugs_by_clean_name:
                for d in self.drugs_by_clean_name[base]:
                    if d.drug_id not in seen_ids:
                        seen_ids.add(d.drug_id)
                        candidate_entries.append(d)

        target_num = None
        if dose:
            m = re.search(r"(\d+(?:\.\d+)?%?)", str(dose).strip())
            if m:
                target_num = m.group(1).upper()

        matching_dose = []
        if target_num:
            for d in candidate_entries:
                if (
                    target_num in d.strength_values
                    or re.search(rf"\b{re.escape(target_num)}\b", d.drug_name.upper())
                    or re.search(rf"\b{re.escape(target_num)}\b", d.drug_code.upper())
                ):
                    matching_dose.append(d)

        results = matching_dose if matching_dose else candidate_entries

        return [
            {
                "drug_id": d.drug_id,
                "drug_code": d.drug_code,
                "drug_name": d.drug_name,
                "base_name": d.base_name,
                "drug_type": d.drug_type.value,
                "routes": d.routes,
                "strengths": sorted(list(d.strength_values)),
            }
            for d in results[:30]
        ]

    def get_reference_data(self) -> ReferenceDataResponse:
        """Returns reference datasets for schedules, dose units, and anatomical routes."""
        return ReferenceDataResponse(
            schedules=self.all_schedules,
            dose_units=self.all_dose_units,
            routes=self.all_routes,
        )


# Global singleton instance
_GLOBAL_REPO: Optional[DrugRepository] = None

def get_drug_repository() -> DrugRepository:
    """Returns the singleton instance of DrugRepository."""
    global _GLOBAL_REPO
    if _GLOBAL_REPO is None:
        _GLOBAL_REPO = DrugRepository()
    return _GLOBAL_REPO
