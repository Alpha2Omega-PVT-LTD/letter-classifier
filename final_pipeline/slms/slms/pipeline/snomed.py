import os
import re
import json
import requests
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Any
from config import (
    FHIR_BASE_URLS, SNOMED_CT_SYSTEM_URI, CACHE_PATH, LOG_PATH,
    MIN_ACCEPT_SCORE, ECL_BY_CATEGORY, ABBREVIATIONS
)
from utils.logger import logger

class SNOMEDMapper:
    """
    Step 5: SNOMED Mapper.
    Maps validated entities of type Diagnosis, Symptom, or Procedure to SNOMED CT codes.
    Uses local cache and remote SNOMED CT FHIR APIs.
    """
    _HISTORY_OF_RE = re.compile(r"\bH/O\b|\bhistory of\b", re.IGNORECASE)
    _DOSE_UNIT_RE = re.compile(
        r"\b\d+(\.\d+)?\s*(mg|microgram|mcg|g|ml|units?|iu)\b", re.IGNORECASE
    )

    def __init__(self):
        self.cache = self._load_cache()
        # Ensure log directory exists
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load SNOMED cache: {e}")
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save SNOMED cache: {e}")

    def expand_abbreviations(self, text: str) -> str:
        for abbr, full in ABBREVIATIONS.items():
            pattern = r"\b" + re.escape(abbr) + r"\b"
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        return text

    def _strip_dose_and_units(self, text: str) -> str:
        return self._DOSE_UNIT_RE.sub(" ", text).strip()

    def _log_rejection(self, term: str, category: str, best_candidate: Optional[Dict[str, Any]], reason: str) -> None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "term": term,
                    "category": category,
                    "best_candidate": best_candidate,
                    "reason": reason,
                }) + "\n")
        except OSError as e:
            logger.error(f"Failed to write rejection log: {e}")

    def get_snomed_possible_codes_for_term(self, search_term: str, category: str = None) -> Optional[List[Dict[str, Any]]]:
        search_term = self.expand_abbreviations(search_term)
        headers = {"Accept": "application/fhir+json"}
        ecl = ECL_BY_CATEGORY.get((category or "").lower())

        for fhir_base in FHIR_BASE_URLS:
            # 1. Try with ECL constraint if available
            if ecl:
                params = {
                    "url": f"{SNOMED_CT_SYSTEM_URI}?fhir_vs=ecl/{ecl}",
                    "filter": search_term,
                    "count": 20
                }
                try:
                    response = requests.get(f"{fhir_base}/ValueSet/$expand", params=params, headers=headers, timeout=4)
                    if response.status_code == 200:
                        data = response.json()
                        possible_concepts = []
                        if data and "expansion" in data and "contains" in data["expansion"]:
                            for concept in data["expansion"]["contains"]:
                                code = concept.get("code")
                                display = concept.get("display")
                                if code and display:
                                    possible_concepts.append({
                                        "snomed_code": code,
                                        "snomed_display": display,
                                        "snomed_system": concept.get("system"),
                                    })
                        if possible_concepts:
                            return possible_concepts
                except Exception as e:
                    logger.debug(f"ECL lookup failed on FHIR base {fhir_base}: {e}")

            # 2. Try unconstrained fallback on the same server
            params_fallback = {
                "url": f"{SNOMED_CT_SYSTEM_URI}?fhir_vs",
                "filter": search_term,
                "count": 20
            }
            try:
                response = requests.get(f"{fhir_base}/ValueSet/$expand", params=params_fallback, headers=headers, timeout=4)
                if response.status_code == 200:
                    data = response.json()
                    possible_concepts = []
                    if data and "expansion" in data and "contains" in data["expansion"]:
                        for concept in data["expansion"]["contains"]:
                            code = concept.get("code")
                            display = concept.get("display")
                            if code and display:
                                possible_concepts.append({
                                    "snomed_code": code,
                                    "snomed_display": display,
                                    "snomed_system": concept.get("system"),
                                    "_unconstrained_fallback": True if ecl else False,
                                })
                    if possible_concepts:
                        return possible_concepts
            except Exception as e:
                logger.debug(f"Unconstrained lookup failed on FHIR base {fhir_base}: {e}")

        return None

    def select_appropriate_snomed_code(
        self, search_term: str, possible_codes: List[Dict[str, Any]],
        category: str = None, status: str = None
    ) -> Optional[Dict[str, Any]]:
        if not possible_codes:
            return None

        search_term = self.expand_abbreviations(search_term)
        search_core = self._strip_dose_and_units(search_term).lower()
        allow_historical = (status or "").strip().lower() == "historical"

        scored_codes = []
        for concept in possible_codes:
            display = concept.get("snomed_display", "")
            display_lower = display.lower()

            if self._HISTORY_OF_RE.search(display) and not allow_historical:
                continue

            display_core = self._strip_dose_and_units(display_lower.split(" (")[0])
            score = SequenceMatcher(None, search_core, display_core).ratio()

            if concept.get("_unconstrained_fallback"):
                score -= 0.05

            semantic_tag = display_lower.rsplit("(", 1)[-1].rstrip(")") if "(" in display_lower else ""
            if category == "procedure" and semantic_tag == "procedure":
                score += 0.15
            elif category == "medication" and semantic_tag in (
                "medicinal product", "clinical drug", "product", "substance"
            ):
                score += 0.15
            elif category in ("diagnosis",) and semantic_tag == "disorder":
                score += 0.1
            elif category in ("symptom",) and semantic_tag in ("finding", "disorder"):
                score += 0.1
            elif semantic_tag not in (
                "procedure", "medicinal product", "clinical drug", "product",
                "substance", "disorder", "finding",
            ) and category in ECL_BY_CATEGORY:
                score -= 0.25

            scored_codes.append({"score": score, **concept})

        if not scored_codes:
            return None

        best_match = max(scored_codes, key=lambda x: x["score"])
        if best_match["score"] > MIN_ACCEPT_SCORE:
            return best_match

        self._log_rejection(search_term, category, best_match, "below_threshold")
        return None

    def get_snomed_code(self, term: str, category: str = None, status: str = None) -> str:
        """
        Runs the full lookup process, first checking the local cache,
        then querying FHIR servers, and finally scoring/persisting matches.
        """
        if not term or term.strip().lower() in ("none", "unknown", ""):
            return "N/A"
        if term.startswith("Local processing error"):
            return "N/A"

        cache_key = f"{category or ''}::{status or ''}::{term.strip().lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        logger.info(f"Step 5: Querying SNOMED CT for '{term}' (category: '{category}', status: '{status}')...")
        possible_codes = self.get_snomed_possible_codes_for_term(term.strip(), category)
        if possible_codes is None:
            logger.warning(f"Step 5: SNOMED API unreachable or no codes found for '{term}'")
            result = "API unreachable / Not found"
            self.cache[cache_key] = result
            self._save_cache()
            return result

        selected = self.select_appropriate_snomed_code(term.strip(), possible_codes, category, status)
        if selected:
            result = f"{selected['snomed_code']} | {selected['snomed_display']}"
            logger.info(f"Step 5: Successfully mapped '{term}' to SNOMED: {result}")
        else:
            result = "Not found"
            logger.warning(f"Step 5: SNOMED mapping not found for '{term}' (below score threshold)")

        self.cache[cache_key] = result
        self._save_cache()
        return result
