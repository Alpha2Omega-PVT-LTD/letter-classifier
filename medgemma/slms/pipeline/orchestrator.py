from typing import Dict, List, Any
from utils.logger import logger
from pipeline.extractor import MedicalEntityExtractor, ExtractedEntity
from pipeline.classifier import EntityClassifier
from pipeline.status import StatusClassifier
from pipeline.validator import Validator
from pipeline.snomed import SNOMEDMapper
from pipeline.letter import LetterClassifier

class ClinicalPipeline:
    """
    Clinical Extraction Pipeline.
    Orchestrates the 6-stage clinical letter processing:
    1. Extract Entities (Qwen via Ollama)
    2. Classify Categories (Diagnosis, Symptom, Procedure, Medication, Reject, Unknown)
    3. Classify Statuses
    4. Validate Entities (Reject inferred concepts)
    5. Map SNOMED CT Codes (FHIR API mapping)
    6. Formats output for writing (delegated to ExcelWriter)
    """
    def __init__(self):
        self.extractor = MedicalEntityExtractor()
        self.classifier = EntityClassifier()
        self.status_classifier = StatusClassifier()
        self.validator = Validator()
        self.snomed_mapper = SNOMEDMapper()
        self.letter_classifier = LetterClassifier()

    def process_letter(self, clinical_text: str) -> Dict[str, Any]:
        """Runs the entire pipeline on a single clinical letter."""
        logger.info("--- Beginning Processing for Clinical Letter ---")
        
        # Classify Letter Type
        letter_type = self.letter_classifier.classify(clinical_text)
        
        # Step 1: Medical Entity Extraction
        raw_entities = self.extractor.extract(clinical_text)
        
        diagnoses = []
        symptoms = []
        procedures = []
        medications = []
        vitals = []

        for entity in raw_entities:
            # Vitals Special Handling
            if entity.provisional_type.lower() == "vital":
                logger.info(f"Vitals Handling: Bypassing classifiers for Vital '{entity.entity}'")
                vitals.append({
                    "entity": entity.entity,
                    "value": entity.evidence
                })
                continue

            # Step 2: Entity Classification
            category = self.classifier.classify(entity)
            if category.lower() in ("reject", "unknown"):
                logger.warning(f"Step 2: Entity '{entity.entity}' classified as '{category}'. Rejecting.")
                continue

            # Step 3: Status Classification
            status = self.status_classifier.classify_status(entity, category)

            # Step 4: Validation
            is_valid = self.validator.validate(entity, category, status)
            if not is_valid:
                logger.warning(f"Step 4: Entity '{entity.entity}' failed validation and was rejected.")
                continue

            # Step 5: SNOMED Mapping & Grouping
            if category == "Diagnosis":
                snomed_code = self.snomed_mapper.get_snomed_code(entity.entity, category, status)
                diagnoses.append({
                    "entity": entity.entity,
                    "status": status,
                    "snomed": snomed_code
                })
            elif category == "Symptom":
                snomed_code = self.snomed_mapper.get_snomed_code(entity.entity, category, status)
                symptoms.append({
                    "entity": entity.entity,
                    "status": status,
                    "snomed": snomed_code
                })
            elif category == "Procedure":
                snomed_code = self.snomed_mapper.get_snomed_code(entity.entity, category, status)
                procedures.append({
                    "entity": entity.entity,
                    "status": status,
                    "snomed": snomed_code
                })
            elif category == "Medication":
                medications.append({
                    "entity": entity.entity,
                    "status": status
                })
            else:
                logger.warning(f"Pipeline: Entity '{entity.entity}' has unhandled validated category '{category}'.")

        logger.info(
            f"Letter processing complete. Extracted: "
            f"{len(diagnoses)} Diagnoses, {len(symptoms)} Symptoms, "
            f"{len(procedures)} Procedures, {len(medications)} Medications, {len(vitals)} Vitals."
        )

        return {
            "letter_type": letter_type,
            "diagnoses": diagnoses,
            "symptoms": symptoms,
            "procedures": procedures,
            "medications": medications,
            "vitals": vitals
        }
