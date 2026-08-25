import os
import ollama
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from config import MODEL_NAME, TEMPERATURE, NUM_CTX, PROMPTS_DIR
from utils.json_parser import parse_and_validate_json
from utils.retry import retry_with_backoff
from utils.logger import logger

class ExtractedEntity(BaseModel):
    id: str
    entity: str
    provisional_type: Literal["Diagnosis", "Symptom", "Procedure", "Medication", "Vital", "Unknown"]
    evidence: str
    sentence: str
    confidence: float

class Step1Output(BaseModel):
    entities: List[ExtractedEntity]

class MedicalEntityExtractor:
    """
    Step 1: Medical Entity Extractor.
    Extracts raw medical entities with provisional types from the clinical text.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.prompt_path = os.path.join(PROMPTS_DIR, "step1_extraction.txt")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _call_llm(self, clinical_text: str) -> Optional[Step1Output]:
        prompt = self.prompt_template.replace("{clinical_text}", clinical_text)
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={
                    'temperature': TEMPERATURE,
                    'num_ctx': NUM_CTX,
                }
            )
            raw_content = response['message']['content']
            return parse_and_validate_json(raw_content, Step1Output)
        except Exception as e:
            logger.error(f"Error calling Ollama in MedicalEntityExtractor: {e}")
            return None

    def extract(self, clinical_text: str) -> List[ExtractedEntity]:
        """Runs the extraction with retry logic and returns a list of ExtractedEntity objects."""
        logger.info("Step 1: Starting Medical Entity Extraction...")
        result = retry_with_backoff(lambda: self._call_llm(clinical_text), retries=3)
        if result and result.entities:
            logger.info(f"Step 1: Successfully extracted {len(result.entities)} entities.")
            return result.entities
        logger.warning("Step 1: No entities extracted or extraction failed.")
        return []
